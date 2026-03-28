"""Squat analysis pipeline — processes frames, renders overlay, returns annotated JPEG."""
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from squat_coach.utils.enums import Phase
from squat_coach.pose.mediapipe_blazepose3d import MediaPipeBlazePose3D
from squat_coach.preprocessing.smoothing import EMALandmarkSmoother
from squat_coach.preprocessing.normalization import normalize_to_hip_center
from squat_coach.preprocessing.calibration import Calibrator
from squat_coach.preprocessing.sequence_buffer import SequenceBuffer
from squat_coach.biomechanics.squat_features import SquatFeatureExtractor
from squat_coach.models.inference_manager import InferenceManager
from squat_coach.phases.phase_detector import PhaseDetector
from squat_coach.phases.rep_segmenter import RepSegmenter
from squat_coach.faults.evidence_engine import EvidenceEngine
from squat_coach.scoring.ideal_reference import build_ideal_reference
from squat_coach.scoring.score_components import (
    compute_depth_score, compute_trunk_control_score,
    compute_posture_stability_score, compute_movement_consistency_score,
)
from squat_coach.scoring.score_fusion import compute_rep_quality_score
from squat_coach.scoring.rationale import build_rationale
from squat_coach.scoring.trend_analysis import TrendTracker
from squat_coach.events.event_builder import build_rep_summary
from squat_coach.events.gemini_payloads import (
    format_gemini_payload, send_to_gemini_async, _get_client,
)
from squat_coach.events.coaching_priority import CoachingPrioritizer
from squat_coach.rendering.draw_pose import draw_skeleton
from squat_coach.session.session_state import SessionState
from squat_coach.server.protocol import FrameResult, RepData, CalibrationMessage

import squat_coach.models.temporal_tcn   # noqa: F401
import squat_coach.models.temporal_gru   # noqa: F401

logger = logging.getLogger("squat_coach.pipeline")


class SquatCoachPipeline:
    """Headless squat analysis — receives frames, returns structured results."""

    def __init__(self, config_dir: str = "squat_coach/config") -> None:
        self._config = self._load_config(f"{config_dir}/default.yaml")
        self._model_config = self._load_config(f"{config_dir}/model.yaml")
        self._scoring_config = self._load_config(f"{config_dir}/scoring.yaml")

        self._pose = MediaPipeBlazePose3D()
        self._smoother = EMALandmarkSmoother(
            alpha=self._config["preprocessing"]["ema_alpha"]
        )
        self._calibrator = Calibrator(
            num_frames=int(self._config["preprocessing"]["calibration_duration_s"] * 24)
        )
        self._session = SessionState()
        self._fault_engine = EvidenceEngine()
        self._coach = CoachingPrioritizer()
        self._trend_tracker = TrendTracker()
        self._seq_buf = SequenceBuffer(
            seq_len=self._model_config["sequence"]["length"],
            feature_dim=self._model_config["sequence"]["feature_dim"],
        )

        self._feature_extractor = None
        self._inference_mgr = None
        self._phase_detector = None
        self._rep_segmenter = None
        self._ideal_ref = None
        self._cal_result = None

        self._seq = 0

        self._rep_min_knee = 180.0
        self._rep_max_torso = 0.0
        self._rep_max_head_offset = 0.0
        self._rep_features_snapshot: dict = {}
        self._live_score_ema = 50.0

        self._all_rep_scores: list[float] = []
        self._last_rep_score = 0.0
        self._best_rep_score = 0.0

        self._last_cue = ""
        self._last_cue_time = 0.0
        self._cue_display_duration = 5.0

        self._pending_coaching: str | None = None
        self._coaching_lock = threading.Lock()

        gemini_cfg = self._config.get("gemini", {})
        if gemini_cfg.get("enabled", False):
            key = gemini_cfg.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if key:
                _get_client(key)

    @property
    def is_calibrated(self) -> bool:
        return self._session.is_calibrated

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        seq = self._seq
        self._seq += 1
        self._session.frame_index += 1

        pose_result = self._pose.estimate(frame, timestamp)

        if not pose_result.detected:
            self._session.dropped_frame_count += 1
            if not self._session.is_calibrated:
                progress = self._calibrator.frame_count / max(self._calibrator.num_frames, 1)
                return FrameResult(
                    seq=seq, timestamp=timestamp,
                    calibration=CalibrationMessage(status="in_progress", progress=progress),
                )
            # No pose detected but calibrated — still send the raw frame
            _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return FrameResult(seq=seq, timestamp=timestamp, rendered_jpeg=jpeg_buf.tobytes())

        self._session.dropped_frame_count = 0

        smoothed = self._smoother.smooth(pose_result.world_landmarks)
        normalized = normalize_to_hip_center(smoothed)

        if not self._session.is_calibrated:
            self._calibrator.add_frame(pose_result)
            progress = self._calibrator.frame_count / max(self._calibrator.num_frames, 1)

            if self._calibrator.is_ready:
                self._cal_result = self._calibrator.compute()
                if self._cal_result:
                    self._session.is_calibrated = True
                    self._session.view_type = self._cal_result.view_type
                    self._init_post_calibration()

                    return FrameResult(
                        seq=seq, timestamp=timestamp,
                        calibration=CalibrationMessage(
                            status="complete", progress=1.0,
                            view_type=self._cal_result.view_type.value,
                        ),
                    )

            return FrameResult(
                seq=seq, timestamp=timestamp,
                calibration=CalibrationMessage(status="in_progress", progress=progress),
            )

        features = self._feature_extractor.extract(normalized, pose_result.visibility)
        model_features = features["model_features"]
        self._seq_buf.push(model_features)

        fused = None
        if self._seq_buf.is_ready and self._inference_mgr and self._inference_mgr.has_models:
            fused = self._inference_mgr.infer(self._seq_buf.get_sequence())

        img_lm = pose_result.image_landmarks
        hip_y_img = float((img_lm[23][1] + img_lm[24][1]) / 2.0)
        knee_angle = features.get("primary_knee_angle", 170.0)

        if fused is not None:
            phase = self._phase_detector.detect(fused.phase_probs, hip_y_img, knee_angle)
        else:
            dummy_probs = np.array([0.25, 0.25, 0.25, 0.25])
            phase = self._phase_detector.detect(dummy_probs, hip_y_img, knee_angle)

        self._session.current_phase = phase

        if phase in (Phase.DESCENT, Phase.BOTTOM):
            self._rep_min_knee = min(self._rep_min_knee, features.get("primary_knee_angle", 180))
            self._rep_max_torso = max(self._rep_max_torso, features.get("torso_inclination_deg", 0))
            self._rep_max_head_offset = max(self._rep_max_head_offset, features.get("head_to_trunk_offset", 0))
            self._rep_features_snapshot = dict(features)

        if phase != Phase.TOP:
            target = self._ideal_ref.target_knee_angle if self._ideal_ref else 90
            best_depth_score = compute_depth_score(self._rep_min_knee, target)
            if best_depth_score > self._live_score_ema:
                self._live_score_ema = best_depth_score
            self._session.current_score = self._live_score_ema

        fault_config = self._scoring_config.get("faults", {}).get("thresholds", {})
        fault_config["baseline_torso_angle"] = self._ideal_ref.trunk_neutral_angle if self._ideal_ref else 10.0
        faults = self._fault_engine.evaluate(features, fault_config)

        new_cue = self._coach.select_cue(faults)
        now_cue = time.monotonic()
        if new_cue:
            self._last_cue = new_cue
            self._last_cue_time = now_cue
            self._session.current_cue = new_cue
        elif now_cue - self._last_cue_time < self._cue_display_duration:
            self._session.current_cue = self._last_cue
        else:
            self._session.current_cue = ""

        # Render only skeleton on the frame (UI handles metrics/text)
        annotated = frame.copy()
        draw_skeleton(annotated, pose_result.image_landmarks, pose_result.visibility)

        # Encode annotated frame as JPEG
        _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        rendered_jpeg = jpeg_buf.tobytes()

        result = FrameResult(
            seq=seq,
            timestamp=timestamp,
            phase=phase.value.upper(),
            knee_angle=features.get("primary_knee_angle"),
            hip_angle=features.get("primary_hip_angle"),
            torso_angle=features.get("torso_inclination_deg"),
            score=self._session.current_score,
            confidence=pose_result.pose_confidence,
            rendered_jpeg=rendered_jpeg,
        )

        rep_result = self._rep_segmenter.update(phase, timestamp)
        if rep_result and rep_result.valid:
            self._session.rep_count = rep_result.rep_index
            result.rep = self._score_rep(rep_result, features, faults, fused, pose_result)

        with self._coaching_lock:
            if self._pending_coaching:
                result.coaching_text = self._pending_coaching
                self._pending_coaching = None

        return result

    def cleanup(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass

    def _init_post_calibration(self) -> None:
        cal = self._cal_result
        scoring = self._scoring_config

        self._feature_extractor = SquatFeatureExtractor(cal, fps=24.0)
        self._ideal_ref = build_ideal_reference(
            cal, target_knee_angle=scoring["calibration"]["target_knee_angle_deg"],
        )
        self._phase_detector = PhaseDetector(
            min_phase_duration_s=scoring["calibration"]["min_phase_duration_s"],
            calibrated_knee_angle=cal.baseline_knee_angle,
        )
        self._rep_segmenter = RepSegmenter(
            min_rep_duration_s=scoring["calibration"]["min_rep_duration_s"],
            cooldown_s=scoring["calibration"]["rep_cooldown_s"],
        )

        ckpt_dir = self._model_config["checkpoints"]["dir"]
        stats_path = str(Path(ckpt_dir) / "feature_stats.json")
        self._inference_mgr = InferenceManager(
            model_configs=self._model_config["models"],
            ensemble_config=self._model_config["ensemble"],
            checkpoint_dir=ckpt_dir,
            stats_path=stats_path,
            view=cal.view_type.value,
        )

    def _score_rep(self, rep_result, features, faults, fused, pose_result) -> RepData:
        ideal = self._ideal_ref
        cal = self._cal_result

        depth = compute_depth_score(
            self._rep_min_knee, ideal.target_knee_angle if ideal else 90
        )
        trunk = compute_trunk_control_score(
            features.get("trunk_stability", 0),
            self._rep_max_torso,
            ideal.trunk_neutral_angle if ideal else 10,
        )
        posture = compute_posture_stability_score(
            features.get("rounded_back_risk", 0),
            self._rep_max_head_offset,
            cal.body_scale if cal else 0.4,
        )
        consistency = compute_movement_consistency_score(
            rep_result.descent_duration, rep_result.ascent_duration,
        )

        scores = {
            "depth": round(depth, 1),
            "trunk_control": round(trunk, 1),
            "posture_stability": round(posture, 1),
            "movement_consistency": round(consistency, 1),
        }
        model_q = fused.quality_score if fused else 0.5
        rep_quality = compute_rep_quality_score(scores, model_quality=model_q)
        scores["total"] = round(rep_quality, 1)

        self._trend_tracker.update(rep_quality)
        self._session.current_score = rep_quality
        self._last_rep_score = rep_quality
        self._best_rep_score = max(self._best_rep_score, rep_quality)
        self._all_rep_scores.append(rep_quality)

        comparison = {
            "depth": f"{max(0, self._rep_min_knee - (ideal.target_knee_angle if ideal else 90)):.0f} deg from target",
            "trunk": f"{max(0, self._rep_max_torso - (ideal.trunk_neutral_angle if ideal else 10)):.0f} deg over baseline",
            "tempo": f"descent {rep_result.descent_duration:.1f}s / ascent {rep_result.ascent_duration:.1f}s",
        }
        trend_val = self._trend_tracker.get_trend()
        trend_str = f"{'up' if trend_val > 0 else 'down'}{abs(trend_val):.0f} over recent reps"
        rationale = build_rationale(rep_result.rep_index, scores, faults, comparison, trend_str)

        confidence = fused.confidence if fused else 0.5
        rep_event = build_rep_summary(
            rep_result, rationale, faults, self._rep_features_snapshot,
            pose_result.pose_confidence, confidence,
        )
        gemini_cfg = self._config.get("gemini", {})
        if gemini_cfg.get("enabled", False):
            gemini_payload = format_gemini_payload(rep_event)
            key = gemini_cfg.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            model = gemini_cfg.get("model", "gemini-2.0-flash")

            def _on_feedback(fb: str) -> None:
                with self._coaching_lock:
                    self._pending_coaching = fb

            send_to_gemini_async(
                gemini_payload, api_key=key, model=model,
                speak_enabled=False,
                on_feedback=_on_feedback,
            )

        fault_names = [f.fault_type.value for f in faults]

        self._rep_min_knee = 180.0
        self._rep_max_torso = 0.0
        self._rep_max_head_offset = 0.0
        self._live_score_ema = 100.0

        return RepData(
            rep_index=rep_result.rep_index,
            scores=scores,
            faults=fault_names,
            coaching_text=rationale.coaching_cue,
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
