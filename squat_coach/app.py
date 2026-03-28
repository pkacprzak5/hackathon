# squat_coach/app.py
"""Main application orchestrator — ties all subsystems together."""
import logging
import time
import cv2
import yaml
import numpy as np
from pathlib import Path

from squat_coach.utils.logging_utils import setup_logging
from squat_coach.utils.timing import FPSTracker
from squat_coach.utils.enums import Phase
from squat_coach.camera.webcam_stream import WebcamStream
from squat_coach.camera.video_replay import VideoReplay
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
from squat_coach.events.formatter import format_frame_log, format_rep_summary
from squat_coach.events.gemini_payloads import format_gemini_payload, send_to_gemini
from squat_coach.events.coaching_priority import CoachingPrioritizer
from squat_coach.rendering.overlay import render_overlay
from squat_coach.session.session_state import SessionState
from squat_coach.session.rep_history import RepHistory
from squat_coach.session.jsonl_logger import JSONLLogger

# Import models to register in factory
import squat_coach.models.temporal_tcn   # noqa: F401
import squat_coach.models.temporal_gru   # noqa: F401

logger = logging.getLogger("squat_coach")


class SquatCoachApp:
    """Main application: video -> pose -> features -> models -> scoring -> output."""

    def __init__(
        self,
        mode: str = "webcam",
        video_path: str | None = None,
        debug: bool = False,
        log_features: bool = False,
    ) -> None:
        self._mode = mode
        self._video_path = video_path
        self._debug = debug
        self._log_features = log_features

        setup_logging(debug=debug)

        # Load configs
        self._config = self._load_config("squat_coach/config/default.yaml")
        self._model_config = self._load_config("squat_coach/config/model.yaml")
        self._scoring_config = self._load_config("squat_coach/config/scoring.yaml")

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def run(self) -> None:
        """Main loop."""
        if self._mode == "train":
            from squat_coach.training.train_all import train_all
            train_all()
            return

        # Initialize subsystems
        camera = self._create_camera()
        pose = MediaPipeBlazePose3D()
        smoother = EMALandmarkSmoother(alpha=self._config["preprocessing"]["ema_alpha"])
        calibrator = Calibrator(
            num_frames=int(self._config["preprocessing"]["calibration_duration_s"] * 30)
        )
        session = SessionState()
        fps_tracker = FPSTracker()
        jsonl_logger = JSONLLogger(self._config["session"]["log_dir"])
        rep_history = RepHistory()
        trend_tracker = TrendTracker()
        coach = CoachingPrioritizer()

        # These are initialized after calibration
        feature_extractor = None
        inference_mgr = None
        phase_detector = None
        rep_segmenter = None
        fault_engine = EvidenceEngine()
        ideal_ref = None
        cal_result = None

        seq_buf = SequenceBuffer(
            seq_len=self._model_config["sequence"]["length"],
            feature_dim=self._model_config["sequence"]["feature_dim"],
        )

        log_throttle_interval = 1.0 / self._config["logging"]["terminal_throttle_hz"]
        last_log_time = 0.0

        # Track min values within a rep for scoring
        rep_min_knee = 180.0
        rep_max_torso = 0.0
        rep_max_head_offset = 0.0
        rep_features_snapshot: dict = {}
        last_cue = ""
        last_cue_time = 0.0
        cue_display_duration = 5.0  # Show cue for 5 seconds minimum
        current_features: dict = {}

        logger.info("Starting Squat Coach in %s mode", self._mode)

        try:
            while camera.is_opened():
                ret, frame = camera.read()
                if not ret or frame is None:
                    if self._mode == "replay":
                        logger.info("Video replay complete")
                        break
                    continue

                fps_tracker.tick()
                session.frame_index += 1
                timestamp = session.frame_index / 30.0

                # Pose estimation
                pose_result = pose.estimate(frame, timestamp)

                if not pose_result.detected:
                    session.dropped_frame_count += 1
                    if session.dropped_frame_count > self._config["preprocessing"]["max_dropped_frames"]:
                        session.current_cue = "Repositioning needed"
                    # Still render overlay with last known state
                    frame = render_overlay(
                        frame, None, None,
                        session.current_phase.value,
                        session.rep_count,
                        session.current_score,
                        session.current_cue,
                        features=current_features,
                    )
                    cv2.imshow("Squat Coach", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                session.dropped_frame_count = 0

                # Smooth landmarks
                smoothed = smoother.smooth(pose_result.world_landmarks)
                normalized = normalize_to_hip_center(smoothed)

                # Calibration phase
                if not session.is_calibrated:
                    calibrator.add_frame(pose_result)
                    if calibrator.is_ready:
                        cal_result = calibrator.compute()
                        if cal_result:
                            session.is_calibrated = True
                            session.view_type = cal_result.view_type
                            logger.info("Calibration complete: %s view", cal_result.view_type.value)

                            # Initialize post-calibration subsystems
                            feature_extractor = SquatFeatureExtractor(cal_result, fps=30.0)
                            ideal_ref = build_ideal_reference(
                                cal_result,
                                target_knee_angle=self._scoring_config["calibration"]["target_knee_angle_deg"],
                            )
                            phase_detector = PhaseDetector(
                                min_phase_duration_s=self._scoring_config["calibration"]["min_phase_duration_s"],
                            )
                            rep_segmenter = RepSegmenter(
                                min_rep_duration_s=self._scoring_config["calibration"]["min_rep_duration_s"],
                                cooldown_s=self._scoring_config["calibration"]["rep_cooldown_s"],
                            )

                            ckpt_dir = self._model_config["checkpoints"]["dir"]
                            stats_path = str(Path(ckpt_dir) / "feature_stats.json")
                            inference_mgr = InferenceManager(
                                model_configs=self._model_config["models"],
                                ensemble_config=self._model_config["ensemble"],
                                checkpoint_dir=ckpt_dir,
                                stats_path=stats_path,
                                view=cal_result.view_type.value,
                            )
                    else:
                        # Show calibration prompt
                        cv2.putText(
                            frame, "Stand still for calibration...",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                        )
                        cv2.imshow("Squat Coach", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                # Feature extraction
                features = feature_extractor.extract(normalized, pose_result.visibility)
                model_features = features["model_features"]

                # Sequence buffer
                seq_buf.push(model_features)

                # Temporal inference
                fused = None
                if seq_buf.is_ready and inference_mgr and inference_mgr.has_models:
                    fused = inference_mgr.infer(seq_buf.get_sequence())

                # Phase detection
                if fused is not None:
                    hip_y = float((normalized[23][1] + normalized[24][1]) / 2.0)
                    phase = phase_detector.detect(fused.phase_probs, hip_y)
                else:
                    phase = session.current_phase

                prev_phase = session.current_phase
                session.current_phase = phase

                # Track per-rep extremes
                if phase in (Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT):
                    rep_min_knee = min(rep_min_knee, features.get("primary_knee_angle", 180))
                    rep_max_torso = max(rep_max_torso, features.get("torso_inclination_deg", 0))
                    rep_max_head_offset = max(rep_max_head_offset, features.get("head_to_trunk_offset", 0))
                    rep_features_snapshot = dict(features)

                # Fault detection
                fault_config = self._scoring_config.get("faults", {}).get("thresholds", {})
                fault_config["baseline_torso_angle"] = ideal_ref.trunk_neutral_angle if ideal_ref else 10.0
                faults = fault_engine.evaluate(features, fault_config)

                # Coaching cue — persist for cue_display_duration seconds
                new_cue = coach.select_cue(faults)
                now_cue = time.monotonic()
                if new_cue:
                    last_cue = new_cue
                    last_cue_time = now_cue
                    session.current_cue = new_cue
                elif now_cue - last_cue_time < cue_display_duration:
                    session.current_cue = last_cue  # Keep showing
                else:
                    session.current_cue = ""
                current_features = features

                # Rep segmentation
                rep_result = rep_segmenter.update(phase, timestamp)
                if rep_result and rep_result.valid:
                    session.rep_count = rep_result.rep_index

                    # Score the rep
                    depth = compute_depth_score(rep_min_knee, ideal_ref.target_knee_angle if ideal_ref else 90)
                    trunk = compute_trunk_control_score(
                        features.get("trunk_stability", 0),
                        rep_max_torso,
                        ideal_ref.trunk_neutral_angle if ideal_ref else 10,
                    )
                    posture = compute_posture_stability_score(
                        features.get("rounded_back_risk", 0),
                        rep_max_head_offset,
                        cal_result.body_scale if cal_result else 0.4,
                    )
                    consistency = compute_movement_consistency_score(
                        rep_result.descent_duration,
                        rep_result.ascent_duration,
                    )

                    scores = {
                        "depth": depth,
                        "trunk_control": trunk,
                        "posture_stability": posture,
                        "movement_consistency": consistency,
                    }
                    model_q = fused.quality_score if fused else 0.5
                    rep_quality = compute_rep_quality_score(scores, model_quality=model_q)
                    scores["rep_quality"] = rep_quality

                    trend_tracker.update(rep_quality)
                    scores["overall_form"] = trend_tracker.ema_score

                    session.current_score = rep_quality

                    # Build rationale
                    comparison = {
                        "depth": f"{max(0, rep_min_knee - (ideal_ref.target_knee_angle if ideal_ref else 90)):.0f} deg from target",
                        "trunk": f"{max(0, rep_max_torso - (ideal_ref.trunk_neutral_angle if ideal_ref else 10)):.0f} deg over baseline",
                        "tempo": f"descent {rep_result.descent_duration:.1f}s / ascent {rep_result.ascent_duration:.1f}s",
                    }
                    trend_val = trend_tracker.get_trend()
                    trend_str = f"{'up' if trend_val > 0 else 'down'}{abs(trend_val):.0f} over recent reps"

                    rationale = build_rationale(
                        rep_result.rep_index, scores, faults, comparison, trend_str
                    )

                    # Build events
                    confidence = fused.confidence if fused else 0.5
                    rep_event = build_rep_summary(
                        rep_result, rationale, faults, rep_features_snapshot,
                        pose_result.pose_confidence, confidence,
                    )

                    # Log to terminal
                    logger.info("\n%s", format_rep_summary(rep_event))

                    # Log to JSONL
                    jsonl_logger.log_rep(rep_event)
                    rep_history.add(rep_event)

                    # Gemini coaching feedback
                    gemini = format_gemini_payload(rep_event)
                    gemini_cfg = self._config.get("gemini", {})
                    if gemini_cfg.get("enabled", False):
                        gemini_key = gemini_cfg.get("api_key", "")
                        gemini_model = gemini_cfg.get("model", "gemini-2.0-flash")
                        feedback = send_to_gemini(gemini, api_key=gemini_key, model=gemini_model)
                        if feedback:
                            logger.info("🤖 GEMINI: %s", feedback)
                            # Show Gemini feedback as the coaching cue
                            last_cue = feedback
                            last_cue_time = time.monotonic()
                            session.current_cue = feedback
                    else:
                        logger.debug("Gemini payload: %s", gemini)

                    # Reset per-rep tracking
                    rep_min_knee = 180.0
                    rep_max_torso = 0.0
                    rep_max_head_offset = 0.0

                # Terminal frame logging (throttled)
                now = time.monotonic()
                if now - last_log_time >= log_throttle_interval:
                    confidence = fused.confidence if fused else 0.0
                    logger.info(format_frame_log(
                        session.frame_index, phase.value, features, confidence
                    ))
                    last_log_time = now

                # Render overlay
                frame = render_overlay(
                    frame,
                    pose_result.image_landmarks,
                    pose_result.visibility,
                    phase.value,
                    session.rep_count,
                    session.current_score,
                    session.current_cue,
                    features=current_features,
                )

                cv2.imshow("Squat Coach", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            pose.close()
            jsonl_logger.close()
            cv2.destroyAllWindows()
            logger.info("Session ended. %d reps completed.", session.rep_count)

    def _create_camera(self):
        cam_cfg = self._config["camera"]
        if self._mode == "replay" and self._video_path:
            return VideoReplay(self._video_path)
        return WebcamStream(
            device_id=cam_cfg["device_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
        )
