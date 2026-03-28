"""Full squat feature extraction orchestrator.

Computes all biomechanics features from a single frame of world landmarks.
Produces both the D=42 model input vector and extended features for
rule-based scoring, fault detection, and terminal logging.
"""
import numpy as np
from numpy.typing import NDArray
from collections import deque

from squat_coach.preprocessing.calibration import CalibrationResult
from squat_coach.biomechanics.angles import compute_joint_angles
from squat_coach.biomechanics.vectors import (
    compute_head_to_trunk_offset,
    compute_shoulder_hip_deltas,
    compute_nose_to_shoulder_offset,
    compute_neck_forward_offset,
)
from squat_coach.biomechanics.distances import (
    compute_hip_depth_ratios,
    compute_pairwise_distance_subset,
)
from squat_coach.biomechanics.kinematics import KinematicsTracker
from squat_coach.biomechanics.posture_analysis import compute_rounded_back_risk
from squat_coach.biomechanics.side_view_constraints import compute_side_view_features
from squat_coach.biomechanics.front_view_constraints import compute_front_view_features
from squat_coach.pose.landmarks import LOWER_BODY_LANDMARKS, TORSO_LANDMARKS
from squat_coach.utils.enums import ViewType


class SquatFeatureExtractor:
    """Extracts full biomechanics feature set from world landmarks.

    Produces:
        - 'model_features': NDArray of shape (42,) for temporal model input
        - Named features dict for scoring, faults, logging
    """

    # Rolling window for trunk stability computation
    _TRUNK_WINDOW_SIZE = 30

    def __init__(self, calibration: CalibrationResult, fps: float = 30.0) -> None:
        self._cal = calibration
        self._kinematics = KinematicsTracker(fps=fps)
        self._trunk_angle_window: deque[float] = deque(maxlen=self._TRUNK_WINDOW_SIZE)

    def extract(
        self,
        world_landmarks: NDArray[np.float64],
        visibility: NDArray[np.float64],
    ) -> dict:
        """Extract all features from a single frame.

        Args:
            world_landmarks: (33, 3) world landmarks.
            visibility: (33,) per-landmark visibility.

        Returns:
            Dict containing:
                'model_features': np.ndarray of shape (42,)
                Plus all named scalar features.
        """
        # A. Core joint angles
        angles = compute_joint_angles(world_landmarks)

        # Primary angles (dominant side from calibration)
        if self._cal.dominant_side == "left" or self._cal.dominant_side == "both":
            primary_knee = angles["left_knee_angle"]
            primary_hip = angles["left_hip_angle"]
        else:
            primary_knee = angles["right_knee_angle"]
            primary_hip = angles["right_hip_angle"]

        # B. Trunk and head offsets
        head_offset = compute_head_to_trunk_offset(world_landmarks)
        sh_h_delta, sh_v_delta = compute_shoulder_hip_deltas(world_landmarks)
        nose_offset = compute_nose_to_shoulder_offset(world_landmarks)
        neck_offset = compute_neck_forward_offset(world_landmarks)

        # C. Hip depth
        hip_vs_knee, hip_vs_ankle = compute_hip_depth_ratios(world_landmarks)

        # D. Trunk stability window
        self._trunk_angle_window.append(angles["torso_inclination_deg"])

        # E. View-specific features (indices 16-19 in model vector)
        if self._cal.view_type == ViewType.SIDE:
            view_feats = compute_side_view_features(
                world_landmarks, list(self._trunk_angle_window)
            )
            back_risk = compute_rounded_back_risk(
                world_landmarks, self._cal.baseline_torso_angle, self._cal.body_scale
            )
            feat_16 = view_feats["forward_lean_angle"]
            feat_17 = back_risk.risk_score
            feat_18 = view_feats["trunk_stability"]
            feat_19 = view_feats["ankle_shin_angle"]
        else:
            view_feats = compute_front_view_features(world_landmarks)
            back_risk = None
            feat_16 = view_feats["knee_valgus_angle"]
            feat_17 = view_feats["stance_width_ratio"]
            feat_18 = view_feats["left_right_symmetry"]
            feat_19 = view_feats["hip_shift_lateral"]

        # F. Kinematics
        mid_hip_y = float((world_landmarks[23][1] + world_landmarks[24][1]) / 2.0)
        kin_input = {
            "hip_vertical": mid_hip_y,
            "trunk_angle": angles["torso_inclination_deg"],
            "knee_angle": primary_knee,
            "hip_angle": primary_hip,
        }
        kin = self._kinematics.update(kin_input)

        # G. Visibility / confidence features
        vis_mean = float(np.mean(visibility))
        lower_vis = float(np.mean(visibility[LOWER_BODY_LANDMARKS]))
        torso_vis = float(np.mean(visibility[TORSO_LANDMARKS]))
        reliability = vis_mean * min(lower_vis, torso_vis)
        view_validity = 1.0 if self._cal.view_type != ViewType.UNKNOWN else 0.0
        occlusion_risk = 1.0 - min(lower_vis, torso_vis)

        # H. Pairwise distances
        pw_dists = compute_pairwise_distance_subset(world_landmarks)

        # Build D=42 model feature vector
        model_features = np.array([
            # 0-5: angles
            angles["left_knee_angle"], angles["right_knee_angle"], primary_knee,
            angles["left_hip_angle"], angles["right_hip_angle"], primary_hip,
            # 6: ankle
            angles["ankle_angle_proxy"],
            # 7-8: torso
            angles["torso_inclination_deg"], angles["shoulder_hip_line_angle"],
            # 9-11: offsets
            head_offset, sh_h_delta, sh_v_delta,
            # 12-13: hip depth
            hip_vs_knee, hip_vs_ankle,
            # 14-15: nose/neck
            nose_offset, neck_offset,
            # 16-19: view-dependent
            feat_16, feat_17, feat_18, feat_19,
            # 20-27: kinematics
            kin.get("hip_vertical_velocity", 0.0),
            kin.get("hip_vertical_acceleration", 0.0),
            kin.get("trunk_angle_velocity", 0.0),
            kin.get("trunk_angle_acceleration", 0.0),
            kin.get("knee_angle_velocity", 0.0),
            kin.get("knee_angle_acceleration", 0.0),
            kin.get("hip_angle_velocity", 0.0),
            kin.get("hip_angle_acceleration", 0.0),
            # 28-33: quality
            vis_mean, lower_vis, torso_vis, reliability, view_validity, occlusion_risk,
            # 34-41: pairwise distances
            *pw_dists,
        ], dtype=np.float64)

        # Build full named features dict
        features: dict = {
            **angles,
            "primary_knee_angle": primary_knee,
            "primary_hip_angle": primary_hip,
            "head_to_trunk_offset": head_offset,
            "shoulder_to_hip_h_delta": sh_h_delta,
            "shoulder_to_hip_v_delta": sh_v_delta,
            "hip_depth_vs_knee": hip_vs_knee,
            "hip_depth_vs_ankle": hip_vs_ankle,
            "nose_to_shoulder_offset": nose_offset,
            "neck_forward_offset": neck_offset,
            **view_feats,
            "rounded_back_risk": back_risk.risk_score if back_risk else 0.0,
            **kin,
            "landmark_visibility_mean": vis_mean,
            "lower_body_visibility": lower_vis,
            "torso_visibility": torso_vis,
            "frame_reliability_score": reliability,
            "view_validity_score": view_validity,
            "occlusion_risk_score": occlusion_risk,
            "model_features": model_features,
        }
        if back_risk:
            features["rounded_back_assessment"] = back_risk

        return features

    def reset(self) -> None:
        """Reset temporal state (kinematics, windows)."""
        self._kinematics.reset()
        self._trunk_angle_window.clear()
