"""Per-fault rule definitions with configurable thresholds."""
from squat_coach.faults.fault_types import FaultDetection, FAULT_CUES
from squat_coach.utils.enums import FaultType
from typing import Optional


def check_insufficient_depth(features: dict, config: dict) -> Optional[FaultDetection]:
    knee = features.get("primary_knee_angle", 180.0)
    threshold = config.get("knee_angle_max_deg", 110.0)
    if knee > threshold:
        severity = min((knee - threshold) / 40.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.INSUFFICIENT_DEPTH,
            severity=severity,
            confidence=0.8,
            evidence=[f"Knee angle {knee:.0f}\u00b0 > {threshold:.0f}\u00b0 target"],
            explanation_token=FAULT_CUES[FaultType.INSUFFICIENT_DEPTH],
        )
    return None


def check_excessive_forward_lean(features: dict, config: dict) -> Optional[FaultDetection]:
    torso = features.get("torso_inclination_deg", 0.0)
    baseline = config.get("baseline_torso_angle", 10.0)
    threshold = config.get("trunk_deviation_deg", 25.0)
    deviation = torso - baseline
    if deviation > threshold:
        severity = min(deviation / 40.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.EXCESSIVE_FORWARD_LEAN,
            severity=severity,
            confidence=0.75,
            evidence=[f"Trunk angle {torso:.0f}\u00b0 ({deviation:.0f}\u00b0 over baseline)"],
            explanation_token=FAULT_CUES[FaultType.EXCESSIVE_FORWARD_LEAN],
        )
    return None


def check_rounded_back(features: dict, config: dict) -> Optional[FaultDetection]:
    risk = features.get("rounded_back_risk", 0.0)
    threshold = config.get("risk_threshold", 0.5)
    if risk > threshold:
        return FaultDetection(
            fault_type=FaultType.ROUNDED_BACK_RISK,
            severity=min(risk, 1.0),
            confidence=risk * 0.9,
            evidence=[f"Rounded back risk score: {risk:.2f}"],
            explanation_token=FAULT_CUES[FaultType.ROUNDED_BACK_RISK],
        )
    return None


def check_unstable_torso(features: dict, config: dict) -> Optional[FaultDetection]:
    stability = features.get("trunk_stability", 0.0)
    threshold = config.get("variance_threshold", 8.0)
    if stability > threshold:
        severity = min(stability / 20.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.UNSTABLE_TORSO,
            severity=severity,
            confidence=0.7,
            evidence=[f"Trunk variance: {stability:.1f}"],
            explanation_token=FAULT_CUES[FaultType.UNSTABLE_TORSO],
        )
    return None


def check_heel_fault(features: dict, config: dict) -> Optional[FaultDetection]:
    threshold = config.get("displacement_threshold", 0.05)
    # Proxy: ankle/foot displacement from expected position
    ankle_shin = features.get("ankle_shin_angle", 90.0)
    if ankle_shin < 60.0:  # Excessive dorsiflexion or heel lift proxy
        severity = min((60.0 - ankle_shin) / 30.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.HEEL_FAULT,
            severity=severity,
            confidence=0.6,
            evidence=[f"Ankle/shin angle {ankle_shin:.0f}\u00b0 indicates possible heel issue"],
            explanation_token=FAULT_CUES[FaultType.HEEL_FAULT],
        )
    return None


def check_knee_valgus(features: dict, config: dict) -> Optional[FaultDetection]:
    angle = features.get("knee_valgus_angle", 0.0)
    threshold = config.get("angle_threshold_deg", 10.0)
    if angle > threshold:
        severity = min(angle / 20.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.KNEE_VALGUS,
            severity=severity,
            confidence=0.7,
            evidence=[f"Knee valgus angle {angle:.1f}\u00b0 > {threshold:.0f}\u00b0 threshold"],
            explanation_token=FAULT_CUES[FaultType.KNEE_VALGUS],
        )
    return None


def check_inconsistent_tempo(features: dict, config: dict) -> Optional[FaultDetection]:
    # This is evaluated at rep level, not per-frame; uses rep timing data passed in features
    descent_s = features.get("last_descent_duration", 0.0)
    ascent_s = features.get("last_ascent_duration", 0.0)
    if descent_s > 0 and ascent_s > 0:
        ratio = min(descent_s, ascent_s) / max(descent_s, ascent_s)
        threshold = config.get("timing_deviation_ratio", 0.4)
        if ratio < threshold:
            severity = min((threshold - ratio) / threshold, 1.0)
            return FaultDetection(
                fault_type=FaultType.INCONSISTENT_TEMPO,
                severity=severity,
                confidence=0.65,
                evidence=[f"Descent/ascent ratio {ratio:.2f} (threshold {threshold:.2f})"],
                explanation_token=FAULT_CUES[FaultType.INCONSISTENT_TEMPO],
            )
    return None


def check_poor_trunk_control(features: dict, config: dict) -> Optional[FaultDetection]:
    torso = features.get("torso_inclination_deg", 0.0)
    baseline = config.get("baseline_torso_angle", 10.0)
    trunk_stability = features.get("trunk_stability", 0.0)
    # Combined: high variance AND deviation from baseline at bottom
    deviation = max(0, torso - baseline)
    if trunk_stability > 5.0 and deviation > 15.0:
        severity = min((trunk_stability / 15.0 + deviation / 30.0) / 2.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.POOR_TRUNK_CONTROL,
            severity=severity,
            confidence=0.7,
            evidence=[f"Trunk variance {trunk_stability:.1f} + deviation {deviation:.0f}\u00b0"],
            explanation_token=FAULT_CUES[FaultType.POOR_TRUNK_CONTROL],
        )
    return None


def check_low_confidence(features: dict, config: dict) -> Optional[FaultDetection]:
    reliability = features.get("frame_reliability_score", 1.0)
    if reliability < 0.5:
        return FaultDetection(
            fault_type=FaultType.LOW_CONFIDENCE,
            severity=1.0 - reliability,
            confidence=0.9,
            evidence=[f"Frame reliability {reliability:.2f}"],
            explanation_token=FAULT_CUES[FaultType.LOW_CONFIDENCE],
            affects_overlay=True,
            affects_gemini=False,
        )
    return None


def check_invalid_view(features: dict, config: dict) -> Optional[FaultDetection]:
    validity = features.get("view_validity_score", 1.0)
    if validity < 0.5:
        return FaultDetection(
            fault_type=FaultType.INVALID_VIEW,
            severity=1.0 - validity,
            confidence=0.9,
            evidence=[f"View validity {validity:.2f}"],
            explanation_token=FAULT_CUES[FaultType.INVALID_VIEW],
            affects_overlay=True,
            affects_gemini=False,
        )
    return None


# Registry of all rule functions
FAULT_RULES = [
    check_insufficient_depth,
    check_excessive_forward_lean,
    check_rounded_back,
    check_unstable_torso,
    check_heel_fault,
    check_knee_valgus,
    check_inconsistent_tempo,
    check_poor_trunk_control,
    check_low_confidence,
    check_invalid_view,
]
