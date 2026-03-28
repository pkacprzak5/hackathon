"""Construct events from system state."""
from squat_coach.events.schemas import SquatEvent, RepSummaryEvent
from squat_coach.phases.rep_segmenter import RepResult
from squat_coach.scoring.rationale import RepRationale
from squat_coach.faults.fault_types import FaultDetection


def build_rep_summary(
    rep_result: RepResult,
    rationale: RepRationale,
    faults: list[FaultDetection],
    features_snapshot: dict[str, float],
    pose_confidence: float,
    assessment_confidence: float,
) -> RepSummaryEvent:
    """Build a complete rep summary event."""
    return RepSummaryEvent(
        timestamp=rep_result.end_time,
        rep_index=rep_result.rep_index,
        phase_durations={
            "descent_s": round(rep_result.descent_duration, 2),
            "bottom_s": round(rep_result.bottom_duration, 2),
            "ascent_s": round(rep_result.ascent_duration, 2),
        },
        features={
            "knee_angle_min_deg": round(features_snapshot.get("primary_knee_angle", 0), 1),
            "hip_angle_min_deg": round(features_snapshot.get("primary_hip_angle", 0), 1),
            "torso_inclination_peak_deg": round(features_snapshot.get("torso_inclination_deg", 0), 1),
            "rounded_back_risk": round(features_snapshot.get("rounded_back_risk", 0), 2),
        },
        faults=[
            {
                "type": f.fault_type.value,
                "severity": round(f.severity, 2),
                "confidence": round(f.confidence, 2),
            }
            for f in faults
        ],
        scores=rationale.scores,
        confidence={
            "pose_confidence": round(pose_confidence, 2),
            "assessment_confidence": round(assessment_confidence, 2),
        },
        coaching_cue=rationale.coaching_cue,
    )


def build_phase_transition_event(
    timestamp: float, from_phase: str, to_phase: str
) -> SquatEvent:
    return SquatEvent(
        event_type="phase_transition",
        timestamp=timestamp,
        data={"from": from_phase, "to": to_phase},
    )
