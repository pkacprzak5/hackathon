"""Format events for terminal logging."""
from squat_coach.events.schemas import SquatEvent, RepSummaryEvent


def format_frame_log(
    frame_idx: int,
    phase: str,
    features: dict,
    confidence: float,
) -> str:
    """Format a single frame log line for terminal output."""
    return (
        f"FRAME {frame_idx:>5d} | phase={phase:<8s} | "
        f"knee={features.get('primary_knee_angle', 0):>5.1f}\u00b0 | "
        f"hip={features.get('primary_hip_angle', 0):>5.1f}\u00b0 | "
        f"torso={features.get('torso_inclination_deg', 0):>4.1f}\u00b0 | "
        f"depth={features.get('hip_depth_vs_knee', 0):>5.2f} | "
        f"back_risk={features.get('rounded_back_risk', 0):>4.2f} | "
        f"conf={confidence:.2f}"
    )


def format_rep_summary(event: RepSummaryEvent) -> str:
    """Format a rep summary for terminal output."""
    s = event.scores
    lines = [
        "\u2550" * 60,
        f"  REP {event.rep_index} COMPLETE | quality={s.get('rep_quality', 0):.0f} | "
        f"depth={s.get('depth', 0):.0f} | trunk={s.get('trunk_control', 0):.0f} | "
        f"consistency={s.get('movement_consistency', 0):.0f} | "
        f"posture={s.get('posture_stability', 0):.0f}",
    ]
    for fault in event.faults:
        lines.append(
            f"  FAULT: {fault['type']} (severity={fault['severity']:.2f}, conf={fault['confidence']:.2f})"
        )
    lines.append(f'  CUE: "{event.coaching_cue}"')
    lines.append("\u2550" * 60)
    return "\n".join(lines)
