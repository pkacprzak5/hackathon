"""Build structured rationale objects for rep scoring."""
from dataclasses import dataclass, field
from squat_coach.faults.fault_types import FaultDetection


@dataclass
class RepRationale:
    """Structured explanation of a rep's score."""
    rep_index: int
    scores: dict[str, float]
    dominant_fault: str
    fault_evidence: str
    comparison_to_ideal: dict[str, str]
    trend: str
    coaching_cue: str


def build_rationale(
    rep_index: int,
    scores: dict[str, float],
    faults: list[FaultDetection],
    ideal_comparison: dict[str, str],
    trend_str: str,
) -> RepRationale:
    """Build a rationale object for a completed rep."""
    # Dominant fault = highest severity (only significant ones)
    significant = [f for f in faults if f.severity >= 0.3]
    if significant:
        dominant = max(significant, key=lambda f: f.severity)
        dominant_name = dominant.fault_type.value
        evidence_str = "; ".join(dominant.evidence)
        cue = dominant.explanation_token
    else:
        dominant_name = "none"
        evidence_str = "No significant faults detected"
        rep_q = scores.get("rep_quality", 50)
        if rep_q >= 80:
            cue = "Great rep!"
        elif rep_q >= 60:
            cue = "Good form, keep it up!"
        else:
            cue = "Keep working on it!"

    return RepRationale(
        rep_index=rep_index,
        scores=scores,
        dominant_fault=dominant_name,
        fault_evidence=evidence_str,
        comparison_to_ideal=ideal_comparison,
        trend=trend_str,
        coaching_cue=cue,
    )
