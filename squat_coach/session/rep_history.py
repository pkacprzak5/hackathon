"""Per-rep history storage for trend analysis."""
from dataclasses import dataclass
from squat_coach.events.schemas import RepSummaryEvent


class RepHistory:
    """Store and query rep summaries."""

    def __init__(self) -> None:
        self._reps: list[RepSummaryEvent] = []

    def add(self, rep: RepSummaryEvent) -> None:
        self._reps.append(rep)

    @property
    def count(self) -> int:
        return len(self._reps)

    def get_last(self, n: int = 1) -> list[RepSummaryEvent]:
        return self._reps[-n:]

    def get_all(self) -> list[RepSummaryEvent]:
        return list(self._reps)
