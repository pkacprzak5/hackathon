"""Rolling trend analysis across reps."""

class TrendTracker:
    """Track score trends with EMA and delta tracking."""

    def __init__(self, ema_alpha: float = 0.3, window: int = 5) -> None:
        self._alpha = ema_alpha
        self._window = window
        self._scores: list[float] = []
        self._ema: float = 50.0

    def update(self, score: float) -> None:
        """Add a new rep score."""
        self._scores.append(score)
        self._ema = self._alpha * score + (1.0 - self._alpha) * self._ema

    def get_trend(self) -> float:
        """Get trend direction. Positive = improving, negative = declining."""
        if len(self._scores) < 2:
            return 0.0
        recent = self._scores[-min(self._window, len(self._scores)):]
        if len(recent) < 2:
            return 0.0
        # Simple linear slope
        n = len(recent)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(recent) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, recent))
        den = sum((xi - x_mean) ** 2 for xi in x)
        return num / den if den > 0 else 0.0

    @property
    def ema_score(self) -> float:
        return self._ema

    @property
    def last_score(self) -> float:
        return self._scores[-1] if self._scores else 50.0
