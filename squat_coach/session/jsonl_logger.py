"""JSONL session file writer."""
import json
import logging
from datetime import datetime
from pathlib import Path
from squat_coach.events.schemas import RepSummaryEvent
from dataclasses import asdict

logger = logging.getLogger("squat_coach.session")


class JSONLLogger:
    """Write rep summaries as JSONL lines to a session file."""

    def __init__(self, log_dir: str = "sessions") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = self._dir / f"{ts}.jsonl"
        self._file = open(self._path, "a")
        logger.info("Session log: %s", self._path)

    def log_rep(self, event: RepSummaryEvent) -> None:
        """Write a rep summary as one JSON line."""
        data = asdict(event)
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
