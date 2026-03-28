"""Structured logging configuration."""
import logging
import sys

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the squat_coach logger."""
    logger = logging.getLogger("squat_coach")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
