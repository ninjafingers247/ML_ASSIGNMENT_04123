"""Shared ingestion utilities: logging setup and a retry/backoff decorator."""
import functools
import logging
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LOGS_DIR  # noqa: E402


def get_logger(name: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(LOGS_DIR / logfile)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def retry(max_attempts: int = 3, base_delay: float = 1.0, logger: logging.Logger = None):
    """Retry decorator with exponential backoff, for transient ingestion failures."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if logger:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s",
                            attempt, max_attempts, func.__name__, exc,
                        )
                    if attempt >= max_attempts:
                        if logger:
                            logger.error(
                                "All %d attempts exhausted for %s", max_attempts, func.__name__
                            )
                        raise
                    time.sleep(base_delay * (2 ** (attempt - 1)))

        return wrapper

    return decorator
