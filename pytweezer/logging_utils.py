import logging
import os


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _parse_level(level_name: str) -> int:
    level = getattr(logging, str(level_name).upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once for the pytweezer process."""
    root = logging.getLogger()
    if root.handlers:
        return

    env_level = os.getenv("PYTWEEZER_LOG_LEVEL", "INFO")
    resolved_level = _parse_level(level or env_level)
    logging.basicConfig(level=resolved_level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)


def get_logger(name: str) -> logging.Logger:
    """Return a logger and ensure default logging is configured."""
    configure_logging()
    return logging.getLogger(name)
