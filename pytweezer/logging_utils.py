"""Shared logging setup for the package.

Restored after the repo clean-up dropped this module while ``coordinators.base``
and ``coordinators.rearrangement`` still imported ``get_logger`` from it.

The old version also attached a ``StructuredMessageHandler`` that republished
every record as JSON on the ZMQ "Logs" topic via ``servers.messageclient``. That
client was removed in the same clean-up, so the handler is gone too and logging
is plain stdlib to stderr.
"""

import logging
import os
import socket


_ENV_LOG_LEVEL = "PYTWEEZER_LOG_LEVEL"
_DEFAULT_FORMAT = (
    f"%(asctime)s | %(levelname)-8s | {socket.gethostname()} | "
    "%(name)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _parse_level(level_name: str) -> int:
    level = getattr(logging, str(level_name).upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once; level from ``PYTWEEZER_LOG_LEVEL`` or INFO."""
    root = logging.getLogger()
    env_level = os.getenv(_ENV_LOG_LEVEL, "INFO")
    resolved_level = _parse_level(level or env_level)
    root.setLevel(resolved_level)

    if not root.handlers:
        logging.basicConfig(
            level=resolved_level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT
        )


def get_logger(name: str) -> logging.Logger:
    """Return a logger and ensure logging is configured."""
    configure_logging()
    return logging.getLogger(name)
