import datetime
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any


_LOG_TOPIC = "Logs"
_ENV_LOG_LEVEL = "PYTWEEZER_LOG_LEVEL"
_LOG_DIR_ENV = "PYTWEEZER_LOG_DIR"
_LOG_FILE_PREFIX = "pytweezer"
_LOG_FILE_EXTENSION = ".jsonl"
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


def _local_timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _get_log_dir() -> Path:
    env_path = os.getenv(_LOG_DIR_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[1] / "logs"


def get_daily_log_path(log_date: datetime.date | None = None) -> Path:
    if log_date is None:
        log_date = datetime.date.today()
    log_dir = _get_log_dir()
    filename = f"{_LOG_FILE_PREFIX}-{log_date.isoformat()}{_LOG_FILE_EXTENSION}"
    return log_dir / filename


def _build_payload(record: logging.LogRecord) -> dict[str, Any]:
    payload = {
        "timestamp": _local_timestamp(),
        "level": record.levelname,
        "host": socket.gethostname(),
        "logger": record.name,
        "module": record.module,
        "message": record.getMessage(),
    }
    if record.exc_info:
        formatter = logging.Formatter()
        payload["exception"] = formatter.formatException(record.exc_info)
    return payload


def _publish_payload(topic: str, payload: dict[str, Any]) -> None:
    # Lazy import to avoid circular dependencies at import time.
    from pytweezer.servers.messageclient import get_message_client

    get_message_client().send_json(topic, payload)


class StructuredMessageHandler(logging.Handler):
    def __init__(self, topic: str = _LOG_TOPIC) -> None:
        super().__init__()
        self.topic = topic

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = _build_payload(record)
            _publish_payload(self.topic, payload)
        except Exception:
            self.handleError(record)


class DailyJsonFileHandler(logging.Handler):
    def __init__(self, log_dir: Path | None = None) -> None:
        super().__init__()
        self._log_dir = log_dir or _get_log_dir()
        self._current_date: datetime.date | None = None
        self._path: Path | None = None

    def _ensure_path(self) -> None:
        today = datetime.date.today()
        if self._current_date != today or self._path is None:
            self._current_date = today
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._path = get_daily_log_path(today)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = _build_payload(record)
            self._ensure_path()
            if self._path is None:
                return
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            self.handleError(record)




def configure_logging(level: str | None = None) -> None:
    """Configure root logging and publish structured logs to the Message hub."""
    root = logging.getLogger()
    env_level = os.getenv(_ENV_LOG_LEVEL, "INFO")
    resolved_level = _parse_level(level or env_level)
    root.setLevel(resolved_level)

    if not root.handlers:
        logging.basicConfig(
            level=resolved_level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT
        )

    if any(isinstance(handler, StructuredMessageHandler) for handler in root.handlers):
        return

    handler = StructuredMessageHandler()
    handler.setLevel(resolved_level)
    root.addHandler(handler)

    if not any(isinstance(handler, DailyJsonFileHandler) for handler in root.handlers):
        file_handler = DailyJsonFileHandler()
        file_handler.setLevel(resolved_level)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger and ensure structured logging is configured."""
    configure_logging()
    return logging.getLogger(name)
