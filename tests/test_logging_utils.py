"""Where log records go while the test suite runs."""

from pathlib import Path

from pytweezer import logging_utils
from pytweezer.logging_utils import get_daily_log_path, get_logger

REPO_LOGS = Path(logging_utils.__file__).resolve().parents[1] / "logs"


def test_test_records_never_reach_the_message_hub(monkeypatch):
    published = []
    monkeypatch.setattr(
        logging_utils, "_publish_payload", lambda *args: published.append(args)
    )
    get_logger("tests.probe").warning("this must stay off the lab's log stream")
    assert published == []


def test_test_records_are_not_written_to_the_lab_log_files():
    get_logger("tests.probe").warning("this must stay out of logs/")
    assert get_daily_log_path().exists()
    assert REPO_LOGS not in get_daily_log_path().parents
