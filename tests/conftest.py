"""Shared pytest fixtures.

Sets ``QT_QPA_PLATFORM=offscreen`` *before* any PyQt6 import so GUI widgets can
be constructed in a headless CI/terminal (see CLAUDE.md), and exposes a single
shared ``QApplication`` for tests that build widgets.
"""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Set before any pytweezer import, and inherited by the subprocesses tests start,
# so test log records stay out of the lab's log stream and log files.
os.environ["PYTWEEZER_LOG_PUBLISH"] = "0"
os.environ["PYTWEEZER_LOG_DIR"] = tempfile.mkdtemp(prefix="pytweezer-test-logs-")

import pytest

from tests.helpers import RecordingDB


@pytest.fixture(scope="session")
def qapp():
    """A process-wide QApplication (Qt allows only one). Session-scoped so every
    widget test shares it."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def recording_db():
    """A fresh :class:`RecordingDB`, so tests never reach a real database."""
    return RecordingDB()


class FakeDeviceClient:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close_rpc(self):
        self.closed = True


@pytest.fixture
def fake_devices(monkeypatch):
    """Experiments get a :class:`FakeDeviceClient` per device opened; returns them all."""
    opened = []

    def get_device(name, timeout):
        opened.append(FakeDeviceClient(name))
        return opened[-1]

    monkeypatch.setattr("pytweezer.experiment.experiment._get_device", get_device)
    return opened
