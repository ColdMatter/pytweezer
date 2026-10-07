"""Shared pytest fixtures.

Sets ``QT_QPA_PLATFORM=offscreen`` *before* any PyQt6 import so GUI widgets can
be constructed in a headless CI/terminal (see CLAUDE.md), and exposes a single
shared ``QApplication`` for tests that build widgets.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A process-wide QApplication (Qt allows only one). Session-scoped so every
    widget test shares it."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class RecordingDB:
    """Stand-in for :class:`~pytweezer.database.writer.DBWriter` that records rows."""

    def __init__(self):
        self.runs = []
        self.points = []
        self.closed = False

    def record_run(self, run):
        self.runs.append(dict(run))

    def record_point(self, rid, point_index, t_start, t_end, scan_values, scalars):
        self.points.append(
            {
                "rid": rid,
                "point_index": point_index,
                "t_start": t_start,
                "t_end": t_end,
                "scan_values": scan_values,
                "scalars": scalars,
            }
        )

    def write(self, measurement, fields, tags=None, time=None):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def recording_db():
    """A fresh :class:`RecordingDB`, so tests never reach a real database."""
    return RecordingDB()
