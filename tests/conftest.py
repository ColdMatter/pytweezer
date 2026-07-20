"""Shared pytest fixtures.

Sets ``QT_QPA_PLATFORM=offscreen`` *before* any PyQt5 import so GUI widgets can
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
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
