"""Startup popup offering the lab setup changes still pending on this PC."""

import math
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from pytweezer.GUI.components import Region

if TYPE_CHECKING:
    from pytweezer.lab_setup import ChangeStatus

_TICK_MS = 100


class LabSetupDialog(QDialog):
    """Lists pending changes; ``exec()`` is Accepted for Apply now.

    Args:
        changes: The pending changes to show.
        timeout_s: Seconds of silence after which the dialog counts as Later.
        parent: Optional parent widget.
    """

    def __init__(
        self, changes: "list[ChangeStatus]", timeout_s: float = 120, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("pytweezer lab setup")
        self.setMinimumWidth(620)
        self._deadline = time.monotonic() + timeout_s

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        noun = "change" if len(changes) == 1 else "changes"
        heading = QLabel(f"{len(changes)} {noun} waiting for this PC")
        heading.setProperty("role", "experimentTitle")
        layout.addWidget(heading)

        region = Region("well")
        for change in changes:
            region.body.addLayout(self._entry(change))
        layout.addWidget(region)

        self.countdown = QLabel()
        self.countdown.setProperty("role", "regionHint")
        layout.addWidget(self.countdown)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        later = QPushButton("Later")
        later.clicked.connect(self.reject)
        apply_now = QPushButton("Apply now")
        apply_now.setObjectName("PrimaryButton")
        apply_now.setDefault(True)
        apply_now.setToolTip(
            "pytweezer closes, the changes are applied in a separate window, "
            "then pytweezer restarts"
        )
        apply_now.clicked.connect(self.accept)
        buttons.addWidget(later)
        buttons.addWidget(apply_now)
        layout.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_TICK_MS)
        self._tick()

    @staticmethod
    def _entry(change: "ChangeStatus") -> QVBoxLayout:
        entry = QVBoxLayout()
        entry.setSpacing(2)
        top = QHBoxLayout()
        title = QLabel(change.title)
        title.setProperty("role", "regionTitle")
        top.addWidget(title)
        if change.admin:
            tag = QLabel("needs admin")
            tag.setProperty("role", "tag")
            top.addWidget(tag)
        top.addStretch(1)
        entry.addLayout(top)
        description = QLabel(change.description)
        description.setProperty("role", "regionHint")
        description.setWordWrap(True)
        description.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        entry.addWidget(description)
        return entry

    def _tick(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._timer.stop()
            self.reject()
            return
        seconds = math.ceil(remaining)
        self.countdown.setText(
            f"Starting pytweezer as normal in {seconds // 60}:{seconds % 60:02d}"
        )
