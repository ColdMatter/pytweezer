"""A titled area of the Experiments tab.

Depth encodes role: a ``well`` (recessed) is somewhere you read and pick
from, a ``sheet`` (raised, accent edge) is where you edit. The look lives in
:mod:`pytweezer.GUI.theme` under ``QFrame#Region``.
"""

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class Region(QFrame):
    def __init__(self, kind, title="", hint="", parent=None):
        super().__init__(parent)
        self.setObjectName("Region")
        self.setProperty("kind", kind)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.header = QHBoxLayout()
        self.header.setSpacing(10)
        self.title = QLabel(title)
        self.title.setProperty("role", "regionTitle")
        self.hint = QLabel(hint)
        self.hint.setProperty("role", "regionHint")
        self.header.addWidget(self.title)
        self.header.addWidget(self.hint)
        self.header.addStretch(1)
        if title:
            layout.addLayout(self.header)
        self.body = layout

    def set_hint(self, text):
        self.hint.setText(text)
