"""Small shared building blocks for pytweezer panels.

Their look lives in :mod:`pytweezer.GUI.theme`; these classes only set the
object names and dynamic properties the stylesheet keys on. See the
``pytweezer-gui-design`` skill for when to use which.
"""

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QVBoxLayout

from pytweezer.GUI.theme import state_style


class Region(QFrame):
    """A titled area of a panel. Depth encodes role.

    ``kind="well"`` (recessed) is somewhere you read and pick from — a list, a
    table, a log. ``kind="sheet"`` (raised, accent edge) is where you edit; a
    panel normally has at most one. Add content to :attr:`body`.
    """

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


def set_state(widget, state):
    """Set the ``state`` property the stylesheet keys on, and restyle the widget.

    Qt doesn't re-evaluate a stylesheet when a dynamic property changes, hence
    the unpolish/polish.
    """
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


_ICONS = {}


def status_icon(state):
    """A coloured dot for a :data:`~pytweezer.GUI.theme.STATE_STYLE` state.

    For item views: the theme's ``::item { color }`` rule overrides per-item
    text colours, but not icons.
    """
    if state not in _ICONS:
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor(state_style(state)[0]))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 10, 10)
        painter.end()
        _ICONS[state] = QtGui.QIcon(pixmap)
    return _ICONS[state]


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Shows 1.5 rather than 1.500000; precision stays at ``decimals()``."""

    def textFromValue(self, value):
        text = f"{value:.{self.decimals()}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
