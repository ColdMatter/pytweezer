"""One cell per physical channel in the NI Cards tab."""

import time
from collections import deque

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pytweezer.drivers.ni_cards import PROTECTED
from pytweezer.GUI.components import CompactDoubleSpinBox, set_state
from pytweezer.GUI.ni_cards.flow_grid import CELL_WIDTH

TRACE_LENGTH = 100
_TRACE_COLOUR = "#5b8def"


def _volts(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{'0' if text == '-0' else text} V"


def _level(value: bool) -> str:
    return "high" if value else "low"


def last_set_text(entry: dict | None, kind: str) -> str:
    """``"last set 1.25 V at 14:02"``, ``"last set high at 14:02"`` or ``"unknown"``."""
    if entry is None:
        return "unknown"
    value = _volts(entry["value"]) if kind == "ao" else _level(entry["value"])
    return (
        f"last set {value} at {time.strftime('%H:%M', time.localtime(entry['time']))}"
    )


def address_text(channel: dict) -> str:
    """The channel's address on its card: ``port3/line5``, ``ao17 · 0…1 V``, ``ai4 · Rse``."""
    local = channel["physical"].split("/", 1)[1]
    if channel["kind"] == "ao" and channel["range"]:
        low, high = channel["range"]
        return f"{local} · {low:g}…{high:g} V"
    if channel["kind"] == "ai" and channel["terminal"]:
        return f"{local} · {channel['terminal']}"
    return local


class Sparkline(QWidget):
    """The last :data:`TRACE_LENGTH` readings as a line, scaled to their own range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: deque[float] = deque(maxlen=TRACE_LENGTH)
        self.setFixedHeight(18)

    def add(self, value: float) -> None:
        self.values.append(value)
        self.update()

    def clear(self) -> None:
        self.values.clear()
        self.update()

    def paintEvent(self, event) -> None:
        if len(self.values) < 2:
            return
        low, high = min(self.values), max(self.values)
        span = (high - low) or 1.0
        width, height = self.width() - 1, self.height() - 2
        step = width / (TRACE_LENGTH - 1)
        offset = width - step * (len(self.values) - 1)
        points = [
            QtCore.QPointF(offset + i * step, 1 + height * (1 - (v - low) / span))
            for i, v in enumerate(self.values)
        ]
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(_TRACE_COLOUR), 1.2))
        painter.drawPolyline(points)
        painter.end()


class ChannelCell(QFrame):
    """A channel's alias, address and control.

    DO cells toggle on click, AO cells take a value (committed on Enter), and
    AI/DI cells toggle watching on click. A held cell is disabled and says who
    holds it.
    """

    write_requested = QtCore.pyqtSignal(str, object)
    watch_toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, channel: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("ChannelCell")
        self.setMinimumWidth(CELL_WIDTH - 12)
        self.physical = channel["physical"]
        self.kind = channel["kind"]
        self.holder: str | None = None
        self.last: dict | None = None
        self.watching = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        name = " / ".join(channel["aliases"]) or self.physical
        self._full_name = name
        self.name = QLabel(name)
        self.name.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.name.setProperty("role", "cellName")
        self.name.setToolTip(f"{name}\n{self.physical}")
        self.address = QLabel(address_text(channel))
        self.address.setProperty("role", "cellAddress")
        layout.addWidget(self.name)
        layout.addWidget(self.address)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.value = QLabel()
        self.value.setProperty("role", "cellValue")
        self.spin = None
        self.trace = None
        if self.kind == "ao":
            self.spin = CompactDoubleSpinBox()
            self.spin.setDecimals(3)
            self.spin.setSingleStep(0.1)
            self.spin.setSuffix(" V")
            self.spin.setRange(*(channel["range"] or (0.0, 0.0)))
            self.spin.setKeyboardTracking(False)
            self.spin.setSpecialValueText("—")
            self.spin.setValue(self.spin.minimum())
            self.spin.lineEdit().returnPressed.connect(self._commit)
            row.addWidget(self.spin)
        elif self.kind != "do":
            row.addWidget(self.value)
        if self.kind == "ai":
            self.trace = Sparkline()
            row.addWidget(self.trace, 1)
        row.addStretch(0 if self.trace else 1)
        layout.addLayout(row)
        self.detail = QLabel()
        self.detail.setProperty("role", "cellAddress")
        layout.addWidget(self.detail)
        if self.kind in ("ai", "di"):
            self.setToolTip("Click to watch: read live while selected")
        elif self.kind == "do":
            self.setToolTip("Click to set the opposite level")
        self._refresh()

    def detail_text(self) -> str:
        return self.detail.text()

    def set_hold(self, holder: str | None) -> None:
        if holder == self.holder:
            return
        self.holder = holder
        if holder is not None and self.watching:
            self.set_watching(False)
        self.setEnabled(holder is None)
        self._refresh()

    def set_last(self, entry: dict | None) -> None:
        if entry == self.last:
            return
        self.last = entry
        if self.spin is not None and not self.spin.hasFocus():
            # The special text shows at the minimum, standing in for "unknown".
            self.spin.setSpecialValueText("—" if entry is None else "")
            self.spin.setValue(self.spin.minimum() if entry is None else entry["value"])
        self._refresh()

    def set_watching(self, watching: bool) -> None:
        self.watching = watching
        if not watching and self.trace is not None:
            self.trace.clear()
        self._refresh()

    def set_reading(self, value) -> None:
        if not self.watching:
            return
        if self.kind == "ai":
            self.value.setText(_volts(value))
            self.trace.add(value)
        else:
            self.value.setText(_level(value))

    def _commit(self) -> None:
        self.write_requested.emit(self.physical, self.spin.value())

    def _refresh(self) -> None:
        if self.holder is not None:
            detail = (
                "protected" if self.holder == PROTECTED else f"held by {self.holder}"
            )
            state = "held"
        elif self.kind in ("ao", "do"):
            detail = last_set_text(self.last, self.kind)
            high = self.kind == "do" and self.last is not None and self.last["value"]
            state = "high" if high else "idle"
        else:
            detail = "watching" if self.watching else "click to watch"
            state = "watching" if self.watching else "idle"
        if self.kind in ("ai", "di") and not self.watching:
            self.value.setText("")
        self.detail.setText(detail)
        set_state(self, state)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        metrics = self.name.fontMetrics()
        self.name.setText(
            metrics.elidedText(
                self._full_name, QtCore.Qt.TextElideMode.ElideRight, self.name.width()
            )
        )

    def mousePressEvent(self, event) -> None:
        if (
            event.button() != QtCore.Qt.MouseButton.LeftButton
            or self.holder is not None
        ):
            return super().mousePressEvent(event)
        if self.kind == "do":
            high = bool(self.last and self.last["value"])
            self.write_requested.emit(self.physical, not high)
        elif self.kind in ("ai", "di"):
            self.set_watching(not self.watching)
            self.watch_toggled.emit(self.physical, self.watching)
        else:
            super().mousePressEvent(event)
