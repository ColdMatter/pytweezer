"""The Streams tab: live views of every stream type, and the log feed.

Each monitor keeps listening while hidden but only draws while visible.
"""

import datetime
import json
import sys
from collections import deque
from typing import ClassVar

from PyQt6 import QtCore
from PyQt6.QtGui import QAction, QFont, QFontMetrics, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.components import Region, status_icon
from pytweezer.GUI.theme import UI_FONT_FAMILY, UI_FONT_POINT_SIZE
from pytweezer.logging_utils import get_daily_log_path
from pytweezer.servers import CommandClient, DataClient, ImageClient
from pytweezer.servers.messageclient import MessageClient

_POLL_MS = 100
_FILTER_WIDTH = 220
_TOOLTIP_CHARS = 2000
_CONTENT_CHARS = 200


def _make_table(columns, stretch_column):
    table = QTableWidget(0, len(columns))
    table.setObjectName("FeedTable")
    table.setHorizontalHeaderLabels(columns)
    table.setShowGrid(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    for column in range(len(columns)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)
    return table


def _filter_box(placeholder, slot):
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setClearButtonEnabled(True)
    box.setFixedWidth(_FILTER_WIDTH)
    box.textChanged.connect(slot)
    return box


class StreamMonitor(QWidget):
    """The most recent messages on every stream of one type, newest first."""

    COLUMNS: ClassVar = ["Received", "Stream", "Content"]
    CONTENT_COL = 2

    def __init__(self, name, streamtype="Data", parent=None):
        super().__init__(parent)
        clients = {
            "Data": DataClient,
            "Image": ImageClient,
            "Command": CommandClient,
            "Message": MessageClient,
        }
        self.stream = clients[streamtype](name)
        self.stream.subscribe("")  # listen to all streams
        self.max_rows = 40
        self.rows = deque(maxlen=self.max_rows)  # (received, topic, content)

        self.filter = _filter_box("Filter by stream…", self._apply_filter)
        self.pause = QCheckBox("Pause")
        self.pause.setToolTip("Stop collecting messages; those that arrive are dropped")
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)

        self.table = _make_table(self.COLUMNS, self.CONTENT_COL)
        region = Region(
            "well",
            f"{streamtype} streams",
            f"last {self.max_rows} messages, newest first",
        )
        region.header.addWidget(self.filter)
        region.header.addWidget(self.pause)
        region.header.addWidget(self.clear_button)
        region.body.addWidget(self.table, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(region)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_list)
        self.timer.start(_POLL_MS)

    def _update_list(self):
        new = []
        while self.stream.has_new_data():
            msg = self.stream.recv()
            if msg is not None and not self.pause.isChecked():
                received = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                new.append((received, str(msg[0]), repr(msg[1])[:_TOOLTIP_CHARS]))
        self.rows.extend(new)
        if new and self.isVisible():
            for row in new:
                self._insert(row)
            while self.table.rowCount() > self.max_rows:
                self.table.removeRow(self.table.rowCount() - 1)

    def _insert(self, row):
        self.table.insertRow(0)
        for column, text in enumerate(row):
            item = QTableWidgetItem(text[:_CONTENT_CHARS])
            if column == self.CONTENT_COL:
                item.setToolTip(text)
            self.table.setItem(0, column, item)
        self.table.setRowHidden(0, not self._matches(row[1]))

    def _matches(self, topic):
        return self.filter.text().strip().casefold() in topic.casefold()

    def _apply_filter(self):
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(
                row, not self._matches(self.table.item(row, 1).text())
            )

    def clear(self):
        self.rows.clear()
        self.table.setRowCount(0)

    def showEvent(self, event):
        super().showEvent(event)
        self.table.setRowCount(0)
        for row in self.rows:  # oldest first, so each lands above the last
            self._insert(row)


def _format_timestamp(raw):
    """Human-friendly ``YYYY-MM-DD HH:MM:SS`` from an ISO timestamp string.

    Log timestamps arrive as ``isoformat(timespec="milliseconds")`` (e.g.
    ``2026-07-08T13:23:00.123+01:00``); this drops the sub-second precision,
    the timezone offset, and the ``T`` separator. Falls back to the raw value
    if it can't be parsed.
    """
    if not raw:
        return ""
    try:
        return datetime.datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(raw)


class LogMonitor(QWidget):
    """Log records from every process, newest first, with level and text filters."""

    # Column index of the (stretchy) message column, used by the double-click
    # detail dialog.
    MESSAGE_COL = 4
    LEVEL_COL = 1
    #: Unknown level names count as informational.
    LEVELS: ClassVar = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    LEVEL_STATE: ClassVar = {
        "WARNING": "starting",
        "ERROR": "crashed",
        "CRITICAL": "crashed",
    }

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.stream = MessageClient(name)
        self.stream.subscribe("Logs")
        self.max_rows = 200

        self.min_level = QComboBox()
        self.min_level.addItem("All levels", 0)
        self.min_level.addItem("Warnings and errors", 30)
        self.min_level.addItem("Errors only", 40)
        self.min_level.setFixedWidth(_FILTER_WIDTH)
        self.min_level.currentIndexChanged.connect(self._apply_filter)
        self.filter = _filter_box("Filter by text…", self._apply_filter)

        self.table = _make_table(
            ["Timestamp", "Level", "Host", "Process", "Message"], self.MESSAGE_COL
        )
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        # Give each row room for two lines so longer messages wrap rather than
        # being clipped to one; hover/double-click still reveal the full text.
        # Measure with a QFont matching the app stylesheet (QSS font settings
        # don't show up in a widget's default QFontMetrics), and set it on the
        # table so rendered and measured line heights agree. The extra pixels
        # cover the cell's own top/bottom margins.
        self.table.setWordWrap(True)
        row_font = QFont(UI_FONT_FAMILY, UI_FONT_POINT_SIZE)
        self.table.setFont(row_font)
        self._row_height = QFontMetrics(row_font).lineSpacing() * 2 + 12
        self.table.verticalHeader().setDefaultSectionSize(self._row_height)

        copy_action = QAction("Copy", self.table)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.setShortcutContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        copy_action.triggered.connect(self._copy_selection)
        self.table.addAction(copy_action)
        self.table.cellDoubleClicked.connect(self._show_message_dialog)

        region = Region(
            "well",
            "Logs",
            "newest first; double-click a row to read the whole message",
        )
        region.header.addWidget(self.min_level)
        region.header.addWidget(self.filter)
        region.body.addWidget(self.table, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(region)

        self._load_daily_logs()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_list)
        self.timer.start(_POLL_MS)

    def _update_list(self):
        while self.stream.has_new_data():
            msg = self.stream.recv()
            if msg is None:
                continue
            _topic, payload = msg
            if not isinstance(payload, dict):
                continue
            self._append_row(payload, prepend=True)

    def _append_row(self, payload, prepend=True):
        row = 0 if prepend else self.table.rowCount()
        self.table.insertRow(row)

        message = str(payload.get("message", ""))
        level = str(payload.get("level", ""))
        values = [
            _format_timestamp(payload.get("timestamp", "")),
            level,
            payload.get("host", ""),
            payload.get("module", ""),
            message,
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            # Hover any cell in the row to read the full (untruncated) message.
            item.setToolTip(message)
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            if col == self.LEVEL_COL and level.upper() in self.LEVEL_STATE:
                item.setIcon(status_icon(self.LEVEL_STATE[level.upper()]))
            self.table.setItem(row, col, item)
        self.table.setRowHidden(row, not self._matches(level, message))

        if self.table.rowCount() > self.max_rows:
            if prepend:
                self.table.removeRow(self.table.rowCount() - 1)
            else:
                self.table.removeRow(0)

    def _matches(self, level, message):
        severity = self.LEVELS.get(level.upper(), 20)
        needle = self.filter.text().strip().casefold()
        return severity >= self.min_level.currentData() and (
            not needle or needle in message.casefold()
        )

    def _apply_filter(self):
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(
                row,
                not self._matches(
                    self.table.item(row, self.LEVEL_COL).text(),
                    self.table.item(row, self.MESSAGE_COL).text(),
                ),
            )

    def _load_daily_logs(self):
        log_path = get_daily_log_path()
        if not log_path.exists():
            return

        entries = deque(maxlen=self.max_rows)
        try:
            with log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        entries.append(payload)
        except OSError:
            return

        for payload in entries:
            self._append_row(payload, prepend=False)

    def _copy_selection(self):
        item = self.table.currentItem()
        if item is None:
            return
        QApplication.clipboard().setText(item.text())

    def _show_message_dialog(self, row, _column):
        item = self.table.item(row, self.MESSAGE_COL)
        message = item.text() if item else ""

        dialog = QDialog(self)
        dialog.setWindowTitle("Log Message")
        dialog.resize(700, 400)

        layout = QVBoxLayout(dialog)
        text = QTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(message)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()


def make_stream_monitor(name, parent=None):
    """Build the nested stream-monitor tab widget without owning a QApplication.

    Returns a QTabWidget with Image/Data/Command/Message stream views plus a
    Logs view, so it can be shown standalone or embedded as a tab in a larger
    window.
    """
    tabs = QTabWidget(parent)
    tabs.addTab(StreamMonitor(name, "Image"), "Image")
    tabs.addTab(StreamMonitor(name, "Data"), "Data")
    tabs.addTab(StreamMonitor(name, "Command"), "Command")
    tabs.addTab(StreamMonitor(name, "Message"), "Message")
    tabs.addTab(LogMonitor(name), "Logs")
    return tabs


def main(name):
    qApp = QApplication(sys.argv)
    Win = make_stream_monitor(name)
    Win.show()
    qApp.exec()


if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        main("StreamMonitor")
