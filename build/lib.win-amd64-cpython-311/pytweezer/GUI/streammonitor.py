""" Monitor the content of streams """
import sys
import json
import datetime
from collections import deque
from PyQt6 import QtCore
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from pytweezer.servers import DataClient,ImageClient,CommandClient
from pytweezer.servers.messageclient import MessageClient
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.GUI.theme import UI_FONT_FAMILY, UI_FONT_POINT_SIZE
from pytweezer.logging_utils import get_daily_log_path


class StreamMonitor(QWidget):
    def __init__(self,name,streamtype='Data',parent=None):
        super().__init__(parent)
        if streamtype=='Data':    
            self.stream=DataClient(name)
        elif streamtype=='Image':
            self.stream=ImageClient(name)
        elif streamtype =='Command':
            self.stream=CommandClient(name)
        elif streamtype =='Message':
            self.stream=MessageClient(name)
        self.stream.subscribe('')       #listen to all streams
        self.msglist=[]

        layout=QVBoxLayout()
        layout.addWidget(QLabel(name))
        self.text=QTextEdit()
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.resize(800,800)

        timer=QtCore.QTimer(self)
        timer.timeout.connect(self._update_list)
        timer.start(100)
        self.timer=timer

    def _update_list(self):
        
        while self.stream.has_new_data():
            msg=self.stream.recv()
            if msg != None:
                self.msglist=[msg[0]+repr(msg[1])[:80]]+self.msglist
                self.msglist=self.msglist[:40]
                self.text.setPlainText('\n'.join(self.msglist))


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
    # Column index of the (stretchy) message column, used by the double-click
    # detail dialog and the per-item alignment tweak.
    MESSAGE_COL = 4

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.stream = MessageClient(name)
        self.stream.subscribe('Logs')
        self.max_rows = 200

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Logs'))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Timestamp", "Level", "Host", "Process", "Message"]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
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
        copy_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        copy_action.triggered.connect(self._copy_selection)
        self.table.addAction(copy_action)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.MESSAGE_COL, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(1000, 800)

        self._load_daily_logs()

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._update_list)
        timer.start(100)
        self.timer = timer

        self.table.cellDoubleClicked.connect(self._show_message_dialog)

    def _update_list(self):
        while self.stream.has_new_data():
            msg = self.stream.recv()
            if msg is None:
                continue
            topic, payload = msg
            if not isinstance(payload, dict):
                continue
            self._append_row(payload, prepend=True)

    def _append_row(self, payload, prepend=True):
        row = 0 if prepend else self.table.rowCount()
        self.table.insertRow(row)

        message = str(payload.get("message", ""))
        values = [
            _format_timestamp(payload.get("timestamp", "")),
            payload.get("level", ""),
            payload.get("host", ""),
            payload.get("module", ""),
            message,
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            # Hover any cell in the row to read the full (untruncated) message.
            item.setToolTip(message)
            if col == self.MESSAGE_COL:
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            self.table.setItem(row, col, item)

        if self.table.rowCount() > self.max_rows:
            if prepend:
                self.table.removeRow(self.table.rowCount() - 1)
            else:
                self.table.removeRow(0)

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
    tabs.addTab(StreamMonitor(name, 'Image'), "Image")
    tabs.addTab(StreamMonitor(name, 'Data'), "Data")
    tabs.addTab(StreamMonitor(name, 'Command'), "Command")
    tabs.addTab(StreamMonitor(name, 'Message'), "Message")
    tabs.addTab(LogMonitor(name), "Logs")
    return tabs


def main(name):
    qApp = QApplication(sys.argv)
    Win = make_stream_monitor(name)
    Win.show()
    qApp.exec()




if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main('StreamMonitor')









