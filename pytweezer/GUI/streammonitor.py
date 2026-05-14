""" Monitor the content of streams """
import json
from collections import deque
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pytweezer.servers import DataClient,ImageClient,CommandClient
from pytweezer.servers.messageclient import MessageClient
from pytweezer.GUI.pytweezerQt import BWidget
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


class LogMonitor(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.stream = MessageClient(name)
        self.stream.subscribe('Logs')
        self.max_rows = 200

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Logs'))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Timestamp", "Level", "Host", "Process", "Logger", "Message"]
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        copy_action = QAction("Copy", self.table)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        copy_action.triggered.connect(self._copy_selection)
        self.table.addAction(copy_action)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)

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

        values = [
            payload.get("timestamp", ""),
            payload.get("level", ""),
            payload.get("host", ""),
            payload.get("module", ""),
            payload.get("logger", ""),
            payload.get("message", ""),
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if col == 5:
                item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
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
        item = self.table.item(row, 5)
        message = item.text() if item else ""

        dialog = QDialog(self)
        dialog.setWindowTitle("Log Message")
        dialog.resize(700, 400)

        layout = QVBoxLayout(dialog)
        text = QTextEdit(dialog)
        text.setReadOnly(True)
        text.setPlainText(message)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec_()



def main(name):
    qApp = QApplication(sys.argv)
    Win = QTabWidget()
    image = StreamMonitor(name,'Image')
    Win.addTab(image,"Image")
    image = StreamMonitor(name,'Data')
    Win.addTab(image,"Data")
    image = StreamMonitor(name,'Command')
    Win.addTab(image,"Command")
    image = StreamMonitor(name,'Message')
    Win.addTab(image,"Message")
    logs = LogMonitor(name)
    Win.addTab(logs, "Logs")
    Win.show()
    qApp.exec_()




if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main('StreamMonitor')









