""" Monitor the content of streams """
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pytweezer.servers import DataClient,ImageClient,CommandClient
from pytweezer.servers.messageclient import MessageClient
from pytweezer.GUI.pytweezerQt import BWidget


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
        self.table.setWordWrap(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)

        vheader = self.table.verticalHeader()
        vheader.setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(self.table)
        self.setLayout(layout)
        self.resize(1000, 800)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._update_list)
        timer.start(100)
        self.timer = timer

    def _update_list(self):
        while self.stream.has_new_data():
            msg = self.stream.recv()
            if msg is None:
                continue
            topic, payload = msg
            if not isinstance(payload, dict):
                continue
            self._append_row(payload)

    def _append_row(self, payload):
        row = 0
        self.table.insertRow(row)
        process = payload.get("process", "")
        pid = payload.get("pid", "")

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

        self.table.resizeRowToContents(row)

        if self.table.rowCount() > self.max_rows:
            self.table.removeRow(self.table.rowCount() - 1)



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









