from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
import os
import zmq

from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import Properties, tweezerpath, icon_path
from pytweezer.servers.configreader import ConfigReader


class AnalysisManagerClient(QtCore.QObject):
    def __init__(self, endpoint: str, timeout_ms: int = 2000):
        super().__init__()
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()

    def request(self, payload: dict):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send_json(payload)
            return socket.recv_json()
        finally:
            socket.close()


class CheckableModel(QtGui.QStandardItemModel):
    pass


class TreeViewWidget(QWidget):
    FROM, SUBJECT, DATE, STREAM = range(4)

    def __init__(self, props, rpc_client, parent=None):
        super().__init__(parent)
        self.props = props
        self._props = props
        self.rpc = rpc_client

        self.dataView = QTreeView()
        self.dataView.setRootIsDecorated(False)
        self.dataView.setAlternatingRowColors(True)

        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self.dataView)
        self.setLayout(dataLayout)

        self.model = self.create_model(self)
        self.dataView.setModel(self.model)

        self.itemdict = {}
        self.filters = {}

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start()

    def create_model(self, parent):
        model = CheckableModel(0, 4, parent)
        model.setHeaderData(self.FROM, Qt.Horizontal, "Name")
        model.setHeaderData(self.SUBJECT, Qt.Horizontal, "Script")
        model.setHeaderData(self.DATE, Qt.Horizontal, "Category")
        model.setHeaderData(self.STREAM, Qt.Horizontal, "Inputstream")
        model.itemChanged.connect(self.on_item_changed)
        return model

    def clear(self):
        self.model.removeRows(0, self.model.rowCount())
        self.itemdict.clear()
        self.filters.clear()

    def populate(self, snapshot: dict):
        self.clear()
        filters = snapshot.get("filters", {})
        running = snapshot.get("running", {})

        for key in sorted(filters.keys(), key=str.casefold):
            entry = filters[key]
            name = entry.get("name")
            category = entry.get("category")
            script = entry.get("script", "")
            streams = entry.get("streams", [])
            active = bool(running.get(key, entry.get("active", False)))
            self.add_item(name, category, script, streams, active)

    def add_item(self, name, category, script, streams, active):
        key = f"{category}/{name}"
        self.filters[name] = [script, category]

        parent_item = self.model.invisibleRootItem()
        check_item = QtGui.QStandardItem(name)
        check_item.setCheckable(True)
        check_item.setCheckState(Qt.Checked if active else Qt.Unchecked)

        parent_item.appendRow(
            [
                check_item,
                QtGui.QStandardItem(script),
                QtGui.QStandardItem(category),
                QtGui.QStandardItem(",".join(streams)),
            ]
        )
        self.itemdict[name] = check_item

    def on_item_changed(self, item):
        name = item.text()
        index = item.index()
        category = self.model.data(self.model.index(index.row(), 2))
        active = item.checkState() == Qt.Checked

        response = self.rpc.request(
            {
                "command": "set_active",
                "category": category,
                "name": name,
                "active": active,
            }
        )
        if not response.get("ok"):
            print_error(f"AnalysisManager RPC error: {response.get('error')}", "error")
            item.setCheckState(Qt.Unchecked if active else Qt.Checked)
            return

        self._props.set(f"{category}/{name}/active", active)

    def refresh_status(self):
        # Keep checkbox state synced with actual process status.
        response = self.rpc.request({"command": "snapshot"})
        if not response.get("ok"):
            return

        running = response.get("running", {})
        for name, item in self.itemdict.items():
            row = item.index().row()
            category = self.model.data(self.model.index(row, 2))
            key = f"{category}/{name}"
            should_be_checked = Qt.Checked if running.get(key, False) else Qt.Unchecked
            if item.checkState() != should_be_checked:
                # Block signals to avoid sending RPC while reflecting status.
                self.model.blockSignals(True)
                item.setCheckState(should_be_checked)
                self.model.blockSignals(False)

    def del_current(self):
        selmodel = self.dataView.selectionModel()
        indexlist = selmodel.selectedRows()
        for index in indexlist:
            name = self.model.data(index)
            category = self.model.data(self.model.index(index.row(), 2))
            response = self.rpc.request(
                {"command": "delete_filter", "category": category, "name": name}
            )
            if not response.get("ok"):
                print_error(f"AnalysisManager delete error: {response.get('error')}", "error")
                continue
            self.model.removeRow(index.row())

    def configure_current(self):
        selmodel = self.dataView.selectionModel()
        indexlist = selmodel.selectedRows()
        if indexlist != []:
            index = indexlist[0]
            name = self.model.data(index)
            category = self.model.data(self.model.index(index.row(), 2))
            dialog = QDialog()
            dialog.setWindowTitle("Dialog")
            layout = QVBoxLayout()
            editor = PropEdit('/Analysis/' + category + '/' + name + '/')
            layout.addWidget(editor)
            dialog.setLayout(layout)
            dialog.exec_()


class AddAnalysisWidget(QtWidgets.QWidget):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager

        layout = QtWidgets.QHBoxLayout()
        addButton = QtWidgets.QPushButton('add')
        addButton.clicked.connect(self.add_filter)
        layout.addWidget(addButton)

        layout.addWidget(QtWidgets.QLabel('name'))
        self.nametext = QtWidgets.QLineEdit('')
        layout.addWidget(self.nametext)

        self.analysistype = QtWidgets.QComboBox()
        self.analysistype.addItem('Image')
        self.analysistype.addItem('Data')
        self.analysistype.currentTextChanged.connect(self.update_streamlist)
        layout.addWidget(self.analysistype)

        analysisdir = manager.analysisdir
        files = [
            f for f in os.listdir(analysisdir)
            if os.path.isfile(analysisdir + f)
            and f[0] != '.'
            and (f.endswith('.py') or f.endswith('.pyx'))
        ]

        self.analysisscript = QtWidgets.QComboBox()
        for f in files:
            self.analysisscript.addItem(f)
        layout.addWidget(self.analysisscript)

        self.streamlist = QComboBox()
        layout.addWidget(self.streamlist)
        self.setLayout(layout)

        self.update_streamlist('Image')

    def update_streamlist(self, category):
        self.streamlist.clear()
        di = self.manager._props.get('/Servers/' + category + 'Stream/active', {})
        for name, value in di.items():
            timedelta = int(max(0, QtCore.QDateTime.currentSecsSinceEpoch() - value['timestamp']))
            self.streamlist.addItem(name + '[%i s]' % timedelta)

    def add_filter(self):
        name = self.nametext.text().strip()
        if not name:
            print_error('AnalysisManager: empty filter name', 'warning')
            return

        category = self.analysistype.currentText()
        script = self.analysisscript.currentText()
        stream = self.streamlist.currentText()
        streams = [stream.split('[')[0]] if stream else ['nostream']

        response = self.manager.rpc.request(
            {
                'command': 'add_filter',
                'category': category,
                'name': name,
                'script': script,
                'streams': streams,
            }
        )
        if not response.get('ok'):
            print_error(f"AnalysisManager add_filter error: {response.get('error')}", 'error')
            return

        self.manager.refresh_snapshot()


class AnalysisManager(BWidget):
    """GUI client for the standalone analysis manager service."""

    def __init__(self, name='Analysis', parent=None):
        super().__init__(name, parent)
        self.conf = ConfigReader.getConfiguration()
        manager_conf = self.conf.get('Servers', {}).get('Analysis Manager', {})
        endpoint = manager_conf.get('rep', 'tcp://127.0.0.1:3111')

        self.rpc = AnalysisManagerClient(endpoint)
        self.analysisdir = tweezerpath + '/pytweezer/analysis/'
        self.init_gui()
        self.refresh_snapshot()

    def init_gui(self):
        layout = QVBoxLayout()

        self.add_widget = AddAnalysisWidget(self)
        layout.addWidget(self.add_widget)

        self.tvw = TreeViewWidget(self._props, self.rpc)
        layout.addWidget(self.tvw)

        delButton = QPushButton('del')
        delButton.clicked.connect(self.del_entry)
        layout.addWidget(delButton)

        configureButton = QPushButton('configure')
        configureButton.clicked.connect(self.configure_filter)
        layout.addWidget(configureButton)

        refreshButton = QPushButton('refresh')
        refreshButton.clicked.connect(self.refresh_snapshot)
        layout.addWidget(refreshButton)

        self.setLayout(layout)

    def refresh_snapshot(self):
        response = self.rpc.request({'command': 'snapshot'})
        if not response.get('ok'):
            print_error(f"AnalysisManager snapshot error: {response.get('error')}", 'error')
            return

        self.analysisdir = response.get('analysisdir', self.analysisdir)
        self.tvw.populate(response)

    def del_entry(self):
        self.tvw.del_current()

    def configure_filter(self):
        self.tvw.configure_current()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile(icon_path + 'pytweezer_analysis_manager_icon.svg')
    app.setWindowIcon(icon)

    window = AnalysisManager()
    window.show()
    app.exec_()


if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
