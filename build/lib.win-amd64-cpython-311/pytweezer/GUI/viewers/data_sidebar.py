"""A hideable key/value sidebar that displays the latest message on a data stream.

Embedded in the image viewers to show, for example, the output of the
``image_stats.py`` analysis (mean, min, max, ...) next to the image. The stream
list is stored in Properties under ``streamkey`` (default ``sidebarstreams``) and
edited through the standard :class:`SubscriptionEditor`.

Call :meth:`update` from the host applet's poll loop to drain new data.
"""

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QDialog, QVBoxLayout

from pytweezer.servers import DataClient
from pytweezer.GUI.subscription_editor import SubscriptionEditor


def _format_value(value):
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    return str(value)


class DataSidebar(QtWidgets.QWidget):
    def __init__(self, props, name, streamkey="sidebarstreams", parent=None):
        super().__init__(parent)
        self.props = props
        self._props = props
        self.name = name
        self.streamkey = streamkey

        self.client = DataClient(name.split("/")[-1] + "_sidebar")

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        self.setLayout(layout)

        self.streamLabel = QtWidgets.QLabel("(no data)")
        self.streamLabel.setWordWrap(True)
        layout.addWidget(self.streamLabel)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["field", "value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self.setMinimumWidth(180)
        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read the stream list from Properties and resubscribe."""
        self.client.unsubscribe()
        streams = self.props.get(self.streamkey, ["--None--"])
        self.client.subscribe(streams)

    def edit_subscriptions(self):
        """Open the shared subscription editor for this sidebar's stream list."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.name} — data sidebar subscriptions")
        layout = QVBoxLayout()
        editor = SubscriptionEditor(self._props, "Data", streamkey=self.streamkey)
        layout.addWidget(editor)
        dialog.setLayout(layout)
        dialog.exec()
        self.update_subscriptions()

    def update(self):
        """Drain any new data messages and show the most recent one."""
        latest = None
        while self.client.has_new_data():
            msg = self.client.recv()
            if msg is None:
                break
            latest = msg
        if latest is None:
            return
        # A stats/params message has no array: recv() -> (streamname, dict).
        streamname, payload = latest[0], latest[1]
        if not isinstance(payload, dict):
            return
        self.streamLabel.setText(streamname)
        self._show(payload)

    def _show(self, payload):
        items = [(k, v) for k, v in payload.items() if not k.startswith("dtype")]
        self.table.setRowCount(len(items))
        for row, (key, value) in enumerate(items):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(_format_value(value)))
        self.table.resizeColumnsToContents()
