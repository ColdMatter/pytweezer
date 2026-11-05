"""The latest message of each subscribed data stream, as a block of key/value pairs.

Embedded in the image viewers, below the image, to show for example the
output of the ``image_stats.py`` analysis or the live cloud fit next to the
frame. The stream list is stored in Properties under ``streamkey`` (default
``sidebarstreams``) and edited through the standard :class:`SubscriptionEditor`.

Call :meth:`DataSidebar.update` from the host applet's poll loop to drain new
data.
"""

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel, QVBoxLayout

from pytweezer.GUI.components import Region
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.servers import DataClient

ROWS_PER_COLUMN = 6
EMPTY_HINT = "No data yet: right-click the image and choose Data sidebar subscriptions"


def _format_value(value):
    if isinstance(value, float):
        return "%.6g" % value
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    return str(value)


class StreamBlock(Region):
    """One stream's latest message: its name as the title, its fields in columns of key | value."""

    def __init__(self, stream, parent=None):
        super().__init__("well", stream, parent=parent)
        self.stream = stream
        self.values: dict[str, QLabel] = {}
        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(2)
        self.body.addLayout(self._grid)

    def show_message(self, payload: dict) -> None:
        items = {k: v for k, v in payload.items() if not k.startswith("dtype")}
        if list(items) != list(self.values):
            self._rebuild(list(items))
        for key, value in items.items():
            self.values[key].setText(_format_value(value))

    def _rebuild(self, keys) -> None:
        while self._grid.count():
            self._grid.takeAt(0).widget().deleteLater()
        self.values = {}
        for index, key in enumerate(keys):
            row, column = index % ROWS_PER_COLUMN, 2 * (index // ROWS_PER_COLUMN)
            name = QLabel(str(key))
            name.setProperty("role", "regionHint")
            value = QLabel()
            self._grid.addWidget(name, row, column)
            self._grid.addWidget(value, row, column + 1)
            self.values[key] = value


class DataSidebar(QtWidgets.QWidget):
    """A row of :class:`StreamBlock`, one per subscribed stream that has sent a message."""

    def __init__(self, props, name, streamkey="sidebarstreams", parent=None):
        super().__init__(parent)
        self.props = props
        self._props = props
        self.name = name
        self.streamkey = streamkey
        self.blocks: dict[str, StreamBlock] = {}

        self.client = DataClient(name.split("/")[-1] + "_sidebar")

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        self.setLayout(layout)
        self.empty = QLabel(EMPTY_HINT)
        self.empty.setProperty("role", "regionHint")
        layout.addWidget(self.empty)
        layout.addStretch(1)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read the stream list from Properties, resubscribe and drop blocks of streams no longer wanted."""
        self.client.unsubscribe()
        streams = self.props.get(self.streamkey, ["--None--"])
        self.client.subscribe(streams)
        for stream in [s for s in self.blocks if not s.startswith(tuple(streams))]:
            self.blocks.pop(stream).deleteLater()
        self.empty.setVisible(not self.blocks)

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
        """Drain new data messages and show the latest of each stream."""
        latest = {}
        while self.client.has_new_data():
            msg = self.client.recv()
            if msg is None:
                break
            # A stats/params message has no array: recv() -> (streamname, dict).
            if isinstance(msg[1], dict):
                latest[msg[0]] = msg[1]
        for stream, payload in latest.items():
            self._block(stream).show_message(payload)
        if latest:
            self.empty.hide()

    def _block(self, stream) -> StreamBlock:
        if stream not in self.blocks:
            block = StreamBlock(stream)
            self.blocks[stream] = block
            # Before the trailing stretch.
            self.layout().insertWidget(
                self.layout().count() - 1, block, alignment=Qt.AlignmentFlag.AlignTop
            )
        return self.blocks[stream]
