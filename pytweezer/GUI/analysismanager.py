"""The Analysis tab: add, start, stop, configure and delete analysis filters.

State comes from the Analysis Manager's snapshot, polled while the tab is
visible; the table is updated in place so selection survives.
"""

import os

import zmq
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from pytweezer.configuration.config import get_config
from pytweezer.GUI.components import Region, set_state, status_icon
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.logging_utils import get_logger
from pytweezer.servers import icon_path, tweezerpath

logger = get_logger("Analysis Manager UI")


class AnalysisManagerClient(QtCore.QObject):
    def __init__(self, endpoint: str, timeout_ms: int = 800):
        super().__init__()
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self._socket = None
        self._poller = zmq.Poller()

    def _connect_socket(self):
        if self._socket is not None:
            self._poller.unregister(self._socket)
            self._socket.close()
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket
        self._poller.register(socket, zmq.POLLIN)

    def _reset_socket(self):
        if self._socket is None:
            return
        try:
            self._poller.unregister(self._socket)
        except Exception:
            pass
        self._socket.close()
        self._socket = None

    def close(self):
        self._reset_socket()

    def request(
        self, payload: dict, retries: int = 2, retry_delay_s: float = 0.12
    ) -> dict:
        last_error = None
        attempts = max(1, int(retries) + 1)

        for attempt in range(attempts):
            if self._socket is None:
                self._connect_socket()
            try:
                self._socket.send_json(payload)
                events = dict(self._poller.poll(self.timeout_ms))
                if events.get(self._socket) == zmq.POLLIN:
                    return self._socket.recv_json()
                raise zmq.Again()
            except zmq.Again as error:
                last_error = error
                self._reset_socket()
                if attempt < attempts - 1:
                    QtCore.QThread.msleep(int(max(0, retry_delay_s) * 1000))
                    continue
                return {
                    "ok": False,
                    "error": (
                        f"Analysis Manager not responding at {self.endpoint} "
                        f"(timeout {self.timeout_ms} ms)"
                    ),
                }
            except Exception as error:
                last_error = error
                self._reset_socket()
                return {"ok": False, "error": str(error)}

        return {
            "ok": False,
            "error": str(last_error) if last_error else "unknown RPC error",
        }


_POLL_MS = 1000
_COLUMNS = ["Name", "State", "Type", "Script", "Input stream"]
_KEY = Qt.ItemDataRole.UserRole
_STATE_TIPS = {
    "crashed": "Was started, but its process has exited",
}


def filter_state(entry, running):
    """The :data:`~pytweezer.GUI.theme.STATE_STYLE` state of one filter."""
    if running:
        return "running"
    return "crashed" if entry.get("active") else "stopped"


class FilterTable(QTableWidget):
    """One row per analysis filter, keyed ``"<category>/<name>"``."""

    def __init__(self, parent=None):
        super().__init__(0, len(_COLUMNS), parent)
        self.setObjectName("FilterTable")
        self.setShowGrid(False)
        self.setHorizontalHeaderLabels(_COLUMNS)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            _COLUMNS.index("Input stream"), QHeaderView.ResizeMode.Stretch
        )
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self._keys = []

    def selected_key(self):
        rows = self.selectionModel().selectedRows()
        return self.item(rows[0].row(), 0).data(_KEY) if rows else None

    def update_filters(self, filters, running):
        keys = sorted(filters, key=str.casefold)
        selected = self.selected_key()
        if keys != self._keys:
            self._keys = keys
            self.setRowCount(len(keys))
            for row in range(len(keys)):
                for column in range(len(_COLUMNS)):
                    self.setItem(row, column, QTableWidgetItem())
        for row, key in enumerate(keys):
            entry = filters[key]
            state = filter_state(entry, running.get(key, False))
            cells = [
                entry["name"],
                state.capitalize(),
                entry["category"],
                entry.get("script", ""),
                ", ".join(entry.get("streams", [])),
            ]
            for column, text in enumerate(cells):
                item = self.item(row, column)
                item.setText(text)
                item.setToolTip(_STATE_TIPS.get(state, "") if column == 1 else "")
            self.item(row, 0).setData(_KEY, key)
            self.item(row, 1).setIcon(status_icon(state))
            if key == selected and not self.item(row, 0).isSelected():
                self.selectRow(row)


class AddFilterSheet(Region):
    """The form that adds a filter: name, type, script and input stream."""

    added = QtCore.pyqtSignal(str, str, str, list)

    def __init__(self, analysisdir, stream_source, parent=None):
        super().__init__(
            "sheet",
            "Add a filter",
            "runs a script from pytweezer/analysis on a live stream",
            parent,
        )
        self._stream_source = stream_source
        self.scripts_by_category = self._scan_scripts(analysisdir)

        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. tweezer atoms")
        self.category = QComboBox()
        self.category.addItems(["Image", "Data"])
        self.script = QComboBox()
        self.stream = QComboBox()
        self.add_button = QPushButton("Add filter")
        self.add_button.setObjectName("PrimaryButton")
        for widget, width in (
            (self.name, 240),
            (self.category, 110),
            (self.script, 240),
            (self.stream, 240),
        ):
            widget.setFixedWidth(width)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        for column, (text, widget) in enumerate(
            [
                ("Name", self.name),
                ("Type", self.category),
                ("Script", self.script),
                ("Input stream", self.stream),
            ]
        ):
            label = QLabel(text)
            label.setProperty("role", "regionHint")
            grid.addWidget(label, 0, column)
            grid.addWidget(widget, 1, column)
        grid.addWidget(self.add_button, 1, 4)
        grid.setColumnStretch(5, 1)
        self.body.addLayout(grid)

        self.category.currentTextChanged.connect(self.update_scripts)
        self.category.currentTextChanged.connect(self.update_streams)
        self.name.textChanged.connect(self._update_enabled)
        self.name.returnPressed.connect(self._add)
        self.add_button.clicked.connect(self._add)
        self.update_scripts()
        self.update_streams()

    @staticmethod
    def _classify_script(path):
        """Guess a script's category from the streams PropertyAttribute it
        declares (or inherits from analysis_base.py's ImageAnalysis/
        DataAnalysis) -- the same 'imagestreams'/'datastreams' key the
        manager stores the filter's input streams under. Scripts that
        declare neither (or, ambiguously, both -- e.g. analysis_base.py
        itself) are left uncategorized and don't appear in either list.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            return None
        has_image = "imagestreams" in text or "ImageAnalysis" in text
        has_data = "datastreams" in text or "DataAnalysis" in text
        if has_image and not has_data:
            return "Image"
        if has_data and not has_image:
            return "Data"
        return None

    @classmethod
    def _scan_scripts(cls, analysisdir):
        by_category = {"Image": [], "Data": []}
        try:
            filenames = sorted(os.listdir(analysisdir))
        except OSError:
            return by_category
        for f in filenames:
            path = os.path.join(analysisdir, f)
            if not os.path.isfile(path) or f[0] == ".":
                continue
            if not f.endswith((".py", ".pyx")):
                continue
            category = cls._classify_script(path)
            if category is not None:
                by_category[category].append(f)
        return by_category

    def update_scripts(self, _category=None):
        self.script.clear()
        self.script.addItems(
            self.scripts_by_category.get(self.category.currentText(), [])
        )
        self._update_enabled()

    def update_streams(self, _category=None):
        """Offer the streams currently publishing, keeping the user's choice."""
        category = self.category.currentText()
        now = QtCore.QDateTime.currentSecsSinceEpoch()
        streams = {
            name: max(0, now - int(value["timestamp"]))
            for name, value in self._stream_source(category).items()
        }
        current = self.stream.currentData()
        if list(streams) != [
            self.stream.itemData(i) for i in range(self.stream.count())
        ]:
            self.stream.clear()
            for name in streams:
                self.stream.addItem(name, name)
            index = self.stream.findData(current)
            if index >= 0:
                self.stream.setCurrentIndex(index)
        for i in range(self.stream.count()):
            age = streams[self.stream.itemData(i)]
            self.stream.setItemText(
                i, self.stream.itemData(i) + (f"  ({age} s ago)" if age < 3600 else "")
            )
        self._update_enabled()

    def _update_enabled(self):
        self.add_button.setEnabled(
            bool(self.name.text().strip()) and self.script.count() > 0
        )

    def _add(self):
        if not self.add_button.isEnabled():
            return
        stream = self.stream.currentData()
        self.added.emit(
            self.category.currentText(),
            self.name.text().strip(),
            self.script.currentText(),
            [stream] if stream else ["nostream"],
        )

    def clear_name(self):
        self.name.clear()


class AnalysisManager(BWidget):
    """GUI client for the standalone analysis manager service."""

    def __init__(self, name="Analysis", parent=None):
        super().__init__(name, parent)
        self.conf = get_config()
        manager_conf = self.conf["Servers"]["Analysis Manager"]
        endpoint = f"tcp://{manager_conf['host']}:{manager_conf['port']}"
        self.rpc = AnalysisManagerClient(endpoint)
        self.analysisdir = tweezerpath + "/pytweezer/analysis/"
        self._last_snapshot_error = ""
        self._filters = {}
        self._running = {}
        self.init_gui()
        self.refresh_snapshot()

        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._poll_tick)
        self._poll.start(_POLL_MS)

    def init_gui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)

        self.status = QLabel("Waiting for the Analysis Manager…")
        self.status.setObjectName("StatusLabel")
        layout.addWidget(self.status)

        self.table = FilterTable()
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(lambda *_: self.configure_filter())
        self.filters_region = Region("well", "Filters", "")
        self.filters_region.body.addWidget(self.table, 1)

        self.toggle_button = QPushButton("Start")
        self.toggle_button.setObjectName("ToggleButton")
        self.toggle_button.clicked.connect(self.toggle_filter)
        self.configure_button = QPushButton("Configure…")
        self.configure_button.setToolTip("Edit this filter's properties")
        self.configure_button.clicked.connect(self.configure_filter)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("DangerButton")
        self.delete_button.setToolTip("Stop the filter and remove it from the manager")
        self.delete_button.clicked.connect(self.del_entry)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.toggle_button)
        buttons.addSpacing(14)
        buttons.addWidget(self.configure_button)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_button)
        self.filters_region.body.addLayout(buttons)
        layout.addWidget(self.filters_region, 1)

        self.add_sheet = AddFilterSheet(self.analysisdir, self._active_streams)
        self.add_sheet.added.connect(self.add_filter)
        layout.addWidget(self.add_sheet)
        self._update_buttons()

    def _active_streams(self, category):
        return self._props.get("/Servers/" + category + "Stream/active", {})

    # -- state ---------------------------------------------------------------

    def _poll_tick(self):
        # Hidden behind another tab: skip, the manager may be across a network.
        if self.isVisible():
            self.refresh_snapshot()
            self.add_sheet.update_streams()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_snapshot()
        self.add_sheet.update_streams()

    def _show_status(self, text, state=""):
        self.status.setText(text)
        set_state(self.status, state)

    def refresh_snapshot(self):
        response = self.rpc.request({"command": "snapshot"}, retries=0, retry_delay_s=0)
        if not response.get("ok"):
            error_text = str(response.get("error", "unknown error"))
            self._show_status(
                f"The Analysis Manager is not responding: {error_text}", "crashed"
            )
            # Avoid flooding the console with the same timeout while service starts.
            if error_text != self._last_snapshot_error:
                logger.warning(f"AnalysisManager snapshot error: {error_text}")
                self._last_snapshot_error = error_text
            return

        self._last_snapshot_error = ""
        self.analysisdir = response.get("analysisdir", self.analysisdir)
        filters = response.get("filters", {})
        running = response.get("running", {})
        self.table.update_filters(filters, running)
        self._filters = filters
        self._running = running
        n_running = sum(bool(running.get(key)) for key in filters)
        summary = f"{n_running} running, {len(filters) - n_running} not"
        self.filters_region.set_hint(summary if filters else "none yet: add one below")
        self._show_status("Connected to the Analysis Manager", "running")
        self._update_buttons()

    def _update_buttons(self):
        key = self.table.selected_key()
        filters = self._filters
        selected = key in filters
        self.configure_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)
        self.toggle_button.setEnabled(selected)
        is_running = selected and self._running.get(key, False)
        self.toggle_button.setText("Stop" if is_running else "Start")
        self.toggle_button.setProperty("kind", "stop" if is_running else "start")
        self.toggle_button.style().unpolish(self.toggle_button)
        self.toggle_button.style().polish(self.toggle_button)

    # -- actions -------------------------------------------------------------

    def _selected_entry(self):
        key = self.table.selected_key()
        return self._filters.get(key)

    def _request(self, what, payload):
        response = self.rpc.request(payload)
        if not response.get("ok"):
            error = response.get("error", "unknown error")
            logger.error(f"AnalysisManager {what} error: {error}")
            self._show_status(f"{what}: {error}", "crashed")
            return None
        return response

    def toggle_filter(self):
        entry = self._selected_entry()
        if entry is None:
            return
        key = f"{entry['category']}/{entry['name']}"
        active = not self._running.get(key, False)
        if self._request(
            "Starting" if active else "Stopping",
            {
                "command": "set_active",
                "category": entry["category"],
                "name": entry["name"],
                "active": active,
            },
        ):
            self._props.set(f"{entry['category']}/{entry['name']}/active", active)
            self.refresh_snapshot()

    def add_filter(self, category, name, script, streams):
        if f"{category}/{name}" in self._filters:
            self._show_status(
                f"Adding: a {category} filter called {name} already exists", "crashed"
            )
            return
        if self._request(
            "Adding",
            {
                "command": "add_filter",
                "category": category,
                "name": name,
                "script": script,
                "streams": streams,
            },
        ):
            self.add_sheet.clear_name()
            self.refresh_snapshot()

    def del_entry(self):
        entry = self._selected_entry()
        if entry is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete filter",
            f"Stop and delete the filter {entry['name']}? Its properties are lost.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._request(
            "Deleting",
            {
                "command": "delete_filter",
                "category": entry["category"],
                "name": entry["name"],
            },
        ):
            self.refresh_snapshot()

    def configure_filter(self):
        entry = self._selected_entry()
        if entry is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Configure {entry['name']}")
        dialog.resize(700, 600)
        layout = QVBoxLayout(dialog)
        layout.addWidget(PropEdit(f"/Analysis/{entry['category']}/{entry['name']}/"))
        dialog.exec()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile(icon_path + "pytweezer_analysis_manager_icon.svg")
    app.setWindowIcon(icon)

    window = AnalysisManager()
    window.show()
    app.exec()


if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        main()
