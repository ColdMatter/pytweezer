import argparse
import os
import subprocess
import sys

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.GUI.theme import apply_dot_style, apply_label_style
from pytweezer.servers import tweezerpath
from pytweezer.logging_utils import get_logger

logger = get_logger("Applet Launcher")

# States that mean "the applet is up", shared with the tile toggle logic.
RUNNING_STATES = ("running", "up")


DEFAULT_APPLETS = [
    {
        "name": "Image Monitor",
        "script": "pytweezer/GUI/viewers/image_monitor.py",
        "description": "Display image streams",
    },
    {
        "name": "Live Plot",
        "script": "pytweezer/GUI/viewers/live_plot.py",
        "description": "Live plot of data streams",
    },
    {
        "name": "Image Plot Monitor",
        "script": "pytweezer/GUI/viewers/image_plot_monitor.py",
        "description": "Image with x/y projection plots linked to the image axes",
    },
    {
        "name": "Scalar History",
        "script": "pytweezer/GUI/viewers/scalar_history.py",
        "description": "Rolling history of a scalar from a data stream header",
    },
]


class AddAppletWidget(QtWidgets.QWidget):
    def __init__(self, launcher, parent=None):
        super().__init__(parent)
        self.launcher = launcher

        layout = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("add")
        add_button.clicked.connect(self.add_applet)
        layout.addWidget(add_button)

        layout.addWidget(QtWidgets.QLabel("name"))
        self.name_edit = QtWidgets.QLineEdit("")
        layout.addWidget(self.name_edit)

        layout.addWidget(QtWidgets.QLabel("template"))
        self.template_combo = QtWidgets.QComboBox()
        self.template_combo.addItem("Custom")
        for entry in self.launcher.templates:
            self.template_combo.addItem(entry["name"], entry)
        self.template_combo.currentIndexChanged.connect(self.apply_template)
        layout.addWidget(self.template_combo)

        layout.addWidget(QtWidgets.QLabel("script"))
        self.script_edit = QtWidgets.QLineEdit("")
        layout.addWidget(self.script_edit)

        self.setLayout(layout)

    def apply_template(self, index):
        if index <= 0:
            return
        entry = self.template_combo.itemData(index)
        if not entry:
            return
        self.script_edit.setText(entry["script"])
        if not self.name_edit.text().strip():
            self.name_edit.setText(entry["name"])

    def add_applet(self):
        name = self.name_edit.text().strip()
        script = self.script_edit.text().strip()
        if not script:
            logger.warning("Applet Launcher: script is required")
            return
        if not name:
            name = self.template_combo.currentText().strip() or "Applet"
        name = self.launcher.unique_name(name)
        self.launcher.add_applet(name, script)
        self.name_edit.setText("")


class AppletRow(QtWidgets.QFrame):
    """One applet as a tile matching the Servers/Devices rows.

    Layout mirrors :class:`bin.managed_panel.ManagedRow` (name + detail on the
    left; status dot, status label and a Start/Stop toggle on the right) with
    an extra remove button, since applets — unlike servers — are added and
    deleted at runtime. All process bookkeeping stays in :class:`AppletLauncher`;
    this widget just renders state and forwards button clicks.
    """

    def __init__(self, name, script, launcher, parent=None):
        super().__init__(parent)
        self.name = name
        self.launcher = launcher
        self.setObjectName("ProcessTile")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        name_label = QtWidgets.QLabel(name)
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)

        self.scriptLabel = QtWidgets.QLabel(script)
        self.scriptLabel.setToolTip(script)
        layout.addWidget(self.scriptLabel, 1)

        self.dot = QtWidgets.QLabel("●")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)

        self.stateLabel = QtWidgets.QLabel()
        self.stateLabel.setObjectName("StatusLabel")
        self.stateLabel.setMinimumWidth(80)
        layout.addWidget(self.stateLabel)

        self.toggleButton = QtWidgets.QPushButton("Start")
        self.toggleButton.setObjectName("ToggleButton")
        self.toggleButton.setFixedWidth(72)
        self.toggleButton.clicked.connect(self._toggle)
        layout.addWidget(self.toggleButton)

        remove = QtWidgets.QPushButton("✕")
        remove.setObjectName("KillButton")
        remove.setFixedSize(24, 24)
        remove.setToolTip("Remove applet")
        remove.clicked.connect(lambda: self.launcher.delete_applet(self.name))
        layout.addWidget(remove)

        self.setLayout(layout)
        self.set_state("stopped")

    def set_script(self, script):
        self.scriptLabel.setText(script)
        self.scriptLabel.setToolTip(script)

    def _toggle(self):
        if self.property("state") in RUNNING_STATES:
            self.launcher._stop_applet(self.name)
        else:
            self.launcher._start_applet(self.name)

    def set_state(self, state, label=None):
        try:
            apply_dot_style(self.dot, state)
            apply_label_style(self.stateLabel, state, text=label)
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)
            running = state in RUNNING_STATES
            self.toggleButton.setText("Stop" if running else "Start")
            self.toggleButton.setProperty("kind", "stop" if running else "start")
            self.toggleButton.style().unpolish(self.toggleButton)
            self.toggleButton.style().polish(self.toggleButton)
        except RuntimeError:
            # Widget may already be deleted during teardown.
            pass


def _applet_header():
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(10, 0, 10, 0)
    row.setSpacing(10)
    applet = QtWidgets.QLabel("Applet")
    applet.setProperty("role", "heading")
    applet.setMinimumWidth(150)
    row.addWidget(applet)
    script = QtWidgets.QLabel("Script")
    script.setProperty("role", "heading")
    row.addWidget(script, 1)
    status = QtWidgets.QLabel("Status")
    status.setProperty("role", "heading")
    status.setMinimumWidth(14 + 10 + 80)
    row.addWidget(status)
    control = QtWidgets.QLabel("Control")
    control.setProperty("role", "heading")
    control.setMinimumWidth(72)
    row.addWidget(control)
    return row


class AppletLauncher(BWidget):
    def __init__(self, name="Applet Launcher", parent=None):
        super().__init__(name, parent)
        self._processes = {}
        self._rows = {}

        self._ensure_defaults()
        # The applet *list* (name -> script/description) is shared across clients
        # via Properties. Which applets are *running* is deliberately per-machine
        # and lives in local QSettings, so starting an applet on one lab PC does
        # not launch it on every other client, and each client restores only the
        # applets it had running last session.
        self._applets = self._load_applets()
        self._active = self._load_active()

        self.init_gui()
        self.populate()
        self._start_active_applets()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start()

    @property
    def templates(self):
        return list(DEFAULT_APPLETS)

    def _load_applets(self):
        data = self._props.get("Applets", {})
        if not isinstance(data, dict):
            return {}
        # Drop any legacy shared "active" flag: running state is now local
        # (see _load_active). Keeping it out of the in-memory model stops it
        # from being re-published to other clients on the next save.
        applets = {}
        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            applets[name] = {k: v for k, v in entry.items() if k != "active"}
        return applets

    def _save_applets(self, applets):
        self._props.set("Applets", applets)

    def _local_settings(self):
        return QtCore.QSettings("pytweezer", self._name)

    def _load_active(self):
        """Names of applets this machine had running last session (local)."""
        value = self._local_settings().value("active_applets", [])
        if isinstance(value, str):  # QSettings collapses a 1-element list to a str
            value = [value]
        if not value:  # empty list round-trips back as None
            return set()
        return set(value)

    def _save_active(self):
        settings = self._local_settings()
        settings.setValue("active_applets", sorted(self._active))
        # The GUI shell hard-exits (os._exit) after the Qt loop, which skips
        # QSettings' deferred write, so force it to disk now.
        settings.sync()

    def _ensure_defaults(self):
        data = self._props.get("Applets", {})
        if isinstance(data, dict) and data:
            return
        applets = {}
        for entry in DEFAULT_APPLETS:
            applets[entry["name"]] = {
                "script": entry["script"],
                "description": entry.get("description", ""),
            }
        self._save_applets(applets)

    def init_gui(self):
        layout = QtWidgets.QVBoxLayout()

        self.add_widget = AddAppletWidget(self)
        layout.addWidget(self.add_widget)

        layout.addLayout(_applet_header())

        # Rows live in a scrollable container so a long applet list stays usable.
        self._rows_container = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll, 1)

        self.setLayout(layout)

    def populate(self):
        for row in self._rows.values():
            row.setParent(None)
        self._rows.clear()
        for name in sorted(self._applets.keys(), key=str.casefold):
            entry = self._applets[name]
            self._add_row(name, entry.get("script", ""))

    def _add_row(self, name, script):
        row = AppletRow(name, script, self)
        # Insert above the trailing stretch so rows stack top-down.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows[name] = row

    def unique_name(self, base):
        existing = set(self._applets.keys())
        if base not in existing:
            return base
        i = 2
        name = f"{base} {i}"
        while name in existing:
            i += 1
            name = f"{base} {i}"
        return name

    def add_applet(self, name, script):
        if name in self._applets:
            logger.warning("Applet Launcher: applet '%s' already exists", name)
            return
        self._applets[name] = {"script": script}
        self._save_applets(self._applets)
        self._add_row(name, script)

    def delete_applet(self, name):
        self._stop_applet(name)
        self._applets.pop(name, None)
        self._save_applets(self._applets)
        row = self._rows.pop(name, None)
        if row is not None:
            row.setParent(None)

    def _resolve_script_path(self, script):
        if not script:
            return ""
        if os.path.isabs(script):
            return script
        return os.path.normpath(os.path.join(tweezerpath, script))

    def _set_active(self, name, active):
        if active:
            self._active.add(name)
        else:
            self._active.discard(name)
        self._save_active()

    def _set_status(self, name, status, label=None):
        row = self._rows.get(name)
        if row is not None:
            row.set_state(status, label=label)

    def _start_applet(self, name):
        entry = self._applets.get(name)
        if not entry:
            return
        script = entry.get("script", "")
        script_path = self._resolve_script_path(script)
        if not script_path or not os.path.exists(script_path):
            logger.error("Applet Launcher: script not found for '%s': %s", name, script)
            self._set_active(name, False)
            self._set_status(name, "unknown", label="Missing")
            return
        process = self._processes.get(name)
        if process is not None and process.poll() is None:
            return
        process = subprocess.Popen([sys.executable, script_path, name], cwd=tweezerpath)
        self._processes[name] = process
        self._set_active(name, True)
        self._set_status(name, "running")

    def _terminate_process(self, name):
        """Kill the child process, leaving the local active set untouched."""
        process = self._processes.pop(name, None)
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except Exception:
            process.kill()

    def _stop_applet(self, name):
        """User-initiated stop: kill it and remember it should stay stopped."""
        self._terminate_process(name)
        self._set_active(name, False)
        self._set_status(name, "stopped")

    def _start_active_applets(self):
        for name in sorted(self._active):
            if name in self._applets:
                self._start_applet(name)
        # Applets deleted on another client linger in the local set; drop them.
        stale = self._active - set(self._applets)
        if stale:
            self._active -= stale
            self._save_active()

    def refresh_status(self):
        for name, process in list(self._processes.items()):
            if process.poll() is None:
                continue
            self._processes.pop(name, None)
            self._set_active(name, False)
            self._set_status(name, "stopped")

        for name in self._applets.keys():
            if name in self._processes:
                self._set_status(name, "running")

    def closeEvent(self, event):
        # Keep the active set intact so the next session restores exactly what
        # was running when the GUI closed.
        for name in list(self._processes.keys()):
            self._terminate_process(name)
        super().closeEvent(event)


def main(name):
    app = QtWidgets.QApplication(sys.argv)
    window = AppletLauncher(name)
    window.show()
    app.exec()


if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "name",
            nargs="?",
            default="Applet Launcher",
            help="name of this program instance",
        )
        args = parser.parse_args()
        main(args.name)
