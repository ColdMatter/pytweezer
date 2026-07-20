import argparse
import os
import subprocess
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import tweezerpath


DEFAULT_APPLETS = [
    {
        "name": "Image Monitor",
        "script": "pytweezer/GUI/viewers/image_monitor.py",
        "description": "Display image streams",
    },
    {
        "name": "Updating Plot",
        "script": "pytweezer/GUI/viewers/updating_plot.py",
        "description": "Live plot of data streams",
    },
]


class AppletModel(QtGui.QStandardItemModel):
    pass


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
            print_error("Applet Launcher: script is required", "warning")
            return
        if not name:
            name = self.template_combo.currentText().strip() or "Applet"
        name = self.launcher.unique_name(name)
        self.launcher.add_applet(name, script)
        self.name_edit.setText("")


class AppletLauncher(BWidget):
    def __init__(self, name="Applet Launcher", parent=None):
        super().__init__(name, parent)
        self._processes = {}
        self._item_map = {}
        self._script_map = {}
        self._status_map = {}

        self._ensure_defaults()
        self._applets = self._load_applets()

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
        return data

    def _save_applets(self, applets):
        self._props.set("Applets", applets)

    def _ensure_defaults(self):
        data = self._props.get("Applets", {})
        if isinstance(data, dict) and data:
            return
        applets = {}
        for entry in DEFAULT_APPLETS:
            applets[entry["name"]] = {
                "script": entry["script"],
                "active": False,
                "description": entry.get("description", ""),
            }
        self._save_applets(applets)

    def init_gui(self):
        layout = QtWidgets.QVBoxLayout()

        self.add_widget = AddAppletWidget(self)
        layout.addWidget(self.add_widget)

        self.table = QtWidgets.QTreeView()
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        self.model = AppletModel(0, 3, self)
        self.model.setHeaderData(0, Qt.Horizontal, "Name")
        self.model.setHeaderData(1, Qt.Horizontal, "Script")
        self.model.setHeaderData(2, Qt.Horizontal, "Status")
        self.model.itemChanged.connect(self.on_item_changed)
        self.table.setModel(self.model)

        layout.addWidget(self.table)

        button_row = QtWidgets.QHBoxLayout()
        delete_button = QtWidgets.QPushButton("del")
        delete_button.clicked.connect(self.delete_selected)
        button_row.addWidget(delete_button)

        restart_button = QtWidgets.QPushButton("restart")
        restart_button.clicked.connect(self.restart_selected)
        button_row.addWidget(restart_button)

        layout.addLayout(button_row)
        self.setLayout(layout)

    def populate(self):
        self.model.blockSignals(True)
        try:
            self.model.removeRows(0, self.model.rowCount())
            self._item_map.clear()
            self._script_map.clear()
            self._status_map.clear()

            for name in sorted(self._applets.keys(), key=str.casefold):
                entry = self._applets[name]
                script = entry.get("script", "")
                active = bool(entry.get("active", False))
                self._add_row(name, script, active)
        finally:
            self.model.blockSignals(False)

    def _add_row(self, name, script, active):
        name_item = QtGui.QStandardItem(name)
        name_item.setCheckable(True)
        name_item.setEditable(False)
        name_item.setCheckState(Qt.Checked if active else Qt.Unchecked)

        script_item = QtGui.QStandardItem(script)
        status_item = QtGui.QStandardItem("running" if active else "stopped")
        status_item.setEditable(False)

        self.model.appendRow([name_item, script_item, status_item])
        self._item_map[name] = name_item
        self._script_map[name] = script_item
        self._status_map[name] = status_item

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
            print_error(f"Applet Launcher: applet '{name}' already exists", "warning")
            return
        self._applets[name] = {"script": script, "active": False}
        self._save_applets(self._applets)
        self._add_row(name, script, False)

    def delete_selected(self):
        selection = self.table.selectionModel().selectedRows()
        if not selection:
            return
        row = selection[0].row()
        name_item = self.model.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()
        self._stop_applet(name)
        self._applets.pop(name, None)
        self._save_applets(self._applets)
        self.model.removeRow(row)
        self._item_map.pop(name, None)
        self._script_map.pop(name, None)
        self._status_map.pop(name, None)

    def restart_selected(self):
        selection = self.table.selectionModel().selectedRows()
        if not selection:
            return
        row = selection[0].row()
        name_item = self.model.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()
        self._stop_applet(name)
        self._start_applet(name)

    def _resolve_script_path(self, script):
        if not script:
            return ""
        if os.path.isabs(script):
            return script
        return os.path.normpath(os.path.join(tweezerpath, script))

    def _set_active(self, name, active):
        if name in self._applets:
            self._applets[name]["active"] = bool(active)
            self._save_applets(self._applets)

    def _set_checked(self, name, checked):
        item = self._item_map.get(name)
        if item is None:
            return
        self.model.blockSignals(True)
        try:
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        finally:
            self.model.blockSignals(False)

    def _set_status(self, name, status):
        item = self._status_map.get(name)
        if item is None:
            return
        item.setText(status)

    def _start_applet(self, name):
        entry = self._applets.get(name)
        if not entry:
            return
        script = entry.get("script", "")
        script_path = self._resolve_script_path(script)
        if not script_path or not os.path.exists(script_path):
            print_error(
                f"Applet Launcher: script not found for '{name}': {script}",
                "error",
            )
            self._set_active(name, False)
            self._set_checked(name, False)
            self._set_status(name, "missing")
            return
        process = self._processes.get(name)
        if process is not None and process.poll() is None:
            return
        process = subprocess.Popen([sys.executable, script_path, name], cwd=tweezerpath)
        self._processes[name] = process
        self._set_active(name, True)
        self._set_checked(name, True)
        self._set_status(name, "running")

    def _stop_applet(self, name):
        process = self._processes.get(name)
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except Exception:
                process.kill()
            self._processes.pop(name, None)
        self._set_active(name, False)
        self._set_status(name, "stopped")
        self._set_checked(name, False)

    def _start_active_applets(self):
        for name, entry in self._applets.items():
            if entry.get("active"):
                self._start_applet(name)

    def on_item_changed(self, item):
        row = item.row()
        column = item.column()
        name_item = self.model.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()

        if column == 0:
            active = name_item.checkState() == Qt.Checked
            if active:
                self._start_applet(name)
            else:
                self._stop_applet(name)
            return

        if column == 1:
            script = item.text().strip()
            if name in self._applets:
                self._applets[name]["script"] = script
                self._save_applets(self._applets)

    def refresh_status(self):
        for name, process in list(self._processes.items()):
            if process.poll() is None:
                continue
            self._processes.pop(name, None)
            self._set_active(name, False)
            self._set_checked(name, False)
            self._set_status(name, "stopped")

        for name in self._applets.keys():
            if name in self._processes:
                self._set_status(name, "running")
            else:
                self._set_status(name, "stopped")

    def closeEvent(self, event):
        for name in list(self._processes.keys()):
            self._stop_applet(name)
        super().closeEvent(event)


def main(name):
    app = QtWidgets.QApplication(sys.argv)
    window = AppletLauncher(name)
    window.show()
    app.exec_()


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
