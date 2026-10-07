"""Table of the running, waiting and recently finished tasks, with their controls."""

from datetime import datetime

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.theme import state_style

COLUMNS = [
    "RID",
    "Experiment",
    "Label",
    "Status",
    "Progress",
    "Priority",
    "Time",
    "Submitter",
]

# Task status -> theme state, so tasks share the process-status traffic lights.
_STATUS_STATE = {
    "running": "running",
    "paused": "starting",
    "queued": "stopped",
    "held": "stopped",
    "terminated": "starting",
    "interrupted": "starting",
    "aborted": "starting",
    "failed": "crashed",
    "crashed": "crashed",
}

_TASK = QtCore.Qt.ItemDataRole.UserRole


class QueueView(QWidget):
    """Shows a manager snapshot; buttons emit ``action_requested(command, fields)``."""

    action_requested = QtCore.pyqtSignal(str, dict)
    edit_requested = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(self._show_details)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.buttons = {}
        for command, text in [
            ("pause", "Pause"),
            ("resume", "Resume"),
            ("terminate", "Terminate"),
            ("abort", "Abort"),
            ("hold", "Hold"),
            ("release", "Release"),
            ("delete", "Delete"),
            ("raise", "Priority +"),
            ("lower", "Priority −"),
            ("edit", "Edit as new"),
        ]:
            button = QPushButton(text)
            button.clicked.connect(lambda _checked, c=command: self._clicked(c))
            buttons.addWidget(button)
            self.buttons[command] = button
        self.buttons["terminate"].setToolTip("Stop after the current point")
        self.buttons["abort"].setToolTip(
            "Kill the task now. A device call already in progress still completes."
        )
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._update_buttons()

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        selected = self.selected_task()
        selected_rid = selected["rid"] if selected else None
        rows = []
        if snapshot.get("running"):
            rows.append(snapshot["running"])
        rows += snapshot.get("queue", [])
        rows += snapshot.get("history", [])
        self.table.setRowCount(len(rows))
        reselect = None
        for row, task in enumerate(rows):
            for column, text in enumerate(_cells(task)):
                item = QTableWidgetItem(text)
                item.setData(_TASK, task)
                if column == 3 and task["status"] in _STATUS_STATE:
                    item.setForeground(
                        QtGui.QColor(state_style(_STATUS_STATE[task["status"]])[0])
                    )
                if task.get("error"):
                    item.setToolTip(task["error"].strip().splitlines()[-1])
                self.table.setItem(row, column, item)
            if task["rid"] == selected_rid:
                reselect = row
        if reselect is not None:
            self.table.selectRow(reselect)
        self._update_buttons()

    def selected_task(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(_TASK)

    def _update_buttons(self):
        task = self.selected_task()
        status = task["status"] if task else None
        allowed = {
            "pause": status == "running" and task.get("requested") != "pause",
            "resume": status == "paused" or (task or {}).get("requested") == "pause",
            "terminate": status in ("running", "paused"),
            "abort": status in ("running", "paused"),
            "hold": status == "queued",
            "release": status == "held",
            "delete": status in ("queued", "held"),
            "raise": status in ("queued", "held", "running", "paused"),
            "lower": status in ("queued", "held", "running", "paused"),
            "edit": task is not None,
        }
        for command, button in self.buttons.items():
            button.setEnabled(bool(allowed[command]))

    def _clicked(self, command):
        task = self.selected_task()
        if task is None:
            return
        rid = task["rid"]
        if command == "edit":
            self.edit_requested.emit(task)
        elif command in ("raise", "lower"):
            step = 1 if command == "raise" else -1
            self.action_requested.emit(
                "set_priority", {"rid": rid, "priority": task["priority"] + step}
            )
        elif command == "abort" and not self.confirm(
            f"Abort task {rid} now? Its current point is lost."
        ):
            return
        else:
            self.action_requested.emit(command, {"rid": rid})

    def confirm(self, question):
        answer = QMessageBox.question(self, "Abort task", question)
        return answer == QMessageBox.StandardButton.Yes

    def _show_details(self, row, _column):
        task = self.table.item(row, 0).data(_TASK)
        text = (
            f"Task {task['rid']}: {task['experiment']}.{task['class_name']}\n"
            f"Status: {task['status']}\n"
            f"File: {task.get('h5_path') or '—'}\n"
        )
        if task.get("error"):
            text += f"\n{task['error']}"
        box = QMessageBox(self)
        box.setWindowTitle(f"Task {task['rid']}")
        box.setText(text)
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        box.open()


def _cells(task):
    total = task.get("points_total")
    progress = f"{task['points_done']}/{total}" if total else ""
    when = task.get("t_start") or task.get("due_time") or task.get("t_submit")
    status = task["status"]
    if task.get("requested") and status in ("running", "paused"):
        status = f"{status} ({task['requested']} requested)"
    return [
        str(task["rid"]),
        f"{task['class_name']}  ({task['experiment'].rsplit('.', 1)[-1]})",
        task.get("label", ""),
        status,
        progress,
        str(task["priority"]),
        _short_time(when),
        task.get("submitter", ""),
    ]


def _short_time(iso):
    if not iso:
        return ""
    moment = datetime.fromisoformat(iso).astimezone()
    if moment.date() == datetime.now().astimezone().date():
        return f"{moment:%H:%M:%S}"
    return f"{moment:%Y-%m-%d %H:%M}"
