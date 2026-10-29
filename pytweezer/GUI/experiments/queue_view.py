"""Table of the running, waiting and recently finished tasks, with their controls."""

from datetime import datetime

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.queue import UNKNOWN_DEVICES
from pytweezer.GUI.components import status_icon
from pytweezer.GUI.grafana import open_in_browser, run_url
from pytweezer.GUI.theme import STATE_STYLE

COLUMNS = [
    "RID",
    "Pipeline",
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
    "completed": "disabled",
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
_STATUS_COLUMN = COLUMNS.index("Status")


class PipelineChips(QWidget):
    """One checkable chip per pipeline, after "All"; exactly one is checked."""

    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        #: Chip for each pipeline, keyed by name; ``None`` is the "All" chip.
        self.buttons = {}
        self._add_chip(None, "All").setChecked(True)

    def selected(self):
        """The selected pipeline's name, or ``None`` for All."""
        return next(name for name, b in self.buttons.items() if b.isChecked())

    def set_pipelines(self, counts):
        """Show a chip per pipeline; ``counts`` maps name to (running, waiting)."""
        selected = self.selected()
        for name in [n for n in self.buttons if n is not None and n not in counts]:
            chip = self.buttons.pop(name)
            self._group.removeButton(chip)
            self._layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        order = sorted(counts, key=str.casefold)
        for name in order:
            if name not in self.buttons:
                self._add_chip(name, name)
        for name in order:
            self._layout.removeWidget(self.buttons[name])
        for index, name in enumerate(order, start=1):
            self._layout.insertWidget(index, self.buttons[name])
            running, waiting = counts[name]
            text = name + (" ●" if running else "") + (f" {waiting}" if waiting else "")
            self.buttons[name].setText(text)
        self.buttons[selected if selected in self.buttons else None].setChecked(True)
        if self.selected() != selected:
            self.selection_changed.emit(self.selected())

    def _add_chip(self, name, text):
        chip = QToolButton()
        chip.setObjectName("PipelineChip")
        chip.setCheckable(True)
        chip.setText(text)
        chip.clicked.connect(
            lambda _checked: self.selection_changed.emit(self.selected())
        )
        self._group.addButton(chip)
        self.buttons[name] = chip
        self._layout.insertWidget(len(self.buttons) - 1, chip)
        return chip


class QueueView(QWidget):
    """Shows a manager snapshot; buttons emit ``action_requested(command, fields)``."""

    action_requested = QtCore.pyqtSignal(str, dict)
    edit_requested = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chips = PipelineChips()
        self.chips.selection_changed.connect(self._apply_filter)
        layout.addWidget(self.chips)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("QueueTable")
        self.table.setShowGrid(False)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # The label is the user's own note, so it gets the spare width.
        header.setSectionResizeMode(
            COLUMNS.index("Label"), QHeaderView.ResizeMode.Stretch
        )
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(self._show_details)
        layout.addWidget(self.table, 1)

        # Grouped by what they act on; the two that lose work sit apart.
        groups = [
            [("pause", "Pause"), ("resume", "Resume"), ("terminate", "Terminate")],
            [("hold", "Hold"), ("release", "Release"), ("raise", "Priority +"),
             ("lower", "Priority −")],
            [("edit", "Edit as new"), ("grafana", "Open in Grafana")],
            None,
            [("abort", "Abort"), ("delete", "Delete")],
        ]  # fmt: skip
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.buttons = {}
        for group in groups:
            if group is None:
                buttons.addStretch(1)
                continue
            if self.buttons:
                buttons.addSpacing(14)
            for command, text in group:
                button = QPushButton(text)
                button.clicked.connect(lambda _checked, c=command: self._clicked(c))
                buttons.addWidget(button)
                self.buttons[command] = button
        for command in ("abort", "delete"):
            self.buttons[command].setObjectName("DangerButton")
        self.buttons["terminate"].setToolTip("Stop after the current point")
        self.buttons["abort"].setToolTip(
            "Kill the task now. A device call already in progress still completes."
        )
        self.buttons["delete"].setToolTip("Remove a waiting task from the queue")
        self.buttons["edit"].setToolTip("Copy this task's settings into the editor")
        self.buttons["grafana"].setToolTip(
            "Show this task's results and the readings during it in Grafana"
        )
        layout.addLayout(buttons)
        self._update_buttons()

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        selected = self.selected_task()
        selected_rid = selected["rid"] if selected else None
        running = sorted(snapshot.get("running") or [], key=lambda t: t["rid"])
        queue = snapshot.get("queue", [])
        rows = running + queue + snapshot.get("history", [])
        self.chips.set_pipelines(_pipeline_counts(running, queue, rows))
        unresolved_heads = {
            task["rid"]
            for task in _heads(running, queue)
            if task.get("devices") is None
        }
        self.table.setRowCount(len(rows))
        reselect = None
        for row, task in enumerate(rows):
            reading = task["rid"] in unresolved_heads
            blocked = reading or _blocked(task)
            shade = _ROW_SHADES.get("blocked" if blocked else _row_kind(task))
            for column, text in enumerate(_cells(task, reading)):
                item = QTableWidgetItem(text)
                item.setData(_TASK, task)
                if shade is not None:
                    item.setBackground(shade)
                if column == _STATUS_COLUMN and task["status"] in _STATUS_STATE:
                    state = "starting" if blocked else _STATUS_STATE[task["status"]]
                    item.setIcon(status_icon(state))
                if task.get("error"):
                    item.setToolTip(task["error"].strip().splitlines()[-1])
                self.table.setItem(row, column, item)
            if task["rid"] == selected_rid:
                reselect = row
        self._apply_filter()
        # Qt keeps the selected row index, which may now hold a different task.
        if reselect is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(reselect)
        self._update_buttons()

    def _apply_filter(self):
        pipeline = self.chips.selected()
        for row in range(self.table.rowCount()):
            task = self.table.item(row, 0).data(_TASK)
            self.table.setRowHidden(
                row, pipeline is not None and _pipeline(task) != pipeline
            )
        selected = self.selected_task()
        if selected is not None and pipeline not in (None, _pipeline(selected)):
            self.table.clearSelection()
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
            "grafana": task is not None and bool(task.get("t_start")),
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
        elif command == "grafana":
            open_in_browser(run_url(rid, task["t_start"], task.get("t_end")))
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
            f"Pipeline: {_pipeline(task)}\n"
            f"Devices: {_devices_text(task.get('devices'))}\n"
            f"File: {task.get('h5_path') or '—'}\n"
        )
        if task.get("error"):
            text += f"\n{task['error']}"
        box = QMessageBox(self)
        box.setWindowTitle(f"Task {task['rid']}")
        box.setText(text)
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        box.open()


def _amber_shade():
    """The theme's warning colour, dimmed into the dark rows so text stays legible."""
    amber = QtGui.QColor(STATE_STYLE["starting"][0])
    base = QtGui.QColor("#141519")
    mix = 0.3
    return QtGui.QColor(
        *(
            round(a * mix + b * (1 - mix))
            for a, b in zip(amber.getRgb()[:3], base.getRgb()[:3])
        )
    )


# Row backgrounds by lifecycle: the running task stands out, finished ones
# recede. (Text colour can't vary per item: the theme's ::item rule sets it.)
_ROW_SHADES = {
    "running": QtGui.QColor("#1f2c45"),
    "finished": QtGui.QColor("#111215"),
    "blocked": _amber_shade(),
}


def _pipeline(task):
    return task.get("pipeline") or "main"


def _blocked(task):
    return bool(task.get("waiting_for"))


def _devices_text(devices):
    if devices is None:
        return "not yet known"
    return ", ".join(devices) or "none"


def _heads(running, queue):
    """Each idle pipeline's next due task, as the manager picks them."""
    busy = {_pipeline(task) for task in running}
    heads = {}
    at = datetime.now().astimezone()
    for task in queue:
        pipeline = _pipeline(task)
        if pipeline in busy or pipeline in heads:
            continue
        due = task.get("due_time")
        if task["status"] == "queued" and (
            due is None or datetime.fromisoformat(due) <= at
        ):
            heads[pipeline] = task
    return list(heads.values())


def _pipeline_counts(running, queue, tasks):
    counts = {_pipeline(task): [False, 0] for task in tasks}
    for task in running:
        counts[_pipeline(task)][0] = True
    for task in queue:
        counts[_pipeline(task)][1] += 1
    return {name: tuple(count) for name, count in counts.items()}


def _row_kind(task):
    if task["status"] in ("running", "paused"):
        return "running"
    if task["status"] in ("queued", "held"):
        return "waiting"
    return "finished"


def queue_summary(snapshot):
    """One line for the queue's title: what's running and how much is waiting."""
    running = snapshot.get("running") or []
    waiting = len(snapshot.get("queue", []))
    rids = ", ".join(
        str(task["rid"]) for task in sorted(running, key=lambda t: t["rid"])
    )
    return ", ".join(
        [
            f"{len(running)} running ({rids})" if running else "nothing running",
            f"{waiting} waiting" if waiting else "nothing waiting",
        ]
    )


def _cells(task, reading=False):
    total = task.get("points_total")
    progress = f"{task['points_done']}/{total}" if total else ""
    when = task.get("t_start") or task.get("due_time") or task.get("t_submit")
    status = task["status"]
    if task.get("requested") and status in ("running", "paused"):
        status = f"{status} ({task['requested']} requested)"
    if _blocked(task):
        status = "waiting: " + ", ".join(
            f"{'all devices' if device == UNKNOWN_DEVICES else device} (RID {rid})"
            for device, rid in task["waiting_for"].items()
        )
    elif reading:
        status = "waiting: reading experiments"
    return [
        str(task["rid"]),
        _pipeline(task),
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
