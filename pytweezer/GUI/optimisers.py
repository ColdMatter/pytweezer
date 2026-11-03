"""The Optimisers tab: set up and run each optimiser, watch it, accept what it proposes.

Optimisers are the classes under :mod:`pytweezer.optimisers`, listed by the
manager's optimisers catalogue; they are set up and run only here, never from
the Experiments tab. The tab follows the manager's published snapshot through an
:class:`~pytweezer.GUI.experiments.feed.ExperimentFeed`, refetching the
optimisers, recipes and saved parameters when their versions change. A running
optimiser is plotted from the feed's points as they arrive; a finished one from
its measurement file, read through a results archive as the Results tab does,
off the GUI thread.
"""

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.archive import default_archive
from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    submitter_name,
)
from pytweezer.experiment.optimisers import SEARCH
from pytweezer.experiment.queue import now
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.task import Task, TaskRequest, TaskStatus
from pytweezer.GUI.background import BackgroundReader
from pytweezer.GUI.components import Region, set_state, status_icon
from pytweezer.GUI.experiments.arg_editor import ValueField
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.saved_parameters import display_value, when_text
from pytweezer.GUI.theme import PLOT_BACKGROUND, PLOT_FOREGROUND
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.optimisers")

NOT_RESPONDING = "the Experiment Manager is not responding"
TASK_HISTORY_LIMIT = 500
ACCEPT_HISTORY_LIMIT = 500
LAST_RUN = "Last run"
DEFAULTS = "Defaults"
UNGROUPED = "Measurement"
_KEY = QtCore.Qt.ItemDataRole.UserRole
_ACCENT = "#5b8def"
_POINTS = "#9aa0b4"
_RUNNING_SHADE = QtGui.QColor("#1f2c45")
_TOP_HINT = "Pick one to set it up and run it"
_DISCARDED_KEPT = 200
_PLOT_COLUMNS = 2
_LIVE = (TaskStatus.RUNNING, TaskStatus.PAUSED)
#: Results plotted; per-setting results are added for Bayesian runs.
_PLOTTED = ["stage", "setting", "objective"]

_ICON_STATE = {
    "queued": "stopped",
    "running": "running",
    "not accepted": "starting",
    "partial": "starting",
    "accepted": "stopped",
    "failed": "crashed",
}


def partial_label(constants: dict) -> str:
    """How far a stopped run got: ``stage 1 of 2 finished`` or ``34 of 60 evaluations``."""
    if "stages_total" in constants:
        return (
            f"stage {constants.get('stages_done', 0)} of "
            f"{constants['stages_total']} finished"
        )
    if "evaluations_total" in constants:
        return (
            f"{constants.get('evaluations_done', 0)} of "
            f"{constants['evaluations_total']} evaluations"
        )
    return "partial"


def proposals(constants: dict, keys: list[str]) -> dict:
    """The values a run proposes for ``keys``, as plain Python values; arrays as nested lists."""
    proposed = {}
    for key in keys:
        value = constants.get(f"proposed.{key}")
        if value is not None:
            proposed[key] = value.tolist() if hasattr(value, "tolist") else value
    return proposed


def live_results(points: list[dict], names: list[str]) -> dict[str, np.ndarray]:
    """``{name: one value per point}`` from the feed's points, in point order; NaN where missing."""
    ordered = sorted(points, key=lambda point: point["index"])
    return {
        name: np.array(
            [float(point.get("scalars", {}).get(name, np.nan)) for point in ordered]
        )
        for name in names
    }


def _number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return display_value(value)


def _grouped(schema: dict) -> dict[str, list[str]]:
    """Argument names by group: the search first, its per-setting arguments leading."""
    settings = schema["optimiser"]["settings"]
    groups: dict[str, list[str]] = {SEARCH: []}
    for name, argument in schema["arguments"].items():
        groups.setdefault(argument.get("group") or UNGROUPED, []).append(name)

    def per_setting_first(name):
        for index, setting in enumerate(settings):
            if name.startswith(f"{setting}_"):
                return index
        return len(settings)

    groups[SEARCH].sort(key=per_setting_first)
    if UNGROUPED in groups:
        groups[UNGROUPED] = groups.pop(UNGROUPED)
    return {group: names for group, names in groups.items() if names}


class SettingsForm(QWidget):
    """One optimiser's arguments, in display units, grouped with the search first."""

    changed = QtCore.pyqtSignal()

    def __init__(self, schema, parent=None):
        super().__init__(parent)
        self.setObjectName("OptimiserSettings")
        self.schema = schema
        self.fields: dict[str, ValueField] = {}
        self.labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for group, names in _grouped(schema).items():
            box = QGroupBox(group)
            box.setObjectName("EditorGroup")
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(10)
            grid.setColumnMinimumWidth(0, 150)
            grid.setColumnStretch(2, 1)
            for row, name in enumerate(names):
                argument = schema["arguments"][name]
                label = QLabel(name)
                label.setProperty("role", "argument")
                label.setToolTip(argument.get("tooltip") or name)
                value = ValueField(argument, width=110)
                value.changed.connect(self._value_changed)
                grid.addWidget(label, row, 0)
                grid.addWidget(value, row, 1)
                self.fields[name] = value
                self.labels[name] = label
            layout.addWidget(box)
        layout.addStretch(1)

    def values(self) -> dict:
        """Every argument's value, in SI."""
        return {name: value.value() for name, value in self.fields.items()}

    def set_values(self, values: dict) -> None:
        """Show ``values``; an argument missing from them, or not valid, shows its default."""
        for name, value in self.fields.items():
            default = value.schema["default"]
            try:
                value.set_value(values.get(name, default))
            except (TypeError, ValueError):
                value.set_value(default)
        self._value_changed()

    def planned(self) -> str:
        """How many points the next run takes: ``30 evaluations`` or ``31 points``."""
        values = self.values()
        optimiser = self.schema["optimiser"]
        if optimiser["kind"] == "fit":
            return "1 fit"
        if optimiser["kind"] == "bayesian":
            count, noun = values.get("n_calls", 0), "evaluation"
        else:
            per_sweep = sum(
                values.get(f"{setting}_points", 0) for setting in optimiser["settings"]
            )
            count, noun = values.get("repeats", 1) * per_sweep, "point"
        return f"{count} {noun}{'' if count == 1 else 's'}"

    def _value_changed(self):
        for name, value in self.fields.items():
            set_state(self.labels[name], "" if value.is_default() else "modified")
        self.changed.emit()


@dataclass
class OptimiserRow:
    """One optimiser's row in the table, and what it last showed."""

    LAST_RUN = 1
    LAST_ACCEPTED = 2
    SAVES = 3

    schema: dict
    item: QTreeWidgetItem
    state: str = "never run"
    latest: Task | None = None
    shown: Task | None = None
    rids: set[int] = field(default_factory=set)
    preset: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.schema["module"], self.schema["class_name"]

    @property
    def proposes(self) -> list[str]:
        return list(self.schema["optimiser"]["proposes"])

    @property
    def active(self) -> bool:
        return self.latest is not None and not self.latest.status.finished


class OptimisersPanel(QWidget):
    """Set up and run each optimiser, watch its run, and accept what it proposes.

    Args:
        client: Talks to the Experiment Manager; a short-timeout client by default.
        feed: The manager's published state, shared with other tabs; one of
            its own by default.
        archive: Reads measurement files; this PC's default archive by default.
    """

    def __init__(self, client=None, feed=None, archive=None, parent=None):
        super().__init__(parent)
        self.client = client or ExperimentManagerClient(timeout_ms=1000, retries=0)
        self._owns_feed = feed is None
        self.feed = feed or ExperimentFeed()
        self.archive = archive if archive is not None else default_archive()
        self.rows: dict[tuple[str, str], OptimiserRow] = {}
        self.forms: dict[tuple[str, str], SettingsForm] = {}
        self.accept_rows: dict[str, QCheckBox] = {}
        self._selected: tuple[str, str] | None = None
        self._tasks: dict[int, Task] = {}
        self._measurements: dict[str, object] = {}
        self._reader = BackgroundReader(self)
        self._reading: set[str] = set()
        self._accepted: dict[int, list] = {}
        self._changes = []
        self._parameters = {}
        self._recipes: dict[tuple[str, str], dict[str, Recipe]] = {}
        self._last_requests: dict[tuple[str, str], TaskRequest | None] = {}
        self._versions: dict[str, object] = {}
        self._stale = {"optimisers", "recipes", "saved", "tasks"}
        self._plots = None
        self._sheet_rid = None
        self._sheet_shown = None
        self._proposed = {}
        self._settings = QtCore.QSettings("pytweezer", "Optimisers")

        self.table = QTreeWidget()
        self.table.setObjectName("OptimiserTable")
        self.table.setHeaderLabels(["Optimiser", "Last run", "Last accepted", "Saves"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        header = self.table.header()
        header.setStretchLastSection(True)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentItemChanged.connect(self._current_changed)
        self.optimisers_region = Region("well", "Optimisers", _TOP_HINT)
        self.optimisers_region.body.addWidget(self.table, 1)

        self.settings_region = self._build_settings()

        self.plots = QWidget()
        self.plots.setObjectName("OptimiserPlots")
        self.plots_layout = QGridLayout(self.plots)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_layout.setSpacing(16)
        self.plot_message = QLabel()
        self.plot_message.setProperty("role", "regionHint")
        self.plot_message.setWordWrap(True)
        self.run_region = Region("well", "Pick an optimiser", "")
        self.run_region.body.addWidget(self.plots, 1)
        self.run_region.body.addWidget(self.plot_message)

        self.sheet = Region("well", "Accept", "nothing is saved until you accept")
        self.proposal_grid_holder = QVBoxLayout()
        self.sheet.body.addLayout(self.proposal_grid_holder)
        self._proposal_widget = None
        self.sheet.body.addStretch(1)
        self.accept_button = QPushButton("Accept")
        self.accept_button.setObjectName("PrimaryButton")
        self.accept_button.setToolTip(
            "Save the ticked values for the whole lab, noting this run as the reason"
        )
        self.accept_button.clicked.connect(self.accept)
        self.accept_count = QLabel()
        self.accept_count.setProperty("role", "regionHint")
        self.discard_button = QPushButton("Discard")
        self.discard_button.setToolTip(
            "Hide these proposals on this PC; nothing is saved or deleted"
        )
        self.discard_button.clicked.connect(self.discard)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.accept_button)
        actions.addWidget(self.accept_count)
        actions.addStretch(1)
        actions.addWidget(self.discard_button)
        self.sheet.body.addLayout(actions)
        self.status = QLabel()
        self.status.setObjectName("StatusLabel")
        self.status.setWordWrap(True)
        self.sheet.body.addWidget(self.status)

        # Left: pick and set up; right: watch and accept.
        left = QSplitter(QtCore.Qt.Orientation.Vertical)
        left.setObjectName("PanelSplitter")
        left.addWidget(self.optimisers_region)
        left.addWidget(self.settings_region)
        left.setStretchFactor(0, 0)
        left.setStretchFactor(1, 1)
        left.setSizes([220, 700])
        right = QSplitter(QtCore.Qt.Orientation.Vertical)
        right.setObjectName("PanelSplitter")
        right.addWidget(self.run_region)
        right.addWidget(self.sheet)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        right.setSizes([650, 250])
        main = QSplitter(QtCore.Qt.Orientation.Horizontal)
        main.setObjectName("PanelSplitter")
        main.addWidget(left)
        main.addWidget(right)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([560, 1000])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(main, 1)

        self._show_selected()
        self.feed.queue_changed.connect(self._queue_changed)
        self.feed.point_received.connect(self._point_received)
        self.feed.connection_changed.connect(self._connection_changed)

    def _build_settings(self):
        region = Region("sheet", "Settings", "")
        self.preset = QComboBox()
        self.preset.setFixedWidth(170)
        self.preset.setToolTip(
            "Load a saved preset, the defaults or the last run's settings into the form"
        )
        self.preset.activated.connect(self._preset_chosen)
        self.save_preset_button = QPushButton("Save as preset…")
        self.save_preset_button.setToolTip("Save these settings for every PC")
        self.save_preset_button.clicked.connect(self.save_preset)
        self.delete_preset_button = QPushButton("Delete preset…")
        self.delete_preset_button.setObjectName("DangerButton")
        self.delete_preset_button.setToolTip("Delete the chosen preset for every PC")
        self.delete_preset_button.clicked.connect(self.delete_preset)
        presets = QHBoxLayout()
        presets.setSpacing(10)
        presets.addWidget(QLabel("Load"))
        presets.addWidget(self.preset)
        presets.addWidget(self.save_preset_button)
        presets.addStretch(1)
        presets.addWidget(self.delete_preset_button)
        region.body.addLayout(presets)

        self.settings_stack = QStackedWidget()
        self.settings_empty = QLabel("Pick an optimiser above to set it up")
        self.settings_empty.setProperty("role", "regionHint")
        self.settings_empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.settings_stack.addWidget(self.settings_empty)
        scroll = QScrollArea()
        scroll.setObjectName("OptimiserSettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self.settings_stack)
        region.body.addWidget(scroll, 1)

        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run_selected)
        self.run_count = QLabel()
        self.run_count.setProperty("role", "regionHint")
        run = QHBoxLayout()
        run.setSpacing(10)
        run.addWidget(self.run_button)
        run.addWidget(self.run_count)
        run.addStretch(1)
        region.body.addLayout(run)
        self.settings_status = QLabel()
        self.settings_status.setObjectName("StatusLabel")
        self.settings_status.setWordWrap(True)
        region.body.addWidget(self.settings_status)
        return region

    # -- following the manager -----------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _queue_changed(self, snapshot):
        self._merge_tasks(snapshot)
        for name, stale in (
            ("optimisers_version", "optimisers"),
            ("recipes_version", "recipes"),
            ("saved_parameters_version", "saved"),
        ):
            version = snapshot.get(name)
            if version is not None and version != self._versions.get(name):
                self._versions[name] = version
                self._stale.add(stale)
        if self.isVisible():
            self._fetch_stale()
            self._update()

    def _point_received(self, point):
        row = self._selected_row()
        if self.isVisible() and row is not None:
            run = self._shown_run(row)
            if run is not None and run.rid == point.get("rid"):
                self._show_selected()

    def _connection_changed(self, connected):
        if not connected:
            self._show_top_error(NOT_RESPONDING)
        elif self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        """Refetch everything the tab shows from the manager."""
        self._stale |= {"optimisers", "recipes", "saved", "tasks"}
        self._measurements = {
            path: m for path, m in self._measurements.items() if not _failed(m)
        }
        self._fetch_stale()
        self._update()

    def _merge_tasks(self, snapshot):
        active = [*(snapshot.get("running") or []), *snapshot.get("queue", [])]
        active_rids = {entry["rid"] for entry in active}
        for rid, known in list(self._tasks.items()):
            if not known.status.finished and rid not in active_rids:
                del self._tasks[rid]
        for entry in [*active, *snapshot.get("history", [])]:
            self._tasks[entry["rid"]] = Task.model_validate(entry)

    def _fetch_stale(self):
        try:
            if "optimisers" in self._stale:
                self._set_optimisers(self.client.optimisers())
            if "tasks" in self._stale:
                for past in self.client.history(limit=TASK_HISTORY_LIMIT):
                    self._tasks.setdefault(past.rid, past)
            if "recipes" in self._stale:
                self._recipes = {}
                for recipe in self.client.recipes():
                    key = (recipe.experiment, recipe.class_name)
                    self._recipes.setdefault(key, {})[recipe.name] = recipe
            if "saved" in self._stale:
                self._parameters = {
                    parameter.key: parameter
                    for parameter in self.client.saved_parameters()
                }
                self._changes = self.client.saved_parameter_history(
                    limit=ACCEPT_HISTORY_LIMIT
                )
                self._accepted = {}
        except ManagerUnavailable:
            self._show_top_error(NOT_RESPONDING)
            return
        except ManagerError as error:
            self._show_top_error(str(error))
            return
        self._stale.clear()
        self.optimisers_region.set_hint(_TOP_HINT)
        set_state(self.optimisers_region.hint, "")

    def _show_top_error(self, why):
        self.optimisers_region.set_hint(f"Optimisers: {why}")
        set_state(self.optimisers_region.hint, "error")

    # -- the table -----------------------------------------------------------

    def _set_optimisers(self, modules):
        schemas = [
            schema
            for module in modules
            for schema in module.get("classes", [])
            if schema.get("optimiser")
        ]
        schemas.sort(key=lambda s: (s["class_name"], s["module"]))
        keys = [(s["module"], s["class_name"]) for s in schemas]
        for key in set(self.forms) - set(keys):
            self._drop_form(key)
        if keys == list(self.rows):
            for schema in schemas:
                row = self.rows[schema["module"], schema["class_name"]]
                if schema["arguments"] != row.schema["arguments"]:
                    self._rebuild_form(row.key, schema)
                row.schema = schema
            return
        self.table.blockSignals(True)
        try:
            self.table.clear()
            self.rows = {}
            for schema in schemas:
                row = self._make_row(schema)
                self.rows[row.key] = row
            if self._selected in self.rows:
                self.table.setCurrentItem(self.rows[self._selected].item)
        finally:
            self.table.blockSignals(False)
        if self._selected not in self.rows:
            self._selected = None

    def _make_row(self, schema):
        saves = ", ".join(schema["optimiser"]["proposes"])
        item = QTreeWidgetItem([schema["class_name"], "never run", "never", saves])
        item.setData(0, _KEY, (schema["module"], schema["class_name"]))
        item.setToolTip(0, schema.get("doc") or schema["module"])
        item.setToolTip(OptimiserRow.SAVES, f"Proposes values for {saves}")
        self.table.addTopLevelItem(item)
        return OptimiserRow(schema, item)

    def _update(self):
        current = now()
        for key, row in self.rows.items():
            tasks = sorted(
                (
                    t
                    for t in self._tasks.values()
                    if (t.experiment, t.class_name) == key
                ),
                key=lambda t: t.rid,
            )
            row.rids = {t.rid for t in tasks}
            row.latest = tasks[-1] if tasks else None
            row.shown = next(
                (t for t in reversed(tasks) if t.status.finished and t.h5_path), None
            )
            self._show_row(row, current)
        self._show_selected()

    def _show_row(self, row, current):
        latest = row.latest
        row.state, text = self._state(latest, current)
        finished = latest is not None and latest.status.finished and latest.h5_path
        if finished and _failed(self._measurement(latest)):
            text += "; its file cannot be read"
        item = row.item
        item.setText(row.LAST_RUN, text)
        icon = _ICON_STATE.get(row.state)
        item.setIcon(row.LAST_RUN, status_icon(icon) if icon else QtGui.QIcon())
        if latest is not None and latest.error:
            item.setToolTip(row.LAST_RUN, latest.error.strip().splitlines()[-1])
        else:
            item.setToolTip(row.LAST_RUN, "")
        shade = _RUNNING_SHADE if row.state == "running" else QtGui.QBrush()
        for column in range(self.table.columnCount()):
            item.setBackground(column, shade)
        item.setText(row.LAST_ACCEPTED, self._last_accepted(row, current))

    def _state(self, task, current):
        if task is None:
            return "never run", "never run"
        status = task.status
        when = when_text(task.t_end or task.t_submit, current)
        if status in (TaskStatus.QUEUED, TaskStatus.HELD):
            return "queued", str(status)
        if status in _LIVE:
            total = task.points_total if task.points_total is not None else "?"
            return "running", f"{status}, {task.points_done} of {total}"
        if status == TaskStatus.COMPLETED:
            if self._acceptance(task.rid):
                return "accepted", f"completed {when}, accepted"
            return "not accepted", f"completed {when}, not accepted"
        if status == TaskStatus.TERMINATED:
            measurement = self._measurement(task)
            if measurement is _READING or _failed(measurement):
                return "partial", f"terminated {when}"
            label = partial_label(measurement.constants)
            return "partial", f"terminated {when}, {label}"
        return "failed", f"{status} {when}"

    def _last_accepted(self, row, current):
        for change in self._changes:
            if change.rid in row.rids:
                return when_text(change.time, current)
        return "never"

    def _measurement(self, task):
        """The run's measurement, the exception reading it raised, or :data:`_READING`.

        Only finished files are read, so each is read once, off the GUI thread
        (over RPC a read can wait out a whole timeout); the tab redraws when it
        arrives. One that could not be read is tried again on the next
        :meth:`refresh`.
        """
        path = task.h5_path
        if not path:
            return FileNotFoundError(f"run {task.rid} wrote no file")
        if path in self._measurements:
            return self._measurements[path]
        if path not in self._reading:
            self._reading.add(path)
            row = self.rows.get((task.experiment, task.class_name))
            results = _plotted_names(row.schema) if row else _PLOTTED
            self._reader.start(
                f"Optimisers read {path}",
                lambda: self.archive.load(path, results=results),
                lambda outcome: self._file_arrived(path, outcome),
            )
        return _READING

    def _file_arrived(self, path, outcome):
        self._reading.discard(path)
        if isinstance(outcome, Exception):
            logger.warning("Cannot read %s", path, exc_info=outcome)
        self._measurements[path] = outcome
        self._update()

    def _acceptance(self, rid):
        """The saved-parameter changes accepted from run ``rid``."""
        if rid not in self._accepted:
            try:
                self._accepted[rid] = self.client.saved_parameter_history(rid=rid)
            except (ManagerUnavailable, ManagerError):
                return []
        return self._accepted[rid]

    # -- the selected optimiser ----------------------------------------------

    def select(self, module: str, class_name: str) -> None:
        """Select optimiser ``module.class_name``: its settings, and its latest run."""
        row = self.rows.get((module, class_name))
        if row is not None:
            self.table.setCurrentItem(row.item)

    def _current_changed(self, item, _previous):
        self._selected = None if item is None else tuple(item.data(0, _KEY))
        self._show_selected()

    def _selected_row(self):
        return self.rows.get(self._selected)

    def _shown_run(self, row):
        """The run plotted for ``row``: one in progress, else the latest finished."""
        if row.latest is not None and row.latest.status in _LIVE:
            return row.latest
        return row.shown

    def _show_selected(self):
        row = self._selected_row()
        self._show_settings(row)
        if row is None:
            self.run_region.title.setText("Pick an optimiser")
            self.run_region.set_hint("")
            self._show_plots(None, None)
            self._show_proposals(None, "Pick an optimiser to see its latest run")
            return
        self.sheet.setVisible(True)
        run = self._shown_run(row)
        name = row.schema["class_name"]
        if run is None:
            self.run_region.title.setText(name)
            self.run_region.set_hint(
                f"run {row.latest.rid} is {row.latest.status}"
                if row.latest
                else "never run; set it up and press Run"
            )
            self._show_plots(None, None)
            self._show_proposals(None, "Nothing to accept until a run finishes")
            return
        self.run_region.title.setText(f"{name}, run {run.rid}")
        if run.status in _LIVE:
            total = run.points_total if run.points_total is not None else "?"
            self.run_region.set_hint(f"{run.status}, {run.points_done} of {total}")
            points = self.feed.points(run.rid)
            self._show_plots(row, run, live_results(points, _plotted_names(row.schema)))
            self._show_proposals(
                None,
                f"Run {run.rid} is {run.status}; its proposals follow when it ends",
            )
            return
        measurement = self._measurement(run)
        if measurement is _READING:
            self.run_region.set_hint(f"{run.status}, reading its file…")
            self._show_plots(None, None)
            self._show_proposals(None, f"Reading run {run.rid}'s file…")
            return
        when = when_text(run.t_end or run.t_submit, now())
        hint = f"{run.status} {when}, {run.points_done} points"
        if row.latest.rid != run.rid:
            hint += f"; run {row.latest.rid} is {row.latest.status}"
        discarded = run.h5_path in self._discarded()
        if discarded:
            hint += "; proposals discarded on this PC"
        self.run_region.set_hint(hint)
        if _failed(measurement):
            self._show_plots(row, run, error=measurement)
        else:
            self._show_plots(row, run, measurement.results, measurement.constants)
        if discarded:
            self._show_proposals(None, "")
            self.sheet.setVisible(False)
            return
        if _failed(measurement):
            self._show_proposals(
                None, f"Run {run.rid}: its file cannot be read ({measurement})"
            )
        elif run.status not in (TaskStatus.COMPLETED, TaskStatus.TERMINATED):
            self._show_proposals(
                None, f"Run {run.rid} {run.status}: its values cannot be accepted"
            )
        else:
            proposed = proposals(measurement.constants, row.proposes)
            if not proposed:
                self._show_proposals(None, f"Run {run.rid} proposed no values")
            else:
                self._show_proposals(run.rid, "", proposed)
                accepted = self._acceptance(run.rid)
                if accepted and not self.status.text():
                    change = accepted[0]
                    self._show_status(
                        f"Accepted {when_text(change.time, now())} by {change.who}"
                    )

    def _show_proposals(self, rid, message, proposed=None):
        """Fill the Accept area with run ``rid``'s proposals, or show ``message``.

        Left alone while nothing it shows has changed; ticks survive while the
        same run's proposals are shown again.
        """
        proposed = proposed or {}
        saved = {key: self._parameters.get(key) for key in proposed}
        row = self._selected_row()
        proposal_units = row.schema["optimiser"].get("units", {}) if row else {}
        # A fit's values were fitted together, so they are accepted together.
        together = row is not None and row.schema["optimiser"]["kind"] == "fit"
        shown = (rid, message, proposed, {k: p and p.value for k, p in saved.items()})
        if shown == self._sheet_shown:
            return
        self._sheet_shown = shown
        same = rid is not None and rid == self._sheet_rid and proposed == self._proposed
        ticks = {key: box.isChecked() for key, box in self.accept_rows.items()}
        if not same:
            self._show_status("")
        if self._proposal_widget is not None:
            self._proposal_widget.deleteLater()
        self._sheet_rid, self._proposed = rid, proposed
        self.accept_rows = {}
        holder = QWidget()
        holder.setObjectName("OptimiserProposals")
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        if proposed:
            for column, text in enumerate(["", "Saved parameter", "Current", "New"]):
                heading = QLabel(text)
                heading.setProperty("role", "regionHint")
                grid.addWidget(heading, 0, column)
        for line, (key, value) in enumerate(proposed.items(), start=1):
            box = QCheckBox()
            box.setChecked(together or not same or ticks.get(key, True))
            box.setEnabled(not together)
            box.setToolTip(
                "A fit's values are saved together" if together else f"Save {key}"
            )
            box.toggled.connect(self._ticks_changed)
            self.accept_rows[key] = box
            name = QLabel(key)
            name.setFont(_monospace())
            parameter = saved[key]
            unit = parameter.unit if parameter else proposal_units.get(key, "")
            unit = f" {unit}" if unit else ""
            current = QLabel(
                _number(parameter.value) + unit if parameter else "not saved"
            )
            current.setProperty("role", "regionHint")
            new = QLabel(_number(value) + unit)
            font = new.font()
            font.setBold(True)
            new.setFont(font)
            for column, widget in enumerate((box, name, current, new)):
                grid.addWidget(widget, line, column)
        if message:
            empty = QLabel(message)
            empty.setProperty("role", "regionHint")
            empty.setWordWrap(True)
            grid.addWidget(empty, 0, 0, 1, 5)
        grid.setColumnStretch(4, 1)
        self.proposal_grid_holder.addWidget(holder)
        self._proposal_widget = holder
        self.discard_button.setEnabled(rid is not None)
        self._ticks_changed()

    def _ticks_changed(self):
        ticked = sum(box.isChecked() for box in self.accept_rows.values())
        self.accept_button.setEnabled(ticked > 0)
        self.accept_count.setText(
            f"saves {ticked} value{'' if ticked == 1 else 's'} for the whole lab"
            if self.accept_rows
            else ""
        )

    def _show_status(self, text, state=""):
        self.status.setText(text)
        set_state(self.status, state)

    def _show_plots(self, row, run, results=None, constants=None, error=None):
        """Plot ``results`` of ``run``, updating the plots in place while the run is the same."""
        key = (
            None if run is None or error else (run.rid, row.schema["optimiser"]["kind"])
        )
        if self._plots is None or self._plots.key != key:
            while self.plots_layout.count():
                widget = self.plots_layout.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            self._plots = None
            if key is not None:
                kind = {"staged": _StagedPlots, "fit": _FitPlots}.get(
                    key[1], _EvaluationPlots
                )
                self._plots = kind(key, row.schema)
                for index, plot in enumerate(self._plots.widgets):
                    self.plots_layout.addWidget(
                        plot, index // _PLOT_COLUMNS, index % _PLOT_COLUMNS
                    )
        if error is not None:
            self.plot_message.setText(f"Its file cannot be read: {error}")
        elif self._plots is None:
            self.plot_message.setText("")
        else:
            measured = self._plots.update(results or {}, constants or {})
            self.plot_message.setText("" if measured else "No points measured yet")

    # -- settings ------------------------------------------------------------

    def _form(self, row):
        if row.key not in self.forms:
            form = SettingsForm(row.schema)
            form.changed.connect(self._settings_changed)
            self.forms[row.key] = form
            self.settings_stack.addWidget(form)
            request = self._last_request(row.key)
            if request is not None:
                row.preset = LAST_RUN
                form.set_values(request.args)
            else:
                row.preset = DEFAULTS
        return self.forms[row.key]

    def _last_request(self, key):
        if key not in self._last_requests:
            try:
                self._last_requests[key] = self.client.last_request(*key)
            except (ManagerUnavailable, ManagerError):
                return None
        return self._last_requests[key]

    def _rebuild_form(self, key, schema):
        old = self.forms.get(key)
        if old is None:
            return
        values = old.values()
        self._drop_form(key)
        form = SettingsForm(schema)
        form.changed.connect(self._settings_changed)
        form.set_values(values)
        self.forms[key] = form
        self.settings_stack.addWidget(form)

    def _drop_form(self, key):
        form = self.forms.pop(key)
        self.settings_stack.removeWidget(form)
        form.deleteLater()

    def _show_settings(self, row):
        if row is None:
            self.settings_region.title.setText("Settings")
            self.settings_region.set_hint("")
            self.settings_stack.setCurrentWidget(self.settings_empty)
            for widget in (
                self.preset,
                self.save_preset_button,
                self.delete_preset_button,
                self.run_button,
            ):
                widget.setEnabled(False)
            self.run_count.setText("")
            return
        form = self._form(row)
        self.settings_region.title.setText(f"Settings: {row.schema['class_name']}")
        self.settings_region.set_hint("the next run uses these")
        self.settings_stack.setCurrentWidget(form)
        self._set_presets(row)
        self.preset.setEnabled(True)
        self.save_preset_button.setEnabled(True)
        self._settings_changed()

    def _set_presets(self, row):
        wanted = [DEFAULTS, *sorted(self._recipes.get(row.key, {}))]
        if self._last_request(row.key) is not None:
            wanted.insert(0, LAST_RUN)
        combo = self.preset
        combo.blockSignals(True)
        if [combo.itemText(i) for i in range(combo.count())] != wanted:
            combo.clear()
            combo.addItems(wanted)
        combo.setCurrentText(row.preset if row.preset in wanted else DEFAULTS)
        combo.blockSignals(False)
        self.delete_preset_button.setEnabled(
            combo.currentText() not in (LAST_RUN, DEFAULTS)
        )

    def _preset_chosen(self, _index):
        row = self._selected_row()
        if row is None:
            return
        name = self.preset.currentText()
        if name == LAST_RUN:
            request = self._last_request(row.key)
            values = request.args if request is not None else {}
        elif name == DEFAULTS:
            values = {}
        else:
            values = self._recipes[row.key][name].args
        row.preset = name
        self.forms[row.key].set_values(values)
        self._set_presets(row)
        self._show_settings_status("")

    def _settings_changed(self):
        row = self._selected_row()
        if row is None or row.key not in self.forms:
            return
        self.run_count.setText(self.forms[row.key].planned())
        self.run_button.setEnabled(not row.active)
        self.run_button.setToolTip(
            f"Run {row.latest.rid} is {row.latest.status}"
            if row.active
            else "Queue this optimiser with these settings"
        )

    def _show_settings_status(self, text, state=""):
        self.settings_status.setText(text)
        set_state(self.settings_status, state)

    # -- actions -------------------------------------------------------------

    def run_selected(self) -> None:
        """Queue the selected optimiser with the settings shown."""
        row = self._selected_row()
        if row is None or row.active:
            return
        module, class_name = row.key
        request = TaskRequest(
            experiment=module,
            class_name=class_name,
            args=self.forms[row.key].values(),
            submitter=submitter_name(),
        )
        try:
            rid = self.client.submit(request)
        except ManagerUnavailable:
            self._show_settings_status(f"Running: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_settings_status(f"Running: {error}", "error")
            return
        self._last_requests[row.key] = request
        self._show_settings_status(f"Queued as run {rid}")
        self.run_button.setEnabled(False)

    def load_request(self, request) -> bool:
        """Select ``request``'s optimiser with its settings in the form; False if it is not one."""
        row = self.rows.get((request.experiment, request.class_name))
        if row is None:
            return False
        self.select(*row.key)
        self._form(row).set_values(request.args)
        row.preset = ""
        self._set_presets(row)
        self._show_settings_status(f"Loaded the settings of {request.class_name}")
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, QDockWidget):
            widget = widget.parentWidget()
        if widget is not None:
            widget.raise_()
        return True

    def save_preset(self) -> None:
        """Save the form as a named preset (a recipe) for every PC."""
        row = self._selected_row()
        if row is None:
            return
        current = self.preset.currentText()
        name = self.ask_preset_name(
            "" if current in (LAST_RUN, DEFAULTS) else current
        ).strip()
        if not name:
            return
        exists = name in self._recipes.get(row.key, {})
        if exists and not self.confirm(
            "Replace preset", f"Replace the preset '{name}' for every PC?"
        ):
            return
        module, class_name = row.key
        recipe = Recipe(
            experiment=module,
            class_name=class_name,
            args=self.forms[row.key].values(),
            name=name,
            submitter=submitter_name(),
        )
        try:
            self.client.save_recipe(recipe, overwrite=exists)
        except ManagerUnavailable:
            self._show_settings_status(f"Saving preset: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_settings_status(f"Saving preset: {error}", "error")
            return
        row.preset = name
        self._stale.add("recipes")
        self._fetch_stale()
        self._set_presets(row)
        self._show_settings_status(f"Saved preset '{name}'")

    def delete_preset(self) -> None:
        """Delete the chosen preset for every PC, after asking."""
        row = self._selected_row()
        name = self.preset.currentText()
        if row is None or name not in self._recipes.get(row.key, {}):
            return
        if not self.confirm(
            "Delete preset", f"Delete the preset '{name}' for every PC?"
        ):
            return
        try:
            self.client.delete_recipe(*row.key, name)
        except ManagerUnavailable:
            self._show_settings_status(f"Deleting preset: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_settings_status(f"Deleting preset: {error}", "error")
            return
        row.preset = DEFAULTS
        self._stale.add("recipes")
        self._fetch_stale()
        self._set_presets(row)
        self._show_settings_status(f"Deleted preset '{name}'")

    def ask_preset_name(self, default: str) -> str:
        name, ok = QInputDialog.getText(
            self, "Save as preset", "Preset name:", text=default
        )
        return name if ok else ""

    def confirm(self, title: str, question: str) -> bool:
        answer = QMessageBox.question(self, title, question)
        return answer == QMessageBox.StandardButton.Yes

    def accept(self) -> None:
        """Save the ticked proposals of the shown run for the whole lab.

        A fit's keys not saved yet are created, in its group and with their
        units; a search only sets keys that exist. The store sets every value
        or, refusing any, none; a refusal is shown in the Accept area. The run is recorded as accepted even if every value
        matches the one saved.
        """
        rid = self._sheet_rid
        values = {
            key: self._proposed[key]
            for key, box in self.accept_rows.items()
            if box.isChecked()
        }
        row = self._selected_row()
        if rid is None or not values or row is None:
            return
        optimiser = row.schema["optimiser"]
        try:
            changes = self.client.set_saved_parameters(
                values,
                f"accepted from run {rid}",
                submitter_name(),
                create=optimiser["kind"] == "fit",
                group=optimiser.get("group", ""),
                units=optimiser.get("units", {}),
                rid=rid,
            )
        except ManagerUnavailable:
            self._show_status(f"Accepting: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_status(f"Accepting: {error}", "error")
            return
        changed = sum(change.old != change.new for change in changes)
        if changed:
            self._show_status(
                f"Accepted: {changed} value{'' if changed == 1 else 's'} changed"
            )
        else:
            self._show_status("Accepted: nothing changed")
        self._stale.add("saved")
        self._fetch_stale()
        self._update()

    def discard(self) -> None:
        """Hide the shown run's proposals on this PC; nothing is saved or deleted."""
        row = self._selected_row()
        if self._sheet_rid is None or row is None or row.shown is None:
            return
        path = row.shown.h5_path
        discarded = [p for p in self._discarded() if p != path] + [path]
        self._settings.setValue("discarded", discarded[-_DISCARDED_KEPT:])
        self._show_selected()

    def _discarded(self) -> list[str]:
        """Files of the runs whose proposals this PC discarded."""
        stored = self._settings.value("discarded") or []
        return [stored] if isinstance(stored, str) else list(stored)

    def closeEvent(self, event):
        if self._owns_feed:
            self.feed.close()
        self.client.close()
        close = getattr(self.archive, "close", None)
        if close is not None:
            close()
        super().closeEvent(event)


class _StagedPlots:
    """One plot per stage: objective against its setting, and the value it chose."""

    def __init__(self, key, schema):
        self.key = key
        optimiser = schema["optimiser"]
        self.stages = []
        self.widgets = []
        for index, (setting, saved) in enumerate(
            zip(optimiser["settings"], optimiser["proposes"], strict=True)
        ):
            plot = _plot(
                f"Stage {index + 1}: {setting} → {saved}",
                setting,
                optimiser["objective"],
            )
            shots = plot.plot(
                [], [], pen=None, symbol="o", symbolSize=6, symbolBrush=_POINTS
            )
            means = plot.plot([], [], pen=pg.mkPen(_POINTS, width=1.5))
            chosen = plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush=_ACCENT,
                symbolPen=_ACCENT,
            )
            line = _chosen_line()
            plot.addItem(line)
            self.stages.append((saved, shots, means, chosen, line))
            self.widgets.append(plot)

    def update(self, results, constants) -> bool:
        stages = np.asarray(results.get("stage", []))
        settings = np.asarray(results.get("setting", []), dtype=float)
        objectives = np.asarray(results.get("objective", []), dtype=float)
        for index, (saved, shots, means, chosen, line) in enumerate(self.stages):
            here = stages == index
            x, y = settings[here], objectives[here]
            shots.setData(x, y)
            means_x = np.unique(x)
            means_y = np.array([np.nanmean(y[x == value]) for value in means_x])
            means.setData(means_x, means_y)
            value = constants.get(f"proposed.{saved}")
            if value is None:
                chosen.setData([], [])
                line.setVisible(False)
                continue
            value = float(value)
            at_value = np.isclose(means_x, value)
            chosen.setData(means_x[at_value], means_y[at_value])
            line.setValue(value)
            line.setVisible(True)
        return len(objectives) > 0


class _EvaluationPlots:
    """The objective per evaluation with the best so far, and against each setting."""

    def __init__(self, key, schema):
        self.key = key
        optimiser = schema["optimiser"]
        self.goal = optimiser["goal"]
        self.settings = optimiser["settings"]
        objective = optimiser["objective"]
        progress = _plot(
            f"Objective per evaluation; best so far ({self.goal})",
            "evaluation",
            objective,
        )
        self.shots = progress.plot(
            [], [], pen=None, symbol="o", symbolSize=6, symbolBrush=_POINTS
        )
        self.best = progress.plot([], [], pen=pg.mkPen(_ACCENT, width=2))
        self.widgets = [progress]
        self.per_setting = []
        for setting, saved in zip(
            optimiser["settings"], optimiser["proposes"], strict=True
        ):
            plot = _plot(f"{setting} → {saved}", setting, objective)
            shots = plot.plot(
                [], [], pen=None, symbol="o", symbolSize=6, symbolBrush=_POINTS
            )
            best = plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush=_ACCENT,
                symbolPen=_ACCENT,
            )
            self.per_setting.append((setting, shots, best))
            self.widgets.append(plot)

    def update(self, results, constants) -> bool:
        objective = np.asarray(results.get("objective", []), dtype=float)
        number = np.arange(1, len(objective) + 1)
        self.shots.setData(number, objective)
        finite = np.where(np.isfinite(objective), objective, np.nan)
        best_index = None
        if np.isfinite(finite).any():
            accumulate = (
                np.fmax.accumulate if self.goal == "max" else np.fmin.accumulate
            )
            self.best.setData(
                np.append(number, number[-1] + 1), accumulate(finite), stepMode="center"
            )
            choose = np.nanargmax if self.goal == "max" else np.nanargmin
            best_index = int(choose(finite))
        else:
            self.best.setData([], [])
        for setting, shots, best in self.per_setting:
            values = np.asarray(results.get(setting, []), dtype=float)
            shots.setData(values, objective[: len(values)])
            if best_index is None or best_index >= len(values):
                best.setData([], [])
            else:
                best.setData([values[best_index]], [objective[best_index]])
        return len(objective) > 0


class _FitPlots:
    """The image a fit was made on, with a square the summing window's size at each site."""

    def __init__(self, key, schema):
        self.key = key
        plot = _plot("Fitted trap sites", "x (px)", "y (px)")
        plot.getViewBox().setAspectLocked(True)
        plot.getViewBox().invertY(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.squares = pg.ScatterPlotItem(
            symbol="s", pxMode=False, brush=None, pen=pg.mkPen(_ACCENT, width=1)
        )
        plot.addItem(self.image)
        plot.addItem(self.squares)
        self.widgets = [plot]

    def update(self, results, constants) -> bool:
        images = results.get("calibration_image")
        sites = results.get("roi_sites")
        # While the run is live these are the feed's per-point scalars (NaN),
        # not the recorded image and sites.
        if (
            images is None
            or sites is None
            or not len(images)
            or np.ndim(images[-1]) != 2
            or np.ndim(sites[-1]) != 2
        ):
            self.image.clear()
            self.squares.clear()
            return False
        image = np.asarray(images[-1], dtype=float)
        self.image.setImage(image)
        # Sites are in pixel-centre coordinates; an ImageItem puts centres at +0.5.
        height, width = image.shape
        self.image.setRect(QtCore.QRectF(-0.5, -0.5, width, height))
        size = float(constants.get("proposed.trap_window_size", 5))
        rows, columns = np.asarray(sites[-1], dtype=float).T
        self.squares.setData(x=columns, y=rows, size=size)
        return True


def _plotted_names(schema) -> list[str]:
    optimiser = schema["optimiser"]
    if optimiser.get("kind") == "fit":
        return ["calibration_image", "roi_sites"]
    if optimiser.get("kind") == "bayesian":
        return [*_PLOTTED, *optimiser["settings"]]
    return list(_PLOTTED)


#: What :meth:`OptimisersPanel._measurement` gives while the file is being read.
_READING = object()


def _failed(measurement) -> bool:
    return isinstance(measurement, Exception)


def _plot(title, x_label, objective):
    plot = pg.PlotWidget(background=PLOT_BACKGROUND)
    plot.setMinimumSize(280, 200)
    plot.setTitle(title, color=PLOT_FOREGROUND, size="10pt")
    for axis in ("left", "bottom"):
        plot.getAxis(axis).setPen(PLOT_FOREGROUND)
        plot.getAxis(axis).setTextPen(PLOT_FOREGROUND)
        plot.getAxis(axis).enableAutoSIPrefix(False)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", objective or "objective")
    return plot


def _chosen_line():
    line = pg.InfiniteLine(
        0.0,
        angle=90,
        pen=pg.mkPen(_ACCENT, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        label="{value:.4g}",
        labelOpts={"color": _ACCENT, "position": 0.92},
    )
    line.setVisible(False)
    return line


def _monospace():
    font = QtGui.QFont()
    font.setFamilies(["Cascadia Mono", "Consolas", "DejaVu Sans Mono"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font
