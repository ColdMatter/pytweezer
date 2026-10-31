"""The Optimisers tab: run each optimiser, look at its latest run, accept what it proposes.

Optimisers are the catalogue entries whose schema has an ``"optimiser"``
entry. The tab follows the manager's published snapshot through an
:class:`~pytweezer.GUI.experiments.feed.ExperimentFeed`, refetching the
catalogue, recipes and saved parameters when their versions change, and reads
each run's measurement file through a results archive, as the Results tab does.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
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
from pytweezer.experiment.queue import now
from pytweezer.experiment.task import Task, TaskRequest, TaskStatus
from pytweezer.GUI.components import Region, set_state, status_icon
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.saved_parameters import display_value, when_text
from pytweezer.GUI.theme import PLOT_BACKGROUND, PLOT_FOREGROUND
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.optimisers")

NOT_RESPONDING = "the Experiment Manager is not responding"
TASK_HISTORY_LIMIT = 500
ACCEPT_HISTORY_LIMIT = 500
DEFAULTS = "Defaults"
_KEY = QtCore.Qt.ItemDataRole.UserRole
_ACCENT = "#5b8def"
_POINTS = "#9aa0b4"
_RUNNING_SHADE = QtGui.QColor("#1f2c45")
_TOP_HINT = "Run one to recalibrate; its results wait below until you accept them"
_DISCARDED_KEPT = 200
_TABLE_HEIGHT_LIMIT = 420
_RUN_HEIGHT_MIN = 400
#: Results read from a run's file; per-setting results are added for Bayesian runs.
_PLOTTED = ["stage", "setting", "objective"]
_UNREADABLE = (OSError, KeyError, ValueError)

_ICON_STATE = {
    "queued": "stopped",
    "running": "running",
    "not accepted": "starting",
    "partial": "starting",
    "accepted": "stopped",
    "failed": "crashed",
}


def generated_settings(schema: dict, suffixes: tuple[str, ...]) -> list[str]:
    """The settings with every generated argument ``<setting><suffix>``, in order.

    ``("_start", "_stop", "_points")`` gives a staged scan's stages and
    ``("_low", "_high")`` a Bayesian optimisation's dimensions.
    """
    names = schema.get("arguments", {})
    first, *rest = suffixes
    settings = [name[: -len(first)] for name in names if name.endswith(first)]
    return [s for s in settings if all(s + suffix in names for suffix in rest)]


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
    """The values a run proposes for ``keys``, as plain Python values."""
    proposed = {}
    for key in keys:
        value = constants.get(f"proposed.{key}")
        if value is not None:
            proposed[key] = value.item() if hasattr(value, "item") else value
    return proposed


def _number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return display_value(value)


@dataclass
class OptimiserRow:
    """One optimiser's row in the table, and what it last showed."""

    LAST_RUN = 2
    LAST_ACCEPTED = 3

    schema: dict
    item: QTreeWidgetItem
    preset: QComboBox
    run_button: QPushButton
    state: str = "never run"
    latest: Task | None = None
    shown: Task | None = None
    rids: set[int] = field(default_factory=set)

    @property
    def key(self) -> tuple[str, str]:
        return self.schema["module"], self.schema["class_name"]

    @property
    def proposes(self) -> list[str]:
        return list(self.schema["optimiser"]["proposes"])


class OptimisersPanel(QWidget):
    """Run each optimiser, look at its latest run, and accept what it proposes.

    Args:
        client: Talks to the Experiment Manager; a short-timeout client by default.
        feed: The manager's published state, shared with other tabs; one of
            its own by default.
        archive: Reads measurement files; this PC's default archive by default.
        experiments_panel: The Experiments tab, which **Edit arguments…** opens
            the optimiser in; without it that button is hidden.
    """

    def __init__(
        self,
        client=None,
        feed=None,
        archive=None,
        experiments_panel=None,
        parent=None,
    ):
        super().__init__(parent)
        self.client = client or ExperimentManagerClient(timeout_ms=1000, retries=0)
        self._owns_feed = feed is None
        self.feed = feed or ExperimentFeed()
        self.archive = archive if archive is not None else default_archive()
        self.experiments_panel = experiments_panel
        self.rows: dict[tuple[str, str], OptimiserRow] = {}
        self.accept_rows: dict[str, QCheckBox] = {}
        self._selected: tuple[str, str] | None = None
        self._tasks: dict[int, Task] = {}
        self._measurements: dict[str, object] = {}
        self._accepted: dict[int, list] = {}
        self._changes = []
        self._parameters = {}
        self._recipes: dict[tuple[str, str], list[str]] = {}
        self._versions: dict[str, object] = {}
        self._stale = {"catalogue", "recipes", "saved", "tasks"}
        self._plotted = None
        self._sheet_rid = None
        self._sheet_shown = None
        self._proposed = {}
        self._settings = QtCore.QSettings("pytweezer", "Optimisers")

        self.table = QTreeWidget()
        self.table.setObjectName("OptimiserTable")
        self.table.setHeaderLabels(
            ["Optimiser", "Saves", "Last run", "Last accepted", "Preset", ""]
        )
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        header = self.table.header()
        header.setStretchLastSection(True)
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(4, 170)
        self.table.currentItemChanged.connect(self._current_changed)
        self.optimisers_region = Region("well", "Optimisers", _TOP_HINT)
        self.optimisers_region.body.addWidget(self.table, 1)

        self.edit_button = QPushButton("Edit arguments…")
        self.edit_button.setToolTip(
            "Open this optimiser in the Experiments tab, with its last arguments"
        )
        self.edit_button.clicked.connect(self.edit_arguments)
        self.edit_button.setVisible(experiments_panel is not None)
        self.plots = QWidget()
        self.plots.setObjectName("OptimiserPlots")
        self.plots_layout = QHBoxLayout(self.plots)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_layout.setSpacing(16)
        self.plot_message = QLabel()
        self.plot_message.setProperty("role", "regionHint")
        self.plot_message.setWordWrap(True)
        self.run_region = Region("well", "Pick an optimiser above", "")
        self.run_region.header.addWidget(self.edit_button)
        self.run_region.body.addWidget(self.plots, 1)
        self.run_region.body.addWidget(self.plot_message)

        self.sheet = Region("sheet", "Accept", "nothing is saved until you accept")
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
        self.sheet.setMinimumWidth(440)

        lower = QSplitter(QtCore.Qt.Orientation.Horizontal)
        lower.setObjectName("PanelSplitter")
        lower.addWidget(self.run_region)
        lower.addWidget(self.sheet)
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 0)
        lower.setSizes([1000, 470])
        self._main = main = QSplitter(QtCore.Qt.Orientation.Vertical)
        main.setObjectName("PanelSplitter")
        main.addWidget(self.optimisers_region)
        main.addWidget(lower)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([260, 600])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(main, 1)

        self._show_proposals(None, "Pick an optimiser above to see its latest run")
        self.feed.queue_changed.connect(self._queue_changed)
        self.feed.connection_changed.connect(self._connection_changed)

    # -- following the manager -----------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _queue_changed(self, snapshot):
        self._merge_tasks(snapshot)
        for name, stale in (
            ("catalogue_version", "catalogue"),
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

    def _connection_changed(self, connected):
        if not connected:
            self._show_top_error(NOT_RESPONDING)
        elif self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        """Refetch everything the tab shows from the manager."""
        self._stale |= {"catalogue", "recipes", "saved", "tasks"}
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
            if "catalogue" in self._stale:
                reply = self.client.call("catalogue")
                self._set_catalogue(reply["modules"])
            if "tasks" in self._stale:
                for past in self.client.history(limit=TASK_HISTORY_LIMIT):
                    self._tasks.setdefault(past.rid, past)
            if "recipes" in self._stale:
                self._recipes = {}
                for recipe in self.client.recipes():
                    key = (recipe.experiment, recipe.class_name)
                    self._recipes.setdefault(key, []).append(recipe.name)
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

    def _set_catalogue(self, modules):
        schemas = [
            schema
            for module in modules
            for schema in module.get("classes", [])
            if schema.get("optimiser")
        ]
        schemas.sort(key=lambda s: (s["class_name"], s["module"]))
        keys = [(s["module"], s["class_name"]) for s in schemas]
        if keys == list(self.rows):
            for schema in schemas:
                self.rows[schema["module"], schema["class_name"]].schema = schema
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
        QtCore.QTimer.singleShot(0, self._fit_table)

    def _fit_table(self):
        """Give the table the height of its rows, up to a limit; the run gets the rest."""
        if not self.rows:
            return
        first = next(iter(self.rows.values())).item
        row_height = self.table.visualItemRect(first).height()
        chrome = (
            self.optimisers_region.sizeHint().height() - self.table.sizeHint().height()
        )
        table = (
            self.table.header().sizeHint().height()
            + len(self.rows) * row_height
            + 2 * self.table.frameWidth()
        )
        needed = min(chrome + table + 4, _TABLE_HEIGHT_LIMIT)
        total = max(sum(self._main.sizes()), needed + _RUN_HEIGHT_MIN)
        self._main.setSizes([needed, total - needed])

    def _make_row(self, schema):
        item = QTreeWidgetItem(
            [
                schema["class_name"],
                ", ".join(schema["optimiser"]["proposes"]),
                "never run",
                "never",
            ]
        )
        item.setData(0, _KEY, (schema["module"], schema["class_name"]))
        item.setToolTip(0, schema.get("doc") or schema["module"])
        item.setToolTip(1, "The saved parameters it proposes values for")
        self.table.addTopLevelItem(item)
        preset = QComboBox()
        preset.setFixedWidth(150)
        preset.setToolTip("The recipe Run submits, or the optimiser's defaults")
        preset.addItem(DEFAULTS)
        run_button = QPushButton("Run")
        run_button.setFixedWidth(70)
        row = OptimiserRow(schema, item, preset, run_button)
        run_button.clicked.connect(lambda: self._run(row.key))
        self.table.setItemWidget(item, 4, _padded(preset))
        self.table.setItemWidget(item, 5, _padded(run_button))
        return row

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
            self._set_presets(row)
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
        row.run_button.setEnabled(latest is None or latest.status.finished)
        row.run_button.setToolTip(
            "Queue this optimiser with the chosen preset"
            if row.run_button.isEnabled()
            else f"Run {latest.rid} is {latest.status}"
        )

    def _state(self, task, current):
        if task is None:
            return "never run", "never run"
        status = task.status
        when = when_text(task.t_end or task.t_submit, current)
        if status in (TaskStatus.QUEUED, TaskStatus.HELD):
            return "queued", str(status)
        if status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
            total = task.points_total if task.points_total is not None else "?"
            return "running", f"{status}, {task.points_done} of {total}"
        if status == TaskStatus.COMPLETED:
            if self._acceptance(task.rid):
                return "accepted", f"completed {when}, accepted"
            return "not accepted", f"completed {when}, not accepted"
        if status == TaskStatus.TERMINATED:
            measurement = self._measurement(task)
            if _failed(measurement):
                return "partial", f"terminated {when}"
            label = partial_label(measurement.constants)
            return "partial", f"terminated {when}, {label}"
        return "failed", f"{status} {when}"

    def _last_accepted(self, row, current):
        for change in self._changes:
            if change.rid in row.rids:
                return when_text(change.time, current)
        return "never"

    def _set_presets(self, row):
        wanted = [DEFAULTS, *sorted(self._recipes.get(row.key, []))]
        combo = row.preset
        if [combo.itemText(i) for i in range(combo.count())] == wanted:
            return
        chosen = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(wanted)
        combo.setCurrentText(chosen if chosen in wanted else DEFAULTS)
        combo.blockSignals(False)

    def _measurement(self, task):
        """The run's measurement, or the exception reading it raised.

        Only finished files are read, so each is read once; one that could not
        be read is tried again on the next :meth:`refresh`.
        """
        path = task.h5_path
        if path not in self._measurements:
            row = self.rows.get((task.experiment, task.class_name))
            schema = row.schema if row else {}
            wanted = _PLOTTED + generated_settings(schema, ("_low", "_high"))
            try:
                self._measurements[path] = self.archive.load(path, results=wanted)
            except _UNREADABLE as error:
                logger.warning("Cannot read %s: %s", path, error)
                self._measurements[path] = error
        return self._measurements[path]

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
        """Show the latest run of optimiser ``module.class_name`` below the table."""
        row = self.rows.get((module, class_name))
        if row is not None:
            self.table.setCurrentItem(row.item)

    def _current_changed(self, item, _previous):
        self._selected = None if item is None else tuple(item.data(0, _KEY))
        self._show_selected()

    def _selected_row(self):
        return self.rows.get(self._selected)

    def _show_selected(self):
        row = self._selected_row()
        if row is None:
            self.run_region.title.setText("Pick an optimiser above")
            self.run_region.set_hint("")
            self.edit_button.setEnabled(False)
            self._show_plots(None, None)
            self._show_proposals(None, "Pick an optimiser above to see its latest run")
            return
        self.edit_button.setEnabled(True)
        self.sheet.setVisible(True)
        run = row.shown
        name = row.schema["class_name"]
        if run is None:
            self.run_region.title.setText(name)
            self.run_region.set_hint(
                "no finished run yet" if row.latest else "never run; press Run above"
            )
            self._show_plots(None, None)
            self._show_proposals(None, "Nothing to accept until a run finishes")
            return
        self.run_region.title.setText(f"{name}, run {run.rid}")
        measurement = self._measurement(run)
        when = when_text(run.t_end or run.t_submit, now())
        hint = f"{run.status} {when}, {run.points_done} points"
        if row.latest.rid != run.rid:
            hint += f"; run {row.latest.rid} is {row.latest.status}"
        discarded = run.h5_path in self._discarded()
        if discarded:
            hint += "; proposals discarded on this PC"
        self.run_region.set_hint(hint)
        self._show_plots(row, run, measurement)
        if discarded:
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
                if accepted and not self.status.property("state"):
                    change = accepted[0]
                    self._show_status(
                        f"Accepted {when_text(change.time, now())} by {change.who}"
                    )

    def _show_proposals(self, rid, message, proposed=None):
        """Fill the Accept sheet with run ``rid``'s proposals, or show ``message``.

        Left alone while nothing it shows has changed; ticks survive while the
        same run's proposals are shown again.
        """
        proposed = proposed or {}
        saved = {key: self._parameters.get(key) for key in proposed}
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
            box.setChecked(ticks.get(key, True) if same else True)
            box.setToolTip(f"Save {key}")
            box.toggled.connect(self._ticks_changed)
            self.accept_rows[key] = box
            name = QLabel(key)
            name.setFont(_monospace())
            parameter = saved[key]
            unit = f" {parameter.unit}" if parameter and parameter.unit else ""
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

    def _show_plots(self, row, run, measurement=None):
        shown = None if run is None else (run.rid, id(measurement))
        if shown == self._plotted and shown is not None:
            return
        self._plotted = shown
        while self.plots_layout.count():
            widget = self.plots_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.plot_message.setText("")
        if run is None:
            return
        if _failed(measurement):
            self.plot_message.setText(f"Its file cannot be read: {measurement}")
            return
        if row.schema["optimiser"]["kind"] == "staged":
            self._plot_stages(row, measurement)
        else:
            self._plot_evaluations(row, measurement)

    def _plot_stages(self, row, measurement):
        results = measurement.results
        stages = np.asarray(results.get("stage", []))
        settings = generated_settings(row.schema, ("_start", "_stop", "_points"))
        for index, key in enumerate(row.proposes):
            chosen = measurement.constants.get(f"proposed.{key}")
            setting = settings[index] if index < len(settings) else "setting"
            title = f"Stage {index + 1}: {setting} → {key}"
            here = stages == index
            if not here.any():
                continue
            x = np.asarray(results["setting"], dtype=float)[here]
            y = np.asarray(results["objective"], dtype=float)[here]
            plot = _plot(title, setting)
            plot.plot(x, y, pen=None, symbol="o", symbolSize=6, symbolBrush=_POINTS)
            means_x = np.unique(x)
            if len(means_x) < len(x):
                means_y = [np.nanmean(y[x == value]) for value in means_x]
                plot.plot(means_x, means_y, pen=pg.mkPen(_POINTS, width=1.5))
            else:
                plot.plot(x, y, pen=pg.mkPen(_POINTS, width=1.5))
            if chosen is not None:
                _mark(plot, float(chosen), x, y)
            self.plots_layout.addWidget(plot, 1)
        if not self.plots_layout.count():
            self.plot_message.setText("No stage finished a point")

    def _plot_evaluations(self, row, measurement):
        objective = np.asarray(measurement.results.get("objective", []), dtype=float)
        if not len(objective):
            self.plot_message.setText("No evaluations recorded")
            return
        number = np.arange(1, len(objective) + 1)
        goal = _bayesian_goal(row, measurement)
        finite = np.where(np.isfinite(objective), objective, np.nan)
        accumulate = np.fmax.accumulate if goal == "max" else np.fmin.accumulate
        best = accumulate(finite)
        plot = _plot(f"Objective per evaluation; best so far ({goal})", "evaluation")
        plot.plot(
            number, objective, pen=None, symbol="o", symbolSize=6, symbolBrush=_POINTS
        )
        plot.plot(
            np.append(number, number[-1] + 1),
            best,
            pen=pg.mkPen(_ACCENT, width=2),
            stepMode="center",
        )
        self.plots_layout.addWidget(plot, 1)

    # -- actions -------------------------------------------------------------

    def run_selected(self) -> None:
        """Queue the selected optimiser with its chosen preset."""
        if self._selected is not None:
            self._run(self._selected)

    def _run(self, key):
        row = self.rows[key]
        module, class_name = key
        preset = row.preset.currentText()
        try:
            if preset == DEFAULTS:
                request = TaskRequest(
                    experiment=module, class_name=class_name, submitter=submitter_name()
                )
                rid = self.client.submit(request)
            else:
                rid = self.client.submit_recipe(
                    module, class_name, preset, submitter=submitter_name()
                )
        except ManagerUnavailable:
            self._show_top_error(f"Running {class_name}: {NOT_RESPONDING}")
            return
        except ManagerError as error:
            self._show_top_error(f"Running {class_name}: {error}")
            return
        self.optimisers_region.set_hint(f"Queued {class_name} as run {rid}")
        set_state(self.optimisers_region.hint, "")
        row.run_button.setEnabled(False)

    def edit_arguments(self) -> None:
        """Open the selected optimiser in the Experiments tab with its last arguments."""
        row = self._selected_row()
        if row is None or self.experiments_panel is None:
            return
        module, class_name = row.key
        try:
            request = self.client.last_request(module, class_name)
        except (ManagerUnavailable, ManagerError):
            request = None
        if request is None:
            request = TaskRequest(experiment=module, class_name=class_name)
        self.experiments_panel.load_request(request)

    def accept(self) -> None:
        """Save the ticked proposals of the shown run for the whole lab.

        The store sets every value or, refusing any, none; a refusal is shown
        in the sheet.
        """
        rid = self._sheet_rid
        values = {
            key: self._proposed[key]
            for key, box in self.accept_rows.items()
            if box.isChecked()
        }
        if rid is None or not values:
            return
        try:
            self.client.set_saved_parameters(
                values, f"accepted from run {rid}", submitter_name(), rid=rid
            )
        except ManagerUnavailable:
            self._show_status(f"Accepting: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_status(f"Accepting: {error}", "error")
            return
        count = len(values)
        self._show_status(f"Saved {count} value{'' if count == 1 else 's'}")
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


def _failed(measurement) -> bool:
    return isinstance(measurement, Exception)


def _bayesian_goal(row, measurement) -> str:
    """``"min"`` if the proposal is the lowest objective, else ``"max"``.

    The schema does not carry the goal, so it is read off which evaluation
    the run proposed.
    """
    objective = np.asarray(measurement.results.get("objective", []), dtype=float)
    finite = objective[np.isfinite(objective)]
    settings = generated_settings(row.schema, ("_low", "_high"))
    proposed = proposals(measurement.constants, row.proposes)
    if not len(finite) or len(settings) != len(row.proposes):
        return "max"
    matches = np.ones(len(objective), dtype=bool)
    for setting, key in zip(settings, row.proposes, strict=True):
        if setting not in measurement.results or key not in proposed:
            return "max"
        matches &= np.isclose(
            np.asarray(measurement.results[setting], dtype=float), proposed[key]
        )
    chosen = objective[matches & np.isfinite(objective)]
    if (
        len(chosen)
        and math.isclose(chosen[0], finite.min())
        and finite.min() != finite.max()
    ):
        return "min"
    return "max"


def _plot(title, x_label):
    plot = pg.PlotWidget(background=PLOT_BACKGROUND)
    plot.setMinimumSize(280, 200)
    plot.setTitle(title, color=PLOT_FOREGROUND, size="10pt")
    for axis in ("left", "bottom"):
        plot.getAxis(axis).setPen(PLOT_FOREGROUND)
        plot.getAxis(axis).setTextPen(PLOT_FOREGROUND)
        plot.getAxis(axis).enableAutoSIPrefix(False)
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", "objective")
    return plot


def _mark(plot, value, x, y):
    at_value = np.isclose(x, value)
    if at_value.any():
        plot.plot(
            [value],
            [np.nanmean(y[at_value])],
            pen=None,
            symbol="o",
            symbolSize=10,
            symbolBrush=_ACCENT,
            symbolPen=_ACCENT,
        )
    line = pg.InfiniteLine(
        value,
        angle=90,
        pen=pg.mkPen(_ACCENT, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
        label=f"{value:.4g}",
        labelOpts={"color": _ACCENT, "position": 0.92},
    )
    plot.addItem(line)


def _padded(widget):
    holder = QWidget()
    holder.setObjectName("OptimiserCell")
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.addWidget(widget)
    layout.addStretch(1)
    return holder


def _monospace():
    font = QtGui.QFont()
    font.setFamilies(["Cascadia Mono", "Consolas", "DejaVu Sans Mono"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font
