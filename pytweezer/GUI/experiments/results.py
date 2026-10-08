"""The Results tab: browse measurement files, inspect them, quick-plot a result.

Reads files directly from :func:`~pytweezer.experiment.storage.data_root`,
which on a client PC is the server's data share. With an
:class:`~pytweezer.GUI.experiments.feed.ExperimentFeed` connected, a running
measurement is plotted from the points the manager publishes rather than by
re-reading its file (which HDF5 does not support while it is being written,
least of all over a network share); its file is read once for its metadata,
and again when the manager reports it finished. The tree updates whenever the
queue changes, with a slow poll for files written outside the manager.

Without a connected feed the tab polls the data root every couple of seconds
while visible, re-reading a selected running measurement each time.
"""

import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from pytweezer.experiment.storage import (
    data_root,
    load_measurement,
    read_header,
    read_planned_points,
)
from pytweezer.experiment.task import TaskRequest, TaskStatus
from pytweezer.GUI.components import Region, status_icon
from pytweezer.GUI.grafana import open_in_browser, run_url
from pytweezer.GUI.theme import PLOT_BACKGROUND, PLOT_FOREGROUND
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments.results")

_PATH = QtCore.Qt.ItemDataRole.UserRole
_FILLED = QtCore.Qt.ItemDataRole.UserRole + 1
_POLL_MS = 2000
#: With a connected feed, how often to look for files the manager didn't write.
_FALLBACK_POLL_S = 30.0
#: Statuses a file can still move on from; finished files are read once.
_UNSETTLED = {"running", "unreadable", ""}
_NONE = "(none)"
_POINT_INDEX = "point index"
_CURVE_COLOURS = ["#5b8def", "#f5a623", "#2ecc71", "#c678dd", "#e5c07b", "#56b6c2"]


def aggregate(x, y, series=None):
    """Mean and standard error of ``y`` at each distinct ``x``, per distinct ``series``.

    Returns ``{series value: (x values, means, standard errors, counts)}``.
    Non-finite ``y`` are ignored; a single sample has a NaN standard error.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    series = np.zeros(len(y)) if series is None else np.asarray(series)
    finite = np.isfinite(y)
    curves = {}
    for value in np.unique(series):
        in_series = (series == value) & finite
        xs = np.unique(x[in_series])
        means, sems, counts = [], [], []
        for x_value in xs:
            samples = y[in_series & (x == x_value)]
            means.append(samples.mean())
            counts.append(len(samples))
            sems.append(
                samples.std(ddof=1) / np.sqrt(len(samples))
                if len(samples) > 1
                else np.nan
            )
        curves[value.item() if hasattr(value, "item") else value] = (
            xs,
            np.array(means),
            np.array(sems),
            np.array(counts),
        )
    return curves


def resubmission(measurement):
    """A :class:`TaskRequest` that repeats ``measurement``."""
    scanned = {axis.argument for axis in measurement.scan.axes}
    return TaskRequest(
        experiment=measurement.attrs["experiment"],
        class_name=measurement.attrs["class_name"],
        args={k: v for k, v in measurement.arguments.items() if k not in scanned},
        scan=measurement.scan,
        label=str(measurement.attrs.get("label", "")),
    )


def describe(measurement):
    attrs = measurement.attrs
    lines = [
        f"Task {attrs['rid']}: {attrs['experiment']}.{attrs['class_name']}",
        f"Status: {attrs['status']}    points {attrs['n_done']}/{attrs['n_points']}",
        f"Label: {attrs.get('label') or '—'}",
        f"Started {attrs.get('t_start') or '—'}, ended {attrs.get('t_end') or '—'}",
        f"Submitted by {attrs.get('submitter') or '—'} on {attrs.get('host', '?')}",
        f"Code: {attrs.get('git_commit') or '?'}"
        + (" (uncommitted changes)" if attrs.get("git_dirty") else ""),
        "",
        "Arguments:",
    ]
    scanned = {axis.argument for axis in measurement.scan.axes}
    for name, value in measurement.arguments.items():
        shown = _shown(measurement, name, value)
        lines.append(f"  {name} = {'(scanned)' if name in scanned else shown}")
    scan = measurement.scan
    lines += [
        "",
        f"Scan: {scan.order}, {scan.repetitions} repetition(s) by {scan.repeat}",
    ]
    for axis in scan.axes:
        if axis.kind == "linear":
            lines.append(
                f"  {axis.argument}: {_shown(measurement, axis.argument, axis.start)} to "
                f"{_shown(measurement, axis.argument, axis.stop)} in {axis.n} steps"
            )
        else:
            values = ", ".join(
                _shown(measurement, axis.argument, v) for v in axis.values
            )
            lines.append(f"  {axis.argument}: {values}")
    lines += ["", "Results:"]
    for name, shape in measurement.result_shapes.items():
        unit = measurement.units.get(name, "")
        lines.append(f"  {name}  {shape or 'scalar'}  {unit}")
    if measurement.constants:
        lines += ["", "Constants: " + ", ".join(measurement.constants)]
    if attrs.get("error"):
        lines += ["", attrs["error"]]
    return "\n".join(lines)


def _shown(measurement, name, value):
    """``value`` of argument ``name`` in its display unit."""
    schema = measurement.argument_schema.get(name, {})
    if schema.get("kind") != "number":
        return str(value)
    unit = schema.get("unit", "")
    return f"{value / (schema.get('scale') or 1.0):g} {unit}".strip()


def _header_summary(path):
    try:
        header = read_header(path)
    except OSError:
        # Also what a file being created looks like; it is retried.
        return "unreadable", ""
    return str(header.get("status", "?")), str(header.get("label", ""))


def _show_file(item, path, status, label):
    item.setText(0, path.stem.replace("_", "  ", 1))
    item.setText(1, status)
    item.setText(2, label)
    failed = status in ("failed", "crashed", "unreadable")
    if failed:
        item.setIcon(1, status_icon("crashed"))
    elif status == "running":
        item.setIcon(1, status_icon("running"))
    else:
        item.setIcon(1, QtGui.QIcon())


class ResultsPanel(QWidget):
    resubmit_requested = QtCore.pyqtSignal(object)

    def __init__(self, root=None, feed=None, parent=None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else None
        self.feed = feed
        self.measurement = None
        self.path = None
        #: A file selected but not yet readable, retried on the poll.
        self._wanted = None
        #: The full /points table of a running measurement, for live plotting.
        self._planned = None
        self._task_statuses = None
        self._last_refresh = 0.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("PanelSplitter")
        layout.addWidget(splitter)

        self.show_unfinished = QCheckBox("Show running")
        self.show_unfinished.setToolTip("Also list measurements that are still running")
        self.show_unfinished.toggled.connect(self._show_unfinished_toggled)
        self.tree = QTreeWidget()
        self.tree.setObjectName("MeasurementTree")
        self.tree.setHeaderLabels(["Measurement", "Status", "Label"])
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemExpanded.connect(self._fill_day)
        self.tree.currentItemChanged.connect(self._selected)
        self.browser = Region("well", "Measurements", "newest first")
        self.browser.header.addWidget(self.show_unfinished)
        self.browser.body.addWidget(self.tree, 1)
        splitter.addWidget(self.browser)

        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setPlaceholderText("Pick a measurement on the left to inspect it")
        self.simulation_banner = QLabel("Simulation: devices were simulated")
        self.simulation_banner.setObjectName("SimulationBanner")
        self.simulation_banner.setVisible(False)
        self.resubmit_button = QPushButton("Resubmit with these arguments")
        self.resubmit_button.setObjectName("PrimaryButton")
        self.resubmit_button.setToolTip(
            "Copy this measurement's arguments and scan into the Experiments tab"
        )
        self.resubmit_button.clicked.connect(self._resubmit)
        self.resubmit_button.setEnabled(False)
        self.grafana_button = QPushButton("Open in Grafana")
        self.grafana_button.clicked.connect(self._open_in_grafana)
        self.grafana_button.setEnabled(False)
        self.detail_region = Region("well", "Measurement", "")
        self.detail_region.header.insertWidget(2, self.simulation_banner)
        self.detail_region.body.addWidget(self.metadata, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.grafana_button)
        buttons.addWidget(self.resubmit_button)
        self.detail_region.body.addLayout(buttons)

        self.y_choice = QComboBox()
        self.x_choice = QComboBox()
        self.series_choice = QComboBox()
        controls = QHBoxLayout()
        controls.setSpacing(8)
        for text, combo in (
            ("Plot", self.y_choice),
            ("against", self.x_choice),
            ("one curve per", self.series_choice),
        ):
            combo.setFixedWidth(160)
            combo.currentTextChanged.connect(self.replot)
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(QLabel(text))
            controls.addWidget(combo)
        controls.addStretch(1)
        self.plot = pg.PlotWidget(background=PLOT_BACKGROUND)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(PLOT_FOREGROUND)
            self.plot.getAxis(axis).setTextPen(PLOT_FOREGROUND)
            # Values are already in their display units.
            self.plot.getAxis(axis).enableAutoSIPrefix(False)
        self.plot.addLegend()
        self.plot_message = QLabel()
        self.plot_message.setProperty("role", "regionHint")
        plot_region = Region("well", "Plot", "mean and standard error over repetitions")
        plot_region.body.addLayout(controls)
        plot_region.body.addWidget(self.plot, 1)
        plot_region.body.addWidget(self.plot_message)

        detail = QSplitter(QtCore.Qt.Orientation.Vertical)
        detail.setObjectName("PanelSplitter")
        detail.addWidget(self.detail_region)
        detail.addWidget(plot_region)
        detail.setStretchFactor(0, 1)
        detail.setStretchFactor(1, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 1000])

        #: Every measurement file seen: path -> (status, label, tree item or None if hidden).
        self._files = {}
        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._poll_tick)
        self._poll.start(_POLL_MS)
        if feed is not None:
            feed.queue_changed.connect(self._queue_changed)
            feed.point_received.connect(self._point_received)
        QtCore.QTimer.singleShot(0, self.refresh)

    @property
    def data_root(self):
        return self.root if self.root is not None else data_root()

    # -- browsing ----------------------------------------------------------

    @property
    def live(self):
        """Whether running measurements come from the feed instead of the file."""
        return self.feed is not None and self.feed.connected

    def _poll_tick(self):
        # Hidden behind another tab: skip, the data root may be a network share.
        if not self.isVisible():
            return
        if self.measurement is None and self._wanted is not None:
            self.load(self._wanted)
        if not self.live or time.monotonic() - self._last_refresh > _FALLBACK_POLL_S:
            self.refresh()

    def _queue_changed(self, snapshot):
        tasks = [snapshot.get("running"), *snapshot.get("queue", [])]
        tasks += snapshot.get("history", [])
        statuses = {task["rid"]: task["status"] for task in tasks if task}
        if statuses == self._task_statuses:
            return
        self._task_statuses = statuses
        if self.isVisible():
            self.refresh()
        m = self.measurement
        if m is not None and m.status == "running":
            status = statuses.get(m.attrs["rid"])
            if status is not None and TaskStatus(status).finished:
                self.load(self.path)

    def _point_received(self, point):
        m = self.measurement
        if m is not None and m.status == "running" and point["rid"] == m.attrs["rid"]:
            self._show_progress()
            self._fill_choices(keep_selection=True)
            self.replot()

    def _live_rows(self):
        """The feed's points for the selected measurement, if it is running."""
        m = self.measurement
        if m is None or m.status != "running" or not self.live or not self._planned:
            return None
        return self.feed.points(m.attrs["rid"])

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        """Bring the tree up to date with the files on disk, in place."""
        self._last_refresh = time.monotonic()
        root = self.data_root
        try:
            days = sorted((p for p in root.glob("*/*/*") if p.is_dir()), reverse=True)
            found = root.is_dir()
        except OSError:
            logger.debug("Cannot list %s", root, exc_info=True)
            days, found = [], False
        if not found:
            self.browser.set_hint(f"{root} not found")
            self.browser.hint.setToolTip(
                "On a PC other than the Experiment Manager's, set client_data_root "
                "in CONFIG (or PYTWEEZER_DATA_DIR) to the data share"
            )
            return
        self.browser.set_hint("newest first")
        self.browser.hint.setToolTip("")
        newest_was_open = self._newest_day_open()
        for day in days:
            item = self._day_item(day)
            if item is None and any(day.glob("*.h5")):
                item = self._add_day(day)
            if item is not None and item.data(0, _FILLED):
                self._sync_day(item)
        if newest_was_open and self.tree.topLevelItemCount():
            self.tree.topLevelItem(0).setExpanded(True)
        self._reload_if_running()

    def _newest_day_open(self):
        if not self.tree.topLevelItemCount():
            return True  # first fill: open the newest day
        return self.tree.topLevelItem(0).isExpanded()

    def _day_item(self, day):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, _PATH) == str(day):
                return item
        return None

    def _add_day(self, day):
        item = QTreeWidgetItem(["-".join(day.parts[-3:])])
        item.setData(0, _PATH, str(day))
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        index = 0
        while index < self.tree.topLevelItemCount() and self.tree.topLevelItem(
            index
        ).data(0, _PATH) > str(day):
            index += 1
        self.tree.insertTopLevelItem(index, item)
        return item

    def _fill_day(self, day_item):
        if not day_item.data(0, _FILLED):
            day_item.setData(0, _FILLED, True)
            self._sync_day(day_item)

    def _sync_day(self, day_item):
        for path in Path(day_item.data(0, _PATH)).glob("*.h5"):
            known = self._files.get(path)
            if known is not None and known[0] not in _UNSETTLED:
                continue  # a finished file never changes
            status, label = _header_summary(path)
            item = known[2] if known else None
            if item is not None and (status, label) == known[:2]:
                continue
            visible = status != "running" or self.show_unfinished.isChecked()
            if visible and item is None:
                item = self._add_file_item(day_item, path)
            elif not visible and item is not None:
                day_item.removeChild(item)
                item = None
            if item is not None:
                _show_file(item, path, status, label)
            self._files[path] = (status, label, item)

    def _add_file_item(self, day_item, path):
        item = QTreeWidgetItem()
        item.setData(0, _PATH, str(path))
        index = 0
        while index < day_item.childCount() and day_item.child(index).data(
            0, _PATH
        ) > str(path):
            index += 1
        day_item.insertChild(index, item)
        return item

    def _show_unfinished_toggled(self):
        # Re-decide visibility of every running file on the next sync.
        for path, (status, label, item) in list(self._files.items()):
            if status == "running":
                self._files[path] = ("", label, item)
        self.refresh()

    def select_path(self, path):
        path = Path(path)
        day = self._day_item(path.parent)
        if day is None:
            return False
        day.setExpanded(True)
        self._fill_day(day)
        known = self._files.get(path)
        if known is None or known[2] is None:
            return False
        self.tree.setCurrentItem(known[2])
        return True

    def _selected(self, item, _previous):
        path = Path(item.data(0, _PATH)) if item else None
        if path is None or path.suffix != ".h5":
            return
        self.load(path)

    def load(self, path):
        try:
            measurement = load_measurement(path, results=[])
        except (OSError, KeyError, ValueError) as error:
            self._wanted = Path(path)
            self.metadata.setPlainText(f"Could not read {path}:\n{error}")
            self.detail_region.set_hint("unreadable")
            self.simulation_banner.setVisible(False)
            self.measurement = None
            self.resubmit_button.setEnabled(False)
            self.grafana_button.setEnabled(False)
            return
        first_load = self.path != Path(path)
        self._wanted = None
        self._planned = None
        if measurement.status == "running":
            try:
                self._planned = read_planned_points(path)
            except (OSError, KeyError):
                pass
        self.path = Path(path)
        self.measurement = measurement
        self.metadata.setPlainText(describe(measurement))
        attrs = measurement.attrs
        self._show_progress()
        self.simulation_banner.setVisible(bool(attrs.get("simulated")))
        self.resubmit_button.setEnabled(True)
        queued = attrs["rid"] >= 0
        self.grafana_button.setEnabled(queued)
        self.grafana_button.setToolTip(
            "Show this run's results and the readings during it in Grafana"
            if queued
            else "Run with run_local(), so it was never recorded in the database"
        )
        self._fill_choices(keep_selection=not first_load)
        self.replot()

    def _show_progress(self):
        attrs = self.measurement.attrs
        rows = self._live_rows()
        done = attrs["n_done"] if rows is None else max(len(rows), attrs["n_done"])
        self.detail_region.set_hint(
            f"task {attrs['rid']}, {attrs['status']}, {done}/{attrs['n_points']} points"
        )

    def _reload_if_running(self):
        # Live, the feed supplies a running measurement's points instead.
        m = self.measurement
        if m is not None and m.status == "running" and not self.live:
            self.load(self.path)

    # -- plotting ------------------------------------------------------------

    def _fill_choices(self, keep_selection=False):
        m = self.measurement
        scalars = [
            name
            for name, shape in m.result_shapes.items()
            if shape == () and m.result_kinds[name] in "biuf"
        ]
        for row in self._live_rows() or []:
            scalars += [name for name in row["scalars"] if name not in scalars]
        axes = [axis.argument for axis in m.scan.axes]
        for combo, options in (
            (self.y_choice, scalars),
            (self.x_choice, [*axes, _POINT_INDEX]),
            (self.series_choice, [_NONE, *axes, "repetition"]),
        ):
            current = combo.currentText()
            if keep_selection and options == [
                combo.itemText(i) for i in range(combo.count())
            ]:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(options)
            if keep_selection and current in options:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def replot(self):
        self.plot.clear()
        self.plot_message.clear()
        m = self.measurement
        y_name = self.y_choice.currentText()
        if m is None or not y_name:
            if m is not None:
                self.plot_message.setText("No scalar results to plot.")
            return
        rows = self._live_rows()
        points = m.points
        if rows is not None:
            points = self._planned
            y = np.full(len(points["index"]), np.nan)
            for row in rows:
                y[row["index"]] = row["scalars"].get(y_name, np.nan)
        else:
            try:
                y = load_measurement(self.path, results=[y_name]).results[y_name]
                y = np.asarray(y, dtype=float)
            except (OSError, ValueError, TypeError):
                self.plot_message.setText(f"{y_name} is not numeric.")
                return
        x_name = self.x_choice.currentText()
        x = points["index"] if x_name == _POINT_INDEX else points[x_name]
        series_name = self.series_choice.currentText()
        series = None if series_name in ("", _NONE) else points[series_name]

        categorical = x.dtype.kind not in "biuf"
        if categorical:
            labels = list(dict.fromkeys(x.tolist()))
            x = np.array([labels.index(v) for v in x.tolist()])
            self.plot.getAxis("bottom").setTicks([list(enumerate(labels))])
        else:
            self.plot.getAxis("bottom").setTicks(None)
            schema = m.argument_schema.get(x_name, {})
            x = x / (schema.get("scale") or 1.0)

        n = min(len(x), len(y))
        curves = aggregate(x[:n], y[:n], None if series is None else series[:n])
        for i, (value, (xs, means, sems, _counts)) in enumerate(curves.items()):
            colour = _CURVE_COLOURS[i % len(_CURVE_COLOURS)]
            name = (
                None
                if series is None
                else f"{series_name} = {_shown(m, series_name, value)}"
            )
            self.plot.plot(
                xs, means, pen=colour, symbol="o", symbolBrush=colour, name=name
            )
            if np.isfinite(sems).any():
                self.plot.addItem(
                    pg.ErrorBarItem(
                        x=xs, y=means, height=2 * np.nan_to_num(sems), pen=colour
                    )
                )
        x_unit = m.argument_schema.get(x_name, {}).get("unit", "")
        self.plot.setLabel(
            "bottom", x_name, units=None if categorical else x_unit or None
        )
        self.plot.setLabel("left", y_name, units=m.units.get(y_name) or None)

    def _open_in_grafana(self):
        attrs = self.measurement.attrs
        open_in_browser(run_url(attrs["rid"], attrs["t_start"], attrs["t_end"]))

    def _resubmit(self):
        if self.measurement is not None:
            self.resubmit_requested.emit(resubmission(self.measurement))
