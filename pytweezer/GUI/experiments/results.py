"""The Results tab: browse measurement files, inspect them, quick-plot a result.

The Analysis dropdown offers "Plot against" for every measurement, then the
analyses its experiment declares (:mod:`pytweezer.GUI.experiments.analyses`),
which become available once the measurement finishes.

Reads files through an archive (:mod:`pytweezer.experiment.archive`): on the
Experiment Manager's PC directly, elsewhere fetched from the manager over RPC.
With an :class:`~pytweezer.GUI.experiments.feed.ExperimentFeed` connected, a
running measurement is plotted from the points the manager publishes rather
than by re-reading its file (which HDF5 does not support while it is being
written); its file is read once for its metadata, and again when the manager
reports it finished. The tree updates whenever the queue changes, with a slow
poll for files written outside the manager.

Without a connected feed the tab polls the archive every couple of seconds
while visible, re-reading a selected running measurement each time.

Every archive read runs off the GUI thread (one at a time per kind: listing,
loading, plotting), since over RPC a read can wait the archive's whole timeout
for an unreachable manager. Asking again while one is in flight replaces any
earlier pending request, and only the newest request's answer is shown.
"""

import json
import time
from pathlib import PurePosixPath

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
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.archive import (
    ArchiveUnavailable,
    ResultsArchive,
    default_archive,
)
from pytweezer.experiment.summary import aggregate, describe, display_value
from pytweezer.experiment.task import TaskRequest, TaskStatus
from pytweezer.GUI.background import BackgroundReader
from pytweezer.GUI.components import Region, status_icon
from pytweezer.GUI.experiments.analyses import ANALYSES, declared_analyses
from pytweezer.GUI.grafana import open_in_browser, run_url
from pytweezer.GUI.theme import PLOT_BACKGROUND, PLOT_FOREGROUND
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments.results")

_PATH = QtCore.Qt.ItemDataRole.UserRole
_FILLED = QtCore.Qt.ItemDataRole.UserRole + 1
_POLL_MS = 2000
#: With a connected feed, how often to look for files the manager didn't write.
_FALLBACK_POLL_S = 30.0
_NONE = "(none)"
_POINT_INDEX = "point index"
_PLOT_AGAINST = "Plot against"
_PLOT_HINT = "mean and standard error over repetitions"
_CURVE_COLOURS = ["#5b8def", "#f5a623", "#2ecc71", "#c678dd", "#e5c07b", "#56b6c2"]


def resubmission(measurement):
    """A :class:`TaskRequest` that repeats ``measurement``.

    Arguments the run took from saved parameters are left unset, so they take
    the saved values current when it runs again.
    """
    scanned = {axis.argument for axis in measurement.scan.axes}
    from_saved = json.loads(measurement.attrs.get("saved_arguments", "{}"))
    return TaskRequest(
        experiment=measurement.attrs["experiment"],
        class_name=measurement.attrs["class_name"],
        args={
            k: None if k in from_saved else v
            for k, v in measurement.arguments.items()
            if k not in scanned
        },
        scan=measurement.scan,
        label=str(measurement.attrs.get("label", "")),
    )


def _show_file(item, path, status, label):
    item.setText(0, PurePosixPath(path).stem.replace("_", "  ", 1))
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

    def __init__(self, root=None, feed=None, archive=None, parent=None):
        super().__init__(parent)
        if archive is None:
            archive = ResultsArchive(root) if root is not None else default_archive()
        self.archive = archive
        #: A second connection, so loading a large file never holds up the listing.
        self._loader = archive.clone()
        self._reader = BackgroundReader(self)
        self._reading = set()
        self._pending_reads = {}
        #: A path to select as soon as a listing shows it.
        self._select_when_listed = None
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
        self.browser = Region("well", "Measurements", "fetching…")
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
        plot_page = QWidget()
        plot_layout = QVBoxLayout(plot_page)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addLayout(controls)
        plot_layout.addWidget(self.plot, 1)
        plot_layout.addWidget(self.plot_message)

        self.analysis_choice = QComboBox()
        self.analysis_choice.setFixedWidth(220)
        self.analysis_choice.setToolTip(
            "Plot a result against the scan, or one of the analyses this "
            "experiment offers"
        )
        self.analysis_choice.addItem(_PLOT_AGAINST, None)
        self.analysis_choice.currentIndexChanged.connect(self._analysis_chosen)
        self.analysis_stack = QStackedWidget()
        self.analysis_stack.addWidget(plot_page)
        #: Each analysis's view, built the first time it is chosen.
        self._views = {}
        #: The analysis last chosen for each experiment class, this session.
        self._chosen = {}
        self.plot_region = Region("well", "Analysis", _PLOT_HINT)
        self.plot_region.header.insertWidget(1, self.analysis_choice)
        self.plot_region.body.addWidget(self.analysis_stack, 1)

        detail = QSplitter(QtCore.Qt.Orientation.Vertical)
        detail.setObjectName("PanelSplitter")
        detail.addWidget(self.detail_region)
        detail.addWidget(self.plot_region)
        detail.setStretchFactor(0, 1)
        detail.setStretchFactor(1, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 1000])

        #: Every measurement seen: path -> (status, label, tree item or None if hidden).
        self._files = {}
        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._poll_tick)
        self._poll.start(_POLL_MS)
        if feed is not None:
            feed.queue_changed.connect(self._queue_changed)
            feed.point_received.connect(self._point_received)
        QtCore.QTimer.singleShot(0, self.refresh)

    # -- browsing ----------------------------------------------------------

    @property
    def live(self):
        """Whether running measurements come from the feed instead of the file."""
        return self.feed is not None and self.feed.connected

    def _poll_tick(self):
        # Hidden behind another tab: skip, each refresh may be a round trip.
        if not self.isVisible():
            return
        if self.measurement is None and self._wanted is not None:
            self.load(self._wanted)
        if not self.live or time.monotonic() - self._last_refresh > _FALLBACK_POLL_S:
            self.refresh()

    def _queue_changed(self, snapshot):
        tasks = [*(snapshot.get("running") or []), *snapshot.get("queue", [])]
        tasks += snapshot.get("history", [])
        statuses = {task["rid"]: task["status"] for task in tasks}
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

    # -- reading off the GUI thread -------------------------------------------

    def _read_in_background(self, kind, read, show):
        """Run ``read()`` on a worker thread, then ``show(result or exception)`` here."""
        if kind in self._reading:
            self._pending_reads[kind] = (read, show)
            return
        self._reading.add(kind)
        self._reader.start(
            f"Results {kind}",
            read,
            lambda outcome: self._finish_read(kind, show, outcome),
        )

    def _finish_read(self, kind, show, outcome):
        self._reading.discard(kind)
        newer = self._pending_reads.pop(kind, None)
        if newer is not None:
            self._read_in_background(kind, *newer)
            return
        show(outcome)

    # -- browsing ------------------------------------------------------------

    def refresh(self):
        """Bring the tree up to date with the archive, in place."""
        self._last_refresh = time.monotonic()
        filled = [
            item.data(0, _PATH) for item in self._day_items() if item.data(0, _FILLED)
        ]
        self._read_in_background(
            "listing", lambda: self._list(filled), self._show_listing
        )

    def _list(self, filled_days):
        days = self.archive.days()
        runs = {}
        for day in filled_days:
            try:
                runs[day] = self.archive.runs(day)
            except OSError:
                logger.debug("Cannot list %s", day, exc_info=True)
        return days, runs

    def _show_listing(self, outcome):
        if isinstance(outcome, ArchiveUnavailable):
            logger.debug("Cannot list %s", self.archive.location, exc_info=outcome)
            self.browser.set_hint("Experiment Manager not reachable")
            self.browser.hint.setToolTip(
                f"Measurements are fetched from {self.archive.location}; "
                "check it is running in the Servers tab"
            )
            return
        if isinstance(outcome, OSError):
            logger.debug("Cannot list %s", self.archive.location, exc_info=outcome)
            self.browser.set_hint(f"{self.archive.location} not found")
            self.browser.hint.setToolTip("")
            return
        if isinstance(outcome, Exception):
            logger.error("Cannot list %s", self.archive.location, exc_info=outcome)
            return
        days, runs = outcome
        self.browser.set_hint("newest first")
        self.browser.hint.setToolTip("")
        newest_was_open = self._newest_day_open()
        for day in days:
            item = self._day_item(day)
            if item is None:
                item = self._add_day(day)
            if day in runs:
                self._sync_day(item, runs[day])
        if newest_was_open and self.tree.topLevelItemCount():
            self.tree.topLevelItem(0).setExpanded(True)
        if self._select_when_listed is not None:
            self.select_path(self._select_when_listed)
        self._reload_if_running()

    def _day_items(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def _newest_day_open(self):
        if not self.tree.topLevelItemCount():
            return True  # first fill: open the newest day
        return self.tree.topLevelItem(0).isExpanded()

    def _day_item(self, day):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, _PATH) == day:
                return item
        return None

    def _add_day(self, day):
        item = QTreeWidgetItem([day.replace("/", "-")])
        item.setData(0, _PATH, day)
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        index = 0
        while (
            index < self.tree.topLevelItemCount()
            and self.tree.topLevelItem(index).data(0, _PATH) > day
        ):
            index += 1
        self.tree.insertTopLevelItem(index, item)
        return item

    def _fill_day(self, day_item):
        if not day_item.data(0, _FILLED):
            day_item.setData(0, _FILLED, True)
            self.refresh()

    def _sync_day(self, day_item, runs):
        for run in runs:
            path, status, label = run["path"], run["status"], run["label"]
            known = self._files.get(path)
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
        item.setData(0, _PATH, path)
        index = 0
        while (
            index < day_item.childCount()
            and day_item.child(index).data(0, _PATH) > path
        ):
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
        """Select the measurement at data-root-relative ``path``, now or once it is listed."""
        known = self._files.get(path)
        if known is not None and known[2] is not None:
            self._select_when_listed = None
            self.tree.setCurrentItem(known[2])
            return
        self._select_when_listed = path
        day = self._day_item(PurePosixPath(path).parent.as_posix())
        if day is not None:
            day.setExpanded(True)
            self._fill_day(day)

    def _selected(self, item, _previous):
        path = item.data(0, _PATH) if item else None
        if path is None or not path.endswith(".h5"):
            return
        self.load(path)

    def load(self, path):
        self._read_in_background(
            "load",
            lambda: self._read_measurement(path),
            lambda outcome: self._show_measurement(path, outcome),
        )

    def _read_measurement(self, path):
        measurement = self._loader.load(path, results=[])
        planned = None
        if measurement.status == "running":
            try:
                planned = self._loader.planned_points(path)
            except (OSError, KeyError):
                pass
        return measurement, planned

    def _show_measurement(self, path, outcome):
        if isinstance(outcome, Exception):
            if not isinstance(outcome, (OSError, KeyError, ValueError)):
                logger.error("Could not read %s", path, exc_info=outcome)
            self._wanted = path
            self.metadata.setPlainText(f"Could not read {path}:\n{outcome}")
            self.detail_region.set_hint("unreadable")
            self.simulation_banner.setVisible(False)
            self.measurement = None
            self.resubmit_button.setEnabled(False)
            self.grafana_button.setEnabled(False)
            return
        measurement, planned = outcome
        first_load = self.path != path
        self._wanted = None
        self._planned = planned
        self.path = path
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
        self._fill_analyses()

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

    # -- analyses ------------------------------------------------------------

    def _fill_analyses(self):
        """Offer the plot and the measurement's analyses, keeping its class's last choice."""
        m = self.measurement
        combo = self.analysis_choice
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(_PLOT_AGAINST, None)
        for name in declared_analyses(m):
            entry = ANALYSES.get(name)
            combo.addItem(entry.label if entry else name, name)
            item = combo.model().item(combo.count() - 1)
            if entry is None:
                item.setEnabled(False)
                item.setToolTip("Not known to this GUI; update pytweezer")
            elif m.status == "running":
                item.setEnabled(False)
                item.setToolTip("Available when the measurement finishes")
        index = combo.findData(self._chosen.get(m.attrs.get("class_name")))
        if index < 0 or not combo.model().item(index).isEnabled():
            index = 0
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._show_analysis()

    def _analysis_chosen(self):
        if self.measurement is not None:
            class_name = self.measurement.attrs.get("class_name")
            self._chosen[class_name] = self.analysis_choice.currentData()
        self._show_analysis()

    def _show_analysis(self):
        name = self.analysis_choice.currentData()
        if name is None:
            self.analysis_stack.setCurrentIndex(0)
            self.plot_region.set_hint(_PLOT_HINT)
            self.replot()
            return
        view = self._views.get(name)
        if view is None:
            view = self._views[name] = ANALYSES[name].view()
            self.analysis_stack.addWidget(view)
        self.analysis_stack.setCurrentWidget(view)
        self.plot_region.set_hint("")
        if self.measurement is not None:
            view.set_measurement(self.archive, self.path, self.measurement)

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
        if self.analysis_stack.currentIndex() != 0:
            return
        self.plot.clear()
        self.plot_message.clear()
        m = self.measurement
        y_name = self.y_choice.currentText()
        if m is None or not y_name:
            if m is not None:
                self.plot_message.setText("No scalar results to plot.")
            return
        rows = self._live_rows()
        if rows is not None:
            points = self._planned
            y = np.full(len(points["index"]), np.nan)
            for row in rows:
                y[row["index"]] = row["scalars"].get(y_name, np.nan)
            self._draw(m, points, y_name, y)
            return
        path = self.path
        self._read_in_background(
            "plot",
            lambda: np.asarray(
                self._loader.load(path, results=[y_name]).results[y_name], dtype=float
            ),
            lambda outcome: self._show_result_values(m, y_name, outcome),
        )

    def _show_result_values(self, m, y_name, outcome):
        if m is not self.measurement or y_name != self.y_choice.currentText():
            return
        if isinstance(outcome, ArchiveUnavailable):
            self.plot_message.setText("Experiment Manager not reachable.")
        elif isinstance(outcome, (OSError, ValueError, TypeError)):
            self.plot_message.setText(f"{y_name} is not numeric.")
        elif isinstance(outcome, Exception):
            logger.error("Could not plot %s", y_name, exc_info=outcome)
            self.plot_message.setText(f"Could not read {y_name}.")
        else:
            self._draw(m, m.points, y_name, outcome)

    def _draw(self, m, points, y_name, y):
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
                else f"{series_name} = {display_value(m, series_name, value)}"
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
