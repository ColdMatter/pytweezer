"""The Results tab: browse measurement files, inspect them, quick-plot a result.

Reads files directly from :func:`~pytweezer.experiment.storage.data_root`, so
on a client PC set ``PYTWEEZER_DATA_DIR`` to the server's data share. The tree
keeps itself up to date (every couple of seconds while the tab is visible):
new measurements appear, and running ones update as they finish. Files are
opened without locking, so the selected measurement can be watched while it
runs.
"""

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.storage import data_root, load_measurement, read_header
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.experiments.queue_view import status_icon
from pytweezer.GUI.theme import PLOT_BACKGROUND, PLOT_FOREGROUND
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments.results")

_PATH = QtCore.Qt.ItemDataRole.UserRole
_FILLED = QtCore.Qt.ItemDataRole.UserRole + 1
_POLL_MS = 2000
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
        *(["SIMULATED (devices were simulated)"] if attrs.get("simulated") else []),
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
    item.setIcon(1, status_icon("crashed") if failed else QtGui.QIcon())


class ResultsPanel(QWidget):
    resubmit_requested = QtCore.pyqtSignal(object)

    def __init__(self, root=None, parent=None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else None
        self.measurement = None
        self.path = None
        layout = QHBoxLayout(self)
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        browser = QWidget()
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.show_unfinished = QCheckBox("Show running")
        self.show_unfinished.toggled.connect(self._show_unfinished_toggled)
        top.addWidget(self.show_unfinished)
        top.addStretch(1)
        browser_layout.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Measurement", "Status", "Label"])
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree.itemExpanded.connect(self._fill_day)
        self.tree.currentItemChanged.connect(self._selected)
        browser_layout.addWidget(self.tree, 1)
        splitter.addWidget(browser)

        detail = QSplitter(QtCore.Qt.Orientation.Vertical)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        detail.addWidget(self.metadata)

        plot_box = QWidget()
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        controls = QFormLayout()
        self.y_choice = QComboBox()
        self.x_choice = QComboBox()
        self.series_choice = QComboBox()
        controls.addRow("Plot", self.y_choice)
        controls.addRow("against", self.x_choice)
        controls.addRow("one curve per", self.series_choice)
        for combo in (self.y_choice, self.x_choice, self.series_choice):
            combo.currentTextChanged.connect(self.replot)
        plot_layout.addLayout(controls)
        self.plot = pg.PlotWidget(background=PLOT_BACKGROUND)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(PLOT_FOREGROUND)
            self.plot.getAxis(axis).setTextPen(PLOT_FOREGROUND)
            # Values are already in their display units.
            self.plot.getAxis(axis).enableAutoSIPrefix(False)
        self.plot.addLegend()
        plot_layout.addWidget(self.plot, 1)
        self.plot_message = QLabel()
        plot_layout.addWidget(self.plot_message)
        buttons = QHBoxLayout()
        self.resubmit_button = QPushButton("Resubmit with these arguments")
        self.resubmit_button.clicked.connect(self._resubmit)
        self.resubmit_button.setEnabled(False)
        buttons.addStretch(1)
        buttons.addWidget(self.resubmit_button)
        plot_layout.addLayout(buttons)
        detail.addWidget(plot_box)
        splitter.addWidget(detail)
        splitter.setStretchFactor(1, 3)

        #: Every measurement file seen: path -> (status, label, tree item or None if hidden).
        self._files = {}
        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._poll_tick)
        self._poll.start(_POLL_MS)
        QtCore.QTimer.singleShot(0, self.refresh)

    @property
    def data_root(self):
        return self.root if self.root is not None else data_root()

    # -- browsing ----------------------------------------------------------

    def _poll_tick(self):
        # Hidden behind another tab: skip, the data root may be a network share.
        if self.isVisible():
            self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        """Bring the tree up to date with the files on disk, in place."""
        try:
            days = sorted(
                (p for p in self.data_root.glob("*/*/*") if p.is_dir()), reverse=True
            )
        except OSError:
            logger.debug("Cannot list %s", self.data_root, exc_info=True)
            return
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
            self.metadata.setPlainText(f"Could not read {path}:\n{error}")
            self.measurement = None
            self.resubmit_button.setEnabled(False)
            return
        first_load = self.path != Path(path)
        self.path = Path(path)
        self.measurement = measurement
        self.metadata.setPlainText(describe(measurement))
        self.resubmit_button.setEnabled(True)
        self._fill_choices(keep_selection=not first_load)
        self.replot()

    def _reload_if_running(self):
        if self.measurement is not None and self.measurement.status == "running":
            self.load(self.path)

    # -- plotting ------------------------------------------------------------

    def _fill_choices(self, keep_selection=False):
        m = self.measurement
        scalars = [
            name
            for name, shape in m.result_shapes.items()
            if shape == () and m.result_kinds[name] in "biuf"
        ]
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
        try:
            y = load_measurement(self.path, results=[y_name]).results[y_name]
            y = np.asarray(y, dtype=float)
        except (OSError, ValueError, TypeError):
            self.plot_message.setText(f"{y_name} is not numeric.")
            return
        x_name = self.x_choice.currentText()
        x = m.points["index"] if x_name == _POINT_INDEX else m.points[x_name]
        series_name = self.series_choice.currentText()
        series = None if series_name in ("", _NONE) else m.points[series_name]

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

    def _resubmit(self):
        if self.measurement is not None:
            self.resubmit_requested.emit(resubmission(self.measurement))
