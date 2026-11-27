"""The plots of an optimiser run, drawn the same in the Optimisers and Results tabs.

A staged scan gets one plot per stage, a Bayesian run its objective per
evaluation and against each setting, and a fit the image it was made on. Each
plot is drawn from the run's results (the file's, or the feed's while it runs)
and its ``proposed.*`` constants.
"""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from pytweezer.GUI.theme import PLOT_BACKGROUND, plot_title_style, style_plot

_ACCENT = "#5b8def"
_POINTS = "#9aa0b4"
_PLOT_COLUMNS = 2
#: Results plotted; per-setting results are added for Bayesian runs.
PLOTTED = ["stage", "setting", "objective", "reference"]


def live_results(points: list[dict], names: list[str]) -> dict[str, np.ndarray]:
    """``{name: one value per point}`` from the feed's points, in point order; NaN where missing."""
    ordered = sorted(points, key=lambda point: point["index"])
    return {
        name: np.array(
            [float(point.get("scalars", {}).get(name, np.nan)) for point in ordered]
        )
        for name in names
    }


class OptimiserPlots(QWidget):
    """A grid of one optimiser run's plots, with a line below saying why it is empty."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_widget = QWidget()
        self.grid_widget.setObjectName("OptimiserPlots")
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(16)
        self.message = QLabel()
        self.message.setProperty("role", "regionHint")
        self.message.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_widget, 1)
        layout.addWidget(self.message)
        #: The drawing of the run shown, or ``None``.
        self.drawing = None

    def show_run(self, key, schema=None, results=None, constants=None, error=None):
        """Draw the run ``key`` of an optimiser of ``schema``, updating in place while ``key`` is the same.

        Args:
            key: Identifies the run drawn; ``None`` clears the plots.
            schema: The optimiser's schema, as ``Experiment.schema()`` gives it.
            results: ``{name: one value per point}``, at least :func:`plotted_names`.
            constants: The run's constants, for its ``proposed.*`` values.
            error: Why the run's file could not be read, shown instead of plots.
        """
        if error is not None:
            key = None
        if self.drawing is None or self.drawing.key != key:
            while self.grid.count():
                widget = self.grid.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            self.drawing = None
            if key is not None:
                kind = {"staged": _StagedPlots, "fit": _FitPlots}.get(
                    schema["optimiser"]["kind"], _EvaluationPlots
                )
                self.drawing = kind(key, schema)
                for index, plot in enumerate(self.drawing.widgets):
                    self.grid.addWidget(
                        plot, index // _PLOT_COLUMNS, index % _PLOT_COLUMNS
                    )
        if error is not None:
            self.message.setText(f"Its file cannot be read: {error}")
        elif self.drawing is None:
            self.message.setText("")
        else:
            measured = self.drawing.update(results or {}, constants or {})
            self.message.setText("" if measured else "No points measured yet")


@dataclass
class _StagePlot:
    """One stage's plot and the items drawn on it."""

    title: str
    setting: str
    saved: str
    unit: str
    scale: float
    saved_unit: str
    plot: pg.PlotWidget
    shots: pg.PlotDataItem
    means: pg.PlotDataItem
    chosen: pg.PlotDataItem
    line: pg.InfiniteLine


class _StagedPlots:
    """One plot per stage: objective against its setting, and the value it chose.

    The setting is shown in its stage's unit, relative to the points'
    ``reference`` when they record one, which the title then gives in the
    saved parameter's unit.
    """

    def __init__(self, key, schema):
        self.key = key
        optimiser = schema["optimiser"]
        setting_units = optimiser.get("setting_units") or [["", 1.0]] * len(
            optimiser["settings"]
        )
        self.stages = []
        self.widgets = []
        for index, (setting, saved, (unit, scale)) in enumerate(
            zip(
                optimiser["settings"], optimiser["proposes"], setting_units, strict=True
            )
        ):
            title = f"Stage {index + 1}: {setting} → {saved}"
            plot = _plot(title, setting, optimiser["objective"])
            plot.setLabel("bottom", setting, units=unit or None)
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
            self.stages.append(
                _StagePlot(
                    title,
                    setting,
                    saved,
                    unit,
                    float(scale),
                    optimiser.get("units", {}).get(saved, ""),
                    plot,
                    shots,
                    means,
                    chosen,
                    line,
                )
            )
            self.widgets.append(plot)

    def update(self, results, constants) -> bool:
        stages = np.asarray(results.get("stage", []))
        settings = np.asarray(results.get("setting", []), dtype=float)
        objectives = np.asarray(results.get("objective", []), dtype=float)
        references = np.asarray(
            results.get("reference", np.full(len(settings), np.nan)), dtype=float
        )
        for index, stage in enumerate(self.stages):
            here = stages == index
            reference = _first_finite(references[here])
            self._show_reference(stage, reference)
            origin = 0.0 if reference is None else reference
            x = (settings[here] - origin) / stage.scale
            y = objectives[here]
            stage.shots.setData(x, y)
            means_x = np.unique(x)
            means_y = np.array([np.nanmean(y[x == value]) for value in means_x])
            stage.means.setData(means_x, means_y)
            value = constants.get(f"proposed.{stage.saved}")
            if value is None:
                stage.chosen.setData([], [])
                stage.line.setVisible(False)
                continue
            value = (float(value) - origin) / stage.scale
            at_value = np.isclose(means_x, value)
            stage.chosen.setData(means_x[at_value], means_y[at_value])
            stage.line.setValue(value)
            stage.line.setVisible(True)
        return len(objectives) > 0

    @staticmethod
    def _show_reference(stage, reference):
        if reference is None:
            stage.plot.setTitle(stage.title, **plot_title_style())
            stage.plot.setLabel("bottom", stage.setting, units=stage.unit or None)
            return
        stage.plot.setTitle(
            f"{stage.title}<br>from {reference:.10g} {stage.saved_unit}".rstrip(),
            **plot_title_style(),
        )
        stage.plot.setLabel("bottom", f"Δ{stage.setting}", units=stage.unit or None)


def _first_finite(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite[0]) if len(finite) else None


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


def plotted_names(schema: dict) -> list[str]:
    """The results an optimiser run's plots need, for an optimiser of ``schema``."""
    optimiser = schema["optimiser"]
    if optimiser.get("kind") == "fit":
        return ["calibration_image", "roi_sites"]
    if optimiser.get("kind") == "bayesian":
        return [*PLOTTED, *optimiser["settings"]]
    return list(PLOTTED)


def _plot(title, x_label, objective):
    plot = pg.PlotWidget(background=PLOT_BACKGROUND)
    plot.setMinimumSize(280, 200)
    plot.setTitle(title, **plot_title_style())
    style_plot(plot)
    for axis in ("left", "bottom"):
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
