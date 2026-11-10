"""Results-tab views of the MOT cloud fits a measurement recorded."""

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from pytweezer.analysis.cloud_fit import contour_ellipses
from pytweezer.analysis.cloud_results import (
    SIZES,
    cloud_cameras,
    cloud_sizes,
    cloud_temperature,
    guess_time_axis,
    needed_results,
    point_image,
    scanned_arguments,
)
from pytweezer.experiment.summary import display_value
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.experiments.analyses import ComputedAnalysisView
from pytweezer.GUI.theme import (
    CURVE_COLORS,
    PLOT_BACKGROUND,
    PLOT_FOREGROUND,
    PLOT_LINE_WIDTH,
    PLOT_SYMBOL_SIZE,
    style_plot,
)

POINT_INDEX = "index"
OUTLINE_PEN = pg.mkPen("w", width=1)
SIZE_LABELS = {"sigma_x": "σx", "sigma_y": "σy"}
CENTRE_LABELS = {"x0": "x0", "y0": "y0"}
MOTMASTER_TICK_US = 10.0
SECOND_US = 1e6
TIME_UNITS = ("s", "ms", "us", "µs", "ns")


def themed_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget(background=PLOT_BACKGROUND)
    style_plot(plot)
    return plot


def refill(combo, items, keep=True):
    """Replace ``combo``'s ``(text, data)`` items silently, keeping the choice if ``keep``."""
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for text, data in items:
        combo.addItem(text, data)
    index = combo.findData(current) if keep else -1
    combo.setCurrentIndex(max(index, 0))
    combo.blockSignals(False)


def holds_seconds(measurement, argument: str) -> bool:
    """Whether scanned ``argument`` holds SI seconds, not MOTMaster clock ticks.

    Native arguments are held in SI whatever unit they display in; MOTMaster
    parameters are sent raw, so theirs stay in ticks.
    """
    schema = measurement.argument_schema.get(argument, {})
    return schema.get("unit") in TIME_UNITS and "motmaster_parameter" not in schema


def axis_scale(measurement, axis: str) -> tuple[float, str]:
    """``(scale, unit)`` that put ``axis`` in its display unit."""
    schema = measurement.argument_schema.get(axis, {})
    return schema.get("scale") or 1.0, schema.get("unit", "")


@dataclass(frozen=True)
class CloudFitDrawing:
    """What :class:`CloudFitView` draws for one camera, axis and point."""

    sizes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    size_unit: str
    point_x: float
    image: np.ndarray | None
    outlines: list[np.ndarray]


class CloudFitView(ComputedAnalysisView):
    """A point's mean image with its fit contours, beside σx and σy against a scanned argument."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = QComboBox()
        self.camera.setToolTip("Camera whose cloud fits to show")
        self.against = QComboBox()
        self.against.setToolTip(
            "Scanned argument to plot the sizes against; points sharing a value "
            "are pooled"
        )
        self.point = QComboBox()
        self.point.setToolTip("Point whose mean image and fit to show")
        controls = QHBoxLayout()
        controls.setSpacing(8)
        for text, combo, width in (
            ("Camera", self.camera, 160),
            ("Plot against", self.against, 160),
            ("Point", self.point, 240),
        ):
            combo.setFixedWidth(width)
            combo.currentIndexChanged.connect(self.recompute)
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(QLabel(text))
            controls.addWidget(combo)
        controls.addStretch(1)

        self.image_plot = themed_plot()
        image_view = self.image_plot.getPlotItem()
        image_view.setLabel("bottom", "x", units="px")
        image_view.setLabel("left", "y", units="px")
        for axis in ("left", "bottom"):
            image_view.getAxis(axis).enableAutoSIPrefix(False)
        image_view.invertY(True)
        image_view.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setColorMap(pg.colormap.get("magma"))
        image_view.addItem(self.image_item)
        self.outline_curves: list[pg.PlotCurveItem] = []

        self.size_plot = themed_plot()
        self.size_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.size_plot.addLegend()
        self.size_curves = {}
        self.size_errors = {}
        for i, field in enumerate(SIZES):
            colour = CURVE_COLORS[i]
            self.size_curves[field] = self.size_plot.plot(
                [],
                [],
                pen=pg.mkPen(colour, width=PLOT_LINE_WIDTH),
                symbol="o",
                symbolSize=PLOT_SYMBOL_SIZE,
                symbolBrush=colour,
                name=SIZE_LABELS[field],
            )
            self.size_errors[field] = pg.ErrorBarItem(
                x=np.empty(0),
                y=np.empty(0),
                height=np.empty(0),
                pen=pg.mkPen(colour, width=PLOT_LINE_WIDTH),
            )
            self.size_plot.addItem(self.size_errors[field])
        self.point_marker = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(PLOT_FOREGROUND, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self.point_marker.setVisible(False)
        self.size_plot.addItem(self.point_marker)

        plots = QHBoxLayout()
        plots.setSpacing(8)
        plots.addWidget(self.image_plot, 1)
        plots.addWidget(self.size_plot, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(plots, 1)
        layout.addWidget(self.message)

    def set_measurement(self, archive, path, measurement):
        refill(self.camera, [(camera, camera) for camera in cloud_cameras(measurement)])
        refill(
            self.against,
            [(axis, axis) for axis in scanned_arguments(measurement)]
            + [("point index", POINT_INDEX)],
        )
        refill(self.point, self._point_labels(measurement), keep=False)
        super().set_measurement(archive, path, measurement)

    @staticmethod
    def _point_labels(measurement):
        points = measurement.points
        axes = [axis.argument for axis in measurement.scan.axes]
        repeated = measurement.scan.repetitions > 1
        labels = []
        for row, index in enumerate(points["index"]):
            values = [
                f"{axis} = {display_value(measurement, axis, points[axis][row])}"
                for axis in axes
            ]
            if repeated:
                values.append(f"repetition {points['repetition'][row]}")
            text = f"point {index}" + (f": {', '.join(values)}" if values else "")
            labels.append((text, row))
        return labels

    def unavailable_reason(self, measurement):
        if not cloud_cameras(measurement):
            return "This file recorded no cloud fits."
        return None

    def needed_results(self, measurement):
        return needed_results(measurement)

    def settings(self):
        return (
            self.camera.currentData(),
            self.against.currentData(),
            self.point.currentData(),
        )

    def compute(self, loaded, settings):
        camera, axis, row = settings
        scale, _ = axis_scale(loaded, axis)
        sizes = {
            field: (x / scale, mean, error)
            for field, (x, mean, error) in cloud_sizes(loaded, camera, axis).items()
        }
        image_name = f"{camera}_mean_image"
        mean_image = (
            self.point_result(image_name, row)
            if image_name in loaded.result_shapes
            else None
        )
        image, fit = point_image(loaded, camera, row, mean_image)
        return CloudFitDrawing(
            sizes=sizes,
            size_unit=loaded.units.get(f"{camera}_sigma_x", ""),
            point_x=float(loaded.points[axis][row]) / scale,
            image=image,
            outlines=contour_ellipses(fit) if image is not None and fit.success else [],
        )

    def show_result(self, result):
        for curve in self.outline_curves:
            self.image_plot.removeItem(curve)
        self.outline_curves = []
        if result is None:
            self.image_item.clear()
            for field in SIZES:
                self.size_curves[field].setData([], [])
                self.size_errors[field].setData(
                    x=np.empty(0), y=np.empty(0), height=np.empty(0)
                )
            self.point_marker.setVisible(False)
            return
        self._draw_sizes(result)
        self._draw_image(result)

    def _draw_sizes(self, result):
        for field, (x, mean, error) in result.sizes.items():
            self.size_curves[field].setData(x, mean)
            self.size_errors[field].setData(
                x=x, y=mean, height=2 * np.nan_to_num(error)
            )
        self.point_marker.setValue(result.point_x)
        self.point_marker.setVisible(True)
        axis = self.against.currentData()
        if axis == POINT_INDEX:
            self.size_plot.setLabel("bottom", "point index")
        else:
            _, unit = axis_scale(self._measurement, axis)
            self.size_plot.setLabel("bottom", axis, units=unit or None)
        # Lengths in metres take an SI prefix; in pixels they must not.
        self.size_plot.getAxis("left").enableAutoSIPrefix(result.size_unit == "m")
        self.size_plot.setLabel("left", "σ", units=result.size_unit or None)

    def _draw_image(self, result):
        has_images = f"{self.camera.currentData()}_mean_image" in (
            self._measurement.result_shapes
        )
        self.image_plot.setVisible(has_images)
        if result.image is None:
            self.image_item.clear()
            if has_images:
                self.message.setText("No mean image was recorded for this point.")
            return
        self.image_item.setImage(result.image)
        for x, y in result.outlines:
            curve = pg.PlotCurveItem(x, y, pen=OUTLINE_PEN)
            self.image_plot.addItem(curve)
            self.outline_curves.append(curve)


def format_value(value: float, error: float, scale: float = 1.0) -> str:
    """``value ± error`` times ``scale``, the error to two figures and the value to its decimal place."""
    value, error = value * scale, error * scale
    if not (math.isfinite(error) and error > 0):
        return f"{value:.3g}"
    decimals = max(0, 1 - math.floor(math.log10(error)))
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"


@dataclass(frozen=True)
class TemperatureDrawing:
    """What :class:`CloudTemperatureView` shows for one camera and time argument."""

    result: dict
    argument: str
    seconds_per_unit: float


class CloudTemperatureView(ComputedAnalysisView):
    """The temperature of a cloud from its expansion over a scanned free-fall time.

    σ² against t² for x and y with their fits, and the centre against t² with
    ``g`` fitted to y0 as a check of the time axis. Lengths are shown in mm and
    times in ms.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = QComboBox()
        self.camera.setToolTip("Camera whose cloud fits to analyse")
        self.time_argument = QComboBox()
        self.time_argument.setToolTip("Scanned argument that is the free-fall time")
        self.time_unit = CompactDoubleSpinBox()
        self.time_unit.setRange(1e-3, 1e9)
        self.time_unit.setValue(MOTMASTER_TICK_US)
        self.time_unit.setSuffix(" µs")
        self.time_unit.setToolTip(
            "Length of one unit of the time argument's value. MOTMaster times "
            "are in clock ticks of 10 µs; an argument in seconds is 1000000 µs"
        )
        self._unit_in_seconds = False

        controls = QHBoxLayout()
        controls.setSpacing(8)
        for text, widget, width in (
            ("Camera", self.camera, 160),
            ("Time argument", self.time_argument, 200),
            ("Time unit", self.time_unit, 110),
        ):
            widget.setFixedWidth(width)
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(QLabel(text))
            controls.addWidget(widget)
        controls.addStretch(1)
        self.camera.currentIndexChanged.connect(self.recompute)
        self.time_argument.currentIndexChanged.connect(self._time_argument_changed)
        self.time_unit.valueChanged.connect(self.recompute_soon)

        self.sigma_plot = themed_plot()
        self.sigma_plot.addLegend()
        self.sigma_plot.setLabel("bottom", "t²", units="ms²")
        self.sigma_curves, self.sigma_errors, self.sigma_lines = {}, {}, {}
        self.centre_plot = themed_plot()
        self.centre_plot.addLegend()
        self.centre_plot.setLabel("bottom", "t²", units="ms²")
        self.centre_curves, self.centre_errors = {}, {}
        for plot in (self.sigma_plot, self.centre_plot):
            for axis in ("left", "bottom"):
                plot.getAxis(axis).enableAutoSIPrefix(False)
        for i, field in enumerate(SIZES):
            self.sigma_curves[field], self.sigma_errors[field] = self._series(
                self.sigma_plot, CURVE_COLORS[i], SIZE_LABELS[field]
            )
            self.sigma_lines[field] = self.sigma_plot.plot(
                [], [], pen=pg.mkPen(CURVE_COLORS[i], width=PLOT_LINE_WIDTH)
            )
        for i, field in enumerate(CENTRE_LABELS):
            self.centre_curves[field], self.centre_errors[field] = self._series(
                self.centre_plot, CURVE_COLORS[i], CENTRE_LABELS[field]
            )
        self.gravity_line = self.centre_plot.plot(
            [], [], pen=pg.mkPen(CURVE_COLORS[1], width=PLOT_LINE_WIDTH)
        )

        # Monospaced so the digits hold their columns as the values change.
        self.readout = QLabel()
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.readout.setFont(font)
        self.readout.setContentsMargins(10, 6, 10, 6)
        self.readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        plots = QHBoxLayout()
        plots.setSpacing(8)
        plots.addWidget(self.sigma_plot, 1)
        plots.addWidget(self.centre_plot, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.readout)
        layout.addLayout(plots, 1)
        layout.addWidget(self.message)

    @staticmethod
    def _series(plot, colour, name):
        curve = plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolSize=PLOT_SYMBOL_SIZE,
            symbolBrush=colour,
            name=name,
        )
        errors = pg.ErrorBarItem(
            x=np.empty(0),
            y=np.empty(0),
            height=np.empty(0),
            pen=pg.mkPen(colour, width=PLOT_LINE_WIDTH),
        )
        plot.addItem(errors)
        return curve, errors

    def set_measurement(self, archive, path, measurement):
        refill(self.camera, [(camera, camera) for camera in cloud_cameras(measurement)])
        arguments = scanned_arguments(measurement)
        refill(self.time_argument, [(name, name) for name in arguments], keep=False)
        guess = guess_time_axis(arguments)
        if guess is not None:
            self.time_argument.blockSignals(True)
            self.time_argument.setCurrentIndex(self.time_argument.findData(guess))
            self.time_argument.blockSignals(False)
        self._sync_time_unit(measurement)
        super().set_measurement(archive, path, measurement)

    def unavailable_reason(self, measurement):
        if not cloud_cameras(measurement):
            return "This file recorded no cloud fits."
        if "cloud_mass_amu" not in measurement.constants:
            return (
                "The experiment recorded no species, so there is no mass to "
                "turn the expansion into a temperature."
            )
        if not scanned_arguments(measurement):
            return "This file scanned no argument to use as the free-fall time."
        return None

    def _sync_time_unit(self, measurement):
        """Default the time unit when the argument's kind changes: seconds, or MOTMaster ticks."""
        argument = self.time_argument.currentData()
        seconds = argument is not None and holds_seconds(measurement, argument)
        if seconds != self._unit_in_seconds:
            self._unit_in_seconds = seconds
            self.time_unit.blockSignals(True)
            self.time_unit.setValue(SECOND_US if seconds else MOTMASTER_TICK_US)
            self.time_unit.blockSignals(False)

    def _time_argument_changed(self):
        if self._measurement is not None:
            self._sync_time_unit(self._measurement)
        self.recompute()

    def needed_results(self, measurement):
        return needed_results(measurement)

    def settings(self):
        return (
            self.camera.currentData(),
            self.time_argument.currentData(),
            self.time_unit.value() * 1e-6,
        )

    def compute(self, loaded, settings):
        camera, argument, seconds_per_unit = settings
        return TemperatureDrawing(
            cloud_temperature(loaded, camera, argument, seconds_per_unit),
            argument,
            seconds_per_unit,
        )

    def show_result(self, result):
        if result is None:
            self.readout.setText("")
            for curve in (*self.sigma_curves.values(), *self.centre_curves.values()):
                curve.setData([], [])
            for errors in (*self.sigma_errors.values(), *self.centre_errors.values()):
                errors.setData(x=np.empty(0), y=np.empty(0), height=np.empty(0))
            for line in (*self.sigma_lines.values(), self.gravity_line):
                line.setData([], [])
            return
        drawing = result.result
        metres = drawing["unit"] == "m"
        # σ² and t² in mm² and ms²; centres in mm.
        square, length = (1e6, 1e3) if metres else (1.0, 1.0)
        self.sigma_plot.setLabel("left", "σ²", units="mm²" if metres else "px²")
        self.centre_plot.setLabel("left", "centre", units="mm" if metres else "px")
        for field in SIZES:
            t2, sigma2, error = drawing["sigma2"][field]
            self._draw_series(
                self.sigma_curves[field],
                self.sigma_errors[field],
                (t2 * 1e6, sigma2 * square, error * square),
            )
            fit = drawing["fits"][field]
            self._draw_line(
                self.sigma_lines[field],
                t2,
                fit.intercept,
                fit.slope,
                1e6,
                square,
            )
        for field in CENTRE_LABELS:
            t2, centre, error = drawing["centres"][field]
            self._draw_series(
                self.centre_curves[field],
                self.centre_errors[field],
                (t2 * 1e6, centre * length, error * length),
            )
        gravity = drawing["gravity"]
        t2 = drawing["centres"]["y0"][0]
        self._draw_line(
            self.gravity_line,
            t2,
            gravity.offset,
            gravity.g / 2,
            1e6,
            length,
        )
        self.readout.setText(self._summary(result, metres))

    @staticmethod
    def _draw_series(curve, errors, series):
        t2, value, error = series
        curve.setData(t2, value)
        errors.setData(x=t2, y=value, height=2 * np.nan_to_num(error))

    @staticmethod
    def _draw_line(line, t2, intercept, slope, t_scale, scale):
        if t2.size == 0 or not (np.isfinite(intercept) and np.isfinite(slope)):
            line.setData([], [])
            return
        ends = np.array([0.0, t2.max() * 1.05])
        line.setData(ends * t_scale, (intercept + slope * ends) * scale)

    @staticmethod
    def _summary(result, metres):
        drawing = result.result
        lines = []
        for field in SIZES:
            fit = drawing["fits"][field]
            name = SIZE_LABELS[field][-1]
            if not fit.success:
                lines.append(f"{name}: {fit.message}")
                continue
            sigma0 = (
                f"σ₀ = {format_value(fit.sigma0, fit.sigma0_err, 1e3)} mm"
                if np.isfinite(fit.sigma0)
                else "σ₀ unresolved"
            )
            lines.append(
                f"T{name} = {format_value(fit.temperature_K, fit.temperature_err_K, 1e6)} µK"
                f"    {sigma0}    ({fit.message})"
            )
        gravity = drawing["gravity"]
        if gravity.success and metres:
            lines.append(
                f"g = {format_value(gravity.g, gravity.g_err)} m/s² from y0, "
                "taking +y (down the image) as down; the camera's orientation "
                f"is not recorded    ({gravity.message})"
            )
        else:
            lines.append(f"g: {gravity.message}")
        unit_us = result.seconds_per_unit * 1e6
        lines.append(f"t = {result.argument} × {unit_us:g} µs")
        return "\n".join(lines)
