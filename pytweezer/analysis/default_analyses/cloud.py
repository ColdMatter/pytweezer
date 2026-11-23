"""The MOT cloud fits a measurement file recorded: ``cloud_fit`` and ``cloud_temperature``.

Reads the cloud fits an experiment records per point for each camera, as
:meth:`~pytweezer.experiments.load_tweezers.LoadTweezersMOTImage.record_cloud_fit`
does for the ORCA: ``<camera>_<field>`` for every field of
:data:`~pytweezer.analysis.cloud_fit.FIELDS`, ``<camera>_<field>_error`` for
each of :data:`~pytweezer.analysis.cloud_fit.ERROR_FIELDS`, and optionally
``<camera>_mean_image``.

The file does not record where the mean image sits in the camera frame, so
:func:`point_image` refits it in its own pixel coordinates: the same fit of the
same image as the one recorded, placed where the image is drawn.

:class:`CloudFitView` and :class:`CloudTemperatureView` are the Results-tab
views.
"""

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from scipy.constants import atomic_mass, k

from pytweezer.analysis.cloud_fit import (
    ERROR_FIELDS,
    FIELDS,
    CloudFit,
    contour_ellipses,
    fit_cloud,
)
from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import aggregate, display_value
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.experiments.analyses import AnalysisEntry, ComputedAnalysisView
from pytweezer.GUI.theme import (
    CURVE_COLORS,
    PLOT_BACKGROUND,
    PLOT_FOREGROUND,
    PLOT_LINE_WIDTH,
    PLOT_SYMBOL_SIZE,
    style_plot,
)

SIZES = ("sigma_x", "sigma_y")
CENTRES = ("x0", "y0")
TIME_ARGUMENT_HINTS = ("freeexp", "freefall", "drop")
MIN_TIMES = 3


def cloud_cameras(measurement: Measurement) -> list[str]:
    """The cameras whose cloud fits the file recorded, in recording order."""
    return [
        name.removesuffix("_sigma_x")
        for name, shape in measurement.result_shapes.items()
        if name.endswith("_sigma_x") and shape == ()
    ]


def scanned_arguments(measurement: Measurement) -> list[str]:
    """The numeric scanned arguments, in scan order."""
    return [
        axis.argument
        for axis in measurement.scan.axes
        if np.asarray(measurement.points[axis.argument]).dtype.kind in "biuf"
    ]


def needed_results(measurement: Measurement) -> list[str]:
    """Every camera's recorded fit fields and their errors; no mean images."""
    names = []
    for camera in cloud_cameras(measurement):
        names += [f"{camera}_{field}" for field in FIELDS]
        names += [f"{camera}_{field}_error" for field in ERROR_FIELDS]
    return [name for name in names if name in measurement.result_shapes]


def pooled_field(
    measurement: Measurement, camera: str, field: str, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(axis values, mean, error)`` of the recorded ``<camera>_<field>`` against ``axis``.

    Points sharing a value of ``axis`` are pooled: the mean and standard error
    over those whose fit succeeded. A value with one such point keeps that
    point's recorded per-frame error (NaN when it had a single frame).
    """
    x = np.asarray(measurement.points[axis])
    y = np.asarray(measurement.results[f"{camera}_{field}"], dtype=float)
    recorded_error = np.asarray(
        measurement.results[f"{camera}_{field}_error"], dtype=float
    )
    empty = (x[:0], np.empty(0), np.empty(0), np.empty(0, dtype=int))
    xs, means, errors, counts = next(iter(aggregate(x, y).values()), empty)
    for i in np.flatnonzero(counts == 1):
        errors[i] = recorded_error[(x == xs[i]) & np.isfinite(y)][0]
    return xs, means, errors


def cloud_sizes(
    measurement: Measurement, camera: str, axis: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{"sigma_x": (axis values, mean, error), "sigma_y": ...}`` against ``axis``.

    Pooled as :func:`pooled_field` does.

    Args:
        measurement: Loaded with the camera's ``sigma_x``, ``sigma_y`` and
            their errors.
        camera: As :func:`cloud_cameras` gives.
        axis: A ``/points`` column, such as a scanned argument or ``"index"``.
    """
    return {field: pooled_field(measurement, camera, field, axis) for field in SIZES}


def point_image(
    measurement: Measurement, camera: str, index: int, mean_image: np.ndarray | None
) -> tuple[np.ndarray | None, CloudFit]:
    """Point ``index``'s mean image and its cloud fit.

    With an image, the fit is in the image's pixel coordinates (pixel
    ``(row, col)`` centred at ``x = col + 0.5``, ``y = row + 0.5``), so it
    overlays the image drawn row-major. The image is ``None`` when the file
    recorded none for this point; the fit is then the recorded one, in the
    file's units.

    Args:
        measurement: Loaded with the camera's fields.
        camera: As :func:`cloud_cameras` gives.
        index: Row of the point in ``measurement.points``.
        mean_image: The point's recorded ``<camera>_mean_image``, read on its
            own as it is large; ``None`` if the file has none.
    """
    image = None if mean_image is None else np.asarray(mean_image)
    if image is not None and np.isfinite(image).any():
        return image, fit_cloud(image)
    values = {
        field: float(measurement.results[f"{camera}_{field}"][index])
        for field in FIELDS
    }
    return None, CloudFit(
        **values,
        units=measurement.units.get(f"{camera}_x0", ""),
        success=bool(np.isfinite(values["sigma_x"])),
    )


@dataclass(frozen=True)
class TofFit:
    """Result of :func:`tof_fit`: ``sigma**2 = sigma0**2 + (k_B T / m) t**2``.

    ``slope`` and ``intercept`` are the line in ``(t**2, sigma**2)`` in SI;
    ``sigma0`` is NaN when the intercept is negative. When ``success`` is False
    ``message`` says why; otherwise it says how the errors were found.
    """

    temperature_K: float
    temperature_err_K: float
    sigma0: float
    sigma0_err: float
    success: bool
    message: str
    slope: float = np.nan
    intercept: float = np.nan


@dataclass(frozen=True)
class GravityFit:
    """Result of :func:`fit_gravity`: ``centre = offset + g t**2 / 2``."""

    g: float
    g_err: float
    offset: float
    success: bool
    message: str


@dataclass(frozen=True)
class _LineFit:
    intercept: float
    slope: float
    intercept_err: float
    slope_err: float
    description: str


def guess_time_axis(names: list[str]) -> str | None:
    """The first of ``names`` that looks like a free-fall time, or ``None``.

    Matches ``FreeExp``, ``FreeFall`` or ``Drop`` in any case.
    """
    for name in names:
        if any(hint in name.lower() for hint in TIME_ARGUMENT_HINTS):
            return name
    return None


def _line_fit(u: np.ndarray, v: np.ndarray, v_err: np.ndarray) -> _LineFit:
    """Least-squares line ``v = intercept + slope u`` and its covariance errors.

    Weighted by ``v_err`` when every error is finite and positive, with the
    covariance taken as absolute and then widened by ``sqrt(chi2 / dof)`` if the
    scatter exceeds what those errors allow (they are often few-shot standard
    errors, and an understated error is worse than an overstated one). Otherwise
    unweighted, with the covariance scaled to the residual scatter.
    """
    weighted = bool(np.all(np.isfinite(v_err) & (v_err > 0)))
    u_scale = np.max(np.abs(u))
    design = np.column_stack([np.ones_like(u), u / u_scale])
    if weighted:
        design, target = design / v_err[:, None], v / v_err
    else:
        target = v
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    dof = len(u) - 2
    chi2_per_dof = float(residual @ residual) / dof
    covariance = np.linalg.inv(design.T @ design)
    if weighted:
        factor = max(1.0, chi2_per_dof)
        description = "weighted by the recorded errors" + (
            f", errors widened by {np.sqrt(factor):.2g} to the scatter"
            if factor > 1.0 + 1e-9
            else ""
        )
    else:
        factor = chi2_per_dof
        description = "unweighted: no usable errors, errors from the scatter"
    errors = np.sqrt(np.diag(covariance) * factor)
    return _LineFit(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1] / u_scale),
        intercept_err=float(errors[0]),
        slope_err=float(errors[1] / u_scale),
        description=description,
    )


def _finite_rows(*columns: np.ndarray) -> np.ndarray:
    return np.all([np.isfinite(column) for column in columns], axis=0)


def _too_few_times(t: np.ndarray) -> str | None:
    distinct = len(np.unique(t))
    if distinct < MIN_TIMES:
        return f"Needs at least three distinct times; this file has {distinct}."
    return None


def _failed_fit(message: str) -> TofFit:
    return TofFit(np.nan, np.nan, np.nan, np.nan, False, message)


def tof_fit(
    t_s: np.ndarray, sigma: np.ndarray, sigma_err: np.ndarray, mass_amu: float
) -> TofFit:
    """Temperature of a freely expanding cloud from its size against time.

    Fits ``sigma**2 = sigma0**2 + (k_B T / m) t**2``. Each ``sigma**2`` carries
    ``2 sigma sigma_err``; the fit is weighted by those if all are finite and
    positive, and unweighted otherwise (see :func:`_line_fit` for the errors).

    Args:
        t_s: Free-fall times in seconds.
        sigma: Cloud sizes in metres.
        sigma_err: Their errors in metres; NaN where unknown.
        mass_amu: Mass of the species in atomic mass units.
    """
    t_s, sigma, sigma_err = (
        np.asarray(a, dtype=float) for a in (t_s, sigma, sigma_err)
    )
    keep = _finite_rows(t_s, sigma) & (sigma > 0)
    t_s, sigma, sigma_err = t_s[keep], sigma[keep], sigma_err[keep]
    if (message := _too_few_times(t_s)) is not None:
        return _failed_fit(message)
    line = _line_fit(t_s**2, sigma**2, 2 * sigma * sigma_err)
    if line.slope <= 0:
        return TofFit(
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            "The cloud does not grow with time: no temperature.",
            line.slope,
            line.intercept,
        )
    kelvin_per_slope = mass_amu * atomic_mass / k
    sigma0 = np.sqrt(line.intercept) if line.intercept > 0 else np.nan
    return TofFit(
        temperature_K=line.slope * kelvin_per_slope,
        temperature_err_K=line.slope_err * kelvin_per_slope,
        sigma0=float(sigma0),
        sigma0_err=float(line.intercept_err / (2 * sigma0)),
        success=True,
        message=line.description,
        slope=line.slope,
        intercept=line.intercept,
    )


def fit_gravity(
    t_s: np.ndarray, centre: np.ndarray, centre_err: np.ndarray
) -> GravityFit:
    """``g`` from the cloud centre falling as ``centre = offset + g t**2 / 2``.

    Weighted and with errors as :func:`tof_fit`. The sign is that of the
    centre's coordinate: positive when the cloud moves towards larger values.
    """
    t_s, centre, centre_err = (
        np.asarray(a, dtype=float) for a in (t_s, centre, centre_err)
    )
    keep = _finite_rows(t_s, centre)
    t_s, centre, centre_err = t_s[keep], centre[keep], centre_err[keep]
    if (message := _too_few_times(t_s)) is not None:
        return GravityFit(np.nan, np.nan, np.nan, False, message)
    line = _line_fit(t_s**2, centre, centre_err)
    return GravityFit(
        2 * line.slope, 2 * line.slope_err, line.intercept, True, line.description
    )


def cloud_temperature(
    measurement: Measurement, camera: str, axis: str, seconds_per_unit: float
) -> dict:
    """Time-of-flight temperature of a camera's cloud against scanned argument ``axis``.

    The values of ``axis`` times ``seconds_per_unit`` are the free-fall times.
    Sizes and centres are pooled as :func:`pooled_field` does. The mass is the
    file's constant ``cloud_mass_amu``. Sizes recorded in pixels give failed
    fits: a temperature needs metres.

    Args:
        measurement: Loaded with :func:`needed_results` and the constant.
        camera: As :func:`cloud_cameras` gives.
        axis: A scanned argument holding the free-fall time.
        seconds_per_unit: Seconds in one unit of the recorded values of ``axis``.

    Returns:
        A dict. ``"unit"`` is the length unit recorded. ``"fits"`` maps
        ``"sigma_x"`` and ``"sigma_y"`` to a :class:`TofFit`. ``"sigma2"`` maps
        each size field to ``(t**2, sigma**2, error)`` in SI. ``"centres"`` maps
        ``"x0"`` and ``"y0"`` to ``(t**2, mean, error)``. ``"gravity"`` is the
        :class:`GravityFit` of ``y0``.
    """
    unit = measurement.units.get(f"{camera}_sigma_x", "")
    mass_amu = float(measurement.constants["cloud_mass_amu"])
    sigma2, fits = {}, {}
    for field in SIZES:
        x, sigma, error = pooled_field(measurement, camera, field, axis)
        t = x * seconds_per_unit
        sigma2[field] = (t**2, sigma**2, 2 * sigma * error)
        fits[field] = (
            tof_fit(t, sigma, error, mass_amu)
            if unit == "m"
            else _failed_fit(
                "Sizes were recorded in pixels (the camera has no pixel scale): "
                "no temperature."
            )
        )
    centres = {}
    for field in CENTRES:
        x, mean, error = pooled_field(measurement, camera, field, axis)
        centres[field] = ((x * seconds_per_unit) ** 2, mean, error)
    t2, y0, y0_err = centres["y0"]
    gravity = (
        fit_gravity(np.sqrt(t2), y0, y0_err)
        if unit == "m"
        else GravityFit(
            np.nan, np.nan, np.nan, False, "Centres were recorded in pixels."
        )
    )
    return {
        "unit": unit,
        "fits": fits,
        "sigma2": sigma2,
        "centres": centres,
        "gravity": gravity,
    }


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


ANALYSES = {
    "cloud_fit": AnalysisEntry("MOT Gaussian fit", CloudFitView),
    "cloud_temperature": AnalysisEntry(
        "MOT temperature (time of flight)", CloudTemperatureView
    ),
}
