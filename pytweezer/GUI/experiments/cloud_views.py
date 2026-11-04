"""Results-tab views of the MOT cloud fits a measurement recorded."""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from pytweezer.analysis.cloud_fit import contour_ellipses
from pytweezer.analysis.cloud_results import (
    SIZES,
    cloud_cameras,
    cloud_sizes,
    needed_results,
    point_image,
    scanned_arguments,
)
from pytweezer.experiment.summary import display_value
from pytweezer.GUI.experiments.analyses import ComputedAnalysisView
from pytweezer.GUI.theme import CURVE_COLORS, PLOT_BACKGROUND, PLOT_FOREGROUND

POINT_INDEX = "index"
OUTLINE_PEN = pg.mkPen("w", width=1)
SIZE_LABELS = {"sigma_x": "σx", "sigma_y": "σy"}


def themed_plot() -> pg.PlotWidget:
    plot = pg.PlotWidget(background=PLOT_BACKGROUND)
    for axis in ("left", "bottom"):
        plot.getAxis(axis).setPen(PLOT_FOREGROUND)
        plot.getAxis(axis).setTextPen(PLOT_FOREGROUND)
    return plot


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
                pen=colour,
                symbol="o",
                symbolBrush=colour,
                name=SIZE_LABELS[field],
            )
            self.size_errors[field] = pg.ErrorBarItem(
                x=np.empty(0), y=np.empty(0), height=np.empty(0), pen=colour
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
        self._refill(
            self.camera, [(camera, camera) for camera in cloud_cameras(measurement)]
        )
        self._refill(
            self.against,
            [(axis, axis) for axis in scanned_arguments(measurement)]
            + [("point index", POINT_INDEX)],
        )
        self._refill(self.point, self._point_labels(measurement), keep=False)
        super().set_measurement(archive, path, measurement)

    @staticmethod
    def _refill(combo, items, keep=True):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        index = combo.findData(current) if keep else -1
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

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
        image, fit = point_image(loaded, camera, row)
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
