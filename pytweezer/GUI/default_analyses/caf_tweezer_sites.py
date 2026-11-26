"""The Results-tab view of the light at the CaF tweezer sites, and its ``ANALYSES`` entry.

The calculation is :mod:`pytweezer.analyses.default.caf_tweezer_sites`.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from pytweezer.analyses.default.caf_tweezer_sites import (
    BRIGHT_Z,
    SMOOTHING_PX,
    SiteAnalysis,
    SiteSettings,
    caf_tweezer_sites,
    missing,
    needed_results,
)
from pytweezer.analyses.default.tweezer_histogram import point_groups
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.default_analyses.cloud import themed_plot
from pytweezer.GUI.experiments.analyses import AnalysisEntry, ComputedAnalysisView
from pytweezer.GUI.theme import CURVE_COLORS, PLOT_LINE_WIDTH, PLOT_TEXT

APERTURE_PEN = pg.mkPen("#e74c3c", width=1.5)
RING_PEN = pg.mkPen(231, 76, 60, 110, width=1, style=Qt.PenStyle.DashLine)
TRAP_PEN = pg.mkPen("#5bd1ef", width=1.5)
EMPTY_BRUSH = pg.mkBrush(141, 142, 153, 90)
CIRCLE_POINTS = 64
COLUMNS = ("Site", "x (px)", "y (px)", "Counts", "Score z", f"Shots z > {BRIGHT_Z:g}")


def circle(x: float, y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
    return x + radius * np.cos(angle), y + radius * np.sin(angle)


def mean_and_error(values: np.ndarray, decimals: int = 2) -> str:
    error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return f"{values.mean():.{decimals}f} ± {error:.{decimals}f}"


def image_plot(
    x_label: str, y_label: str, unit: str
) -> tuple[pg.PlotWidget, pg.ImageItem]:
    plot = themed_plot()
    item = plot.getPlotItem()
    item.setLabel("bottom", x_label, units=unit)
    item.setLabel("left", y_label, units=unit)
    for axis in ("left", "bottom"):
        item.getAxis(axis).enableAutoSIPrefix(False)
    item.invertY(True)
    item.setAspectLocked(True)
    image = pg.ImageItem(axisOrder="row-major")
    item.addItem(image)
    return plot, image


class CafTweezerSitesView(ComputedAnalysisView):
    """The simulated array, the mean HamCam frame with its sites, and each site's light shot by shot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        defaults = SiteSettings()
        self.shots_from = QComboBox()
        self.shots_from.setToolTip(
            "Every point, or one scan value with its repetitions pooled"
        )
        self.aperture = self._radius(
            defaults.aperture, "Radius of the disc summed at each site"
        )
        self.ring_inner = self._radius(
            defaults.ring_inner, "Inner radius of the background ring"
        )
        self.ring_outer = self._radius(
            defaults.ring_outer, "Outer radius of the background ring"
        )

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.shots_from.setFixedWidth(240)
        for text, widget in (
            ("Shots from", self.shots_from),
            ("Aperture", self.aperture),
            ("Ring from", self.ring_inner),
            ("to", self.ring_outer),
        ):
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(QLabel(text))
            controls.addWidget(widget)
        self.smooth = QCheckBox(f"Smooth image ({SMOOTHING_PX:g} px)")
        self.smooth.setChecked(True)
        self.smooth.setToolTip(
            "Draw the mean frame smoothed so faint sites show; the scores use the raw frames"
        )
        controls.addSpacing(12)
        controls.addWidget(self.smooth)
        controls.addStretch(1)
        self.shots_from.currentIndexChanged.connect(self.recompute)
        self.smooth.toggled.connect(self._redraw_frame)
        self._result = None

        self.trap_plot, self.trap_image = image_plot("x", "y", "µm")
        self.trap_image.setColorMap(pg.colormap.get("magma"))
        self.trap_plot.setTitle("Simulated trap plane")
        self.frame_plot, self.frame_image = image_plot("x", "y", "px")
        self.frame_image.setColorMap(pg.colormap.get("viridis"))
        self.score_plot = themed_plot()
        self.score_plot.setTitle("Score per shot")
        self.score_plot.setLabel("bottom", "z")
        self.score_plot.setLabel("left", "shots per site")
        self.score_plot.addLegend(offset=(-10, 10))
        self._overlays = []

        plots = QHBoxLayout()
        plots.setSpacing(8)
        for plot in (self.trap_plot, self.frame_plot, self.score_plot):
            plots.addWidget(plot, 1)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setSizeAdjustPolicy(
            QAbstractItemView.SizeAdjustPolicy.AdjustToContents
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(plots, 1)
        layout.addWidget(self.summary)
        table_row = QHBoxLayout()
        table_row.addWidget(self.table)
        table_row.addStretch(1)
        layout.addLayout(table_row)
        layout.addWidget(self.message)

    def _radius(self, value: float, tooltip: str) -> CompactDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(0.5, 100)
        spin.setValue(value)
        spin.setSuffix(" px")
        spin.setToolTip(tooltip)
        spin.setFixedWidth(110)
        spin.valueChanged.connect(self.recompute_soon)
        return spin

    def set_measurement(self, archive, path, measurement):
        self.shots_from.blockSignals(True)
        self.shots_from.clear()
        self.shots_from.addItem("all points")
        if missing(measurement) is None:
            self.shots_from.addItems([label for label, _ in point_groups(measurement)])
        self.shots_from.blockSignals(False)
        super().set_measurement(archive, path, measurement)

    def unavailable_reason(self, measurement):
        return missing(measurement)

    def needed_results(self, measurement):
        return needed_results(measurement)

    def settings(self):
        group = self.shots_from.currentIndex() - 1
        return SiteSettings(
            group=None if group < 0 else group,
            aperture=self.aperture.value(),
            ring_inner=self.ring_inner.value(),
            ring_outer=self.ring_outer.value(),
        )

    def compute(self, loaded, settings):
        return caf_tweezer_sites(loaded, settings), settings

    def show_result(self, result):
        self._result = result
        for plot, item in self._overlays:
            plot.removeItem(item)
        self._overlays = []
        self.score_plot.clear()
        if result is None:
            self.trap_image.clear()
            self.frame_image.clear()
            self.summary.setText("")
            self.table.setRowCount(0)
            return
        analysis, settings = result
        self._show_trap_plane(analysis)
        self._show_frame(analysis, settings)
        self._show_scores(analysis)
        self._show_summary(analysis)

    def _overlay(self, plot, item):
        plot.addItem(item)
        self._overlays.append((plot, item))

    def _show_trap_plane(self, a: SiteAnalysis) -> None:
        x0, x1, y0, y1 = a.trap_plane_extent
        self.trap_image.setImage(a.trap_plane, levels=(0, 1))
        self.trap_image.setRect(pg.QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        for k, (x, y) in enumerate(a.traps_um):
            self._overlay(
                self.trap_plot,
                pg.ScatterPlotItem(
                    [x], [y], symbol="+", size=14, pen=TRAP_PEN, brush=None
                ),
            )
            self._label(self.trap_plot, k, x, y)

    def _redraw_frame(self) -> None:
        if self._result is not None:
            self._draw_frame_image(self._result[0])

    def _draw_frame_image(self, a: SiteAnalysis) -> None:
        image = a.smoothed_image if self.smooth.isChecked() else a.mean_image
        low, high = np.percentile(image, [1, 99.7])
        self.frame_image.setImage(image, levels=(low, high))

    def _show_frame(self, a: SiteAnalysis, s: SiteSettings) -> None:
        self._draw_frame_image(a)
        frames = "image − background" if a.background_subtracted else "image"
        self.frame_plot.setTitle(f"Mean {frames}, {a.shots} shots")
        for k, (y, x) in enumerate(a.sites):
            # Pixel (row, col) is drawn centred at (col + 0.5, row + 0.5).
            cx, cy = x + 0.5, y + 0.5
            for radius, pen in (
                (s.aperture, APERTURE_PEN),
                (s.ring_inner, RING_PEN),
                (s.ring_outer, RING_PEN),
            ):
                self._overlay(
                    self.frame_plot, pg.PlotCurveItem(*circle(cx, cy, radius), pen=pen)
                )
            self._label(self.frame_plot, k, cx + s.aperture, cy - s.aperture)

    def _label(self, plot, k, x, y):
        text = pg.TextItem(str(k), color=PLOT_TEXT, anchor=(0, 1))
        text.setPos(x, y)
        self._overlay(plot, text)

    def _show_scores(self, a: SiteAnalysis) -> None:
        top = max(6.0, float(np.ceil(a.z.max())) + 1) if a.z.size else 6.0
        edges = np.arange(-6.0, top + 0.5, 0.5)
        empty, _ = np.histogram(a.empty_z, edges)
        self.score_plot.plot(
            edges,
            empty * a.shots / max(len(a.empty_z), 1),
            stepMode="center",
            fillLevel=0,
            brush=EMPTY_BRUSH,
            pen=None,
            name="empty positions",
        )
        for k in range(a.z.shape[1]):
            counts, _ = np.histogram(a.z[:, k], edges)
            colour = CURVE_COLORS[k % len(CURVE_COLORS)]
            self.score_plot.plot(
                edges,
                counts,
                stepMode="center",
                pen=pg.mkPen(colour, width=PLOT_LINE_WIDTH),
                name=f"site {k}",
            )
        self.score_plot.addItem(pg.InfiniteLine(BRIGHT_Z, pen=RING_PEN))

    def _show_summary(self, a: SiteAnalysis) -> None:
        combined = a.combined
        observed = combined.mean()
        n_sets = len(a.combined_chance)
        if n_sets:
            reached = int((a.combined_chance >= observed).sum())
            chance = (
                f"{reached} of {n_sets} random sets of {len(a.sites)} empty positions "
                f"reach it; they scatter by {a.combined_chance.std():.2f}"
            )
        else:
            chance = "too few empty positions to judge it"
        self.summary.setText(
            f"All sites together: z = {mean_and_error(combined)} per shot ({chance}).\n"
            f"Frame median {mean_and_error(a.offset, 0)} counts. By chance a site has "
            f"{a.bright_by_chance:.2f} of {a.shots} shots above z = {BRIGHT_Z:g}."
        )
        self.table.setRowCount(len(a.sites))
        for k, (y, x) in enumerate(a.sites):
            cells = (
                str(k),
                f"{x:.1f}",
                f"{y:.1f}",
                mean_and_error(a.counts[:, k], 0),
                mean_and_error(a.z[:, k]),
                str(int((a.z[:, k] > BRIGHT_Z).sum())),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(k, column, item)


ANALYSES = {
    "caf_tweezer_sites": AnalysisEntry(
        "Light at the tweezer sites", CafTweezerSitesView
    ),
}
