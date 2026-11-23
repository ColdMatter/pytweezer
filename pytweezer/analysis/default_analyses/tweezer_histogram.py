"""The tweezer loading histogram of a measurement file: ``tweezer_histogram`` (counts) and ``tweezer_photon_histogram``.

The per-file counterpart of the live ``tweezer_loading`` analysis, giving the
same ``(header, values)`` pair from what a file recorded, in the header's
``unit``:

* ``first_counts``/``second_counts`` per point, each site's summed raw pixel
  values, with the ``trap_sites`` and ``threshold`` (photons) constants. These
  are histogrammed in counts, or converted with
  ``analysis.convert_counts_to_photons`` when ``unit`` is ``"photons"``;
* ``first_photons``/``second_photons`` per point (files from before raw
  counts), always in photons; or
* the ``hamcam_images`` stack per point, ``(2, H, W)`` (``LoadTweezers``
  without trap sites), always in photons. The sites are found in the mean of
  every top-hat filtered first frame in the file.

Whichever image is histogrammed, the header also gives the imaging survival
from the first image to the second, with both scored against the first
image's threshold.

:class:`TweezerLoadingPlots` draws the pair, as the live ``tweezer_loading``
viewer does too: what ``analysis.get_array_loading_statistics(show_histogram=True)``
plots in the optimisation notebooks. The photon histogram of each site, the
histogram of all sites with its two-Gaussian fit and occupancy threshold, and
each site's loading probability at its place in the image, with a line
beneath giving the mean loading, its spread over sites, the fidelity and the
fit. The loading map's colour range can be dragged on its colour bar.
:class:`TweezerHistogramView` is the Results-tab view around it.
"""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import (
    find_sites,
    imaging_survival,
    loading_statistics,
    photons_at_sites,
)
from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import display_value
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.experiments.analyses import AnalysisEntry, ComputedAnalysisView
from pytweezer.GUI.theme import CURVE_COLORS, plot_title_style, style_plot

COUNTS = ("first_counts", "second_counts")
PHOTONS = ("first_photons", "second_photons")
FRAMES = "hamcam_images"
RESULTS = {"counts": COUNTS, "photons": PHOTONS, "frames": (FRAMES,)}


@dataclass(frozen=True)
class HistogramSettings:
    """What to histogram, and how to score files that recorded only frames.

    Attributes:
        image: 0 for the first image of each shot, 1 for the second.
        unit: ``"counts"`` or ``"photons"``; a file of recorded counts is
            histogrammed in this unit, any other file in photons.
        group: Index into :func:`point_groups`; ``None`` for every point.
        sites: Trap sites to find (frames only).
        feature_size: Top-hat filter size in pixels (frames only).
        window_size: Side of the box summed at each site in pixels (frames only).
        detection_step: Step in counts by which the site-finding level is
            lowered (frames only).
        fallback_threshold: Occupancy threshold in photons when the fit fails
            and the file recorded none.
    """

    image: int = 0
    unit: str = "counts"
    group: int | None = None
    sites: int = 64
    feature_size: int = 10
    window_size: int = 5
    detection_step: float = 100.0
    fallback_threshold: float = 100.0


def source(measurement: Measurement) -> str | None:
    """``"counts"``, ``"photons"`` or ``"frames"``, by what the file recorded; ``None`` if none."""
    for kind, names in RESULTS.items():
        if all(name in measurement.result_shapes for name in names):
            return kind
    return None


def needed_results(measurement: Measurement) -> list[str]:
    """The results :func:`tweezer_histogram` reads from this file."""
    return list(RESULTS.get(source(measurement), ()))


def point_groups(measurement: Measurement) -> list[tuple[str, np.ndarray]]:
    """``(label, mask over the points run)`` per distinct value of the scan axes.

    Repetitions of a value pool into one group. Without scan axes each point
    is its own group.
    """
    points = measurement.points
    axes = [axis.argument for axis in measurement.scan.axes]
    if not axes:
        indices = points["index"]
        return [(f"point {index}", indices == index) for index in indices]
    keys = list(zip(*(points[axis].tolist() for axis in axes), strict=True))
    groups = []
    for key in dict.fromkeys(keys):
        label = ", ".join(
            f"{axis} = {display_value(measurement, axis, value)}"
            for axis, value in zip(axes, key, strict=True)
        )
        groups.append((label, np.array([k == key for k in keys], dtype=bool)))
    return groups


def tweezer_histogram(
    measurement: Measurement, settings: HistogramSettings
) -> tuple[dict, np.ndarray | None]:
    """The live analysis's header and ``(shots, sites)`` values for the selected shots.

    When there are no shots or the sites cannot be found, the values are
    ``None`` and the header's ``"error"`` says why.

    Raises:
        KeyError: If the file recorded no counts, photons or frames, or
            ``measurement`` was loaded without them.
    """
    kind = source(measurement)
    if kind is None:
        raise KeyError("this file recorded no tweezer images, counts or photons")
    if settings.group is None:
        mask = np.ones(len(measurement.points["index"]), dtype=bool)
    else:
        mask = point_groups(measurement)[settings.group][1]
    unit = settings.unit if kind == "counts" else "photons"
    head = {"shots": 0, "sites": settings.sites, "source": kind, "unit": unit}

    if kind in ("counts", "photons"):
        images = [
            np.asarray(measurement.results[name], float)[mask].reshape(
                -1, len(measurement.constants["trap_sites"])
            )
            for name in RESULTS[kind]
        ]
        sites = {
            i: (int(row), int(col))
            for i, (row, col) in enumerate(measurement.constants["trap_sites"])
        }
        fallback_threshold = float(
            measurement.constants.get("threshold", settings.fallback_threshold)
        )
        if kind == "counts" and unit == "photons":
            images = [an.convert_counts_to_photons(image) for image in images]
        elif kind == "counts":
            fallback_threshold = float(an.convert_photons_to_counts(fallback_threshold))
    else:
        if not mask.any():
            return head | {"error": "No shots in this selection"}, None
        stack = measurement.results[FRAMES]
        firsts = _filtered(stack[:, 0], settings)
        sites = find_sites(
            np.mean(firsts, axis=0), settings.sites, settings.detection_step
        )
        if sites is None:
            return head | {
                "error": f"Found no {settings.sites} trap sites in the mean first image"
            }, None
        images = [
            photons_at_sites(
                [frame for frame, keep in zip(frames, mask, strict=True) if keep],
                sites,
                settings.window_size,
            )
            for frames in (firsts, _filtered(stack[:, 1], settings))
        ]
        fallback_threshold = settings.fallback_threshold

    photons = images[settings.image]
    head |= {"shots": len(photons), "sites": len(sites)}
    if not len(photons):
        return head | {"error": "No shots in this selection"}, None
    head |= loading_statistics(photons, sites, fallback_threshold)
    first_threshold = (
        head["threshold"]
        if settings.image == 0
        else loading_statistics(images[0], sites, fallback_threshold)["threshold"]
    )
    return head | imaging_survival(*images, first_threshold), photons


def _filtered(frames, settings):
    return [
        an.morphological_tophat_high_pass(
            np.asarray(frame, dtype=np.float64), feature_size=settings.feature_size
        )
        for frame in frames
    ]


TEXT_COLOUR = "#e6e6e6"
DATA_COLOUR = CURVE_COLORS[7]
BACKGROUND_COLOUR = CURVE_COLORS[0]
SIGNAL_COLOUR = CURVE_COLORS[1]
THRESHOLD_COLOUR = CURVE_COLORS[2]


def _translucent(hexcode, alpha):
    colour = QColor(hexcode)
    colour.setAlpha(alpha)
    return colour


def _gaussian(x, mean, variance, weight):
    return (
        weight
        * np.exp(-((x - mean) ** 2) / (2 * variance))
        / np.sqrt(2 * np.pi * variance)
    )


class TweezerLoadingPlots(QtWidgets.QWidget):
    """Per-site and all-site photon histograms beside the array's loading map.

    Args:
        bins: Bins of the all-sites histogram.
        site_bins: Bins of each site's histogram.
        show_site_labels: Number the sites on the loading map.

    The three settings are plain attributes, read on the next draw.
    """

    def __init__(self, parent=None, *, bins=60, site_bins=40, show_site_labels=False):
        super().__init__(parent)
        self.bins = bins
        self.site_bins = site_bins
        self.show_site_labels = show_site_labels
        self.site_curves = []
        self.site_labels = []
        self.head = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        win = pg.GraphicsLayoutWidget()
        layout.addWidget(win)

        self.site_plot = win.addPlot()
        self.total_plot = win.addPlot()
        self.map_plot = win.addPlot()
        self.plot_items = (self.site_plot, self.total_plot, self.map_plot)
        for plot, title in zip(
            self.plot_items, ("Each site", "All sites", "Loading probability")
        ):
            plot.setTitle(title, **plot_title_style())
            style_plot(plot)
        for plot in (self.site_plot, self.total_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("bottom", "photons")
            plot.setLabel("left", "probability density")
            plot.getAxis("left").enableAutoSIPrefix(False)

        self.total_plot.addLegend(offset=(-10, 10))
        self.total_data = self.total_plot.plot(
            stepMode="center",
            fillLevel=0,
            brush=_translucent(DATA_COLOUR, 70),
            pen=pg.mkPen(DATA_COLOUR, width=1),
            name="data",
        )
        self.total_fit = self.total_plot.plot(
            pen=pg.mkPen(TEXT_COLOUR, width=2), name="fit"
        )
        self.background_fit = self.total_plot.plot(
            pen=pg.mkPen(BACKGROUND_COLOUR, width=2, style=Qt.PenStyle.DashLine),
            name="background",
        )
        self.signal_fit = self.total_plot.plot(
            pen=pg.mkPen(SIGNAL_COLOUR, width=2, style=Qt.PenStyle.DashLine),
            name="signal",
        )
        self.threshold_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(THRESHOLD_COLOUR, width=2, style=Qt.PenStyle.DashLine),
            label="threshold {value:.1f}",
            labelOpts={"position": 0.92, "color": THRESHOLD_COLOUR},
        )
        self.threshold_line.hide()
        self.total_plot.addItem(self.threshold_line)

        self.map_plot.invertY(True)
        self.map_plot.setAspectLocked(True)
        self.map_plot.setLabel("bottom", "column / px")
        self.map_plot.setLabel("left", "row / px")
        self.site_map = pg.ScatterPlotItem(pxMode=False, pen=None)
        self.map_plot.addItem(self.site_map)
        self.colour_map = pg.colormap.get("viridis")
        self.colour_bar = pg.ColorBarItem(
            values=(0, 1), colorMap=self.colour_map, label="loading probability"
        )
        self.colour_bar.sigLevelsChanged.connect(self._draw_map)
        win.addItem(self.colour_bar)

        # Monospaced so the digits hold their columns as the values change.
        self.readout = QtWidgets.QLabel()
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.readout.setFont(font)
        self.readout.setContentsMargins(10, 6, 10, 8)
        layout.addWidget(self.readout)

    def show_result(self, head, photons):
        """Draw one result; without photons, clear the plots and say why."""
        loading = head.get("site_loading")
        if (
            loading is None
            or photons is None
            or photons.ndim != 2
            or photons.shape[1] != len(loading)
            or photons.size == 0
        ):
            self.head = None
            self._clear_plots()
            self.readout.setText(head.get("error") or "No photons yet")
            return
        self.head = head
        for plot in (self.site_plot, self.total_plot):
            plot.setLabel("bottom", head.get("unit", "photons"))
        self._draw_histograms(head, photons)
        self._draw_map()
        self.readout.setText(self._summary(head))

    def show_message(self, text):
        """Clear the plots and show ``text`` beneath them."""
        self.head = None
        self._clear_plots()
        self.readout.setText(text)

    def _draw_histograms(self, head, photons):
        low, high = float(photons.min()), float(photons.max())
        if high <= low:
            high = low + 1.0

        n_sites = photons.shape[1]
        if len(self.site_curves) != n_sites:
            self.site_plot.clear()
            self.site_curves = [
                self.site_plot.plot(
                    stepMode="center",
                    pen=pg.mkPen(
                        _translucent(CURVE_COLORS[i % len(CURVE_COLORS)], 150), width=1
                    ),
                )
                for i in range(n_sites)
            ]
        site_bins = max(2, int(self.site_bins))
        for curve, site_photons in zip(self.site_curves, photons.T, strict=True):
            density, edges = np.histogram(
                site_photons, bins=site_bins, range=(low, high), density=True
            )
            curve.setData(edges, density)

        density, edges = np.histogram(
            photons.ravel(),
            bins=max(2, int(self.bins)),
            range=(low, high),
            density=True,
        )
        self.total_data.setData(edges, density)

        background, signal = head.get("background"), head.get("signal")
        if background and signal:
            x = np.linspace(low, high, 500)
            background_pdf, signal_pdf = (
                _gaussian(x, *background),
                _gaussian(x, *signal),
            )
            self.background_fit.setData(x, background_pdf)
            self.signal_fit.setData(x, signal_pdf)
            self.total_fit.setData(x, background_pdf + signal_pdf)
        else:
            for curve in (self.background_fit, self.signal_fit, self.total_fit):
                curve.setData([], [])

        # The line's label only updates while the line is visible.
        self.threshold_line.show()
        self.threshold_line.setValue(head["threshold"])

    def _draw_map(self):
        if self.head is None:
            return
        head = self.head
        rows = np.asarray(head["site_rows"], dtype=float)
        cols = np.asarray(head["site_cols"], dtype=float)
        loading = np.asarray(head["site_loading"], dtype=float)

        low, high = self.colour_bar.levels()
        scaled = np.clip((loading - low) / max(high - low, 1e-9), 0, 1)
        brushes = [
            pg.mkBrush(colour) for colour in self.colour_map.map(scaled, mode="qcolor")
        ]
        self.site_map.setData(
            x=cols, y=rows, size=self._spot_size(rows, cols), brush=brushes
        )

        if bool(self.show_site_labels):
            if len(self.site_labels) != len(loading):
                self._clear_map_labels()
                for i in range(len(loading)):
                    label = pg.TextItem(str(i), color=TEXT_COLOUR, anchor=(0.5, 0.5))
                    self.map_plot.addItem(label)
                    self.site_labels.append(label)
            for label, row, col in zip(self.site_labels, rows, cols, strict=True):
                label.setPos(col, row)
        else:
            self._clear_map_labels()

    @staticmethod
    def _spot_size(rows, cols):
        """80 % of the median nearest-neighbour spacing, so neighbouring sites don't overlap."""
        if len(rows) < 2:
            return 5.0
        separations = np.hypot(rows[:, None] - rows, cols[:, None] - cols)
        np.fill_diagonal(separations, np.inf)
        return 0.8 * float(np.median(separations.min(axis=1)))

    def _clear_map_labels(self):
        for label in self.site_labels:
            self.map_plot.removeItem(label)
        self.site_labels = []

    def _clear_plots(self):
        for curve in self.site_curves:
            curve.setData([], [])
        for curve in (
            self.total_data,
            self.background_fit,
            self.signal_fit,
            self.total_fit,
        ):
            curve.setData([], [])
        self.threshold_line.hide()
        self.site_map.setData([], [])
        self._clear_map_labels()

    @staticmethod
    def _task(head):
        rid = head.get("rid")
        return f"task {rid}" if rid is not None and rid >= 0 else ""

    def _summary(self, head):
        def percent(value, digits=1):
            return f"{value * 100:.{digits}f} %" if value is not None else "--"

        unit = head.get("unit", "photons")
        parts = [
            f"loading {percent(head['loading'])}",
            f"site spread {percent(head.get('loading_spread'))}",
            f"fidelity {percent(head.get('fidelity'), 3)}",
            f"threshold {head['threshold']:.1f} {unit}",
        ]
        if "survival" in head:
            survival = head["survival"]
            parts.insert(
                1,
                f"imaging survival {percent(survival)} ± {percent(head['survival_error'])}"
                if survival is not None
                else "imaging survival --",
            )
        if head.get("background") and head.get("signal"):
            parts.append(
                f"empty {head['background'][0]:.1f} ± {np.sqrt(head['background'][1]):.1f} / full {head['signal'][0]:.1f} ± {np.sqrt(head['signal'][1]):.1f} {unit}"
            )
        else:
            parts.append("fit failed: fixed threshold")
        parts.append(f"{head['shots']} shots, {head['sites']} sites")
        if head.get("sites_match") is False:
            parts.append("sites fitted to another array than the rig's: refit them")
        if task := self._task(head):
            parts.insert(0, task)
        return "     ".join(parts)


class TweezerHistogramView(ComputedAnalysisView):
    """The tweezer loading histogram of a file, as the live viewer draws it, in ``unit``."""

    def __init__(self, unit="counts", parent=None):
        super().__init__(parent)
        self.unit = unit

        defaults = HistogramSettings()
        self.image = QComboBox()
        self.image.addItems(["first", "second"])
        self.image.setToolTip("Which image of each shot to histogram")
        self.shots_from = QComboBox()
        self.shots_from.setToolTip(
            "Every point, or one scan value with its repetitions pooled"
        )
        self.sites = QSpinBox()
        self.sites.setRange(1, 10000)
        self.sites.setValue(defaults.sites)
        self.sites.setToolTip("Trap sites to find in the mean first image")
        self.feature_size = QSpinBox()
        self.feature_size.setRange(1, 1000)
        self.feature_size.setValue(defaults.feature_size)
        self.feature_size.setSuffix(" px")
        self.feature_size.setToolTip("Top-hat filter size")
        self.window_size = QSpinBox()
        self.window_size.setRange(1, 1000)
        self.window_size.setValue(defaults.window_size)
        self.window_size.setSuffix(" px")
        self.window_size.setToolTip("Side of the box summed at each site")
        self.detection_step = CompactDoubleSpinBox()
        self.detection_step.setRange(1e-3, 1e9)
        self.detection_step.setValue(defaults.detection_step)
        self.detection_step.setToolTip(
            "Step in counts by which the site-finding level is lowered"
        )

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._frames_only = []
        for text, widget, frames_only in (
            ("Image", self.image, False),
            ("Shots from", self.shots_from, False),
            ("Sites", self.sites, True),
            ("Feature size", self.feature_size, True),
            ("Window", self.window_size, True),
            ("Detection step", self.detection_step, True),
        ):
            label = QLabel(text)
            widget.setFixedWidth(160 if isinstance(widget, QComboBox) else 110)
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(label)
            controls.addWidget(widget)
            if frames_only:
                self._frames_only += [label, widget]
        controls.addStretch(1)
        self.image.currentIndexChanged.connect(self.recompute)
        self.shots_from.currentIndexChanged.connect(self.recompute)
        for spin in (self.sites, self.feature_size, self.window_size):
            spin.valueChanged.connect(self.recompute_soon)
        self.detection_step.valueChanged.connect(self.recompute_soon)

        self.plots = TweezerLoadingPlots()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.plots, 1)
        layout.addWidget(self.message)

    def set_measurement(self, archive, path, measurement):
        kind = source(measurement)
        for widget in self._frames_only:
            widget.setVisible(kind == "frames")
        self.shots_from.blockSignals(True)
        self.shots_from.clear()
        self.shots_from.addItem("all points")
        self.shots_from.addItems([label for label, _ in point_groups(measurement)])
        self.shots_from.blockSignals(False)
        super().set_measurement(archive, path, measurement)

    def unavailable_reason(self, measurement):
        if source(measurement) is None:
            return "This file recorded no tweezer images, counts or photons."
        return None

    def needed_results(self, measurement):
        return needed_results(measurement)

    def settings(self):
        group = self.shots_from.currentIndex() - 1
        return HistogramSettings(
            image=self.image.currentIndex(),
            unit=self.unit,
            group=None if group < 0 else group,
            sites=self.sites.value(),
            feature_size=self.feature_size.value(),
            window_size=self.window_size.value(),
            detection_step=self.detection_step.value(),
        )

    def compute(self, loaded, settings):
        return tweezer_histogram(loaded, settings)

    def show_result(self, result):
        if result is None:
            self.plots.show_message("")
        else:
            self.plots.show_result(*result)


ANALYSES = {
    "tweezer_histogram": AnalysisEntry(
        "Tweezer loading histogram (counts)",
        lambda: TweezerHistogramView(unit="counts"),
    ),
    "tweezer_photon_histogram": AnalysisEntry(
        "Tweezer loading histogram (photons)",
        lambda: TweezerHistogramView(unit="photons"),
    ),
}
