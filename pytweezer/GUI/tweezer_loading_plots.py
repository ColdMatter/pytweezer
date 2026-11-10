"""Loading histograms and loading map of the tweezer array, from one result.

Draws the ``(header, photons)`` pair that the live ``tweezer_loading``
analysis publishes and :func:`pytweezer.analysis.tweezer_histogram.tweezer_histogram`
returns for a file: what ``analysis.get_array_loading_statistics(show_histogram=True)``
plots in the optimisation notebooks. The photon histogram of each site, the
histogram of all sites with its two-Gaussian fit and occupancy threshold, and
each site's loading probability at its place in the image, with a line
beneath giving the mean loading, its spread over sites, the fidelity and the
fit. The loading map's colour range can be dragged on its colour bar.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from pytweezer.GUI.theme import CURVE_COLORS, plot_title_style, style_plot

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
        if head.get("background") and head.get("signal"):
            parts.append(
                f"empty {head['background'][0]:.1f} / full {head['signal'][0]:.1f} {unit}"
            )
        else:
            parts.append("fit failed: fixed threshold")
        parts.append(f"{head['shots']} shots, {head['sites']} sites")
        if head.get("sites_match") is False:
            parts.append("sites fitted to another array than the rig's: refit them")
        if task := self._task(head):
            parts.insert(0, task)
        return "     ".join(parts)
