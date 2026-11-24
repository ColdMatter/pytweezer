"""The Results-tab view of the radial trap frequency, and its ``ANALYSES`` entry.

:class:`TrapFrequencyView` draws the calculation in
:mod:`pytweezer.analyses.default.trap_frequency`.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout

from pytweezer.analyses.default.cloud import scanned_arguments
from pytweezer.analyses.default.trap_frequency import (
    FREQUENCY_ARGUMENT,
    OCCUPANCY,
    gaussian_dip,
    needed_results,
    trap_frequency,
)
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.default_analyses.cloud import format_value, refill, themed_plot
from pytweezer.GUI.experiments.analyses import AnalysisEntry, ComputedAnalysisView
from pytweezer.GUI.theme import CURVE_COLORS, PLOT_LINE_WIDTH, PLOT_SYMBOL_SIZE

KHZ = 1e3
DATA_COLOUR = CURVE_COLORS[2]
#: Dips not taken as the radial resonance; the accent marks the one that is.
OTHER_DIP_PEN = pg.mkPen("#8d8e99", width=1, style=Qt.PenStyle.DashLine)
OTHER_FIT_PEN = pg.mkPen("#8d8e99", width=PLOT_LINE_WIDTH)
RESONANCE_PEN = pg.mkPen("#5b8def", width=PLOT_LINE_WIDTH)


class TrapFrequencyView(ComputedAnalysisView):
    """Survival against modulation frequency, its dips, and the radial trap frequency.

    Every detected dip is marked and fitted; the one taken as ``2 ν_r`` is in
    the accent colour. Frequencies are shown in kHz.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequency_argument = QComboBox()
        self.frequency_argument.setToolTip(
            "Scanned argument that is the modulation frequency"
        )
        self.min_relative_depth = CompactDoubleSpinBox()
        self.min_relative_depth.setRange(0.05, 1.0)
        self.min_relative_depth.setSingleStep(0.05)
        self.min_relative_depth.setValue(0.5)
        self.min_relative_depth.setToolTip(
            "The radial resonance is the highest-frequency dip at least this "
            "fraction as deep as the deepest dip. Raise it if a weak dip above the "
            "radial one is taken; lower it if the radial dip is shallower than an "
            "axial or subharmonic one"
        )

        controls = QHBoxLayout()
        controls.setSpacing(8)
        for text, widget, width in (
            ("Frequency argument", self.frequency_argument, 200),
            ("Minimum relative depth", self.min_relative_depth, 80),
        ):
            widget.setFixedWidth(width)
            if controls.count():
                controls.addSpacing(12)
            controls.addWidget(QLabel(text))
            controls.addWidget(widget)
        controls.addStretch(1)
        self.frequency_argument.currentIndexChanged.connect(self.recompute)
        self.min_relative_depth.valueChanged.connect(self.recompute_soon)

        self.plot = themed_plot()
        self.plot.setLabel("bottom", "modulation frequency", units="kHz")
        self.plot.setLabel("left", "survival")
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).enableAutoSIPrefix(False)
        self.points = self.plot.plot(
            [],
            [],
            pen=pg.mkPen(DATA_COLOUR, width=1),
            symbol="o",
            symbolSize=PLOT_SYMBOL_SIZE,
            symbolBrush=DATA_COLOUR,
        )
        self.errors = pg.ErrorBarItem(
            x=np.empty(0),
            y=np.empty(0),
            height=np.empty(0),
            pen=pg.mkPen(DATA_COLOUR, width=PLOT_LINE_WIDTH),
        )
        self.plot.addItem(self.errors)
        self.dip_items: list[pg.GraphicsObject] = []

        self.readout = QLabel()
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.readout.setFont(font)
        self.readout.setContentsMargins(10, 6, 10, 6)
        self.readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.readout)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.message)

    def set_measurement(self, archive, path, measurement):
        arguments = scanned_arguments(measurement)
        refill(self.frequency_argument, [(name, name) for name in arguments])
        if FREQUENCY_ARGUMENT in arguments:
            self.frequency_argument.blockSignals(True)
            self.frequency_argument.setCurrentIndex(
                self.frequency_argument.findData(FREQUENCY_ARGUMENT)
            )
            self.frequency_argument.blockSignals(False)
        super().set_measurement(archive, path, measurement)

    def unavailable_reason(self, measurement):
        if not scanned_arguments(measurement):
            return "This file scanned no argument to use as the modulation frequency."
        recorded = measurement.result_shapes
        if "survival" not in recorded and not all(n in recorded for n in OCCUPANCY):
            return (
                "This file recorded no survival: the experiment needs saved trap "
                "sites (run FindTrapSites and accept its proposal)."
            )
        return None

    def needed_results(self, measurement):
        return needed_results(measurement)

    def settings(self):
        return self.frequency_argument.currentData(), self.min_relative_depth.value()

    def compute(self, loaded, settings):
        argument, min_relative_depth = settings
        return trap_frequency(loaded, argument, min_relative_depth)

    def show_result(self, result):
        for item in self.dip_items:
            self.plot.removeItem(item)
        self.dip_items = []
        if result is None:
            self.readout.setText("")
            self.points.setData([], [])
            self.errors.setData(x=np.empty(0), y=np.empty(0), height=np.empty(0))
            return
        f_khz = result.frequency / KHZ
        self.points.setData(f_khz, result.survival)
        self.errors.setData(
            x=f_khz, y=result.survival, height=2 * np.nan_to_num(result.error)
        )
        for index, dip in enumerate(result.dips):
            chosen = index == result.chosen
            self._add_dip_item(
                pg.InfiniteLine(
                    pos=dip.centre_hz / KHZ,
                    angle=90,
                    pen=RESONANCE_PEN if chosen else OTHER_DIP_PEN,
                )
            )
            if dip.curve is not None:
                dense = np.linspace(*dip.fit_range_hz, 200)
                self._add_dip_item(
                    pg.PlotDataItem(
                        dense / KHZ,
                        gaussian_dip(dense, *dip.curve),
                        pen=RESONANCE_PEN if chosen else OTHER_FIT_PEN,
                    )
                )
        self.readout.setText(self._summary(result))

    def _add_dip_item(self, item):
        self.plot.addItem(item)
        self.dip_items.append(item)

    @staticmethod
    def _summary(result):
        if not result.success:
            return result.message
        dip = result.radial_dip
        radial = format_value(dip.trap_hz, dip.trap_err_hz, 1 / KHZ)
        resonance = format_value(dip.centre_hz, dip.centre_err_hz, 1 / KHZ)
        lines = [f"ν_r = {radial} kHz    (2ν_r dip at {resonance} kHz; {dip.message})"]
        others = result.other_dips
        if others:
            lines.append(f"  {'other dips':>18}      {'f/2':>18}")
        for other in others:
            centre = format_value(other.centre_hz, other.centre_err_hz, 1 / KHZ)
            half = format_value(other.trap_hz, other.trap_err_hz, 1 / KHZ)
            fitted = "" if other.curve is not None else "    (lowest point, not fitted)"
            lines.append(f"  {centre:>18} kHz  {half:>18} kHz{fitted}")
        return "\n".join(lines)


ANALYSES = {
    "trap_frequency": AnalysisEntry(
        "Radial trap frequency (parametric heating)", TrapFrequencyView
    ),
}
