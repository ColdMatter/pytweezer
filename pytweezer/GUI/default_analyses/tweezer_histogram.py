"""The Results-tab view of the tweezer loading histogram, and its ``ANALYSES`` entries.

The calculation is :mod:`pytweezer.analyses.default.tweezer_histogram`.
"""

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from pytweezer.analyses.default.tweezer_histogram import (
    HistogramSettings,
    needed_results,
    point_groups,
    source,
    tweezer_histogram,
)
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.experiments.analyses import AnalysisEntry, ComputedAnalysisView
from pytweezer.GUI.views.tweezer_loading import TweezerLoadingPlots


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
