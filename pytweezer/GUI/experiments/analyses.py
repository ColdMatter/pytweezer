"""Analyses the Results tab offers for a measurement, beside its universal plot.

An experiment names its analyses in ``Experiment.analyses``; each file records
those names (root attr ``analyses``) when it is created, so the tab never
imports the experiment. :data:`ANALYSES` maps each name to its view. A view
loads what it needs over its own archive connection and computes off the GUI
thread, so a large file never stalls the tab.

To add one, put its computation in :mod:`pytweezer.analysis` (no Qt), write an
:class:`AnalysisView` that draws it, and register both here.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pytweezer.analysis.tweezer_histogram import (
    HistogramSettings,
    needed_results,
    point_groups,
    source,
    tweezer_histogram,
)
from pytweezer.experiment.archive import ArchiveUnavailable
from pytweezer.experiment.storage import Measurement
from pytweezer.GUI.components import CompactDoubleSpinBox
from pytweezer.GUI.tweezer_loading_plots import TweezerLoadingPlots
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments.analyses")

_RECOMPUTE_DELAY_MS = 300


def declared_analyses(measurement: Measurement) -> list[str]:
    """The analysis names ``measurement``'s file records; none for older files."""
    return json.loads(measurement.attrs.get("analyses", "[]"))


class AnalysisView(QWidget):
    """One analysis of the selected measurement."""

    def set_measurement(self, archive, path: str, measurement: Measurement) -> None:
        """Analyse the file at ``path``; ``measurement`` is its header, loaded without results."""
        raise NotImplementedError


class TweezerHistogramView(AnalysisView):
    """The tweezer loading histogram of a file, as the live viewer draws it, in ``unit``."""

    _result_ready = QtCore.pyqtSignal(int, object)

    def __init__(self, unit="counts", parent=None):
        super().__init__(parent)
        self.unit = unit
        self._archive = self._clone = None
        self._path = self._measurement = None
        #: The last file loaded with its results, so changing a control doesn't reload.
        self._loaded = (None, None)
        self._generation = 0
        # One analysis at a time, each holding a whole file's frames; a request
        # made meanwhile waits here, replaced by any newer one.
        self._running = False
        self._pending = None
        self._result_ready.connect(self._show_result)
        self._recompute_soon = QtCore.QTimer(self)
        self._recompute_soon.setSingleShot(True)
        self._recompute_soon.setInterval(_RECOMPUTE_DELAY_MS)
        self._recompute_soon.timeout.connect(self._recompute)

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
        self.image.currentIndexChanged.connect(self._recompute)
        self.shots_from.currentIndexChanged.connect(self._recompute)
        for spin in (self.sites, self.feature_size, self.window_size):
            spin.valueChanged.connect(self._recompute_soon.start)
        self.detection_step.valueChanged.connect(self._recompute_soon.start)

        self.plots = TweezerLoadingPlots()
        self.message = QLabel()
        self.message.setProperty("role", "regionHint")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.plots, 1)
        layout.addWidget(self.message)

    def set_measurement(self, archive, path, measurement):
        if archive is not self._archive:
            self._archive, self._clone = archive, archive.clone()
        if path != self._path:
            self.plots.show_message("")
        self._path, self._measurement = path, measurement
        kind = source(measurement)
        for widget in self._frames_only:
            widget.setVisible(kind == "frames")
        self.shots_from.blockSignals(True)
        self.shots_from.clear()
        self.shots_from.addItem("all points")
        self.shots_from.addItems([label for label, _ in point_groups(measurement)])
        self.shots_from.blockSignals(False)
        if kind is None:
            self._generation += 1
            self.plots.show_message("")
            self.message.setText(
                "This file recorded no tweezer images, counts or photons."
            )
            return
        self._recompute()

    def _settings(self):
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

    def _recompute(self):
        if self._measurement is None or source(self._measurement) is None:
            return
        self._recompute_soon.stop()
        self._generation += 1
        self.message.setText("Analysing…")
        request = (self._generation, self._path, self._measurement, self._settings())
        if self._running:
            self._pending = request
        else:
            self._start(request)

    def _start(self, request):
        self._running = True
        threading.Thread(target=self._compute, args=request, daemon=True).start()

    def _compute(self, generation, path, measurement, settings):
        try:
            loaded_path, loaded = self._loaded
            if loaded_path != path:
                loaded = self._clone.load(path, results=needed_results(measurement))
                self._loaded = (path, loaded)
            result = tweezer_histogram(loaded, settings)
        except ArchiveUnavailable as error:
            result = error
        except Exception as error:
            logger.exception("Could not analyse %s", path)
            result = error
        self._result_ready.emit(generation, result)

    def _show_result(self, generation, result):
        self._running = False
        if self._pending is not None:
            request, self._pending = self._pending, None
            self._start(request)
        if generation != self._generation:
            return
        if isinstance(result, ArchiveUnavailable):
            self.plots.show_message("")
            self.message.setText("Experiment Manager not reachable.")
        elif isinstance(result, Exception):
            self.plots.show_message("")
            self.message.setText(f"Could not analyse: {result}")
        else:
            self.message.setText("")
            self.plots.show_result(*result)


@dataclass(frozen=True)
class AnalysisEntry:
    """How the Results tab lists an analysis and builds its view."""

    label: str
    view: Callable[[], AnalysisView]


ANALYSES: dict[str, AnalysisEntry] = {
    "tweezer_histogram": AnalysisEntry(
        "Tweezer loading histogram (counts)",
        lambda: TweezerHistogramView(unit="counts"),
    ),
    "tweezer_photon_histogram": AnalysisEntry(
        "Tweezer loading histogram (photons)",
        lambda: TweezerHistogramView(unit="photons"),
    ),
}
