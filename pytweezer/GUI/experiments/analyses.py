"""Analyses the Results tab offers for a measurement, beside its universal plot.

An experiment names its analyses in ``Experiment.analyses``; each file records
those names (root attr ``analyses``) when it is created, so the tab never
imports the experiment. This module holds the views' framework; the analyses
themselves, and the registry mapping each name to its view, are in
:mod:`pytweezer.analysis.default_analyses`. A view loads what it needs over its
own archive connection and computes off the GUI thread, so a large file never
stalls the tab.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtWidgets import QLabel, QWidget

from pytweezer.experiment.archive import ArchiveUnavailable
from pytweezer.experiment.storage import Measurement
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments.analyses")

_RECOMPUTE_DELAY_MS = 300
#: Per-point results a view keeps, most recently read; each may be a whole image.
_POINT_RESULTS_KEPT = 8


def declared_analyses(measurement: Measurement) -> list[str]:
    """The analysis names ``measurement``'s file records; none for older files."""
    return json.loads(measurement.attrs.get("analyses", "[]"))


class AnalysisView(QWidget):
    """One analysis of the selected measurement."""

    def set_measurement(self, archive, path: str, measurement: Measurement) -> None:
        """Analyse the file at ``path``; ``measurement`` is its header, loaded without results."""
        raise NotImplementedError


class ComputedAnalysisView(AnalysisView):
    """An analysis computed off the GUI thread from the file's results.

    Loads the results it needs over its own clone of the archive, once per
    file, and computes in a worker thread. One computation runs at a time; a
    request made meanwhile waits, replaced by any newer one, and a result
    overtaken by a newer request is discarded. Failures are shown in
    :attr:`message`, never raised.

    A subclass lays out its controls, its drawing and :attr:`message`, calls
    :meth:`recompute` (or :meth:`recompute_soon`, for a spin box) when a
    control changes, and implements:

    * :meth:`needed_results`: which results to load from the file;
    * :meth:`settings`: its controls' state, read on the GUI thread;
    * :meth:`compute`: the result from the loaded file and those settings, in
      the worker thread. It may read one point's value of a result it did not
      load, such as an image, with :meth:`point_result`;
    * :meth:`show_result`: draw the result, or clear the drawing for ``None``;
    * :meth:`unavailable_reason`: why a file cannot be analysed, or ``None``.

    To refill controls that depend on the file, override
    :meth:`set_measurement` and call it from the override with the controls'
    signals blocked.
    """

    _result_ready = QtCore.pyqtSignal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._archive = self._clone = None
        self._path = self._measurement = None
        #: The last file loaded with its results, so changing a control doesn't reload.
        self._loaded = (None, None)
        #: Values read by point_result(), keyed (path, name, index), oldest first.
        self._point_results = {}
        #: The file the running compute() analyses.
        self._computing_path = None
        self._generation = 0
        # One analysis at a time, each holding a whole file's results; a request
        # made meanwhile waits here, replaced by any newer one.
        self._running = False
        self._pending = None
        self._result_ready.connect(self._show_result)
        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(_RECOMPUTE_DELAY_MS)
        self._recompute_timer.timeout.connect(self.recompute)
        self.message = QLabel()
        self.message.setProperty("role", "regionHint")

    def needed_results(self, measurement: Measurement) -> list[str]:
        """The results :meth:`compute` reads from ``measurement``'s file."""
        raise NotImplementedError

    def settings(self) -> Any:
        """The controls' state, passed to :meth:`compute`."""
        raise NotImplementedError

    def compute(self, loaded: Measurement, settings: Any) -> Any:
        """The result to show, from the loaded file; runs in a worker thread."""
        raise NotImplementedError

    def show_result(self, result: Any) -> None:
        """Draw a result of :meth:`compute`; ``None`` clears the drawing."""
        raise NotImplementedError

    def unavailable_reason(self, measurement: Measurement) -> str | None:
        """Why ``measurement`` cannot be analysed, or ``None`` if it can."""
        return None

    def point_result(self, name: str, index: int) -> Any:
        """Point ``index``'s value of result ``name`` in the file being analysed.

        Reads that one value over the view's archive connection, keeping the
        few read last. Call it only from :meth:`compute`.
        """
        key = (self._computing_path, name, index)
        value = self._point_results.pop(key, None)
        if value is None:
            value = self._clone.point_result(self._computing_path, name, index)
        self._point_results[key] = value
        while len(self._point_results) > _POINT_RESULTS_KEPT:
            del self._point_results[next(iter(self._point_results))]
        return value

    def set_measurement(self, archive, path, measurement):
        if archive is not self._archive:
            self._archive, self._clone = archive, archive.clone()
        if path != self._path:
            self.show_result(None)
        self._path, self._measurement = path, measurement
        reason = self.unavailable_reason(measurement)
        if reason is not None:
            self._generation += 1
            self.show_result(None)
            self.message.setText(reason)
            return
        self.recompute()

    def recompute_soon(self) -> None:
        """:meth:`recompute` once the controls have been still for a moment."""
        self._recompute_timer.start()

    def recompute(self) -> None:
        """Analyse the current file again with the current settings."""
        measurement = self._measurement
        if measurement is None or self.unavailable_reason(measurement) is not None:
            return
        self._recompute_timer.stop()
        self._generation += 1
        self.message.setText("Analysing…")
        request = (self._generation, self._path, measurement, self.settings())
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
                loaded = self._clone.load(
                    path, results=self.needed_results(measurement)
                )
                self._loaded = (path, loaded)
            self._computing_path = path
            result = self.compute(loaded, settings)
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
            self.show_result(None)
            self.message.setText("Experiment Manager not reachable.")
        elif isinstance(result, Exception):
            self.show_result(None)
            self.message.setText(f"Could not analyse: {result}")
        else:
            self.message.setText("")
            self.show_result(result)


@dataclass(frozen=True)
class AnalysisEntry:
    """How the Results tab lists an analysis and builds its view."""

    label: str
    view: Callable[[], AnalysisView]
