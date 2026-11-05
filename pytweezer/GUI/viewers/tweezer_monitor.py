"""The tweezer camera image with a square at each accepted trap site.

The Image Monitor, plus the trap sites accepted from ``FindTrapSites``: each is
drawn as a square the summing window across (``trap_window_size``), so it is
plain at a glance whether the sites sit on the atoms. Subscribe it to the
tweezer camera's image stream. The sites are kept in full-sensor pixels and
placed in each frame by the ROI origin in its header (``roi_x0``/``roi_y0``; 0
when missing). They are read from the saved parameters when the viewer starts
and again whenever the Experiment Manager says they changed; until some are
accepted, a line on the image says so.

Properties (besides the Image Monitor's):
    *   show_site_labels: (bool) number each square with its site index.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

from pytweezer.analysis.trap_sites import to_roi
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.theme import CURVE_COLORS
from pytweezer.GUI.viewers.image_monitor import ImageDisplay
from pytweezer.logging_utils import get_logger
from pytweezer.servers import PropertyAttribute

logger = get_logger(__name__)

SITE_COLOUR = CURVE_COLORS[2]
NO_SITES = "No trap sites are saved: run FindTrapSites and accept its proposal"


def sites_from_saved(saved: dict) -> tuple[np.ndarray | None, int, str]:
    """The accepted sites (full-sensor ``(y, x)``), their window size, and a message if there are none."""
    if "trap_sites" not in saved or "trap_window_size" not in saved:
        return None, 0, NO_SITES
    return (
        np.asarray(saved["trap_sites"], dtype=float),
        int(saved["trap_window_size"]),
        "",
    )


class SiteOverlay:
    """Squares, optional index labels and a message, drawn over an image in ``plot``."""

    def __init__(self, plot):
        self.plot = plot
        self.squares = pg.ScatterPlotItem(
            symbol="s", pxMode=False, brush=None, pen=pg.mkPen(SITE_COLOUR, width=1)
        )
        plot.addItem(self.squares)
        self.labels: list[pg.TextItem] = []
        self.message = pg.TextItem(color=SITE_COLOUR, anchor=(0, 0))
        plot.addItem(self.message)
        self.sites = None
        self.window_size = 0

    def set_sites(self, sites, window_size: int, message: str = "") -> None:
        self.sites, self.window_size = sites, window_size
        self.message.setText(message)

    def place(self, image_item, head: dict, labels: bool = False) -> None:
        """Put a square on each site's pixel of the frame ``image_item`` shows, read from ``head``'s ROI."""
        for label in self.labels:
            self.plot.removeItem(label)
        self.labels = []
        if self.sites is None:
            self.squares.clear()
            return
        sites = to_roi(self.sites, head.get("roi_y0", 0), head.get("roi_x0", 0))
        # Through the image's own transform, so squares follow however the
        # frame is placed and scaled: row-major puts data[row, col] at local
        # (col, row), and pixel centres are at +0.5.
        centres = [
            image_item.mapToParent(QtCore.QPointF(col + 0.5, row + 0.5))
            for row, col in sites
        ]
        step = image_item.mapToParent(QtCore.QPointF(1, 0)) - image_item.mapToParent(
            QtCore.QPointF(0, 0)
        )
        self.squares.setData(
            x=[centre.x() for centre in centres],
            y=[centre.y() for centre in centres],
            size=self.window_size * float(np.hypot(step.x(), step.y())),
        )
        if labels:
            for index, centre in enumerate(centres):
                label = pg.TextItem(str(index), color=SITE_COLOUR, anchor=(0.5, 1.2))
                label.setPos(centre)
                self.plot.addItem(label)
                self.labels.append(label)


class TweezerDisplay(ImageDisplay):
    """The Image Monitor with the accepted trap sites drawn over each frame."""

    _show_site_labels = PropertyAttribute("show_site_labels", False)

    def init_gui(self):
        super().init_gui()
        self.overlay = SiteOverlay(self.plot)
        self._saved_version = None
        self._placed = None
        self.feed = ExperimentFeed()
        self.feed.queue_changed.connect(self._queue_changed)
        self._load_sites()

    def _queue_changed(self, snapshot):
        version = snapshot.get("saved_parameters_version")
        if version is not None and version != self._saved_version:
            self._saved_version = version
            self._load_sites()

    def _load_sites(self):
        try:
            saved = get_saved_parameters()
        except (ManagerUnavailable, ManagerError) as error:
            logger.warning("Could not read the saved trap sites: %r", error)
            return
        self.overlay.set_sites(*sites_from_saved(saved))
        self._placed = None

    def poll(self):
        super().poll()
        head = getattr(self, "image_head", None)
        labels = bool(self._show_site_labels)
        if head is not None and self._placed != (id(head), labels):
            self.overlay.place(self.image_item, head, labels=labels)
            self._placed = (id(head), labels)

    def closeEvent(self, event):
        self.feed.close()
        super().closeEvent(event)


def main(name):
    run_applet(TweezerDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(TweezerDisplay, default_name="Tweezer Monitor")
