"""The Image Monitor, with a stored background subtracted from every frame.

Right-click the image and choose *store background* to keep the frame on screen
as the background; every frame after it is shown as ``float64(frame) -
background``. *clear background* goes back to the raw frames. A frame whose
shape differs from the background's (the camera's ROI or binning changed) is
shown raw, and a line on the image says so.

The background lives only in this window: it is lost when the window closes, and
other viewers of the same stream are unaffected. To subtract a background for
everyone, use an analysis (``bg_subtract.py``, ``shot_background.py``).

Properties: as the Image Monitor's.
"""

import numpy as np
import pyqtgraph as pg

from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.theme import PLOT_TEXT
from pytweezer.GUI.viewers.image_monitor import ImageDisplay


class BackgroundMonitor(ImageDisplay):
    def init_gui(self):
        self.background = None
        self.raw_frame = None
        super().init_gui()
        self.status = pg.TextItem(color=PLOT_TEXT, anchor=(0, 0))
        self.status.setParentItem(self.plot.getViewBox())
        menu = self.plot.getViewBox().menu
        menu.addAction("store background").triggered.connect(self.store_background)
        menu.addAction("clear background").triggered.connect(self.clear_background)

    def store_background(self):
        if self.raw_frame is not None:
            self.background = self.raw_frame
            self.redraw()

    def clear_background(self):
        self.background = None
        self.redraw()

    def redraw(self):
        if self.raw_frame is not None:
            self.show_frame(self.image_head, self.raw_frame)

    def show_frame(self, head, imgdata):
        self.raw_frame = np.asarray(imgdata, dtype=np.float64)
        shown, status = subtract_background(self.raw_frame, self.background)
        self.status.setText(status)
        super().show_frame(head, shown)


def subtract_background(frame, background):
    """``frame`` minus ``background`` where they match, and a line saying what is shown."""
    if background is None:
        return frame, ""
    if frame.shape != background.shape:
        return frame, (
            f"raw frame: background is {background.shape}, frame is {frame.shape}"
        )
    return frame - background, "background subtracted"


def main(name):
    run_applet(BackgroundMonitor, default_name=name)


if __name__ == "__main__":
    run_applet(BackgroundMonitor, default_name="Background Monitor")
