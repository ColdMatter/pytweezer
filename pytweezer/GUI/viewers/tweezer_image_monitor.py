import argparse
import logging
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from pytweezer.GUI.viewers.image_group import ImageWindow
from pytweezer.GUI.viewers.image_monitor import ImageDisplay
from pytweezer.servers import DataClient, PropertyAttribute


class TweezerImageDisplay(ImageDisplay):
    """ImageDisplay subclass that overlays detected tweezer/atom positions."""

    _positionstreams = PropertyAttribute("positionstreams", ["--None--"])
    _show_tweezer_grid = PropertyAttribute("show_tweezer_grid", True)
    _tweezer_square_size = PropertyAttribute("tweezer_square_size", 6.0)
    _expected_sites = PropertyAttribute("expected_sites", 64)

    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self.latest_atom_count = 0
        self.latest_filling_fraction = 0.0

        self.posstream = DataClient("imageviewerpositions")
        for stream in self._positionstreams:
            self.posstream.subscribe(stream)

        self.tweezer_overlay = pg.ScatterPlotItem(
            pen=pg.mkPen((80, 255, 80, 220), width=1.5),
            brush=pg.mkBrush(0, 0, 0, 0),
            symbol="s",
            size=float(self._tweezer_square_size),
            pxMode=False,
        )
        self.plot.addItem(self.tweezer_overlay)

        subscribe_menu = self.plot.getViewBox().menu.addAction("Tweezer positions")
        subscribe_menu.triggered.connect(self.subscribe_positions)

    def update_image(self):
        super().update_image()
        self.update_positions()

    def update_positions(self):
        if not self._show_tweezer_grid:
            self.tweezer_overlay.setData([], [])
            return

        if not self.posstream.has_new_data():
            return

        positions = None
        while self.posstream.has_new_data():
            msg = self.posstream.recv()
            if msg is None:
                continue
            payload = msg[1] if len(msg) >= 2 else None
            if isinstance(payload, dict) and "positions" in payload:
                positions = np.asarray(payload["positions"], dtype=np.float64)

        if positions is None or positions.size == 0:
            self.tweezer_overlay.setData([], [])
            self._set_tweezer_stats(0)
            return

        self.tweezer_overlay.setData(
            # ImageDisplay uses pyqtgraph ImageItem axis convention where the first
            # index is rendered along x and the second along y.
            x=positions[:, 0],
            y=positions[:, 1],
            size=float(self._tweezer_square_size),
            symbol="s",
        )
        self._set_tweezer_stats(int(positions.shape[0]))

    def _set_tweezer_stats(self, atom_count):
        if atom_count is None:
            atom_count = 0
        self.latest_atom_count = int(atom_count)

        expected_sites = int(self._expected_sites) if int(self._expected_sites) > 0 else 0
        if expected_sites > 0:
            self.latest_filling_fraction = float(atom_count) / float(expected_sites)
        else:
            self.latest_filling_fraction = 0.0

    def subscribe_positions(self):
        self._subscribe_win("positionstreams", "Data")
        self.posstream.unsubscribe()
        self.posstream.subscribe(self._positionstreams)

        active = [s for s in self._positionstreams if s != "--None--"]
        if len(active) == 0:
            self.tweezer_overlay.setData([], [])
            
class TweezerImageWindow(ImageWindow):
    """Image window variant that uses the tweezer overlay display."""

    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self._info_timer = QtCore.QTimer(self)
        self._info_timer.timeout.connect(self.update_infobox)
        self._info_timer.start(100)

    def get_display_class(self):
        return TweezerImageDisplay

    def get_display_name(self):
        # Tweezer viewer configs are often stored directly at /<name>/...
        # rather than /<name>/Viewer/...
        has_streams = self._props.get('/' + self._name + '/imagestreams', None) is not None
        if has_streams:
            return self._name
        return self._displayname

    # def update_infobox(self):

    #     info = {}

    #     info["N atoms"] = int(getattr(self.image_monitor, "latest_atom_count", 0))
    #     info["Filling Fraction"] = float(getattr(self.image_monitor, "latest_filling_fraction", 0.0))

    #     streams = self.imDataBox.props.get("liststreams", [])
    #     if len(streams) == 0:
    #         self.imDataBox.props.set("liststreams", ["N atoms", "Filling Fraction"])

    #     self.imDataBox.setNewData(info)




def main(name):
    logging.info("Started.")
    app = QtWidgets.QApplication(sys.argv)
    cam_win = TweezerImageWindow(name)
    cam_win.show()
    app.exec_()


if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        parser = argparse.ArgumentParser()
        parser.add_argument("name", nargs=1, help="name of this program instance")
        args = parser.parse_args()
        name = args.name[0]
        main(name)
