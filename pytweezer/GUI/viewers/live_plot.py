"""Live-plot applet: subscribes to 1-D data streams and plots them live.

Built on :class:`pytweezer.GUI.applet.Applet` — see ``docs/applets.md``.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor
from pyqtgraph.Qt import QtCore

from pytweezer.servers import DataClient
from pytweezer.GUI.applet import Applet, run_applet


class LivePlot(Applet):
    """Plot applet: draws one curve per subscribed data stream, updating live."""

    stream_category = "Data"
    poll_interval = 10

    def init_gui(self):
        self.datastream = DataClient(self.name)
        self.curvedict = {}

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        pg.setConfigOptions(antialias=True)
        win = pg.GraphicsLayoutWidget()
        layout.addWidget(win)

        self.plot = pg.PlotItem()
        win.addItem(self.plot)

        vb = self.plot.getViewBox()
        vb.menu.addAction("subscriptions").triggered.connect(
            self.open_subscription_editor
        )
        vb.menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read the ``datastreams`` list from Properties and rebuild curves."""
        self.datastream.unsubscribe()
        self.curvedict = {}
        datastreams = self.props.get("datastreams", ["Axial_slice"])
        self.datastream.subscribe(datastreams)
        print("live_plot.py subscribed to:  ", datastreams)

        self.plot.clear()
        for stream in datastreams:
            col = QColor(self.props.get(stream + "/color", int(255 * 256**3 + 255)))
            pen = pg.mkPen(col, width=1)
            self.curvedict[stream] = self.plot.plot(pen=pen)

    def poll(self):
        if self.datastream.has_new_data():
            msg, di, A = self.datastream.recv()
            # generate x axis in case array is one-dimensional
            if A.ndim == 1:
                x = range(len(A))
                A = np.vstack([x, A])

            if msg in self.curvedict:
                if self.props.get("rotated", False):
                    self.curvedict[msg].setData(A[1], A[0])
                else:
                    self.curvedict[msg].setData(A[0], A[1])

    def sizeHint(self):
        return QtCore.QSize(200, 200)


def main(name):
    run_applet(LivePlot, default_name=name)


if __name__ == "__main__":
    run_applet(LivePlot, default_name="LivePlot")
