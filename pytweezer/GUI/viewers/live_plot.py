"""Live-plot applet: subscribes to 1-D data streams and plots them live.

Built on :class:`pytweezer.GUI.applet.Applet` — see ``docs/applets.md``.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

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
        # The plot is the whole window; margins here would just frame it in
        # background colour.
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        win = pg.GraphicsLayoutWidget()
        layout.addWidget(win)

        self.plot = pg.PlotItem()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend(offset=(-10, 10))
        win.addItem(self.plot)

        vb = self.plot.getViewBox()
        vb.menu.addAction("subscriptions").triggered.connect(
            lambda: self.open_subscription_editor()
        )
        vb.menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read the ``datastreams`` list from Properties and rebuild curves."""
        self.datastream.unsubscribe()
        self.curvedict = {}
        datastreams = self.props.get("datastreams", ["Axial_slice"])
        self.datastream.subscribe(datastreams)

        self.plot.clear()
        for index, stream in enumerate(datastreams):
            if stream == "--None--":
                continue
            pen = pg.mkPen(self.stream_color(stream, index), width=1.5)
            self.curvedict[stream] = self.plot.plot(pen=pen, name=stream)

    def poll(self):
        # Drain the queue: at 100 Hz a busy stream delivers several messages per
        # poll, and reading one per tick falls further behind the longer it runs.
        while self.datastream.has_new_data():
            message = self.datastream.recv()
            # recv() gives (channel, header) for a header-only message and
            # (channel, header, array) when an array follows, so a stream
            # carrying only scalars must not be unpacked as three.
            if message is None or len(message) < 3:
                continue
            msg, di, A = message
            # generate x axis in case array is one-dimensional
            if A.ndim == 1:
                x = range(len(A))
                A = np.vstack([x, A])

            if msg in self.curvedict:
                if self.props.get("rotated", False):
                    self.curvedict[msg].setData(A[1], A[0])
                else:
                    self.curvedict[msg].setData(A[0], A[1])


def main(name):
    run_applet(LivePlot, default_name=name)


if __name__ == "__main__":
    run_applet(LivePlot, default_name="LivePlot")
