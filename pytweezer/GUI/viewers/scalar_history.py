"""Rolling history of a scalar carried in a data stream's header.

Built on :class:`pytweezer.GUI.applet.Applet` — see ``docs/applets.md``.

Where :mod:`pytweezer.GUI.viewers.live_plot` draws a whole array per message,
this draws *one number* per message against arrival order, so a quantity an
analysis publishes shot-to-shot (atom number, centre of mass, fit width) can be
watched drifting over a run. Point it at any analysis that publishes scalars in
the header — ``image_stats.py``, ``centre_of_mass.py``, ``gaussian_fit.py``.

The x-axis is messages received, not wall-clock time: a stream that stalls holds
its trace rather than stretching it, which is usually what you want when
comparing shots.

Properties:
    *   datastreams: ([str]) input data streams.
    *   field: (str) header key to plot. Streams lacking it are ignored.
    *   history: (int) number of points kept per stream.
"""

from collections import deque

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont

from pytweezer.servers import DataClient, PropertyAttribute
from pytweezer.GUI.applet import Applet, run_applet


class ScalarHistory(Applet):
    """Strip chart of one header field, one curve per subscribed data stream."""

    stream_category = "Data"
    poll_interval = 50
    default_size = (720, 420)

    _field = PropertyAttribute("field", "atom_number")
    _history = PropertyAttribute("history", 200)
    _datastreams = PropertyAttribute("datastreams", ["--None--"])

    def init_gui(self):
        self.stream = DataClient(self.name)
        self.curves = {}
        self.points = {}

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        win = pg.GraphicsLayoutWidget()
        layout.addWidget(win)
        self.plot = win.addPlot()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", text="messages received")
        self.legend = self.plot.addLegend(offset=(-10, 10))

        # Latest values, monospaced so the digits hold their columns instead of
        # jittering as the numbers change.
        self.readout = QtWidgets.QLabel()
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.readout.setFont(font)
        self.readout.setContentsMargins(10, 6, 10, 8)
        layout.addWidget(self.readout)

        view_box = self.plot.getViewBox()
        view_box.menu.addAction("subscriptions").triggered.connect(
            lambda: self.open_subscription_editor()
        )
        view_box.menu.addAction("configure").triggered.connect(self.open_config_editor)
        view_box.menu.addAction("clear history").triggered.connect(self.clear)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read ``datastreams`` and rebuild one curve per stream."""
        self.stream.unsubscribe()
        self.plot.clear()
        self.legend.clear()
        self.curves = {}
        self.points = {}

        streams = self._datastreams
        self.stream.subscribe(streams)
        self.plot.setLabel("left", text=self._field)

        for index, name in enumerate(streams):
            if name == "--None--":
                continue
            pen = pg.mkPen(self.stream_color(name, index), width=1.5)
            self.curves[name] = self.plot.plot(pen=pen, name=name)
            self.points[name] = deque(maxlen=max(2, int(self._history)))
        self._refresh_readout()

    def clear(self):
        for points in self.points.values():
            points.clear()
        for name, curve in self.curves.items():
            curve.setData([], [])
        self._refresh_readout()

    def poll(self):
        field = self._field
        changed = False
        # Drain: several shots can land between polls, and each is a point.
        while self.stream.has_new_data():
            message = self.stream.recv()
            # A scalar arrives as a header-only message, so recv() hands back
            # two values here and three when an array happens to follow.
            if message is None or len(message) < 2:
                continue
            channel, head = message[0], message[1]
            if channel not in self.points:
                continue
            value = head.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            self.points[channel].append(float(value))
            changed = True

        if changed:
            for name, points in self.points.items():
                self.curves[name].setData(range(len(points)), list(points))
            self._refresh_readout()

    def _refresh_readout(self):
        if not self.points:
            self.readout.setText("no streams subscribed")
            return
        parts = []
        for name, points in self.points.items():
            latest = f"{points[-1]:>12.4g}" if points else f"{'--':>12}"
            parts.append(f"{name}  {latest}")
        self.readout.setText("     ".join(parts))


def main(name):
    run_applet(ScalarHistory, default_name=name)


if __name__ == "__main__":
    run_applet(ScalarHistory, default_name="Scalar History")
