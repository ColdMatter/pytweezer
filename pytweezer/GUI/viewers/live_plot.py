"""Live plot of 1-D data streams and of experiment results.

Data streams: one curve per stream named in ``datastreams``, redrawn from each
message's array (a 1-D array is drawn against its index, a 2xN one as
``[coords, values]``).

Experiment results: one set of markers per scalar result named in
``experimentresults`` (context menu → *experiment results*), from the points
the Experiment Manager publishes as tasks measure them. Points accumulate
across tasks until *clear results*, keeping the newest ``history``; on opening,
the plot starts from the points the manager still holds (each running task's,
and each pipeline's last). They are drawn against the argument scanned first by the
newest point carrying a chosen result, leaving out points that scanned
something else, or in arrival order when that point scans nothing numeric. The RIDs drawn are the plot
title. The history lives in this window only: closing it loses the points.

Properties:
    *   datastreams: ([str]) input data streams.
    *   experimentresults: ([str]) experiment results to plot.
    *   history: (int) number of experiment points kept.
    *   rotated: (bool) swap a data stream's axes.
"""

from numbers import Real

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.servers import DataClient, PropertyAttribute

ARRIVAL_ORDER = "point"


class ResultHistory:
    """Experiment points accumulated across tasks, oldest first, each kept once."""

    def __init__(self):
        self._points = {}

    def add(self, point, limit):
        """Keep ``point`` (``{"rid", "index", "values", "scalars", ...}``), then the newest ``limit``."""
        self._points.setdefault((point["rid"], point["index"]), point)
        while len(self._points) > limit:
            del self._points[next(iter(self._points))]

    def clear(self):
        self._points.clear()

    def result_names(self):
        return {name for point in self._points.values() for name in point["scalars"]}

    def x_argument(self, names):
        """The argument scanned first by the newest point with any of ``names``, or ``None`` if it scans nothing numeric."""
        newest = next(
            (
                point
                for point in reversed(self._points.values())
                if not point["scalars"].keys().isdisjoint(names)
            ),
            None,
        )
        if newest is None:
            return None
        values = newest["values"]
        argument = next(iter(values), None)
        if argument is None or not isinstance(values[argument], Real):
            return None
        return argument

    def curve(self, name, argument):
        """``(x, y, rids)`` of result ``name``: against ``argument``, or arrival order if ``None``."""
        x, y, rids = [], [], set()
        for order, point in enumerate(self._points.values()):
            if name not in point["scalars"]:
                continue
            value = order if argument is None else point["values"].get(argument)
            if not isinstance(value, Real):
                continue
            x.append(value)
            y.append(point["scalars"][name])
            rids.add(point["rid"])
        return np.array(x, dtype=float), np.array(y, dtype=float), rids


def rid_title(rids):
    if not rids:
        return None
    first, last = min(rids), max(rids)
    return f"RID {first}" if first == last else f"RID {first}–{last}"


class LivePlot(Applet):
    """Plot applet: a curve per data stream and markers per experiment result."""

    stream_category = "Data"
    poll_interval = 10

    _history = PropertyAttribute("history", 10000)

    def init_gui(self):
        self.datastream = DataClient(self.name)
        self.curvedict = {}
        self.result_curves = {}
        self.results = ResultHistory()
        self._took_held_points = False

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
        vb.menu.addAction("experiment results").triggered.connect(
            self._edit_experiment_results
        )
        vb.menu.addAction("clear results").triggered.connect(self._clear_results)
        vb.menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.feed = ExperimentFeed()
        self.feed.point_received.connect(self._experiment_point)
        self.feed.queue_changed.connect(self._take_held_points)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Re-read both subscription lists from Properties and rebuild the plot."""
        self.datastream.unsubscribe()
        self.curvedict = {}
        datastreams = self.props.get("datastreams", ["Axial_slice"])
        self.datastream.subscribe(datastreams)
        results = self.props.get("experimentresults", [])

        self.plot.clear()
        for index, stream in enumerate(datastreams):
            if stream == "--None--":
                continue
            pen = pg.mkPen(self.stream_color(stream, index), width=1.5)
            self.curvedict[stream] = self.plot.plot(pen=pen, name=stream)
        self.result_curves = {}
        for index, name in enumerate(results, start=len(datastreams)):
            self.result_curves[name] = self.plot.plot(
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolPen=None,
                symbolBrush=self.stream_color(name, index),
                name=name,
            )
        self._draw_results()

    def _edit_experiment_results(self):
        names = self.results.result_names() | {
            name
            for points in self.feed.held_points().values()
            for point in points
            for name in point["scalars"]
        }
        self.open_subscription_editor(
            streamkey="experimentresults", choices=sorted(names)
        )

    def _take_held_points(self, _snapshot):
        # Only on first connecting: later, they would bring back points cleared.
        if self._took_held_points:
            return
        self._took_held_points = True
        for rid, points in sorted(self.feed.held_points().items()):
            for point in points:
                self.results.add({"rid": rid, **point}, self._history)
        self._draw_results()

    def _experiment_point(self, point):
        self.results.add(point, self._history)
        self._draw_results()

    def _clear_results(self):
        self.results.clear()
        self._draw_results()

    def _draw_results(self):
        argument = self.results.x_argument(self.result_curves)
        drawn = set()
        for name, curve in self.result_curves.items():
            x, y, rids = self.results.curve(name, argument)
            curve.setData(x, y)
            drawn |= rids
        self.plot.setTitle(rid_title(drawn))
        if self.result_curves:
            self.plot.setLabel("bottom", argument or ARRIVAL_ORDER)

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

    def closeEvent(self, event):
        self.feed.close()
        super().closeEvent(event)


def main(name):
    run_applet(LivePlot, default_name=name)


if __name__ == "__main__":
    run_applet(LivePlot, default_name="LivePlot")
