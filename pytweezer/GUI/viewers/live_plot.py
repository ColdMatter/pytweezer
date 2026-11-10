"""Live plots of data streams and of experiment results, in a grid.

The window is a grid of plots, filled row by row (context menu → *add plot*,
*remove plot*). Each plot has its own subscriptions, stored under its key in
``plots``, and can draw three kinds of curve:

Data streams: one curve per stream named in ``<plot>/datastreams`` that sends
an array, redrawn from each message's array (a 1-D array is drawn against its
index, a 2xN one as ``[coords, values]``). A plot with header fields chosen
draws no arrays: a stream such as ``tweezer_loading.py``'s sends a
``(shots, sites)`` array that is not a curve, and its scalars are plotted as
header fields instead.

Header fields: one curve per subscribed stream and field named in
``<plot>/fields`` (context menu → *header fields*, which offers the numeric
fields the plot's streams have sent so far), keeping the newest ``history``
values against the number of messages received. This is how scalars an
analysis publishes in a header (``tweezer_counts.py``'s ``atom_counts`` and
``background_counts``, ``image_stats.py``'s ``mean``) are watched over a run.
A NaN leaves a gap.

Experiment results: one set of markers per scalar result named in
``<plot>/experimentresults`` (context menu → *experiment results*), from the
points the Experiment Manager publishes as tasks measure them. Points
accumulate across tasks, keeping the newest ``history``; on opening, the window
starts from the points the manager still holds (each running task's, and each
pipeline's last). They are drawn against the argument scanned first by the
newest point carrying a chosen result, leaving out points that scanned
something else, or in arrival order when that point scans nothing numeric. The
RIDs drawn are the plot title.

*clear history* empties the header-field and experiment-result histories of
every plot. They live in this window only: closing it loses them.

Properties:
    *   plots: ([str]) keys of the plots, in grid order.
    *   columns: (int) plots per row; 0 arranges them in a near-square grid.
    *   <plot>/datastreams: ([str]) input data streams of that plot.
    *   <plot>/fields: ([str]) header fields of those streams to plot.
    *   <plot>/experimentresults: ([str]) experiment results to plot.
    *   history: (int) points kept per header field, and experiment points kept.
    *   rotated: (bool) swap a data stream's axes.
"""

import math
from collections import deque
from numbers import Real

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.servers import DataClient, PropertyAttribute

ARRIVAL_ORDER = "point"
MESSAGES_RECEIVED = "messages received"


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


def numeric_fields(head):
    """``{field: value}`` of the header's plain numbers (not bools)."""
    return {
        field: float(value)
        for field, value in head.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    }


def grid_position(index, n_plots, columns):
    """``(row, column)`` of plot ``index`` of ``n_plots``, ``columns`` per row (0: near-square)."""
    columns = columns if columns > 0 else math.ceil(math.sqrt(n_plots))
    return divmod(index, columns)


class PlotPanel:
    """One plot of the grid, with its subscriptions under ``<key>/`` in the applet's properties."""

    def __init__(self, applet, key):
        self.applet = applet
        self.key = key
        self.widget = pg.PlotWidget()
        self.plot = self.widget.getPlotItem()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend(offset=(-10, 10))
        self.stream = DataClient(f"{applet.name}_{key}")
        self.datastreams = []
        self.fields = []
        self.array_curves = {}
        self.field_curves = {}
        self.field_points = {}
        self.result_curves = {}
        self.result_argument = None
        self.seen_fields = set()
        self.show_x_label = True
        self.colour_offset = 0

        menu = self.plot.getViewBox().menu
        menu.addAction("subscriptions").triggered.connect(
            lambda: applet.open_subscription_editor(streamkey=f"{key}/datastreams")
        )
        menu.addAction("header fields").triggered.connect(self._edit_fields)
        menu.addAction("experiment results").triggered.connect(
            lambda: applet.edit_experiment_results(key)
        )
        menu.addAction("clear history").triggered.connect(applet.clear_history)
        menu.addAction("add plot").triggered.connect(applet.add_plot)
        menu.addAction("remove plot").triggered.connect(lambda: applet.remove_plot(key))
        menu.addAction("configure").triggered.connect(applet.open_config_editor)

    def _edit_fields(self):
        self.applet.open_subscription_editor(
            streamkey=f"{self.key}/fields",
            choices=sorted(self.seen_fields | set(self.fields)),
        )

    def _new_curve(self, name, markers=False, **style):
        colour = self.applet.stream_color(
            f"{self.key}/{name}", self.colour_offset + len(self.plot.listDataItems())
        )
        if markers:
            style |= {"pen": None, "symbol": "o", "symbolSize": 6}
            style |= {"symbolPen": None, "symbolBrush": colour}
        else:
            style["pen"] = pg.mkPen(colour, width=1.5)
        return self.plot.plot(name=name, **style)

    def update(self, results):
        """Re-read this plot's subscriptions and rebuild its curves, keeping the field histories still wanted."""
        props = self.applet.props
        self.datastreams = [
            stream
            for stream in props.get(f"{self.key}/datastreams", [])
            if stream != "--None--"
        ]
        self.fields = props.get(f"{self.key}/fields", [])
        self.stream.unsubscribe()
        self.stream.subscribe(self.datastreams)

        self.plot.clear()
        self.array_curves = {}
        self.field_curves = {}
        self.field_points = {
            (stream, field): points
            for (stream, field), points in self.field_points.items()
            if stream in self.datastreams and field in self.fields
        }
        for stream, field in self.field_points:
            self._field_curve(stream, field)
        self._draw_fields()
        self.result_curves = {
            name: self._new_curve(name, markers=True)
            for name in props.get(f"{self.key}/experimentresults", [])
        }
        self.draw_results(results)

    def _field_curve(self, stream, field):
        curve = self._new_curve(f"{stream}: {field}", connect="finite")
        self.field_curves[stream, field] = curve
        return curve

    def poll(self, history, rotated):
        changed = False
        while self.stream.has_new_data(timeout_ms=0):
            message = self.stream.recv()
            # (channel, header) for a header-only message, (channel, header,
            # array) when an array follows.
            if message is None or len(message) < 2:
                continue
            channel, head = message[0], message[1]
            if channel not in self.datastreams:
                continue
            values = numeric_fields(head)
            self.seen_fields.update(values)
            for field in self.fields:
                if field not in values:
                    continue
                points = self.field_points.get((channel, field))
                if points is None or points.maxlen != history:
                    points = deque(points or (), maxlen=history)
                    self.field_points[channel, field] = points
                points.append(values[field])
                changed = True
            if len(message) >= 3 and not self.fields:
                self._draw_array(channel, message[2], rotated)
        if changed:
            self._draw_fields()

    def _draw_array(self, stream, array, rotated):
        if array.ndim == 1:
            array = np.vstack([np.arange(len(array)), array])
        if array.ndim != 2 or len(array) < 2:
            return
        curve = self.array_curves.get(stream)
        if curve is None:
            curve = self.array_curves[stream] = self._new_curve(stream)
        x, y = (array[1], array[0]) if rotated else (array[0], array[1])
        curve.setData(x, y)

    def _draw_fields(self):
        for (stream, field), points in self.field_points.items():
            curve = self.field_curves.get((stream, field))
            if curve is None:
                curve = self._field_curve(stream, field)
            curve.setData(np.arange(len(points)), np.array(points))
        self._label_x()

    def clear_fields(self):
        for points in self.field_points.values():
            points.clear()
        self._draw_fields()

    def draw_results(self, results):
        self.result_argument = results.x_argument(self.result_curves)
        drawn = set()
        for name, curve in self.result_curves.items():
            x, y, rids = results.curve(name, self.result_argument)
            curve.setData(x, y)
            drawn |= rids
        self.plot.setTitle(rid_title(drawn))
        self._label_x()

    def _label_x(self):
        if not self.show_x_label:
            self.plot.setLabel("bottom", "")
        elif self.result_curves:
            self.plot.setLabel("bottom", self.result_argument or ARRIVAL_ORDER)
        elif self.fields:
            self.plot.setLabel("bottom", MESSAGES_RECEIVED)
        else:
            self.plot.setLabel("bottom", "")

    def close(self):
        self.stream.unsubscribe()
        self.widget.deleteLater()


class LivePlot(Applet):
    """Grid of plots, each with curves from data streams, header fields and experiment results."""

    stream_category = "Data"
    poll_interval = 50
    default_size = (900, 600)

    _plots = PropertyAttribute("plots", ["plot0"])
    _columns = PropertyAttribute("columns", 0)
    _history = PropertyAttribute("history", 10000)
    _rotated = PropertyAttribute("rotated", False)

    def init_gui(self):
        self.panels = {}
        self.results = ResultHistory()
        self._took_held_points = False

        self.grid = QtWidgets.QGridLayout()
        # The plots are the whole window; margins here would just frame them
        # in background colour.
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        self.setLayout(self.grid)

        self.feed = ExperimentFeed()
        self.feed.point_received.connect(self._experiment_point)
        self.feed.queue_changed.connect(self._take_held_points)

        self.update_subscriptions()

    def update_subscriptions(self):
        """Match the panels to ``plots``, lay them out, and re-read each one's subscriptions."""
        keys = self._plots or ["plot0"]
        for key in set(self.panels) - set(keys):
            self.grid.removeWidget(self.panels[key].widget)
            self.panels.pop(key).close()
        for key in keys:
            if key not in self.panels:
                self.panels[key] = PlotPanel(self, key)
        self.arrange(keys)
        for key in keys:
            self.panels[key].update(self.results)

    def arrange(self, keys):
        """Place the panels of ``keys``, in order, in the grid."""
        for index, key in enumerate(keys):
            row, column = grid_position(index, len(keys), int(self._columns))
            self.grid.addWidget(self.panels[key].widget, row, column)

    def add_plot(self):
        keys = self._plots
        number = 0
        while f"plot{number}" in keys:
            number += 1
        self._plots = keys + [f"plot{number}"]
        self.update_subscriptions()

    def remove_plot(self, key):
        keys = self._plots
        if key not in keys or len(keys) == 1:
            return
        self._plots = [other for other in keys if other != key]
        self.props.delete(key)
        self.update_subscriptions()

    def edit_experiment_results(self, key):
        names = self.results.result_names() | {
            name
            for points in self.feed.held_points().values()
            for point in points
            for name in point["scalars"]
        }
        self.open_subscription_editor(
            streamkey=f"{key}/experimentresults", choices=sorted(names)
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

    def clear_history(self):
        self.results.clear()
        for panel in self.panels.values():
            panel.clear_fields()
        self._draw_results()

    def _draw_results(self):
        for panel in self.panels.values():
            panel.draw_results(self.results)

    def poll(self):
        history = max(2, int(self._history))
        rotated = bool(self._rotated)
        for panel in self.panels.values():
            panel.poll(history, rotated)

    def closeEvent(self, event):
        self.feed.close()
        super().closeEvent(event)


def main(name):
    run_applet(LivePlot, default_name=name)


if __name__ == "__main__":
    run_applet(LivePlot, default_name="LivePlot")
