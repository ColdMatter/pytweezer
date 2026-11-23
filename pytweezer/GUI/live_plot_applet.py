"""An applet drawing data streams, header fields and experiment results on a grid of :class:`~pytweezer.GUI.views.line_plot.LinePlot`.

:class:`LivePlotApplet` holds the wiring every plot-grid applet shares: one
:class:`PlotPanel` per key in ``plots``, each with its own subscriptions, the
experiment feed and the result history. A subclass places the panels by
overriding :meth:`LivePlotApplet.arrange`.

Each panel draws a curve per data stream sending an array, per subscribed
stream and header field (its newest ``history`` values against messages
received), and per experiment result (markers, against the argument the newest
point carrying it scanned first, or in arrival order). The RIDs drawn are the
plot title. :mod:`pytweezer.GUI.viewers.live_plot` describes them as the user
sees them.

Properties:
    *   plots: ([str]) keys of the plots, in order.
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
from PyQt6 import QtWidgets

from pytweezer.GUI.applet import Applet
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.views.line_plot import LinePlot
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
    """One plot of the grid, with its subscriptions under ``<key>/`` in the applet's properties.

    Attributes:
        view: The plot.
        show_x_label: Label the x axis; the applet turns it off on a plot
            whose x axis another one labels.
        colour_offset: How far along the palette this plot's colours start.
    """

    def __init__(self, applet, key):
        self.applet = applet
        self.key = key
        self.view = LinePlot()
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

        self.view.add_menu_action(
            "subscriptions",
            lambda: applet.open_subscription_editor(streamkey=f"{key}/datastreams"),
        )
        self.view.add_menu_action("header fields", self._edit_fields)
        self.view.add_menu_action(
            "experiment results", lambda: applet.edit_experiment_results(key)
        )
        self.view.add_menu_action("clear history", applet.clear_history)
        self.view.add_menu_action("add plot", applet.add_plot)
        self.view.add_menu_action("remove plot", lambda: applet.remove_plot(key))
        self.view.add_menu_action("configure", applet.open_config_editor)

    def _edit_fields(self):
        self.applet.open_subscription_editor(
            streamkey=f"{self.key}/fields",
            choices=sorted(self.seen_fields | set(self.fields)),
        )

    def _new_curve(self, key, name, markers=False, **style):
        colour = self.applet.stream_color(
            f"{self.key}/{name}",
            self.colour_offset + len(self.view.listDataItems()),
        )
        return self.view.curve(key, colour, markers=markers, name=name, **style)

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

        self.view.clear_curves()
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
        # Keyed apart from the data curves, which may share a result's name.
        self.result_curves = {
            name: self._new_curve(f"result {name}", name, markers=True)
            for name in props.get(f"{self.key}/experimentresults", [])
        }
        self.draw_results(results)

    def _field_curve(self, stream, field):
        name = f"{stream}: {field}"
        curve = self._new_curve(name, name, connect="finite")
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
            curve = self.array_curves[stream] = self._new_curve(stream, stream)
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
        self.view.set_title(rid_title(drawn))
        self._label_x()

    def _label_x(self):
        if not self.show_x_label:
            self.view.set_x_label("")
        elif self.result_curves:
            self.view.set_x_label(self.result_argument or ARRIVAL_ORDER)
        elif self.fields:
            self.view.set_x_label(MESSAGES_RECEIVED)
        else:
            self.view.set_x_label("")

    def close(self):
        self.stream.unsubscribe()
        self.view.deleteLater()


class LivePlotApplet(Applet):
    """A grid of :class:`PlotPanel`, one per key in ``plots``, fed by data streams and the experiment feed.

    A subclass places the panels by overriding :meth:`arrange`.

    Attributes:
        panels: ``{key: PlotPanel}`` of the plots shown.
        grid: The layout the panels are placed in.
        results: The experiment points received.
        feed: The Experiment Manager's queue and points.
    """

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

    def update_subscriptions(self):
        """Match the panels to ``plots``, lay them out, and re-read each one's subscriptions."""
        keys = self._plots or ["plot0"]
        for key in set(self.panels) - set(keys):
            self.grid.removeWidget(self.panels[key].view)
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
            self.grid.addWidget(self.panels[key].view, row, column)

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
        """Empty every plot's header-field and experiment-result histories."""
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
