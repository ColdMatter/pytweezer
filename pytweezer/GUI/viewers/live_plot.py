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

from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.live_plot_applet import LivePlotApplet


class LivePlot(LivePlotApplet):
    """Grid of plots, each with curves from data streams, header fields and experiment results."""


def main(name):
    run_applet(LivePlot, default_name=name)


if __name__ == "__main__":
    run_applet(LivePlot, default_name="LivePlot")
