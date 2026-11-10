"""Live plots stacked one above another, sharing their x axis.

The Live Plot's plots, in one column with their x axes linked: zooming or
panning one moves them all, and only the bottom one carries the x tick values
and label. Each plot keeps its own y axis and its own subscriptions, so two
quantities on very different scales, such as ``tweezer_counts.py``'s
``atom_counts`` and ``background_counts``, are read against the same shots.
Everything else, from the context menu to the properties, is as in
:mod:`pytweezer.GUI.viewers.live_plot`, except that there is no ``columns``.

Linking matches the x *values*, so plot the same kind of x in every plot:
header fields against messages received, or experiment results against the
same scanned argument.
"""

from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.viewers.live_plot import LivePlot
from pytweezer.servers import PropertyAttribute

# Fixed so the plots' left edges, and with them their x axes, line up whatever
# the width of each one's y tick labels.
Y_AXIS_WIDTH = 70


class StackedPlotter(LivePlot):
    """The Live Plot in one column, with every plot's x axis linked to the top one's."""

    default_size = (800, 700)

    _plots = PropertyAttribute("plots", ["plot0", "plot1"])

    def arrange(self, keys):
        top = self.panels[keys[0]].plot
        for index, key in enumerate(keys):
            panel = self.panels[key]
            self.grid.addWidget(panel.widget, index, 0)
            panel.plot.setXLink(None if index == 0 else top)
            panel.plot.getAxis("left").setWidth(Y_AXIS_WIDTH)
            panel.show_x_label = index == len(keys) - 1
            panel.plot.getAxis("bottom").setStyle(showValues=panel.show_x_label)


def main(name):
    run_applet(StackedPlotter, default_name=name)


if __name__ == "__main__":
    run_applet(StackedPlotter, default_name="Stacked Plotter")
