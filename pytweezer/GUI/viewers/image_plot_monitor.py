"""Image Monitor with the horizontal and vertical sums above and to the left of the image.

Everything :mod:`pytweezer.GUI.viewers.image_monitor` does, plus two line plots
whose axes are linked to the image's: the sum down the columns above it, sharing
its x-axis, and the sum across the rows to its left, sharing its y-axis. Each
draws the sums as points and the 1-D Gaussian fit as a white line.

Point ``projectionstreams`` at the ``projections.py`` analyses, by the name they
run under: ``<name>_x`` draws in the top plot and ``<name>_y`` in the left one.
Both publish ``[coords, sums, fit_curve]`` in the image's coordinates. The left
plot runs its coordinate along the image's y-axis and its values along x,
increasing towards the image. A profile whose fit was not trusted (its fit curve
is NaN) is drawn without the line.

Properties (besides the Image Monitor's):
    *   projectionstreams: ([str]) projection analyses to draw.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.viewers.image_monitor import ImageDisplay
from pytweezer.servers import DataClient, PropertyAttribute

SIDE_PLOT_SIZE = 140
AXIS_SIZE = 60
FIT_PEN = pg.mkPen("w", width=2)


class ImageProjectionDisplay(ImageDisplay):
    """Image display with the projection plots above and left of the image."""

    image_row = 1
    image_col = 1

    _projectionstreams = PropertyAttribute("projectionstreams", ["--None--"])

    def init_gui(self):
        self.projectionclient = DataClient("imageprojectionviewer")
        self.projection_curves: dict[str, tuple[pg.PlotDataItem, pg.PlotDataItem]] = {}
        self.top_plot = None
        self.left_plot = None
        super().init_gui()

        self.top_plot = self.graphics_layout.addPlot(row=0, col=1)
        self.left_plot = self.graphics_layout.addPlot(row=1, col=0)
        self.top_plot.setFixedHeight(SIDE_PLOT_SIZE)
        self.top_plot.setXLink(self.plot)
        self.top_plot.hideAxis("bottom")
        self.left_plot.setFixedWidth(SIDE_PLOT_SIZE)
        self.left_plot.setYLink(self.plot)
        self.left_plot.hideAxis("left")
        # The image's y-axis is inverted; a linked view follows the range but not
        # the direction.
        self.left_plot.getViewBox().invertY(True)
        # Equal axis widths keep the top plot's view box above the image's.
        for plot in (self.plot, self.top_plot):
            plot.getAxis("left").setWidth(AXIS_SIZE)
        for plot in (self.plot, self.left_plot):
            plot.getAxis("bottom").setHeight(AXIS_SIZE // 2)
        self.graphics_layout.ci.layout.setRowStretchFactor(1, 4)
        self.graphics_layout.ci.layout.setColumnStretchFactor(1, 4)

        self.plot.getViewBox().menu.addAction("Projections").triggered.connect(
            self.subscribe_projections
        )
        self._rebuild_projection_curves()

    def sizeHint(self):
        return QtCore.QSize(820, 760)

    def update_subscriptions(self):
        super().update_subscriptions()
        self.projectionclient.unsubscribe()
        self.projectionclient.subscribe(self._projection_channels())
        self._rebuild_projection_curves()

    def _projection_channels(self) -> list[str]:
        return [
            f"{stream}{axis}"
            for stream in self._projectionstreams
            if stream != "--None--"
            for axis in ("_x", "_y")
        ]

    def subscribe_projections(self):
        active = self._props.get("/Servers/DataStream/active", {})
        choices = sorted(
            name[:-2]
            for name in active
            if name.endswith("_x") and f"{name[:-2]}_y" in active
        )
        self.open_subscription_editor("projectionstreams", choices=choices)

    def _rebuild_projection_curves(self):
        if self.top_plot is None:
            return
        for channel, curves in self.projection_curves.items():
            plot = self.top_plot if channel.endswith("_x") else self.left_plot
            for curve in curves:
                plot.removeItem(curve)
        self.projection_curves.clear()
        streams = [s for s in self._projectionstreams if s != "--None--"]
        for index, stream in enumerate(streams):
            colour = self.stream_color(stream, index)
            for axis, plot in (("_x", self.top_plot), ("_y", self.left_plot)):
                data_curve = plot.plot(
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolPen=None,
                    symbolBrush=colour,
                )
                fit_curve = plot.plot(pen=FIT_PEN)
                self.projection_curves[stream + axis] = (data_curve, fit_curve)

    def poll(self):
        super().poll()
        self.update_projections()

    def update_projections(self):
        """Redraw each projection from its latest message."""
        latest = {}
        while self.projectionclient.has_new_data(0):
            message = self.projectionclient.recv()
            if message is not None and len(message) >= 3:
                latest[message[0]] = message[2]
        for channel, profile in latest.items():
            if channel in self.projection_curves:
                self.show_projection(channel, np.asarray(profile))

    def show_projection(self, channel: str, profile: np.ndarray) -> None:
        """Draw ``profile`` (``[coords, sums, fit_curve]``) in the plot ``channel`` belongs to."""
        if profile.ndim != 2 or profile.shape[0] < 2:
            return
        coords, sums = profile[0], profile[1]
        fit = profile[2] if profile.shape[0] > 2 else np.full(coords.shape, np.nan)
        data_curve, fit_curve = self.projection_curves[channel]
        trusted = np.isfinite(fit).all()
        if not trusted:
            fit, fit_coords = np.empty(0), np.empty(0)
        else:
            fit_coords = coords
        if channel.endswith("_x"):
            data_curve.setData(coords, sums)
            fit_curve.setData(fit_coords, fit)
        else:
            data_curve.setData(sums, coords)
            fit_curve.setData(fit, fit_coords)


def main(name):
    run_applet(ImageProjectionDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(ImageProjectionDisplay, default_name="Image Plot Monitor")
