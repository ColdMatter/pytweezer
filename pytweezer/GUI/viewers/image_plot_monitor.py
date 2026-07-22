"""Image viewer with projection line plots above and to the left of the image.

Built on :class:`pytweezer.GUI.applet.Applet` — see ``docs/applets.md``.

Like :mod:`pytweezer.GUI.viewers.image_monitor` it shows an image stream with a
colormap, histogram LUT and ROIs, but adds two line plots whose axes are linked
to the image:

    * a top plot sharing the image's x-axis (subscribed via ``topstreams``),
    * a left plot sharing the image's y-axis (subscribed via ``leftstreams``).

Point these at the ``projections.py`` outputs (and any ``gaussian_fit.py`` fit
curve) to see the profiles and fits aligned with the image. Data streams carry a
``[coords, values]`` array; the top plot draws ``values`` against ``coords`` on
the shared x-axis, and the left plot draws ``coords`` up the shared y-axis with
``values`` running toward the image.
"""

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout
import PyQt5
import pyqtgraph as pg
import matplotlib.pyplot as plt

from pytweezer.servers import ImageClient, DataClient, PropertyAttribute
from pytweezer.GUI.viewers.archive.zmq_ROI import zmq_ROI
from pytweezer.GUI.applet import Applet, run_applet


class ImagePlotDisplay(Applet):
    """Image display with x/y projection plots linked to the image axes."""

    stream_category = "Image"
    poll_interval = 10

    _roilist = PropertyAttribute("ROIs", ["slice"])
    _imagestreams = PropertyAttribute("imagestreams", ["--None--"])
    _topstreams = PropertyAttribute("topstreams", ["--None--"])
    _leftstreams = PropertyAttribute("leftstreams", ["--None--"])
    _invert_colors = PropertyAttribute("invert_colors", False)

    def init_gui(self):
        self.imgresolution = [1, 1]

        self.imgstream = ImageClient("imageplotviewer")
        self.topclient = DataClient("imageplotviewer_top")
        self.leftclient = DataClient("imageplotviewer_left")
        self.topcurves = {}
        self.leftcurves = {}

        colors = np.array(plt.cm.magma.colors) * 255
        if self._invert_colors:
            colors = colors[::-1]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)

        glw = pg.GraphicsLayoutWidget()

        # Grid: top plot over the image, left plot beside it, LUT on the far right.
        self.top_plot = glw.addPlot(row=0, col=1)
        self.left_plot = glw.addPlot(row=1, col=0)
        self.plot = glw.addPlot(row=1, col=1)
        lut = pg.HistogramLUTItem()
        glw.addItem(lut, row=1, col=2)

        imgdata = np.ones((100, 100))
        self.image_item = pg.ImageItem(imgdata)
        self.image_item.setRect(PyQt5.QtCore.QRect(0, 0, 100, 100))
        lut.setImageItem(self.image_item)
        # The colormap goes on the histogram's gradient, not on the image: a
        # HistogramLUTItem drives its image's lookup table, so a setLookupTable
        # here would be overwritten and the image would come out greyscale.
        lut.gradient.setColorMap(cmap)
        self.plot.addItem(self.image_item)
        self.plot.setLabel("left", text="y", units="m")
        self.plot.setLabel("bottom", text="x", units="m")

        # Keep the projection plots slim and share axes with the image so the
        # profiles line up with what is drawn below/right of them.
        self.top_plot.setMaximumHeight(140)
        self.top_plot.setXLink(self.plot)
        self.top_plot.hideAxis("bottom")
        self.left_plot.setMaximumWidth(140)
        self.left_plot.setYLink(self.plot)
        self.left_plot.hideAxis("left")
        # Left plot's value axis points toward the image.
        self.left_plot.getViewBox().invertX(True)

        view_box = self.plot.getViewBox()
        view_box.setAspectLocked(True)

        for region in self._roilist:
            roi = zmq_ROI(self.name + "/ROI/" + region, invertible=True)
            roi.addScaleHandle((1, 1), center=(0, 0))
            self.plot.addItem(roi)

        layout = QVBoxLayout()
        layout.addWidget(glw)
        self.setLayout(layout)

        view_box.menu.addAction("Image subscriptions").triggered.connect(
            lambda: self.open_subscription_editor()
        )
        view_box.menu.addAction("Top plot subscriptions").triggered.connect(
            lambda: self.open_subscription_editor("topstreams")
        )
        view_box.menu.addAction("Left plot subscriptions").triggered.connect(
            lambda: self.open_subscription_editor("leftstreams")
        )
        view_box.menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

    def sizeHint(self):
        return QtCore.QSize(820, 760)

    def update_subscriptions(self):
        self.imgstream.unsubscribe()
        self.imgstream.subscribe(self._imagestreams)
        self._rebuild_curves(self.topclient, self._topstreams, self.top_plot, self.topcurves)
        self._rebuild_curves(self.leftclient, self._leftstreams, self.left_plot, self.leftcurves)

    def _rebuild_curves(self, client, streams, plot, curvedict):
        client.unsubscribe()
        for curve in curvedict.values():
            plot.removeItem(curve)
        curvedict.clear()
        client.subscribe(streams)
        for index, stream in enumerate(streams):
            if stream == "--None--":
                continue
            pen = pg.mkPen(self.stream_color(stream, index), width=1.5)
            curvedict[stream] = plot.plot(pen=pen)

    def poll(self):
        self._update_image()
        self._update_curves(self.topclient, self.topcurves, orientation="top")
        self._update_curves(self.leftclient, self.leftcurves, orientation="left")

    def _update_image(self):
        while self.imgstream.has_new_data():
            msg, head, imgdata = self.imgstream.recv()
            self.image_item.setImage(imgdata, autoLevels=True)
            self.image_item.resetTransform()
            try:
                self.imgresolution = head["_imgresolution"]
                offset = np.array(head["_offset"]) * np.array(self.imgresolution)
            except Exception:
                self.imgresolution = [1, 1]
                offset = [0, 0]
            h, w = imgdata.shape
            self.image_item.setRect(
                QtCore.QRectF(
                    offset[0],
                    offset[1],
                    w * self.imgresolution[0],
                    h * self.imgresolution[1],
                )
            )

    def _update_curves(self, client, curvedict, orientation):
        while client.has_new_data():
            msg = client.recv()
            if msg is None or len(msg) < 3:
                continue
            streamname, head, arr = msg
            if streamname not in curvedict:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1:
                coords = np.arange(arr.size)
                values = arr
            else:
                coords, values = arr[0], arr[1]
            if orientation == "top":
                curvedict[streamname].setData(coords, values)
            else:
                # coordinate runs up the shared y-axis; value along x.
                curvedict[streamname].setData(values, coords)


def main(name):
    run_applet(ImagePlotDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(ImagePlotDisplay, default_name="Image Plot Monitor")
