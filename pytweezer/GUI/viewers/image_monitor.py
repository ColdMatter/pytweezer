"""Image-viewer applet: subscribes to image streams and displays them live.

Built on :class:`pytweezer.GUI.applet.Applet` — see ``docs/applets.md``.
"""

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QSpinBox,
)
import PyQt5
import pyqtgraph as pg
import matplotlib.pyplot as plt

from pytweezer.servers import ImageClient, tweezerpath, PropertyAttribute
from pytweezer.GUI.viewers.archive.zmq_ROI import zmq_ROI
from pytweezer.GUI.viewers.data_sidebar import DataSidebar
from pytweezer.GUI.applet import Applet, run_applet


class ImageDisplay(Applet):
    """Self-updating image display applet.

    Subscribes to the image streams named in the ``imagestreams`` property and
    to optional mask streams (``maskstreams``), drawing them live with a
    colormap, histogram LUT, and configurable ROIs.
    """

    stream_category = "Image"
    poll_interval = 10

    _maxlevels = PropertyAttribute("maxlevels", 255.0)
    _autolevels = PropertyAttribute("autolevels", False)
    _imgpath = PropertyAttribute("/Servers/Imagepath", tweezerpath + "/../Data")
    _roilist = PropertyAttribute("ROIs", ["slice"])
    _imagestreams = PropertyAttribute("imagestreams", ["--None--"])
    _maskstreams = PropertyAttribute("maskstreams", ["--None--"])
    _invert_colors = PropertyAttribute("invert_colors", False)
    _show_sidebar = PropertyAttribute("show_sidebar", False)

    def init_gui(self):
        self.imDataDict = {}
        self.imgresolution = [1, 1]
        self.image_index = -1

        btn = QPushButton("SaveImage")
        layout = QVBoxLayout()
        layout.addWidget(btn)
        # btn.clicked.connect(self.saveCurrentImage)

        # Image and its data sidebar sit side by side; the sidebar is hidden by
        # default and toggled from the view-box context menu.
        content = QHBoxLayout()
        layout.addLayout(content)
        self.sidebar = DataSidebar(self._props, self.name)
        self.sidebar.setVisible(bool(self._show_sidebar))
        content.addWidget(self.sidebar)

        self.imgstream = ImageClient("imageviewer")
        self.maskstream = ImageClient("imageviewermask")
        self.update_subscriptions()
        if self.imgstream.has_new_data():
            imgdata, head = self.imgstream.recv()
        else:
            imgdata = np.ones((100, 100))

        colors = np.array(plt.cm.magma.colors) * 255
        if self._invert_colors:
            colors = colors[::-1]

        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        graphics_layout_widget = pg.GraphicsLayoutWidget()
        plot: pg.PlotItem = graphics_layout_widget.addPlot()
        plot.setLabel("left", text="y", units="m")
        plot.setLabel("bottom", text="x", units="m")
        image_item = pg.ImageItem(imgdata)
        self.image_item = image_item
        # add a LUT to the image display
        lut = pg.HistogramLUTItem()
        lut.setImageItem(image_item)
        graphics_layout_widget.addItem(lut)
        image_item.setRect(PyQt5.QtCore.QRect(0, 0, imgdata.shape[1], imgdata.shape[0]))
        image_item.setLookupTable(cmap.getLookupTable())
        plot.addItem(image_item)
        self.plot = plot

        # add Image mask
        maskdat = np.zeros(imgdata.shape)
        mask = pg.ImageItem(maskdat)
        mask.setLookupTable(np.array([[0, 0, 255, 0], [0, 255, 255, 255]]))
        plot.addItem(mask)
        self.mask = mask
        for region in self._roilist:
            roi = zmq_ROI(self.name + "/ROI/" + region, invertible=True)
            roi.addScaleHandle((1, 1), center=(0, 0))
            plot.addItem(roi)
        content.addWidget(graphics_layout_widget, 1)

        self.setLayout(layout)
        view_box = plot.getViewBox()
        # we want to keep the aspect ratio correct when plotting images
        view_box.setAspectLocked(True)
        view_box.menu.addAction("subscriptions").triggered.connect(
            self.open_subscription_editor
        )
        view_box.menu.addAction("Image masks").triggered.connect(self.subscribe_mask)
        view_box.menu.addAction("configure").triggered.connect(self.open_config_editor)
        # scroll-wheel selector for the image index when several images are streamed
        view_box.menu.addAction("Image Index").triggered.connect(self.index_selector)
        view_box.menu.addAction("Data sidebar").triggered.connect(self.toggle_sidebar)
        view_box.menu.addAction("Data sidebar subscriptions").triggered.connect(
            self.sidebar.edit_subscriptions
        )

    def toggle_sidebar(self):
        show = not self.sidebar.isVisible()
        self.sidebar.setVisible(show)
        self._show_sidebar = show

    def sizeHint(self):
        return QtCore.QSize(800, 700)

    def update_subscriptions(self):
        """Re-apply image and mask stream subscriptions from Properties."""
        self.imgstream.unsubscribe()
        self.imgstream.subscribe(self._imagestreams)
        self.maskstream.unsubscribe()
        self.maskstream.subscribe(self._maskstreams)
        self.sidebar.update_subscriptions()

    def subscribe_mask(self):
        self.open_subscription_editor("maskstreams")

    def poll(self):
        self.sidebar.update()
        if self.imgstream.has_new_data():
            while self.imgstream.has_new_data():
                msg, head, imgdata = self.imgstream.recv()
                if self.image_index != -1 and head["index"] != self.image_index:
                    continue
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
                self.imgdata = imgdata
        self.update_mask()

    def update_mask(self):
        if self.maskstream.has_new_data():
            while self.maskstream.has_new_data():
                msg, head, imgdata = self.maskstream.recv()
                self.imgresolution = head["_imgresolution"]
                offset = tuple(np.array(head["_offset"]) * np.array(self.imgresolution))

                h, w = imgdata.shape
                self.mask.setRect(
                    QtCore.QRectF(
                        offset[0],
                        offset[1],
                        w * self.imgresolution[0],
                        h * self.imgresolution[1],
                    )
                )
                self.mask.setImage(imgdata, autoLevels=False, levels=(0, 1))

    def saveCurrentImage(self):
        """Save the current image as png using current time as filename"""
        pass

    def index_selector(self):
        d = QDialog()
        layout = QVBoxLayout()
        d.setWindowTitle("Select Image Index")
        index_selector = QSpinBox()
        index_selector.setMinimum(-1)
        index_selector.setValue(self.image_index)
        layout.addWidget(index_selector)
        d.setLayout(layout)
        index_selector.valueChanged.connect(self.update_image_index)
        d.exec_()

    def update_image_index(self, value):
        self.image_index = value


def main(name):
    run_applet(ImageDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(ImageDisplay, default_name="Image Monitor")
