"""Image-viewer applet: subscribes to image streams and displays them live.

Built on :class:`pytweezer.GUI.applet.Applet`.

Right-click an ROI and choose **Copy as camera ROI** to put
``(x0, width, y0, height)`` on the clipboard: the region of the last frame
under that ROI in unbinned sensor pixels, the argument order of the
cameras' ``set_roi`` and of ``CameraShots(roi=...)``. It is worked out from
the frame header's ``roi_x0``/``roi_y0`` and ``binning``, so it is only
meaningful on a camera's own stream.

Frames are drawn row 0 at the top with x along the columns, as in the notebooks
and the Optimisers tab. Pixel column ``c``, row ``r`` of a frame sits at
``x = (origin_x + c + 0.5) * res_x``, ``y = (origin_y + r + 0.5) * res_y``, with
``res`` the header's ``_imgresolution`` (metres per pixel) and ``origin`` its
``_offset`` (binned pixels); without them ``res = (1, 1)`` and ``origin = (0, 0)``.
Rows therefore run down the screen with +y.

Properties:
    *   imagestreams: ([str]) image streams to draw.
    *   maskstreams: ([str]) image streams drawn over the frame as a mask.
    *   overlaystreams: ([str]) data streams of cloud fits, drawn as contour
        ellipses over the frame.
    *   n_contours: (int) contour ellipses per fit.
    *   show_sidebar: (bool) the data panel below the image: the latest message
        of each stream in ``sidebarstreams``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtGui import QAction, QCursor, QGuiApplication
from PyQt6.QtWidgets import QDialog, QMenu, QSpinBox, QToolTip, QVBoxLayout

from pytweezer.analysis.cloud_fit import (
    FIELDS,
    CloudFit,
    contour_ellipses,
    frame_geometry,
)
from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.viewers.archive.zmq_ROI import zmq_ROI
from pytweezer.GUI.viewers.data_sidebar import DataSidebar
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, ImageClient, PropertyAttribute, tweezerpath

logger = get_logger(__name__)

OVERLAY_PEN = pg.mkPen("w", width=1)


def frame_rect(head: dict, shape: tuple[int, ...]) -> QtCore.QRectF:
    """Where a frame of ``shape`` sits in the plot, from its header's ``_imgresolution`` and ``_offset``."""
    resolution, origin = frame_geometry(head)
    res_x, res_y = resolution or (1.0, 1.0)
    rows, cols = shape[:2]
    return QtCore.QRectF(
        origin[0] * res_x, origin[1] * res_y, cols * res_x, rows * res_y
    )


def overlay_paths(head: dict, n: int) -> list[np.ndarray]:
    """The ``n`` contour ellipses (each ``[x, y]``, in plot coordinates) of the cloud fit in ``head``.

    Empty when the fit failed or the header carries no fit.
    """
    if not head.get("success") or any(field not in head for field in FIELDS):
        return []
    fit = CloudFit(
        **{field: head[field] for field in FIELDS},
        units=head.get("units", ""),
        success=True,
    )
    return contour_ellipses(fit, n)


def camera_roi(
    roi: pg.ROI, image_item: pg.ImageItem, head: dict
) -> tuple[int, int, int, int] | None:
    """The camera ROI ``(x0, width, y0, height)`` that reads the part of ``image_item``'s frame under ``roi``.

    In unbinned sensor pixels, from the ROI origin and binning in the frame's
    ``head``; ``None`` when ``roi`` misses the frame.
    """
    rect = roi.mapRectToItem(image_item, roi.boundingRect())
    horizontal = sorted((rect.left(), rect.right()))
    vertical = sorted((rect.top(), rect.bottom()))
    if image_item.axisOrder == "col-major":
        row_span, col_span = horizontal, vertical
    else:
        row_span, col_span = vertical, horizontal
    rows, cols = image_item.image.shape[:2]
    row0, row1 = max(0, round(row_span[0])), min(rows, round(row_span[1]))
    col0, col1 = max(0, round(col_span[0])), min(cols, round(col_span[1]))
    if row1 <= row0 or col1 <= col0:
        return None
    binning = int(head.get("binning", 1))
    return (
        int(head.get("roi_x0", 0)) + col0 * binning,
        (col1 - col0) * binning,
        int(head.get("roi_y0", 0)) + row0 * binning,
        (row1 - row0) * binning,
    )


class CameraROI(zmq_ROI):
    """A :class:`zmq_ROI` whose context menu copies it as a camera ROI.

    ``viewer`` draws the frames in ``viewer.image_item`` and keeps the last
    one's header as ``viewer.image_head`` (``None`` until a frame arrives).
    """

    def __init__(self, name, viewer, **kwargs):
        super().__init__(name, **kwargs)
        self.viewer = viewer
        self.menu = QMenu("ROI")
        copy = QAction("Copy as camera ROI", self.menu)
        copy.triggered.connect(self.copy_as_camera_roi)
        self.menu.addAction(copy)

    def contextMenuEnabled(self):
        return True

    def copy_as_camera_roi(self) -> None:
        head = self.viewer.image_head
        if head is None:
            message = "No frame yet: the camera ROI comes from the last frame"
        elif (sensor_roi := camera_roi(self, self.viewer.image_item, head)) is None:
            message = "The ROI is outside the frame"
        else:
            text = "({}, {}, {}, {})".format(*sensor_roi)
            QGuiApplication.clipboard().setText(text)
            message = f"Copied camera ROI (x0, width, y0, height) = {text}"
            if "binning" not in head:
                message += "\nThe frame does not say its binning; taken as 1"
            logger.info("%s from %s", message, self.name)
        QToolTip.showText(QCursor.pos(), message)


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
    _overlaystreams = PropertyAttribute("overlaystreams", ["--None--"])
    _n_contours = PropertyAttribute("n_contours", 8)
    _show_sidebar = PropertyAttribute("show_sidebar", True)

    def init_gui(self):
        self.imDataDict = {}
        self.image_index = -1
        self.image_head = None

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.sidebar = DataSidebar(self._props, self.name)
        self.sidebar.setVisible(bool(self._show_sidebar))

        self.imgstream = ImageClient("imageviewer")
        self.maskstream = ImageClient("imageviewermask")
        self.overlayclient = DataClient("imageviewer_overlay")
        self.overlay_curves: dict[str, list[pg.PlotCurveItem]] = {}
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
        image_item = pg.ImageItem(imgdata, axisOrder="row-major")
        self.image_item = image_item
        # add a LUT to the image display
        lut = pg.HistogramLUTItem()
        lut.setImageItem(image_item)
        graphics_layout_widget.addItem(lut)
        image_item.setRect(frame_rect({}, imgdata.shape))
        # The colormap goes on the histogram's gradient, not on the image: a
        # HistogramLUTItem drives its image's lookup table, so a setLookupTable
        # here would be overwritten and the image would come out greyscale.
        lut.gradient.setColorMap(cmap)
        plot.addItem(image_item)
        self.plot = plot

        # add Image mask
        maskdat = np.zeros(imgdata.shape)
        mask = pg.ImageItem(maskdat, axisOrder="row-major")
        mask.setLookupTable(np.array([[0, 0, 255, 0], [0, 255, 255, 255]]))
        plot.addItem(mask)
        self.mask = mask
        for region in self._roilist:
            roi = CameraROI(self.name + "/ROI/" + region, self, invertible=True)
            roi.addScaleHandle((1, 1), center=(0, 0))
            plot.addItem(roi)
        layout.addWidget(graphics_layout_widget, 1)
        layout.addWidget(self.sidebar)

        view_box = plot.getViewBox()
        view_box.invertY(True)
        # we want to keep the aspect ratio correct when plotting images
        view_box.setAspectLocked(True)
        view_box.menu.addAction("subscriptions").triggered.connect(
            lambda: self.open_subscription_editor()
        )
        view_box.menu.addAction("Image masks").triggered.connect(self.subscribe_mask)
        view_box.menu.addAction("Overlays").triggered.connect(self.subscribe_overlays)
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
        self.overlayclient.unsubscribe()
        self.overlayclient.subscribe(self._overlaystreams)
        self.sidebar.update_subscriptions()

    def subscribe_mask(self):
        self.open_subscription_editor("maskstreams")

    def subscribe_overlays(self):
        self.open_subscription_editor("overlaystreams", category="Data")

    def poll(self):
        self.sidebar.update()
        while self.imgstream.has_new_data():
            msg, head, imgdata = self.imgstream.recv()
            if self.image_index != -1 and head["index"] != self.image_index:
                continue
            self.image_item.setImage(imgdata, autoLevels=True)
            self.image_item.resetTransform()
            self.image_item.setRect(frame_rect(head, imgdata.shape))
            self.imgdata = imgdata
            self.image_head = head
        self.update_mask()
        self.update_overlays()

    def update_mask(self):
        while self.maskstream.has_new_data():
            msg, head, imgdata = self.maskstream.recv()
            self.mask.setRect(frame_rect(head, imgdata.shape))
            self.mask.setImage(imgdata, autoLevels=False, levels=(0, 1))

    def update_overlays(self):
        """Redraw each overlay stream's contour ellipses from its latest message."""
        latest = {}
        while self.overlayclient.has_new_data(0):
            msg = self.overlayclient.recv()
            if msg is not None and isinstance(msg[1], dict):
                latest[msg[0]] = msg[1]
        for stream, head in latest.items():
            self.show_overlay(stream, overlay_paths(head, int(self._n_contours)))

    def show_overlay(self, stream: str, paths: list[np.ndarray]) -> None:
        """Replace ``stream``'s curves with one white curve per path in ``paths``."""
        curves = self.overlay_curves.pop(stream, [])
        for curve in curves[len(paths) :]:
            self.plot.removeItem(curve)
        curves = curves[: len(paths)]
        for index, (x, y) in enumerate(paths):
            if index == len(curves):
                curves.append(pg.PlotCurveItem(pen=OVERLAY_PEN))
                curves[index].setZValue(10)
                self.plot.addItem(curves[index])
            curves[index].setData(x, y)
        if curves:
            self.overlay_curves[stream] = curves

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
        d.exec()

    def update_image_index(self, value):
        self.image_index = value


def main(name):
    run_applet(ImageDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(ImageDisplay, default_name="Image Monitor")
