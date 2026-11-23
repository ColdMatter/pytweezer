"""An applet that draws image streams live in an :class:`~pytweezer.GUI.views.image.ImageView`.

:class:`ImageApplet` holds the wiring every image applet shares: the image, mask
and overlay subscriptions, persisted ROIs and colour levels, the image index
filter and the data sidebar. A subclass adds what is particular to it through
``self.view`` and the two layouts.

Right-click an ROI and choose **Copy as camera ROI** to put
``(x0, width, y0, height)`` on the clipboard: the region of the last frame
under that ROI in unbinned sensor pixels, the argument order of the
cameras' ``set_roi`` and of the ROI arguments of
:class:`~pytweezer.experiments.load_tweezers.LoadTweezers`. It is worked out from
the frame header's ``roi_x0``/``roi_y0`` and ``binning``, so it is only
meaningful on a camera's own stream.

Frames are drawn row 0 at the top with x along the columns, as in the notebooks
and the Optimisers tab. Pixel column ``c``, row ``r`` of a frame sits at
``x = (origin_x + c + 0.5) * res_x``, ``y = (origin_y + r + 0.5) * res_y``, with
``res`` the header's ``_imgresolution`` (metres per pixel) and ``origin`` its
``_offset`` (binned pixels); without ``_imgresolution``, ``res = (1, 1)`` and the
units are pixels; without ``_offset``, ``origin = (0, 0)``.
Rows therefore run down the screen with +y.

Properties:
    *   imagestreams: ([str]) image streams to draw.
    *   maskstreams: ([str]) image streams drawn over the frame as a mask.
    *   overlaystreams: ([str]) data streams of cloud fits, drawn as contour
        ellipses over the frame.
    *   n_contours: (int) contour ellipses per fit.
    *   autoscale: (bool) scale the colours to each frame (context menu, *auto
        scale*); off holds ``manual_levels``.
    *   manual_levels: ([float]) ``[low, high]`` last set by dragging the
        histogram's levels, which also turns autoscale off.
    *   show_sidebar: (bool) the data panel below the image: the latest message
        of each stream in ``sidebarstreams``.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtGui import QAction, QCursor, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMenu,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from pytweezer.analysis.cloud_fit import FIELDS, CloudFit, contour_ellipses
from pytweezer.GUI.applet import Applet
from pytweezer.GUI.viewers.data_sidebar import DataSidebar
from pytweezer.GUI.views.image import ImageView
from pytweezer.GUI.zmq_roi import zmq_ROI
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, ImageClient, PropertyAttribute, tweezerpath

logger = get_logger(__name__)


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
    """A :class:`zmq_ROI` whose context menu copies it as a camera ROI of the frame ``view`` shows."""

    def __init__(self, name, view: ImageView, **kwargs):
        super().__init__(name, **kwargs)
        self.view = view
        self.menu = QMenu("ROI")
        copy = QAction("Copy as camera ROI", self.menu)
        copy.triggered.connect(self.copy_as_camera_roi)
        self.menu.addAction(copy)

    def contextMenuEnabled(self):
        return True

    def copy_as_camera_roi(self) -> None:
        head = self.view.frame()[1]
        if head is None:
            message = "No frame yet: the camera ROI comes from the last frame"
        elif (sensor_roi := camera_roi(self, self.view.image_item, head)) is None:
            message = "The ROI is outside the frame"
        else:
            text = "({}, {}, {}, {})".format(*sensor_roi)
            QGuiApplication.clipboard().setText(text)
            message = f"Copied camera ROI (x0, width, y0, height) = {text}"
            if "binning" not in head:
                message += "\nThe frame does not say its binning; taken as 1"
            logger.info("%s from %s", message, self.name)
        QToolTip.showText(QCursor.pos(), message)


class ImageApplet(Applet):
    """An applet drawing the image streams in ``imagestreams`` live, with masks, fit overlays and ROIs.

    A subclass may override :meth:`init_gui`, calling ``super().init_gui()``
    first; ``self.view`` exists from then on.

    Attributes:
        view: The image, its colour scale and the ROIs.
        panel_layout: The applet's top-level layout, holding one widget with the
            image and its data sidebar. ``insertWidget(0, ...)`` puts something
            to the left of the image.
        image_layout: The column holding the image and, below it, the data
            sidebar. ``insertWidget(1, ...)`` puts something directly under the
            image.
        sidebar: The data panel below the image.
        imgstream: The image stream client.
        maskstream: The mask stream client.
        overlayclient: The cloud-fit overlay stream client.
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
    _autoscale = PropertyAttribute("autoscale", True)
    _manual_levels = PropertyAttribute("manual_levels", [])

    def init_gui(self):
        self.imDataDict = {}
        self.image_index = -1

        self.panel_layout = QHBoxLayout()
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.panel_layout)
        image_panel = QWidget()
        self.panel_layout.addWidget(image_panel, 1)
        self.image_layout = QVBoxLayout(image_panel)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView(invert_colours=bool(self._invert_colors))
        self.image_layout.addWidget(self.view, 1)
        self.sidebar = DataSidebar(self._props, self.name)

        self.imgstream = ImageClient("imageviewer")
        self.maskstream = ImageClient("imageviewermask")
        self.overlayclient = DataClient("imageviewer_overlay")
        self.update_subscriptions()
        if self.imgstream.has_new_data():
            imgdata, head = self.imgstream.recv()
        else:
            imgdata = np.ones((100, 100))
        # Autoscaling a flat frame gives levels 0-255, and the colour scale's
        # axis would keep the width of those labels once real frames arrive.
        self.view.show_image(imgdata, levels=(1, 2))
        # show_mask scales a mask by the shape of the one before it, so start
        # from a blank one, unplaced: show_mask on an empty item would blow it up.
        self.view.mask_item.setImage(np.zeros(imgdata.shape))

        for region in self._roilist:
            roi = CameraROI(self.name + "/ROI/" + region, self.view, invertible=True)
            roi.addScaleHandle((1, 1), center=(0, 0))
            self.view.add_roi(roi)
        self.image_layout.addWidget(self.sidebar)
        self.sidebar.setVisible(bool(self._show_sidebar))

        self.view.sig_levels_dragged.connect(self._levels_dragged)
        self.autoscale_action = self.view.add_menu_action(
            "auto scale", self.set_autoscale, checkable=True
        )
        with QtCore.QSignalBlocker(self.autoscale_action):
            self.autoscale_action.setChecked(bool(self._autoscale))
        self.set_autoscale(bool(self._autoscale))
        self.view.add_menu_action(
            "subscriptions", lambda: self.open_subscription_editor()
        )
        self.view.add_menu_action("Image masks", self.subscribe_mask)
        self.view.add_menu_action("Overlays", self.subscribe_overlays)
        self.view.add_menu_action("configure", self.open_config_editor)
        # scroll-wheel selector for the image index when several images are streamed
        self.view.add_menu_action("Image Index", self.index_selector)
        self.view.add_menu_action("Data sidebar", self.toggle_sidebar)
        self.view.add_menu_action(
            "Data sidebar subscriptions", self.sidebar.edit_subscriptions
        )

    def set_autoscale(self, enabled):
        """Scale the colours to each frame, or hold the last manual levels."""
        self._autoscale = enabled
        if not enabled and len(self._manual_levels) == 2:
            self.view.set_levels(*self._manual_levels)

    def _levels_dragged(self, low: float, high: float):
        """Dragging the histogram's levels is a manual scale: record it and stop autoscaling."""
        self._manual_levels = [low, high]
        self.autoscale_action.setChecked(False)

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
            levels = None if self._autoscale else self.view.levels()
            self.view.show_image(imgdata, head, levels=levels)
        self.update_mask()
        self.update_overlays()

    def update_mask(self):
        while self.maskstream.has_new_data():
            msg, head, imgdata = self.maskstream.recv()
            self.view.show_mask(imgdata, head)

    def update_overlays(self):
        """Redraw each overlay stream's contour ellipses from its latest message."""
        latest = {}
        while self.overlayclient.has_new_data(0):
            msg = self.overlayclient.recv()
            if msg is not None and isinstance(msg[1], dict):
                latest[msg[0]] = msg[1]
        for stream, head in latest.items():
            self.view.show_paths(stream, overlay_paths(head, int(self._n_contours)))

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
