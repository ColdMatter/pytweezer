"""A pan-and-zoom image plot with a colour scale, placing frames in metres."""

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QAction

from pytweezer.analysis.cloud_fit import frame_geometry

PATH_PEN = pg.mkPen("w", width=1)
PATH_Z_VALUE = 10
AXIS_SIZE = 60
MASK_LOOKUP_TABLE = np.array([[0, 0, 255, 0], [0, 255, 255, 255]])


def frame_rect(head: dict, shape: tuple[int, ...]) -> QtCore.QRectF:
    """Where a frame of ``shape`` sits in the plot, from its header's ``_imgresolution`` and ``_offset``."""
    resolution, origin = frame_geometry(head)
    res_x, res_y = resolution or (1.0, 1.0)
    rows, cols = shape[:2]
    return QtCore.QRectF(
        origin[0] * res_x, origin[1] * res_y, cols * res_x, rows * res_y
    )


def _colour_map(name: str, invert: bool) -> pg.ColorMap:
    colours = plt.get_cmap(name)(np.linspace(0.0, 1.0, 256))[:, :3] * 255
    if invert:
        colours = colours[::-1]
    return pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colours)), color=colours)


class ImageView(QtWidgets.QWidget):
    """An image plot with its colour scale, row 0 at the top and axes in metres.

    Frames are placed by their header's ``_imgresolution`` and ``_offset`` (see
    :func:`frame_rect`). The view draws what it is handed and knows nothing of
    where frames come from.

    Args:
        parent: Parent widget.
        colour_map: Name of a matplotlib colour map.
        invert_colours: Reverse the colour map.
        colour_bar: ``"histogram"`` for a draggable histogram of the pixel
            values, ``"bar"`` for a plain colour bar, ``None`` for none.
        units: Axis units.
        title: Plot title.
        row_zero_at_top: Draw row 0 of the frame at the top; ``False`` puts it
            at the bottom, Y increasing upwards.
        vertical_axis: Label of the vertical axis.

    Attributes:
        graphics: The graphics layout holding the plot and its colour scale.
        plot: The image plot, with the aspect locked and Y inverted when
            ``row_zero_at_top``.
        image_item: The frame being shown.
        lut: The colour scale (``None`` without one).
        mask_item: The mask drawn over the frame.
        sig_levels_dragged: ``(low, high)`` when the user drags the histogram's
            levels, never when a frame moves them.
        sig_mouse_moved: The scene position of every mouse move over the graphics.
    """

    sig_levels_dragged = QtCore.pyqtSignal(float, float)
    sig_mouse_moved = QtCore.pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        *,
        colour_map: str = "magma",
        invert_colours: bool = False,
        colour_bar: str | None = "histogram",
        units: str = "m",
        title: str | None = None,
        row_zero_at_top: bool = True,
        vertical_axis: str = "y",
    ):
        super().__init__(parent)
        self._invert_colours = invert_colours
        self._vertical_axis = vertical_axis
        self._units = None
        self._autoscaling_frame = False
        self._image: np.ndarray | None = None
        self._head: dict | None = None
        self._rois: list[pg.ROI] = []
        self._path_curves: dict[str, list[pg.PlotCurveItem]] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)
        self.graphics.scene().sigMouseMoved.connect(self.sig_mouse_moved)

        self.plot: pg.PlotItem = self.graphics.addPlot(row=1, col=1)
        self.set_units(units)
        if title is not None:
            self.plot.setTitle(title)
        view_box = self.plot.getViewBox()
        view_box.invertY(row_zero_at_top)
        view_box.setAspectLocked(True)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)

        self.lut = None
        if colour_bar == "histogram":
            self.lut = pg.HistogramLUTItem()
            self.lut.setImageItem(self.image_item)
            self.graphics.addItem(self.lut, row=1, col=2)
            self.lut.sigLevelChangeFinished.connect(self._levels_dragged)
        elif colour_bar == "bar":
            self.lut = pg.ColorBarItem(interactive=False, width=15, colorMapMenu=False)
            self.lut.setImageItem(self.image_item, insert_in=self.plot)
        elif colour_bar is not None:
            raise ValueError(
                f"colour_bar is 'histogram', 'bar' or None, not {colour_bar!r}"
            )
        self.set_colour_map(colour_map)

        self.mask_item = pg.ImageItem(axisOrder="row-major")
        self.mask_item.setLookupTable(MASK_LOOKUP_TABLE)
        self.plot.addItem(self.mask_item)

    def set_units(self, units: str) -> None:
        """Label both axes in ``units``."""
        if units == self._units:
            return
        self._units = units
        self.plot.setLabel("left", text=self._vertical_axis, units=units)
        self.plot.setLabel("bottom", text="x", units=units)

    def set_colour_map(self, name: str) -> None:
        """Colour the image with the matplotlib colour map ``name``."""
        colour_map = _colour_map(name, self._invert_colours)
        if isinstance(self.lut, pg.HistogramLUTItem):
            # A HistogramLUTItem drives its image's lookup table, so a
            # setLookupTable would be overwritten and the image come out greyscale.
            self.lut.gradient.setColorMap(colour_map)
        elif self.lut is not None:
            self.lut.setColorMap(colour_map)
        else:
            self.image_item.setColorMap(colour_map)

    def show_image(
        self,
        image: np.ndarray,
        head: dict | None = None,
        *,
        levels: tuple[float, float] | None = None,
    ) -> None:
        """Draw ``image`` placed by ``head``; ``levels=None`` scales the colours to it."""
        self._autoscaling_frame = True
        try:
            if levels is None:
                self.image_item.setImage(image, autoLevels=True)
            else:
                self.image_item.setImage(image, autoLevels=False)
                self.set_levels(*levels)
        finally:
            self._autoscaling_frame = False
        self.image_item.resetTransform()
        self.image_item.setRect(frame_rect(head or {}, image.shape))
        self._image = image
        self._head = head

    def clear_image(self) -> None:
        """Remove the frame."""
        self.image_item.clear()
        self._image = None
        self._head = None

    def frame(self) -> tuple[np.ndarray | None, dict | None]:
        """The last frame shown and the very header object it was shown with."""
        return self._image, self._head

    def set_levels(self, low: float, high: float) -> None:
        """Map ``low`` to the bottom of the colour scale and ``high`` to the top."""
        was_autoscaling = self._autoscaling_frame
        self._autoscaling_frame = True
        try:
            if isinstance(self.lut, pg.HistogramLUTItem):
                self.lut.setLevels(low, high)
            elif self.lut is not None:
                self.lut.setLevels((low, high))
            else:
                self.image_item.setLevels((low, high))
        finally:
            self._autoscaling_frame = was_autoscaling

    def levels(self) -> tuple[float, float]:
        """The pixel values at the ends of the colour scale."""
        if isinstance(self.lut, pg.HistogramLUTItem):
            low, high = self.lut.getLevels()
        elif self.lut is not None:
            low, high = self.lut.levels()
        else:
            levels = self.image_item.getLevels()
            low, high = (0.0, 0.0) if levels is None else levels
        return float(low), float(high)

    def set_bar_ticks(self, ticks: list[tuple[float, str]]) -> None:
        """Label the colour scale's axis with ``(value, text)`` ticks."""
        self.lut.axis.setTicks([ticks])

    def _levels_dragged(self) -> None:
        # pyqtgraph also emits sigLevelChangeFinished when setImage moves the levels.
        if not self._autoscaling_frame:
            self.sig_levels_dragged.emit(*self.levels())

    def show_mask(self, mask: np.ndarray, head: dict | None = None) -> None:
        """Draw the 0/1 ``mask`` over the frame, placed by ``head``."""
        self.mask_item.setRect(frame_rect(head or {}, mask.shape))
        self.mask_item.setImage(mask, autoLevels=False, levels=(0, 1))

    def show_paths(self, key: str, paths: list[np.ndarray], pen=None) -> None:
        """Replace the curves under ``key`` with one curve per ``[x, y]`` in ``paths``."""
        curves = self._path_curves.pop(key, [])
        for curve in curves[len(paths) :]:
            self.plot.removeItem(curve)
        curves = curves[: len(paths)]
        for index, (x, y) in enumerate(paths):
            if index == len(curves):
                curves.append(pg.PlotCurveItem())
                curves[index].setZValue(PATH_Z_VALUE)
                self.plot.addItem(curves[index])
            curves[index].setPen(PATH_PEN if pen is None else pen)
            curves[index].setData(x, y)
        if curves:
            self._path_curves[key] = curves

    def add_roi(self, roi: pg.ROI) -> None:
        """Put ``roi`` on the plot."""
        self.plot.addItem(roi)
        self._rois.append(roi)

    def rois(self) -> list[pg.ROI]:
        """The ROIs added with :meth:`add_roi`."""
        return list(self._rois)

    def add_side_plot(self, side: str, size: int = 140) -> pg.PlotItem:
        """Add a plot above (``"top"``) or left of (``"left"``) the image, sharing its axis.

        The top plot follows the image's x-axis and the left plot its y-axis;
        ``size`` is the side plot's height or width.
        """
        if side == "top":
            side_plot = self.graphics.addPlot(row=0, col=1)
            side_plot.setFixedHeight(size)
            side_plot.setXLink(self.plot)
            side_plot.hideAxis("bottom")
            # Equal axis widths keep the top plot's view box above the image's.
            for plot in (self.plot, side_plot):
                plot.getAxis("left").setWidth(AXIS_SIZE)
            self.graphics.ci.layout.setRowStretchFactor(1, 4)
        elif side == "left":
            side_plot = self.graphics.addPlot(row=1, col=0)
            side_plot.setFixedWidth(size)
            side_plot.setYLink(self.plot)
            side_plot.hideAxis("left")
            # A linked view follows the image's range but not its inverted direction.
            side_plot.getViewBox().invertY(self.plot.getViewBox().yInverted())
            for plot in (self.plot, side_plot):
                plot.getAxis("bottom").setHeight(AXIS_SIZE // 2)
            self.graphics.ci.layout.setColumnStretchFactor(1, 4)
        else:
            raise ValueError(f"side is 'top' or 'left', not {side!r}")
        return side_plot

    def add_menu_action(self, text: str, slot, checkable: bool = False) -> QAction:
        """Add ``text`` to the plot's right-click menu, calling ``slot`` (with the state if ``checkable``)."""
        action = self.plot.getViewBox().menu.addAction(text)
        if checkable:
            action.setCheckable(True)
            action.toggled.connect(slot)
        else:
            action.triggered.connect(slot)
        return action
