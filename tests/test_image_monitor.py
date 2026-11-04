import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6 import QtCore

from pytweezer.GUI.viewers.image_monitor import camera_roi

HEAD = {"roi_x0": 100, "roi_y0": 40, "binning": 2}


def framed_roi(rows, cols, axis_order, shape=(64, 48)):
    """A frame placed as ImageDisplay places it, and an ROI over ``rows`` x ``cols`` of it."""
    plot = pg.PlotItem()
    image = pg.ImageItem(np.zeros(shape), axisOrder=axis_order)
    # ImageDisplay stretches the frame into a (cols, rows) rectangle whatever the axis order.
    image.setRect(QtCore.QRectF(0, 0, shape[1], shape[0]))
    plot.addItem(image)
    if axis_order == "col-major":
        corner, extent = (rows[0], cols[0]), (rows[1] - rows[0], cols[1] - cols[0])
    else:
        corner, extent = (cols[0], rows[0]), (cols[1] - cols[0], rows[1] - rows[0])
    roi = pg.ROI((0, 0))
    plot.addItem(roi)
    roi.setPos(image.mapToParent(QtCore.QPointF(*corner)))
    far = image.mapToParent(
        QtCore.QPointF(corner[0] + extent[0], corner[1] + extent[1])
    )
    roi.setSize(far - roi.pos())
    return plot, roi, image


@pytest.mark.parametrize("axis_order", ["col-major", "row-major"])
def test_the_camera_roi_reads_the_frame_pixels_under_the_roi(qapp, axis_order):
    _plot, roi, image = framed_roi(rows=(10, 30), cols=(5, 21), axis_order=axis_order)
    # Frame pixels are binned by 2 and start at sensor pixel (100, 40).
    assert camera_roi(roi, image, HEAD) == (110, 32, 60, 40)


def test_the_camera_roi_stops_at_the_frame_edge(qapp):
    _plot, roi, image = framed_roi(rows=(50, 90), cols=(-8, 10), axis_order="col-major")
    assert camera_roi(roi, image, {}) == (0, 10, 50, 14)


def test_an_roi_off_the_frame_has_no_camera_roi(qapp):
    _plot, roi, image = framed_roi(rows=(70, 90), cols=(5, 10), axis_order="col-major")
    assert camera_roi(roi, image, HEAD) is None
