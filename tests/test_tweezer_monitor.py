import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6 import QtCore

from pytweezer.GUI.viewers.tweezer_monitor import (
    SiteOverlay,
    show_rows_down,
    sites_from_saved,
)

SITES = np.array([[10.0, 12.0], [10.0, 22.0], [30.0, 12.0]])
ROI = {"roi_x0": 50, "roi_y0": 100}
SAVED = {"trap_sites": SITES + [100, 50], "trap_window_size": 5}


def image_view(shape=(64, 48), axis_order="col-major"):
    """An ImageItem placed as ImageDisplay places a camera frame."""
    plot = pg.PlotItem()
    image = pg.ImageItem(np.zeros(shape), axisOrder=axis_order)
    image.setRect(QtCore.QRectF(0, 0, shape[1], shape[0]))
    plot.addItem(image)
    return plot, image


@pytest.mark.parametrize("axis_order", ["col-major", "row-major"])
def test_each_square_sits_on_its_sites_pixel_in_the_frames_roi(qapp, axis_order):
    plot, image = image_view(axis_order=axis_order)
    overlay = SiteOverlay(plot)
    overlay.set_sites(*sites_from_saved(SAVED)[:2])

    overlay.place(image, ROI)

    pixels = []
    for point in overlay.squares.points():
        local = image.mapFromParent(point.pos())
        x, y = int(np.floor(local.x())), int(np.floor(local.y()))
        # Column-major puts data[i, j] at local (i, j); row-major at (j, i).
        pixels.append((x, y) if axis_order == "col-major" else (y, x))
    assert sorted(pixels) == sorted((int(r), int(c)) for r, c in SITES)


def test_the_viewer_draws_rows_down_the_screen_like_the_optimisers_tab(qapp):
    plot, image = image_view()
    show_rows_down(plot, image)
    assert image.axisOrder == "row-major"
    assert plot.getViewBox().yInverted()


def test_a_square_is_the_summing_window_across(qapp):
    plot, image = image_view()
    overlay = SiteOverlay(plot)
    overlay.set_sites(*sites_from_saved(SAVED)[:2])
    overlay.place(image, ROI)
    across = image.mapToParent(QtCore.QPointF(5, 0)) - image.mapToParent(
        QtCore.QPointF(0, 0)
    )
    assert {point.size() for point in overlay.squares.points()} == {
        np.hypot(across.x(), across.y())
    }


def test_a_frame_without_an_roi_origin_is_read_from_the_sensor_corner(qapp):
    plot, image = image_view((160, 160))
    overlay = SiteOverlay(plot)
    overlay.set_sites(*sites_from_saved(SAVED)[:2])
    overlay.place(image, {})
    local = image.mapFromParent(overlay.squares.points()[0].pos())
    assert (int(local.x()), int(local.y())) == (110, 62)


def test_labels_number_the_sites_only_when_asked(qapp):
    plot, image = image_view()
    overlay = SiteOverlay(plot)
    overlay.set_sites(*sites_from_saved(SAVED)[:2])
    overlay.place(image, ROI, labels=False)
    assert overlay.labels == []
    overlay.place(image, ROI, labels=True)
    assert [label.toPlainText() for label in overlay.labels] == ["0", "1", "2"]


def test_without_saved_sites_there_are_no_squares_and_a_message_says_why(qapp):
    plot, image = image_view()
    overlay = SiteOverlay(plot)
    sites, window_size, message = sites_from_saved({})
    overlay.set_sites(sites, window_size, message)
    overlay.place(image, ROI)
    assert len(overlay.squares.points()) == 0
    assert "FindTrapSites" in overlay.message.toPlainText()
