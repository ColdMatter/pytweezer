import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6 import QtCore

from pytweezer.GUI.viewers import tweezer_monitor
from pytweezer.GUI.viewers.tweezer_monitor import (
    SiteOverlay,
    sites_from_saved,
)
from tests.test_image_monitor import fake_hubs

SITES = np.array([[10.0, 12.0], [10.0, 22.0], [30.0, 12.0]])
ROI = {"roi_x0": 50, "roi_y0": 100}
SAVED = {"trap_sites": SITES + [100, 50], "trap_window_size": 5}


def image_view(shape=(64, 48)):
    """An ImageItem placed as ImageDisplay places a camera frame."""
    plot = pg.PlotItem()
    image = pg.ImageItem(np.zeros(shape), axisOrder="row-major")
    image.setRect(QtCore.QRectF(0, 0, shape[1], shape[0]))
    plot.addItem(image)
    return plot, image


def test_each_square_sits_on_its_sites_pixel_in_the_frames_roi(qapp):
    plot, image = image_view()
    overlay = SiteOverlay(plot)
    overlay.set_sites(*sites_from_saved(SAVED)[:2])

    overlay.place(image, ROI)

    pixels = []
    for point in overlay.squares.points():
        local = image.mapFromParent(point.pos())
        # Row-major puts data[row, col] at local (col, row).
        pixels.append((int(np.floor(local.y())), int(np.floor(local.x()))))
    assert sorted(pixels) == sorted((int(r), int(c)) for r, c in SITES)


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
    assert (int(local.y()), int(local.x())) == (110, 62)


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


class FakeFeed(QtCore.QObject):
    queue_changed = QtCore.pyqtSignal(dict)

    def close(self):
        pass


@pytest.fixture
def monitor(qapp, monkeypatch):
    """A Tweezer Monitor with no hubs, Properties or Experiment Manager behind it."""
    fake_hubs(monkeypatch)
    monkeypatch.setattr(tweezer_monitor, "ExperimentFeed", FakeFeed)
    monkeypatch.setattr(tweezer_monitor, "get_saved_parameters", lambda: SAVED)
    return tweezer_monitor.TweezerDisplay("Test Tweezer Monitor")


ATOMS = np.full((64, 48), 100.0)
ATOMS[10, 12] = ATOMS[30, 12] = 900.0
BACKGROUND = np.full((64, 48), 100.0)
BACKGROUND[0, 0] = 110.0


def broadcast_shot(monitor, *frames, **head):
    for index, frame in enumerate(frames):
        monitor.imgstream.queue.append(("cam", ROI | head | {"index": index}, frame))
    monitor.poll()


def test_a_shots_two_images_are_shown_side_by_side(monitor):
    broadcast_shot(monitor, ATOMS, BACKGROUND, rid=7)

    np.testing.assert_array_equal(monitor.image_item.image, ATOMS)
    np.testing.assert_array_equal(monitor.second_item.image, BACKGROUND)


def test_the_second_image_is_drawn_at_the_first_images_levels(monitor):
    broadcast_shot(monitor, ATOMS, BACKGROUND, rid=7)
    assert monitor.second_item.getLevels() == pytest.approx(
        monitor.image_item.getLevels()
    )

    monitor.lut.setLevels(0.0, 2000.0)

    assert monitor.second_item.getLevels() == pytest.approx([0.0, 2000.0])


def test_live_frames_outside_a_shot_all_go_to_the_first_image(monitor):
    broadcast_shot(monitor, ATOMS, BACKGROUND)

    np.testing.assert_array_equal(monitor.image_item.image, BACKGROUND)
    assert monitor.second_item.image is None


def test_both_images_pan_and_zoom_together_with_the_sites_on_each(monitor):
    broadcast_shot(monitor, ATOMS, BACKGROUND, rid=7)
    monitor.plot.getViewBox().setRange(xRange=(5, 25), yRange=(5, 35), padding=0)

    (x_range, y_range) = monitor.second_plot.getViewBox().viewRange()
    assert x_range == pytest.approx([5, 25], abs=1)
    assert len(monitor.second_overlay.squares.points()) == len(SITES)
