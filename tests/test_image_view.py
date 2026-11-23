import numpy as np
import pyqtgraph as pg
import pytest

from pytweezer.GUI.views.image import ImageView, frame_rect


@pytest.fixture
def view(qapp):
    return ImageView()


def test_a_frame_is_placed_by_its_resolution_and_offset(view):
    head = {"_imgresolution": (2e-6, 3e-6), "_offset": (4, 5)}
    image = np.ones((6, 8))

    view.show_image(image, head)

    assert view.image_item.mapRectToParent(view.image_item.boundingRect()) == (
        frame_rect(head, image.shape)
    )


def test_frame_returns_the_head_it_was_given(view):
    head = {"index": 1}
    image = np.ones((4, 4))

    view.show_image(image, head)

    assert view.frame()[0] is image
    assert view.frame()[1] is head


def test_showing_a_frame_does_not_count_as_a_drag(view):
    dragged = []
    view.sig_levels_dragged.connect(lambda low, high: dragged.append((low, high)))

    view.show_image(np.arange(16.0).reshape(4, 4))
    view.show_image(np.arange(16.0).reshape(4, 4) * 10)

    assert dragged == []


def test_dragging_the_levels_reports_them(view):
    dragged = []
    view.sig_levels_dragged.connect(lambda low, high: dragged.append((low, high)))
    view.show_image(np.arange(16.0).reshape(4, 4))

    view.lut.region.setRegion((1, 2))

    assert dragged == [(1, 2)]


def test_setting_the_levels_does_not_count_as_a_drag(view):
    dragged = []
    view.sig_levels_dragged.connect(lambda low, high: dragged.append((low, high)))
    view.show_image(np.arange(16.0).reshape(4, 4))

    view.set_levels(1, 2)

    assert view.levels() == (1, 2)
    assert dragged == []


def test_fixed_levels_are_kept(view):
    view.show_image(np.arange(16.0).reshape(4, 4), levels=(0, 5))

    assert view.levels() == (0, 5)


def test_a_colour_bar_keeps_fixed_levels(qapp):
    view = ImageView(colour_bar="bar")

    view.show_image(np.arange(16.0).reshape(4, 4), levels=(0, 5))

    assert view.levels() == (0, 5)


def test_paths_replace_by_key(view):
    path = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])

    view.show_paths("a", [path, path])
    view.show_paths("a", [path])

    curves = [item for item in view.plot.items if isinstance(item, pg.PlotCurveItem)]
    assert len(curves) == 1


def test_a_side_plot_shares_the_image_axis(view):
    top = view.add_side_plot("top")
    left = view.add_side_plot("left")

    assert top.getViewBox().linkedView(pg.ViewBox.XAxis) is view.plot.getViewBox()
    assert left.getViewBox().linkedView(pg.ViewBox.YAxis) is view.plot.getViewBox()
    assert left.getViewBox().yInverted()


def test_units_relabel_both_axes(view):
    view.set_units("px")

    assert view.plot.getAxis("bottom").labelUnits == "px"
    assert view.plot.getAxis("left").labelUnits == "px"


def test_row_zero_can_sit_at_the_bottom(qapp):
    view = ImageView(row_zero_at_top=False, vertical_axis="z")

    assert not view.plot.getViewBox().yInverted()
    assert view.plot.getAxis("left").labelText == "z"


@pytest.mark.parametrize("levels", [None, (0, 5)])
def test_levels_without_a_colour_bar_are_the_images(qapp, levels):
    view = ImageView(colour_bar=None)
    frame = np.arange(16.0).reshape(4, 4)

    view.show_image(frame, levels=levels)

    assert view.levels() == (levels or (frame.min(), frame.max()))


def test_the_left_side_plot_follows_the_image_direction(qapp):
    view = ImageView(row_zero_at_top=False)

    left = view.add_side_plot("left")

    assert not left.getViewBox().yInverted()


def test_a_mask_is_placed_for_its_own_shape_whatever_was_shown_before(view):
    view.show_mask(np.zeros((100, 100)))
    head = {"_imgresolution": (2e-6, 3e-6)}
    mask = np.ones((6, 8))

    view.show_mask(mask, head)

    assert view.mask_item.mapRectToParent(view.mask_item.boundingRect()) == (
        frame_rect(head, mask.shape)
    )
