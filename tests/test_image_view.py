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

    view.lut.setLevels(1, 2)
    view.lut.sigLevelChangeFinished.emit(view.lut)

    assert dragged[-1] == (1, 2)


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

    assert len(view._path_curves["a"]) == 1
    curves = [item for item in view.plot.items if isinstance(item, pg.PlotCurveItem)]
    assert len(curves) == 1


def test_a_side_plot_shares_the_image_axis(view):
    top = view.add_side_plot("top")
    left = view.add_side_plot("left")

    assert top.getViewBox().linkedView(pg.ViewBox.XAxis) is view.plot.getViewBox()
    assert left.getViewBox().linkedView(pg.ViewBox.YAxis) is view.plot.getViewBox()
    assert left.getViewBox().yInverted()
