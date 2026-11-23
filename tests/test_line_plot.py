import numpy as np

from pytweezer.GUI.views.line_plot import LinePlot

COLOUR = "#5b8def"


def test_curve_is_reused_by_key(qapp):
    plot = LinePlot()

    assert plot.curve("a", COLOUR) is plot.curve("a", COLOUR)


def test_set_curve_draws_the_data(qapp):
    plot = LinePlot()
    plot.curve("a", COLOUR)

    plot.set_curve("a", [0.0, 1.0, 2.0], [3.0, 5.0, 4.0])

    x, y = plot.curve("a", COLOUR).getData()
    np.testing.assert_array_equal(x, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y, [3.0, 5.0, 4.0])


def test_markers_draw_points_not_lines(qapp):
    plot = LinePlot()

    curve = plot.curve("a", COLOUR, markers=True)

    assert curve.opts["symbol"] == "o"
    assert curve.opts["pen"] is None


def test_remove_curve_forgets_the_key(qapp):
    plot = LinePlot()
    first = plot.curve("a", COLOUR)

    plot.remove_curve("a")

    assert plot.curve("a", COLOUR) is not first
    assert first not in plot.listDataItems()
