import numpy as np

from pytweezer.GUI.views.histogram import HistogramPlot


def test_a_histogram_is_a_normalised_density(qapp):
    plot = HistogramPlot("Values")
    curve = plot.add_histogram("#5b8def")

    HistogramPlot.draw(curve, np.array([0.5, 1.5, 1.5, 3.5]), 4, (0, 4))

    edges, density = curve.getData()
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(density, [0.25, 0.5, 0, 0.25])


def test_clearing_removes_only_the_histograms(qapp):
    plot = HistogramPlot("Values")
    plot.add_histogram("#5b8def")
    plot.add_histogram("#f0a35e", filled=True)
    plot.plot([0, 1], [0, 1])

    plot.clear_histograms()

    assert len(plot.listDataItems()) == 1
