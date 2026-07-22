"""Unit tests for the streaming analysis processes in ``pytweezer/analysis/``.

An analysis's real ``__init__`` opens a :class:`Properties` connection and a
pub/sub client, so constructing one here would need live hubs and would leave
non-daemon threads behind. :func:`build` sidesteps that: it allocates the object
without running ``__init__`` and installs fakes for the two attributes
``process`` actually touches -- ``_props`` (which every ``PropertyAttribute``
reads through) and the publishing clients. What's left under test is the
transform itself, which is the part worth testing.
"""

import numpy as np
import pytest


class FakeProps:
    """Stand-in for a :class:`Properties` connection backed by a plain dict."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, name, default=None):
        return self.values.get(name, default)

    def set(self, name, value):
        self.values[name] = value


class RecordingClient:
    """Stand-in for an ``ImageClient``/``DataClient`` that records publishes."""

    def __init__(self):
        self.sent = []

    def send(self, header, data, channel="", **kwargs):
        self.sent.append((header, data, channel))

    def subscribe(self, channel):
        pass


def build(cls, props=None):
    """Return an instance of an analysis class with ZMQ replaced by fakes.

    ``props`` seeds the property values the analysis reads (the same names its
    ``PropertyAttribute`` declarations use). Both ``client`` and ``dataq`` are
    :class:`RecordingClient`s -- analyses that publish through only one of them
    simply leave the other empty.
    """
    obj = object.__new__(cls)
    obj.name = "Analysis/Test/" + cls.__name__
    obj._props = FakeProps(props)
    obj.client = RecordingClient()
    obj.dataq = RecordingClient()
    return obj


# --------------------------------------------------------------------------- #
# image_stats
# --------------------------------------------------------------------------- #

def test_image_stats_publishes_expected_statistics():
    from pytweezer.analysis.image_stats import ImageStats

    stats_analysis = build(ImageStats)
    frame = np.array([[0, 2], [4, 6]], dtype=np.uint16)

    assert stats_analysis.process({"_imgindex": 3}, frame) is None

    (head, data, channel), = stats_analysis.dataq.sent
    assert data is None
    assert head["n"] == 4
    assert head["mean"] == pytest.approx(3.0)
    assert head["sum"] == pytest.approx(12.0)
    assert head["min"] == pytest.approx(0.0)
    assert head["max"] == pytest.approx(6.0)
    assert head["_imgindex"] == 3


def test_image_stats_drops_empty_frame():
    from pytweezer.analysis.image_stats import ImageStats

    stats_analysis = build(ImageStats)
    assert stats_analysis.process({}, np.zeros((0, 0))) is None
    assert stats_analysis.dataq.sent == []


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #

def test_projections_sums_both_axes():
    from pytweezer.analysis.projections import Projections

    proj = build(Projections)
    frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    assert proj.process({}, frame) is None

    channels = [channel for _head, _data, channel in proj.dataq.sent]
    assert channels == ["_x", "_y"]

    x_head, x_data, _ = proj.dataq.sent[0]
    np.testing.assert_allclose(x_data[1], [5, 7, 9])
    _y_head, y_data, _ = proj.dataq.sent[1]
    np.testing.assert_allclose(y_data[1], [6, 15])


def test_projections_ignores_non_2d_data():
    from pytweezer.analysis.projections import Projections

    proj = build(Projections)
    assert proj.process({}, np.arange(5)) is None
    assert proj.dataq.sent == []


# --------------------------------------------------------------------------- #
# gaussian_fit
# --------------------------------------------------------------------------- #

def test_gaussian_fit_recovers_known_peak():
    from pytweezer.analysis.gaussian_fit import GaussianFit, gaussian

    fit = build(GaussianFit, {"method": "lsq"})
    x = np.linspace(0, 40, 200)
    y = gaussian(x, amplitude=5.0, center=17.0, sigma=3.0, offset=1.0)

    head, data = fit.process({}, np.vstack([x, y]))

    assert data is None
    assert head["center"] == pytest.approx(17.0, abs=0.05)
    assert head["sigma"] == pytest.approx(3.0, abs=0.05)
    assert head["success"] is True


# --------------------------------------------------------------------------- #
# roi_slice
# --------------------------------------------------------------------------- #

def test_roi_slice_crops_and_shifts_offset():
    from pytweezer.analysis.roi_slice import RoiSlice

    roi = build(
        RoiSlice,
        {
            "roipaths": ["/Image Monitor/ROI/slice"],
            "/Image Monitor/ROI/slice/pos": [1, 2],
            "/Image Monitor/ROI/slice/size": [2, 3],
        },
    )
    frame = np.arange(48, dtype=np.uint16).reshape(6, 8)

    assert roi.process({}, frame) is None

    (head, data, channel), = roi.client.sent
    assert channel == "_slice"
    np.testing.assert_array_equal(data, frame[2:5, 1:3])
    assert head["_offset"] == [1, 2]
