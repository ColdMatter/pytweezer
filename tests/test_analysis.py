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

    ((head, data, channel),) = stats_analysis.dataq.sent
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

    ((head, data, channel),) = roi.client.sent
    assert channel == "_slice"
    np.testing.assert_array_equal(data, frame[2:5, 1:3])
    assert head["_offset"] == [1, 2]


# --------------------------------------------------------------------------- #
# tweezer_loading
# --------------------------------------------------------------------------- #

SITE_ROWS, SITE_COLS = np.meshgrid(np.arange(14, 60, 12), np.arange(14, 60, 12))
SITE_CENTRES = list(zip(SITE_ROWS.ravel(), SITE_COLS.ravel(), strict=True))
# Site i is full in the shots where (3 * shot + i) % 10 < LOADED_IN_TEN[i], so
# over any multiple of ten shots it loads exactly LOADED_IN_TEN[i] / 10.
LOADED_IN_TEN = [(2 + i) % 9 + 1 for i in range(len(SITE_CENTRES))]


def tweezer_frame(occupied, shape=(72, 72), seed=0):
    """Camera-like frame: offset, broad glow and noise, plus a spot at each occupied site."""
    rng = np.random.default_rng(seed)
    rows, cols = np.indices(shape)
    frame = 2000.0 + 300.0 * np.exp(-((rows - 36) ** 2 + (cols - 36) ** 2) / 800.0)
    frame += rng.normal(0.0, 10.0, shape)
    for row, col in occupied:
        frame += 400.0 * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / 3.0)
    return frame.astype(np.uint16)


def patterned_frame(shot):
    occupied = [
        centre
        for i, centre in enumerate(SITE_CENTRES)
        if (3 * shot + i) % 10 < LOADED_IN_TEN[i]
    ]
    return tweezer_frame(occupied, seed=shot)


def loading_window(frames, shots=100):
    from pytweezer.analysis.tweezer_loading import LoadingStatistics

    statistics = LoadingStatistics()
    for frame in frames:
        statistics.add(frame, shots=shots, feature_size=10)
    return statistics.summary(
        n_sites=len(SITE_CENTRES),
        window_size=5,
        detection_step=10,
        fallback_threshold=0.0,
    )


def test_tweezer_loading_counts_each_site_as_often_as_it_was_full():
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))

    # Each found site is credited to the nearest true centre, so the test does
    # not depend on how the site finder rounds a spot's position.
    found = np.column_stack([head["site_rows"], head["site_cols"]])
    loading = np.zeros(len(SITE_CENTRES))
    for (row, col), site_loading in zip(found, head["site_loading"], strict=True):
        distances = np.hypot(*(np.array(SITE_CENTRES) - (row, col)).T)
        assert distances.min() <= 1.5
        loading[distances.argmin()] = site_loading
    assert loading == pytest.approx(np.array(LOADED_IN_TEN) / 10)
    assert photons.shape == (40, len(SITE_CENTRES))


def test_tweezer_loading_threshold_separates_full_sites_from_empty_ones():
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))

    full = photons > head["threshold"]
    assert full.sum() == sum(LOADED_IN_TEN) * 4
    assert head["fidelity"] > 0.99


def test_tweezer_loading_scores_only_the_last_shots_in_the_window():
    all_full = [tweezer_frame(SITE_CENTRES, seed=100 + i) for i in range(20)]
    patterned = [patterned_frame(shot) for shot in range(20)]

    head, _ = loading_window(all_full + patterned, shots=20)

    assert head["shots"] == 20
    assert head["loading"] == pytest.approx(sum(LOADED_IN_TEN) / 10 / len(SITE_CENTRES))


def test_tweezer_loading_publishes_no_photons_until_it_finds_the_sites():
    from pytweezer.analysis.tweezer_loading import LoadingStatistics, TweezerLoading

    analysis = build(TweezerLoading, {"sites": len(SITE_CENTRES), "detection_step": 10})
    analysis.statistics = LoadingStatistics()

    analysis.process({"index": 0}, tweezer_frame([]))

    ((head, data, _),) = analysis.dataq.sent
    assert data is None
    assert head == {"shots": 1, "sites": len(SITE_CENTRES)}


def test_tweezer_loading_ignores_the_second_image_of_a_shot():
    from pytweezer.analysis.tweezer_loading import LoadingStatistics, TweezerLoading

    analysis = build(TweezerLoading, {"image_index": 0})
    analysis.statistics = LoadingStatistics()

    analysis.process({"index": 1}, tweezer_frame(SITE_CENTRES))

    assert analysis.dataq.sent == []
    assert len(analysis.statistics.frames) == 0


def test_tweezer_loading_restarts_the_window_when_the_frame_size_changes():
    from pytweezer.analysis.tweezer_loading import LoadingStatistics

    statistics = LoadingStatistics()
    for shot in range(5):
        statistics.add(patterned_frame(shot), shots=100, feature_size=10)
    statistics.add(np.zeros((40, 40), dtype=np.uint16), shots=100, feature_size=10)

    assert len(statistics.frames) == 1
