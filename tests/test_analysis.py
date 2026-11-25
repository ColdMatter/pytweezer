"""Unit tests for the streaming analysis processes in ``pytweezer/analyses/live/``.

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
# analysis_base
# --------------------------------------------------------------------------- #


class _StreamEnded(Exception):
    pass


def test_data_analysis_handles_header_only_messages():
    """Scalar streams (e.g. ImageStats' output) carry a header and no array."""
    from pytweezer.analysis.analysis_base import DataAnalysis

    class Echo(DataAnalysis):
        def process(self, head, data):
            return head, data

    messages = iter([("Stats", {"mean": 3.0}), ("Trace", {"n": 2}, np.arange(2))])

    def recv():
        try:
            return next(messages)
        except StopIteration:
            raise _StreamEnded from None

    analysis = build(Echo)
    analysis.client.recv = recv
    with pytest.raises(_StreamEnded):
        analysis.run()

    (scalar_head, scalar_data, _), (trace_head, trace_data, _) = analysis.client.sent
    assert (scalar_head, scalar_data) == ({"mean": 3.0}, None)
    assert trace_head == {"n": 2}
    np.testing.assert_array_equal(trace_data, np.arange(2))


# --------------------------------------------------------------------------- #
# image_stats
# --------------------------------------------------------------------------- #


def test_image_stats_publishes_expected_statistics():
    from pytweezer.analyses.live.image_stats import ImageStats

    stats_analysis = build(ImageStats)
    frame = np.array([[0, 2], [4, 6]], dtype=np.uint16)

    assert stats_analysis.process({"index": 3, "rid": 7}, frame) is None

    ((head, data, channel),) = stats_analysis.dataq.sent
    assert data is None
    assert head["n"] == 4
    assert head["mean"] == pytest.approx(3.0)
    assert head["sum"] == pytest.approx(12.0)
    assert head["min"] == pytest.approx(0.0)
    assert head["max"] == pytest.approx(6.0)
    assert head["index"] == 3
    assert head["rid"] == 7


def test_image_selector_passes_only_the_chosen_image_of_each_shot():
    from pytweezer.analyses.live.imageselector import ImageSelector

    selector = build(ImageSelector, {"SelectedImage": 1})
    frame = np.ones((2, 2))
    assert selector.process({"index": 0}, frame) is None
    assert selector.process({"index": 1}, frame) == ({"index": 1}, frame)
    assert selector.process({"timestamp": 1.0}, frame) is None


def test_data_average_publishes_the_mean_of_each_shots_frames():
    from pytweezer.analyses.live.data_average import DataAverage

    messages = iter(
        [
            ("Trace", {"index": 0}, np.array([1.0, 2.0])),
            ("Trace", {"index": 1}, np.array([3.0, 6.0])),
            ("Trace", {"timestamp": 1.0}, np.array([9.0, 9.0])),
        ]
    )

    def recv():
        try:
            return next(messages)
        except StopIteration:
            raise _StreamEnded from None

    average = build(DataAverage, {"n_average": 2})
    average.dataq.has_new_data = lambda: True
    average.dataq.recv = recv
    with pytest.raises(_StreamEnded):
        average.run()

    ((head, data, _),) = average.dataq.sent
    assert head == {"index": 1}
    np.testing.assert_array_equal(data, [2.0, 4.0])


def test_image_stats_drops_empty_frame():
    from pytweezer.analyses.live.image_stats import ImageStats

    stats_analysis = build(ImageStats)
    assert stats_analysis.process({}, np.zeros((0, 0))) is None
    assert stats_analysis.dataq.sent == []


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #


def test_projections_sums_both_axes():
    from pytweezer.analyses.live.projections import Projections

    proj = build(Projections)
    frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    assert proj.process({}, frame) is None

    channels = [channel for _head, _data, channel in proj.dataq.sent]
    assert channels == ["_x", "_y"]

    x_head, x_data, _ = proj.dataq.sent[0]
    np.testing.assert_allclose(x_data[1], [5, 7, 9])
    _y_head, y_data, _ = proj.dataq.sent[1]
    np.testing.assert_allclose(y_data[1], [6, 15])


def gaussian_frame(shape, x0, y0, sigma, amplitude, offset=5.0, seed=0):
    """Frame whose peak sits on pixel indices ``(x0, y0)``, with Poisson noise."""
    rows, cols = np.indices(shape)
    model = offset + amplitude * np.exp(
        -0.5 * ((cols - x0) ** 2 + (rows - y0) ** 2) / sigma**2
    )
    return np.random.default_rng(seed).poisson(model).astype(np.uint16)


def test_projections_ignores_non_2d_data():
    from pytweezer.analyses.live.projections import Projections

    proj = build(Projections)
    assert proj.process({}, np.arange(5)) is None
    assert proj.dataq.sent == []


def test_projection_fit_curve_matches_sums_of_a_gaussian():
    from pytweezer.analyses.live.projections import Projections

    resolution, origin = 2e-6, (30.0, 10.0)
    frame = gaussian_frame((60, 90), x0=40.0, y0=25.0, sigma=6.0, amplitude=200.0)
    head = {"_imgresolution": [resolution] * 2, "_offset": list(origin)}

    proj = build(Projections)
    proj.process(head, frame)

    (x_head, x_data, x_channel), (y_head, y_data, _) = proj.dataq.sent
    assert x_channel == "_x"
    assert x_data.shape == (3, 90)
    np.testing.assert_allclose(
        x_data[0], (origin[0] + np.arange(90) + 0.5) * resolution
    )
    np.testing.assert_allclose(x_data[2], x_data[1], atol=0.03 * x_data[1].max())
    assert x_head["success"] is True
    quarter_pixel = 0.25 * resolution
    assert x_head["center"] == pytest.approx(
        (origin[0] + 40.5) * resolution, abs=quarter_pixel
    )
    assert y_head["center"] == pytest.approx(
        (origin[1] + 25.5) * resolution, abs=quarter_pixel
    )
    assert y_data[0, 0] == pytest.approx((origin[1] + 0.5) * resolution)


def test_projection_of_a_flat_noisy_frame_publishes_no_fit_curve():
    from pytweezer.analyses.live.projections import Projections

    frame = np.random.default_rng(3).poisson(1000.0, (60, 90)).astype(np.uint16)

    proj = build(Projections)
    proj.process({}, frame)

    for head, data, _channel in proj.dataq.sent:
        assert head["success"] is False
        assert np.isnan(data[2]).all()
        assert all(
            np.isnan(head[key]) for key in ("center", "sigma", "amplitude", "offset")
        )


# --------------------------------------------------------------------------- #
# gaussian_fit
# --------------------------------------------------------------------------- #


def test_gaussian_fit_recovers_known_peak():
    from pytweezer.analyses.live.gaussian_fit import GaussianFit, gaussian

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
    from pytweezer.analyses.live.roi_slice import RoiSlice

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
    from pytweezer.analyses.live.tweezer_loading import LoadingStatistics

    statistics = LoadingStatistics()
    for frame in frames:
        statistics.add(frame, shots=shots, feature_size=10)
    return statistics.summary(
        sites=dict(enumerate(SITE_CENTRES)), window_size=5, fallback_threshold=0.0
    )


def filtered_patterned_frames(n_shots):
    from pytweezer.analysis import analysis as an

    return [
        an.morphological_tophat_high_pass(
            patterned_frame(shot).astype(np.float64), feature_size=10
        )
        for shot in range(n_shots)
    ]


def notebook_statistics(frames, sites, threshold):
    from pytweezer.analysis import analysis as an

    return an.get_array_loading_statistics(
        frames,
        sites,
        threshold=threshold,
        window_size=5,
        binning=60,
        show_histogram=False,
        threshold_detection=True,
        verbose=False,
    )


def test_loading_statistics_match_the_notebook_analysis():
    from pytweezer.analysis.array_loading import (
        find_sites,
        loading_statistics,
        photons_at_sites,
    )

    frames = filtered_patterned_frames(40)
    sites = find_sites(np.mean(frames, axis=0), len(SITE_CENTRES), 10)
    photons = photons_at_sites(frames, sites, window_size=5)
    head = loading_statistics(photons, sites, fallback_threshold=0.0)

    expected_photons, expected_loading, expected_threshold, expected_fidelity = (
        notebook_statistics(frames, sites, threshold=0.0)
    )
    assert photons == pytest.approx(expected_photons)
    assert head["threshold"] == pytest.approx(expected_threshold, rel=1e-3)
    assert head["fidelity"] == pytest.approx(expected_fidelity, rel=1e-3)
    assert head["site_loading"] == pytest.approx(list(expected_loading))


def test_loading_statistics_fall_back_like_the_notebook_analysis(monkeypatch):
    from pytweezer.analysis import analysis as an
    from pytweezer.analysis.array_loading import (
        find_sites,
        loading_statistics,
        photons_at_sites,
    )

    def no_fit(counts):
        raise ValueError("no fit")

    monkeypatch.setattr(an, "detect_loading_threshold", no_fit)
    frames = filtered_patterned_frames(40)
    sites = find_sites(np.mean(frames, axis=0), len(SITE_CENTRES), 10)
    photons = photons_at_sites(frames, sites, window_size=5)
    midpoint = float((photons.min() + photons.max()) / 2)

    head = loading_statistics(photons, sites, fallback_threshold=midpoint)

    _, expected_loading, _, expected_fidelity = notebook_statistics(
        frames, sites, threshold=midpoint
    )
    assert head["threshold"] == midpoint
    assert head["background"] is None
    assert head["signal"] is None
    assert head["fidelity"] == pytest.approx(expected_fidelity)
    assert head["site_loading"] == pytest.approx(list(expected_loading))


def test_tweezer_loading_counts_each_site_as_often_as_it_was_full():
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))

    assert head["site_loading"] == pytest.approx(np.array(LOADED_IN_TEN) / 10)
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


ROI_ORIGIN = {"roi_x0": 50, "roi_y0": 100}


def saved_sites(**changes):
    from pytweezer.analysis.trap_sites import default_geometry

    return {
        "trap_sites": np.array(SITE_CENTRES, dtype=float)
        + [ROI_ORIGIN["roi_y0"], ROI_ORIGIN["roi_x0"]],
        "trap_threshold": 0.0,
        "trap_window_size": 5,
        "trap_feature_size": 10,
        "trap_geometry": default_geometry(4, 4),
    } | changes


def loading_analysis(monkeypatch, saved, **properties):
    from pytweezer.analyses.live import tweezer_loading
    from pytweezer.analyses.live.tweezer_loading import (
        LoadingStatistics,
        TweezerLoading,
    )

    monkeypatch.setattr(tweezer_loading, "get_saved_parameters", lambda: saved)
    analysis = build(TweezerLoading, {"rig": ""} | properties)
    analysis.statistics = LoadingStatistics()
    return analysis


def test_tweezer_loading_scores_the_saved_sites_in_the_frames_roi(monkeypatch):
    analysis = loading_analysis(monkeypatch, saved_sites())

    for shot in range(20):
        analysis.process({"index": 0, "rid": 4, **ROI_ORIGIN}, patterned_frame(shot))

    head, photons, _ = analysis.dataq.sent[-1]
    assert (head["site_rows"], head["site_cols"]) == (
        [row for row, _ in SITE_CENTRES],
        [col for _, col in SITE_CENTRES],
    )
    assert head["site_loading"] == pytest.approx(np.array(LOADED_IN_TEN) / 10)
    assert photons.shape == (20, len(SITE_CENTRES))


def test_tweezer_loading_publishes_no_photons_without_saved_sites(monkeypatch):
    analysis = loading_analysis(monkeypatch, {})

    analysis.process({"index": 0, "rid": 4}, tweezer_frame([]))

    ((head, data, _),) = analysis.dataq.sent
    assert data is None
    assert head["rid"] == 4
    assert "FindTrapSites" in head["error"]


def test_tweezer_loading_publishes_no_photons_for_sites_outside_the_frame(
    monkeypatch,
):
    analysis = loading_analysis(monkeypatch, saved_sites())

    analysis.process({"index": 0, "roi_x0": 0, "roi_y0": 0}, tweezer_frame([]))

    ((head, data, _),) = analysis.dataq.sent
    assert data is None
    assert "outside" in head["error"]


def test_tweezer_loading_ignores_the_second_image_of_a_shot():
    from pytweezer.analyses.live.tweezer_loading import (
        LoadingStatistics,
        TweezerLoading,
    )

    analysis = build(TweezerLoading, {"image_index": 0})
    analysis.statistics = LoadingStatistics()

    analysis.process({"index": 1}, tweezer_frame(SITE_CENTRES))

    assert analysis.dataq.sent == []
    assert len(analysis.statistics.frames) == 0


def test_tweezer_loading_restarts_the_window_when_the_frame_size_changes():
    from pytweezer.analyses.live.tweezer_loading import LoadingStatistics

    statistics = LoadingStatistics()
    for shot in range(5):
        statistics.add(patterned_frame(shot), shots=100, feature_size=10)
    statistics.add(np.zeros((40, 40), dtype=np.uint16), shots=100, feature_size=10)

    assert len(statistics.frames) == 1


class ArrayRig:
    def __init__(self, array):
        self.array = array

    def get_array(self):
        return self.array


def unreachable_rig(name, timeout):
    raise ConnectionRefusedError(name)


@pytest.mark.parametrize(
    "get_rig, match",
    [
        (lambda name, timeout: ArrayRig(None), True),
        (unreachable_rig, True),
        (
            lambda name, timeout: ArrayRig(
                {
                    "x_um": saved_sites()["trap_geometry"][:, 0],
                    "y_um": saved_sites()["trap_geometry"][:, 1],
                }
            ),
            True,
        ),
        (
            lambda name, timeout: ArrayRig(
                {
                    "x_um": saved_sites()["trap_geometry"][:, 0] + 1.0,
                    "y_um": saved_sites()["trap_geometry"][:, 1],
                }
            ),
            False,
        ),
    ],
    ids=["rig does not know", "rig unreachable", "same array", "another array"],
)
def test_tweezer_loading_says_whether_the_sites_fit_the_array_on_the_rig(
    monkeypatch, get_rig, match
):
    from pytweezer.analyses.live import tweezer_loading

    monkeypatch.setattr(tweezer_loading, "get_device", get_rig)
    analysis = loading_analysis(monkeypatch, saved_sites(), rig="Rb Rearrangement Rig")

    for shot in range(5):
        analysis.process({"index": 0, **ROI_ORIGIN}, patterned_frame(shot))

    head, _, _ = analysis.dataq.sent[-1]
    assert head["sites_match"] is match


def test_tweezer_loading_starts_afresh_when_a_new_task_sends_frames(monkeypatch):
    analysis = loading_analysis(monkeypatch, saved_sites(), shots=100)

    for shot in range(20):
        analysis.process(
            {"index": 0, "rid": 1, **ROI_ORIGIN},
            tweezer_frame(SITE_CENTRES, seed=100 + shot),
        )
    for shot in range(20):
        analysis.process({"index": 0, "rid": 2, **ROI_ORIGIN}, patterned_frame(shot))

    head, _, _ = analysis.dataq.sent[-1]
    assert (head["rid"], head["shots"]) == (2, 20)
    assert head["loading"] == pytest.approx(sum(LOADED_IN_TEN) / 10 / len(SITE_CENTRES))


# --------------------------------------------------------------------------- #
# tweezer_counts
# --------------------------------------------------------------------------- #


def counts_analysis(monkeypatch, saved, **properties):
    from pytweezer.analyses.live import tweezer_counts
    from pytweezer.analyses.live.tweezer_counts import TweezerCounts

    monkeypatch.setattr(tweezer_counts, "get_saved_parameters", lambda: saved)
    return build(TweezerCounts, properties)


def test_tweezer_counts_splits_the_frame_into_site_boxes_and_the_rest(monkeypatch):
    analysis = counts_analysis(monkeypatch, saved_sites(trap_window_size=3))
    frame = np.ones((72, 72), dtype=np.uint16)
    row, col = SITE_CENTRES[0]
    frame[row, col] = 101
    frame[0, 0] = 51

    analysis.process({"index": 0, "rid": 4, **ROI_ORIGIN}, frame)

    ((head, data, _),) = analysis.dataq.sent
    box_pixels = 9 * len(SITE_CENTRES)
    assert data is None
    assert head["atom_pixels"] == box_pixels
    assert head["atom_counts"] == box_pixels + 100
    assert head["background_pixels"] == 72 * 72 - box_pixels
    assert head["background_counts"] == 72 * 72 - box_pixels + 50
    assert head["rid"] == 4


def test_tweezer_counts_counts_overlapping_and_clipped_boxes_once(monkeypatch):
    saved = {"trap_sites": [[0, 0], [0, 1]], "trap_window_size": 3}
    analysis = counts_analysis(monkeypatch, saved)

    analysis.process({}, np.ones((10, 10)))

    ((head, _, _),) = analysis.dataq.sent
    assert head["atom_pixels"] == 2 * 3
    assert head["atom_counts"] + head["background_counts"] == 100


def test_tweezer_counts_skips_other_frames_of_the_shot(monkeypatch):
    analysis = counts_analysis(monkeypatch, saved_sites(), image_index=0)

    analysis.process({"index": 1}, np.ones((40, 40)))

    assert analysis.dataq.sent == []


def test_tweezer_counts_publishes_nan_counts_without_saved_sites(monkeypatch):
    analysis = counts_analysis(monkeypatch, {})

    analysis.process({"rid": 4}, np.ones((40, 40)))

    ((head, _, _),) = analysis.dataq.sent
    assert np.isnan(head["atom_counts"]) and np.isnan(head["background_counts"])
    assert head["error"]


# --------------------------------------------------------------------------- #
# tweezer_occupancy_counts
# --------------------------------------------------------------------------- #


def occupancy_counts_analysis(monkeypatch, saved):
    from pytweezer.analyses.live import tweezer_occupancy_counts
    from pytweezer.analyses.live.tweezer_occupancy_counts import TweezerOccupancyCounts

    monkeypatch.setattr(tweezer_occupancy_counts, "get_saved_parameters", lambda: saved)
    return build(TweezerOccupancyCounts)


def test_tweezer_occupancy_counts_sums_raw_counts_of_occupied_and_empty_sites(
    monkeypatch,
):
    analysis = occupancy_counts_analysis(monkeypatch, saved_sites(trap_window_size=3))
    occupied = SITE_CENTRES[:5]
    frame = tweezer_frame(occupied)

    analysis.process({"index": 0, "rid": 4, **ROI_ORIGIN}, frame)

    ((head, data, _),) = analysis.dataq.sent

    def box_sum(centres):
        return sum(
            int(frame[row - 1 : row + 2, col - 1 : col + 2].sum())
            for row, col in centres
        )

    assert data is None
    assert (head["occupied_sites"], head["empty_sites"]) == (5, len(SITE_CENTRES) - 5)
    assert head["occupied_counts"] == box_sum(occupied)
    assert head["empty_counts"] == box_sum(SITE_CENTRES[5:])
    assert head["mean_occupied_counts"] == pytest.approx(box_sum(occupied) / 5)
    assert head["mean_empty_counts"] == pytest.approx(
        box_sum(SITE_CENTRES[5:]) / (len(SITE_CENTRES) - 5)
    )
    assert head["mean_atom_counts"] == pytest.approx(
        head["mean_occupied_counts"] - head["mean_empty_counts"]
    )
    assert head["occupancy"] == pytest.approx(5 / len(SITE_CENTRES))
    assert head["rid"] == 4


def test_tweezer_occupancy_counts_mean_of_an_empty_group_is_nan(monkeypatch):
    analysis = occupancy_counts_analysis(monkeypatch, saved_sites())

    analysis.process({"index": 0, **ROI_ORIGIN}, tweezer_frame([]))

    ((head, _, _),) = analysis.dataq.sent
    assert head["occupancy"] == 0.0
    assert head["occupied_counts"] == 0.0
    assert np.isnan(head["mean_occupied_counts"])
    assert np.isnan(head["mean_atom_counts"])
    assert np.isfinite(head["mean_empty_counts"])


def test_tweezer_occupancy_counts_skips_all_but_the_first_image(monkeypatch):
    analysis = occupancy_counts_analysis(monkeypatch, saved_sites())

    analysis.process({"index": 1, **ROI_ORIGIN}, tweezer_frame(SITE_CENTRES))

    assert analysis.dataq.sent == []


def test_tweezer_occupancy_counts_publishes_nan_counts_without_saved_sites(
    monkeypatch,
):
    analysis = occupancy_counts_analysis(monkeypatch, {})

    analysis.process({"rid": 4}, np.ones((40, 40)))

    ((head, _, _),) = analysis.dataq.sent
    assert np.isnan(head["occupied_counts"]) and np.isnan(head["empty_counts"])
    assert "FindTrapSites" in head["error"]


# --------------------------------------------------------------------------- #
# cloud_fit
# --------------------------------------------------------------------------- #


def test_cloud_fit_analysis_publishes_fit_fields_with_geometry():
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis
    from pytweezer.analysis.cloud_fit import FIELDS

    resolution, origin = 3e-6, (20.0, 40.0)
    frame = gaussian_frame((80, 100), x0=55.0, y0=30.0, sigma=7.0, amplitude=300.0)
    head = {
        "rid": 4,
        "index": 2,
        "_imgresolution": [resolution] * 2,
        "_offset": list(origin),
    }

    analysis = build(CloudFitAnalysis)
    assert analysis.process(head, frame) is None

    ((out, data, channel),) = analysis.dataq.sent
    assert data is None and channel == ""
    assert set(FIELDS) | {"units", "success"} <= set(out)
    assert out["success"] is True and out["units"] == "m"
    assert out["x0"] == pytest.approx((origin[0] + 55.5) * resolution, rel=1e-3)
    assert out["y0"] == pytest.approx((origin[1] + 30.5) * resolution, rel=1e-3)
    assert out["sigma_x"] == pytest.approx(7.0 * resolution, rel=0.03)
    assert (out["rid"], out["index"]) == (4, 2)
    assert out["_imgresolution"] == [resolution] * 2
    assert out["_offset"] == list(origin)


def test_cloud_fit_analysis_publishes_nan_for_a_frame_with_no_cloud():
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis
    from pytweezer.analysis.cloud_fit import FIELDS

    frame = np.random.default_rng(3).poisson(100.0, (80, 100)).astype(np.uint16)
    analysis = build(CloudFitAnalysis)
    analysis.process({"rid": 4, "_imgresolution": [3e-6, 3e-6]}, frame)

    ((out, _, _),) = analysis.dataq.sent
    assert out["success"] is False and out["rid"] == 4 and out["units"] == "m"
    assert all(np.isnan(out[field]) for field in FIELDS)


def test_offset_without_a_scale_places_pixels_from_the_crop_origin():
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis
    from pytweezer.analyses.live.projections import Projections

    frame = gaussian_frame((60, 90), x0=40.0, y0=25.0, sigma=6.0, amplitude=200.0)
    head = {"_offset": [30.0, 10.0]}

    fit = build(CloudFitAnalysis)
    fit.process(head, frame)
    ((out, _, _),) = fit.dataq.sent
    assert out["units"] == "px"
    assert out["x0"] == pytest.approx(30.0 + 40.5, abs=0.1)
    assert out["y0"] == pytest.approx(10.0 + 25.5, abs=0.1)

    proj = build(Projections)
    proj.process(head, frame)
    (_, x_data, _), (_, y_data, _) = proj.dataq.sent
    np.testing.assert_allclose(x_data[0], 30.0 + np.arange(90) + 0.5)
    np.testing.assert_allclose(y_data[0], 10.0 + np.arange(60) + 0.5)


def test_cloud_fit_analysis_reports_pixels_without_a_scale():
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis

    analysis = build(CloudFitAnalysis)
    analysis.process(
        {}, gaussian_frame((80, 100), x0=55.0, y0=30.0, sigma=7.0, amplitude=300.0)
    )

    ((out, _, _),) = analysis.dataq.sent
    assert out["units"] == "px"
    assert out["x0"] == pytest.approx(55.5, abs=0.1)


@pytest.mark.parametrize(
    "frame",
    [
        np.full((40, 40), np.nan),
        np.ones((1, 1)),
        np.zeros((0, 5)),
    ],
    ids=["nan", "single-pixel", "empty"],
)
def test_cloud_fit_analysis_survives_a_bad_frame(frame):
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis

    analysis = build(CloudFitAnalysis)
    assert analysis.process({"rid": 1, "index": 0}, frame) is None

    ((out, _, _),) = analysis.dataq.sent
    assert out["success"] is False
    assert (out["rid"], out["index"]) == (1, 0)


def test_cloud_fit_analysis_ignores_non_2d_data():
    from pytweezer.analyses.live.cloud_fit import CloudFitAnalysis

    analysis = build(CloudFitAnalysis)
    assert analysis.process({}, np.ones((2, 3, 4))) is None
    assert analysis.dataq.sent == []


# --------------------------------------------------------------------------- #
# shot_background
# --------------------------------------------------------------------------- #


def test_background_frame_is_held_not_published():
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    background = np.full((4, 5), 20, dtype=np.uint16)

    assert analysis.process({"rid": 1, "background": True}, background) is None
    assert analysis.client.sent == []


def test_background_subtraction_is_signed_float():
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    analysis.process({"rid": 1, "background": True}, np.full((4, 5), 20, np.uint16))

    head, data = analysis.process(
        {"rid": 1, "index": 3}, np.full((4, 5), 10, dtype=np.uint16)
    )

    assert data.dtype == np.float64
    np.testing.assert_array_equal(data, np.full((4, 5), -10.0))
    assert (head["rid"], head["index"]) == (1, 3)


def test_frames_of_one_background_shot_are_averaged():
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    for level in (10, 30):
        analysis.process(
            {"rid": 1, "background": True}, np.full((2, 2), level, np.uint16)
        )

    _, data = analysis.process({"rid": 1}, np.full((2, 2), 25, np.uint16))

    np.testing.assert_array_equal(data, np.full((2, 2), 5.0))


def test_new_background_replaces_the_one_of_an_earlier_rid():
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    analysis.process({"rid": 1, "background": True}, np.full((2, 2), 10, np.uint16))
    analysis.process({"rid": 2, "background": True}, np.full((2, 2), 50, np.uint16))

    _, data = analysis.process({"rid": 2}, np.full((2, 2), 60, np.uint16))

    np.testing.assert_array_equal(data, np.full((2, 2), 10.0))


def test_each_stream_keeps_its_own_background():
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    for stream, level in (("CamA", 10), ("CamB", 100)):
        analysis.current_stream = stream
        analysis.process(
            {"rid": 1, "background": True}, np.full((3, 3), level, np.uint16)
        )

    outputs = {}
    for stream in ("CamA", "CamB"):
        analysis.current_stream = stream
        _, outputs[stream] = analysis.process({"rid": 1}, np.full((3, 3), 120.0))

    np.testing.assert_array_equal(outputs["CamA"], np.full((3, 3), 110.0))
    np.testing.assert_array_equal(outputs["CamB"], np.full((3, 3), 20.0))


@pytest.mark.parametrize(
    ("rid", "shape"), [(2, (4, 5)), (1, (5, 4))], ids=["other-rid", "other-shape"]
)
def test_mismatched_background_passes_frame_through(rid, shape):
    from pytweezer.analyses.live.shot_background import ShotBackground

    analysis = build(ShotBackground)
    analysis.process({"rid": 1, "background": True}, np.full((4, 5), 20, np.uint16))
    frame = np.full(shape, 10, dtype=np.uint16)

    head, data = analysis.process({"rid": rid}, frame)

    assert data.dtype == np.float64
    np.testing.assert_array_equal(data, frame)


# --------------------------------------------------------------------------- #
# bg_subtract
# --------------------------------------------------------------------------- #


def bg_subtract(props=None):
    from pytweezer.analyses.live.bg_subtract import BgSubtract

    analysis = build(BgSubtract, props)
    analysis.current_stream = "HamCam"
    return analysis


def test_bg_subtract_publishes_signal_minus_background_once_both_arrive():
    analysis = bg_subtract()

    assert analysis.process({"rid": 1}, np.full((2, 3), 10, np.uint16)) is None
    head, data = analysis.process({"rid": 1}, np.full((2, 3), 25, np.uint16))

    assert data.dtype == np.float64
    np.testing.assert_array_equal(data, np.full((2, 3), -15.0))


def test_bg_subtract_counts_frames_per_rid_not_from_the_header_index():
    analysis = bg_subtract()

    analysis.process({"rid": 1, "index": 57}, np.full((2, 2), 10.0))
    _, data = analysis.process({"rid": 1, "index": 58}, np.full((2, 2), 4.0))

    np.testing.assert_array_equal(data, np.full((2, 2), 6.0))


def test_bg_subtract_uses_the_configured_positions():
    analysis = bg_subtract({"signal_index": 2, "background_index": 0})

    assert analysis.process({"rid": 1}, np.full((2, 2), 5.0)) is None
    assert analysis.process({"rid": 1}, np.full((2, 2), 99.0)) is None
    _, data = analysis.process({"rid": 1}, np.full((2, 2), 12.0))

    np.testing.assert_array_equal(data, np.full((2, 2), 7.0))
    assert analysis.process({"rid": 1}, np.full((2, 2), 1.0)) is None


def test_bg_subtract_new_rid_restarts_the_count():
    analysis = bg_subtract()

    analysis.process({"rid": 1}, np.full((2, 2), 10.0))
    assert analysis.process({"rid": 2}, np.full((2, 2), 8.0)) is None
    _, data = analysis.process({"rid": 2}, np.full((2, 2), 3.0))

    np.testing.assert_array_equal(data, np.full((2, 2), 5.0))


def test_bg_subtract_output_stream_is_prefixed_so_input_subscribers_miss_it():
    from pytweezer.analyses.live.bg_subtract import output_stream

    assert output_stream("HamCam") == "bg_subtract_HamCam"
    assert not output_stream("HamCam").startswith("HamCam")


# --------------------------------------------------------------------------- #
# recapture_rate
# --------------------------------------------------------------------------- #


def notebook_recapture_rate(images, skip, crop=None):
    """``get_recapture_rate_three_image`` from the CaF notebooks, for one zip."""
    bg = images[skip * 3 + 2 :: 3].mean(axis=0)
    first_im = images[skip * 3 :: 3] - bg
    second_im = images[skip * 3 + 1 :: 3] - bg
    if crop is not None:
        x1, x2, y1, y2 = crop
        first_im = first_im[:, x1:x2, y1:y2]
        second_im = second_im[:, x1:x2, y1:y2]
    rate = second_im.sum(axis=(1, 2)) / first_im.sum(axis=(1, 2))
    return rate.mean(), rate.std() / np.sqrt(len(rate))


def recapture_run(shots, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(shots):
        background = rng.poisson(100, (6, 7))
        cloud = np.zeros((6, 7))
        cloud[2:4, 3:5] = rng.uniform(200, 400)
        recaptured = cloud * rng.uniform(0.3, 0.7)
        frames += [background + cloud, background + recaptured, background]
    return np.array(frames, dtype=np.uint16)


def feed_run(analysis, frames, **tags):
    for index, frame in enumerate(frames):
        analysis.process({"index": index, **tags}, frame)


@pytest.mark.parametrize(
    "skip, crop", [(0, []), (2, []), (1, [1, 5, 2, 6])], ids=["all", "skip", "crop"]
)
def test_recapture_rate_matches_the_notebook_at_the_end_of_a_run(skip, crop):
    from pytweezer.analyses.live.recapture_rate import RecaptureRate

    frames = recapture_run(10)
    analysis = build(RecaptureRate, {"skip": skip, "crop": crop})
    feed_run(analysis, frames, pycaf_run=1.0)

    rate, rate_err = notebook_recapture_rate(frames.astype(float), skip, crop or None)
    head, _, _ = analysis.dataq.sent[-1]
    assert len(analysis.dataq.sent) == 10 - skip
    assert head["shots"] == 10 - skip
    assert head["recapture_rate"] == pytest.approx(rate)
    assert head["recapture_rate_err"] == pytest.approx(rate_err)
    assert head["pycaf_run"] == 1.0


def test_recapture_rate_images_are_the_mean_first_and_second_less_background():
    from pytweezer.analyses.live.recapture_rate import RecaptureRate

    frames = recapture_run(4).astype(float)
    analysis = build(RecaptureRate)
    for index, frame in enumerate(frames):
        result = analysis.process({"index": index, "pycaf_run": 1.0}, frame)

    _, image = result
    background = frames[2::3].mean(axis=0)
    np.testing.assert_allclose(image[:, :7], frames[0::3].mean(axis=0) - background)
    np.testing.assert_allclose(image[:, 7:], frames[1::3].mean(axis=0) - background)


def test_recapture_rate_starts_afresh_for_each_run():
    from pytweezer.analyses.live.recapture_rate import RecaptureRate

    analysis = build(RecaptureRate)
    feed_run(analysis, recapture_run(5, seed=1), pycaf_run=1.0)
    second_run = recapture_run(3, seed=2)
    feed_run(analysis, second_run, pycaf_run=2.0)

    rate, _ = notebook_recapture_rate(second_run.astype(float), 0)
    head, _, _ = analysis.dataq.sent[-1]
    assert head["shots"] == 3
    assert head["recapture_rate"] == pytest.approx(rate)


def test_recapture_rate_leaves_out_a_shot_missing_a_frame():
    from pytweezer.analyses.live.recapture_rate import RecaptureRate

    frames = recapture_run(3)
    analysis = build(RecaptureRate)
    for index, frame in enumerate(frames):
        if index != 4:  # second shot's recaptured frame
            analysis.process({"index": index, "pycaf_run": 1.0}, frame)

    complete = np.delete(frames, [3, 4, 5], axis=0).astype(float)
    rate, _ = notebook_recapture_rate(complete, 0)
    head, _, _ = analysis.dataq.sent[-1]
    assert head["shots"] == 2
    assert head["recapture_rate"] == pytest.approx(rate)


def test_recapture_rate_ignores_frames_not_tagged_with_a_run():
    from pytweezer.analyses.live.recapture_rate import RecaptureRate

    analysis = build(RecaptureRate)
    feed_run(analysis, recapture_run(2))

    assert analysis.dataq.sent == []
