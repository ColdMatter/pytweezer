import pytest

from pytweezer.analysis.tweezer_histogram import (
    HistogramSettings,
    needed_results,
    point_groups,
    tweezer_histogram,
)
from pytweezer.experiment import Experiment, ListAxis, Number, Scan, run_local
from pytweezer.experiments.tweezer_array import TweezerImaging
from tests.test_analysis import (
    SITE_CENTRES,
    loading_window,
    patterned_frame,
    tweezer_frame,
)
from tests.test_tweezer_array import FAST, rig  # noqa: F401

FRAMES_SETTINGS = HistogramSettings(sites=len(SITE_CENTRES), detection_step=10)


class ImagesArray(Experiment):
    """Records two frames per point, as LoadTweezers does; the second is empty."""

    detuning = Number(0.0, unit="MHz", scale=1e6)

    def run_point(self):
        self.record("first_image", patterned_frame(self.point.index))
        self.record("second_image", tweezer_frame([], seed=1000 + self.point.index))


def frames_file(scan=None):
    return run_local(ImagesArray, scan or Scan(repetitions=40))


def test_frames_file_gives_the_live_analysis_result():
    head, photons = tweezer_histogram(frames_file(), FRAMES_SETTINGS)

    expected_head, expected_photons = loading_window(
        patterned_frame(shot) for shot in range(40)
    )
    assert head["source"] == "frames"
    assert photons == pytest.approx(expected_photons)
    assert head["threshold"] == pytest.approx(expected_head["threshold"], rel=1e-3)
    assert head["site_loading"] == pytest.approx(expected_head["site_loading"])


def test_frames_second_image_is_scored_at_the_first_images_sites():
    head, _ = tweezer_histogram(frames_file(), FRAMES_SETTINGS)
    second, _ = tweezer_histogram(
        frames_file(), HistogramSettings(**{**vars(FRAMES_SETTINGS), "image": 1})
    )

    assert (second["site_rows"], second["site_cols"]) == (
        head["site_rows"],
        head["site_cols"],
    )


def test_photons_file_uses_the_recorded_sites_and_photons(rig):  # noqa: F811
    measurement = run_local(TweezerImaging, **FAST)

    for image, name in enumerate(["first_photons", "second_photons"]):
        head, photons = tweezer_histogram(measurement, HistogramSettings(image=image))
        assert head["source"] == "photons"
        assert photons == pytest.approx(measurement.results[name][0])
        sites = measurement.constants["trap_sites"]
        assert head["site_rows"] == list(sites[:, 0])
        assert head["site_cols"] == list(sites[:, 1])


def test_a_scan_value_pools_its_repetitions():
    scan = Scan(axes=[ListAxis(argument="detuning", values=[0.0, 2e6])], repetitions=20)
    measurement = frames_file(scan)

    groups = point_groups(measurement)
    head, photons = tweezer_histogram(
        measurement, HistogramSettings(**{**vars(FRAMES_SETTINGS), "group": 1})
    )

    assert [label for label, _ in groups] == ["detuning = 0 MHz", "detuning = 2 MHz"]
    assert head["shots"] == 20
    assert photons.shape == (20, len(SITE_CENTRES))


def test_without_scan_axes_each_point_is_a_group():
    groups = point_groups(frames_file(Scan(repetitions=3)))

    assert [label for label, _ in groups] == ["point 0", "point 1", "point 2"]
    assert [list(mask) for _, mask in groups][1] == [False, True, False]


def test_a_group_with_no_shots_gives_a_header_without_photons():
    measurement = frames_file(Scan(repetitions=3))
    measurement.points = {
        name: values[:0] for name, values in measurement.points.items()
    }
    measurement.results = {
        name: values[:0] for name, values in measurement.results.items()
    }

    head, photons = tweezer_histogram(measurement, FRAMES_SETTINGS)

    assert photons is None
    assert head["shots"] == 0
    assert head["error"]


def test_frames_without_the_sites_give_a_header_without_photons():
    settings = HistogramSettings(sites=len(SITE_CENTRES) + 5, detection_step=10)

    head, photons = tweezer_histogram(frames_file(), settings)

    assert photons is None
    assert f"{len(SITE_CENTRES) + 5} trap sites" in head["error"]


def test_needed_results_names_only_what_the_source_uses(rig):  # noqa: F811
    assert needed_results(frames_file(Scan())) == ["first_image", "second_image"]
    assert needed_results(run_local(TweezerImaging, **FAST)) == [
        "first_photons",
        "second_photons",
    ]
