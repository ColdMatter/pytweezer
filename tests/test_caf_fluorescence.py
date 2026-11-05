from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.experiment import Scan, run_local
from pytweezer.experiment.camera_experiment import CameraShots
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.experiment.scan import ListAxis
from pytweezer.experiments.caf_fluorescence import CaFFluorescence
from pytweezer.experiments.caf_mot import LASERS, CaFMOT

START = {
    "v0": 494.431874,
    "BX": 564.582340,
    "v1": 476.958908,
    "v2": 477.299380,
    "v3": 477.627302,
    "TCool": 494.431874,
}


class FakeCamera:
    """Catches one frame per MOTMaster iteration, ``frame(field, set_points, iteration)``.

    A ``frame`` returning ``None`` is a frame the camera missed.
    """

    def __init__(self):
        self.frames = []
        self.streams = []
        self.broadcasts = []
        self.frame = lambda field, set_points, iteration: np.full(
            (2, 3), 10 + 5 * field
        )

    def close_rpc(self):
        pass

    def is_live(self):
        return False

    def stop_live(self):
        pass

    def frame_geometry(self):
        return 1, (0, 0)

    def setup_acquisition(self, nframes, mode="sequence"):
        pass

    def start_acquisition(self):
        self.frames = []

    def stop_acquisition(self):
        pass

    def catch(self, iterations, field, set_points):
        for iteration in range(iterations):
            image = self.frame(field, set_points, iteration)
            if image is not None:
                self.frames.append(np.asarray(image, dtype=np.uint16))

    def broadcast_image(self, image, info, stream=None):
        self.broadcasts.append((stream, np.asarray(image), info))

    def acquire_n_frames(self, nframes, broadcast=False, header=None, stream=None):
        self.streams.append(stream)
        if len(self.frames) < nframes:
            raise TimeoutError(f"{len(self.frames)} of {nframes} frames arrived")
        return np.stack(self.frames[:nframes])


class FakeCaFMotMaster:
    """Logs ramps and runs; each run makes the camera catch its iterations' frames."""

    def __init__(self, camera):
        self.camera = camera
        self.log = []
        self.set_points = dict(START)
        self.iterations = 1
        self.fail_on_go, self.failure = None, None
        self.refused_ramps = set()

    def close_rpc(self):
        pass

    def set_linked(self, value):
        pass

    def set_motmaster_experiment(self, script):
        pass

    def set_run_until_stopped(self, value):
        pass

    def set_save_toggle(self, value):
        pass

    def set_trigger_mode(self, value):
        pass

    def set_iterations(self, iterations):
        self.iterations = iterations

    def get_params(self):
        return {"MOTCoilsCurrentValue": 1.0}

    def laser_set_points_wml(self):
        return dict(self.set_points)

    def ramp_laser_set_point_wml(self, laser, target):
        if (laser, target) in self.refused_ramps:
            raise RuntimeError(f"{laser} lost lock")
        self.log.append(("ramp", laser, target))
        self.set_points[laser] = target

    def start_motmaster_experiment(self, parameters=None):
        self.log.append(("go", dict(parameters)))
        if len(self.goes()) == self.fail_on_go:
            raise self.failure
        field = parameters.get(
            "MOTCoilsCurrentValue", self.get_params()["MOTCoilsCurrentValue"]
        )
        self.camera.catch(self.iterations, field, dict(self.set_points))

    def goes(self):
        return [entry[1] for entry in self.log if entry[0] == "go"]

    def fields(self):
        """The MOT field of each run: the one sent, else the script's default."""
        default = self.get_params()["MOTCoilsCurrentValue"]
        return [go.get("MOTCoilsCurrentValue", default) for go in self.goes()]

    def ramps_before_last_go(self):
        last_go = max(i for i, entry in enumerate(self.log) if entry[0] == "go")
        return [entry for entry in self.log[:last_go] if entry[0] == "ramp"]


@pytest.fixture
def caf(monkeypatch):
    camera = FakeCamera()
    motmaster = FakeCaFMotMaster(camera)
    fakes = {"CaF MotMaster": motmaster, "CaF ThorCam": camera}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: fakes[name],
    )
    return SimpleNamespace(motmaster=motmaster, camera=camera)


def run(experiment_cls, scan=None, **args):
    return run_local(experiment_cls, scan, **args)


def measuring(**set_points):
    """A CaFFluorescence whose points measure at ``set_points``, as a StagedScan would."""

    class Measuring(CaFFluorescence):
        def run_point(self):
            for name, value in self.measure(**set_points).items():
                self.record(name, value)

    return Measuring


class OneIteration(CaFFluorescence):
    caf = MotMaster("CaF MotMaster", script="0. DDSFourSidebands5W_noDDS")
    thorcam = CameraShots("CaF ThorCam", frames=1, trigger=None, locked=("frames",))


def test_counts_are_background_subtracted_frame_sums(caf):
    signal = [1.0, 2.0, 6.0]

    def frame(field, set_points, iteration):
        if field == 0.0:
            return np.full((1, 2), 10.0 + iteration)
        return np.full((1, 2), 11.0 + signal[iteration])

    caf.camera.frame = frame
    measurement = run(CaFFluorescence, iterations=3)
    assert measurement.status == "completed", measurement.attrs["error"]
    # Background mean is 11 per pixel, so the frame sums are 2 * signal = [2, 4, 12].
    assert measurement.results["counts"][0] == pytest.approx(6.0)
    assert measurement.results["counts_error"][0] == pytest.approx(np.sqrt(28 / 3))


def test_counts_error_is_nan_for_one_frame(caf):
    measurement = run(OneIteration)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert measurement.results["counts"][0] == pytest.approx(30.0)
    assert np.isnan(measurement.results["counts_error"][0])


@pytest.mark.parametrize(
    "set_points, ramps",
    [
        ({}, []),
        ({"v0": 494.44}, [("ramp", "v0", 494.44)]),
        ({"v0": 494.44, "BX": None}, [("ramp", "v0", 494.44)]),
    ],
    ids=["none-given", "v0-given", "BX-None"],
)
def test_none_leaves_a_laser_alone_and_a_value_ramps_it(caf, set_points, ramps):
    measurement = run(measuring(**set_points))
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.ramps_before_last_go() == ramps


@pytest.mark.parametrize(
    "failure, status",
    [
        (None, "completed"),
        (RuntimeError("Go failed"), "failed"),
        (KeyboardInterrupt(), "terminated"),
    ],
    ids=["completed", "failed", "terminated"],
)
def test_lasers_return_to_where_they_started(caf, failure, status):
    if failure is not None:
        # The second run is the first measured shot, after the lasers moved.
        caf.motmaster.fail_on_go, caf.motmaster.failure = 2, failure
    measurement = run(measuring(v0=494.44, BX=564.59, TCool=494.42))
    assert measurement.status == status
    assert caf.motmaster.set_points == START


def test_finish_leaves_a_laser_it_never_moved_alone(caf):
    class HandTuned(CaFFluorescence):
        def run_point(self):
            self.caf.ramp_laser_set_point_wml("v1", 476.96)
            for name, value in self.measure(v0=494.44).items():
                self.record(name, value)

    measurement = run(HandTuned)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.set_points == {**START, "v1": 476.96}
    assert [entry for entry in caf.motmaster.log if entry[0] == "ramp"] == [
        ("ramp", "v1", 476.96),
        ("ramp", "v0", 494.44),
        ("ramp", "v0", START["v0"]),
    ]


def test_a_laser_that_cannot_return_does_not_strand_the_others(caf):
    caf.motmaster.refused_ramps = {("v0", START["v0"])}
    measurement = run(measuring(v0=494.44, BX=564.59))
    assert measurement.status == "failed"
    assert "v0" in measurement.attrs["error"]
    assert caf.motmaster.set_points == {**START, "v0": 494.44}


def test_a_stranded_laser_is_reported_after_a_failed_run(caf):
    caf.motmaster.fail_on_go, caf.motmaster.failure = 2, RuntimeError("Go failed")
    caf.motmaster.refused_ramps = {("v0", START["v0"])}
    measurement = run(measuring(v0=494.44))
    assert measurement.status == "failed"
    assert "Go failed" in measurement.attrs["error"]
    assert f"{{'v0': {START['v0']}}}" in measurement.attrs["error"]


def test_start_set_points_read_back_from_the_file(caf):
    measurement = run(CaFFluorescence)
    assert measurement.constants["start_set_points"] == START


def test_a_shot_missing_frames_fails(caf):
    caf.camera.frame = lambda field, set_points, iteration: (
        np.zeros((2, 3)) if field == 0.0 else None
    )
    measurement = run(CaFFluorescence)
    assert measurement.status == "failed"
    assert "TimeoutError" in measurement.attrs["error"]


def test_the_background_shot_has_the_field_off(caf):
    measurement = run(CaFFluorescence, Scan(repetitions=2))
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.fields() == [0.0, 1.0, 1.0]
    assert all("MOTCoilsCurrentValue" not in go for go in caf.motmaster.goes()[1:])


def gaussian_cloud_frame(field, set_points, iteration):
    """A 100-count cloud on an offset of 20 when the field is on; the offset alone when off."""
    y, x = np.mgrid[:60, :80]
    cloud = 100 * np.exp(-((x - 40) ** 2 / (2 * 6**2) + (y - 30) ** 2 / (2 * 4**2)))
    return 20 + (cloud if field else 0 * cloud)


def test_fluorescence_point_records_thorcam_cloud_fit(caf):
    caf.camera.frame = gaussian_cloud_frame
    measurement = run(CaFFluorescence)

    assert measurement.status == "completed", measurement.attrs["error"]
    results = measurement.results
    assert results["thorcam_sigma_x"][0] == pytest.approx(6, rel=0.03)
    assert results["thorcam_sigma_y"][0] == pytest.approx(4, rel=0.03)
    assert results["thorcam_x0"][0] == pytest.approx(40.5, abs=0.2)
    assert results["thorcam_y0"][0] == pytest.approx(30.5, abs=0.2)
    # The prepare background has the offset, so none is left in the fit.
    assert abs(results["thorcam_offset"][0]) < 2
    assert results["counts"][0] == pytest.approx(
        results["thorcam_mean_image"][0].sum(), rel=1e-4
    )
    assert measurement.constants["cloud_mass_amu"] == 59.0


def test_background_and_subtracted_frames_go_on_their_own_streams(caf):
    measurement = run(CaFFluorescence, iterations=3)

    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.camera.streams == ["background_caf_thorcam", None]
    assert [stream for stream, _, _ in caf.camera.broadcasts] == [
        "subtracted_caf_thorcam"
    ] * 3
    # 10 per pixel at field 0 and 15 at field 1.
    for _, image, info in caf.camera.broadcasts:
        np.testing.assert_array_equal(image, np.full((2, 3), 5.0))
        assert "rid" in info


def test_iterations_set_the_script_iterations_and_the_frames_read(caf):
    measurement = run(CaFMOT, iterations=4)

    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.iterations == 4
    assert len(caf.camera.frames) == 4


def test_iterations_cannot_be_scanned(caf):
    with pytest.raises(ValueError, match="iterations"):
        run(CaFMOT, Scan(axes=[ListAxis(argument="iterations", values=[1, 2])]))


class TestCaFMOT:
    def test_it_takes_no_background_and_leaves_frames_on_the_camera_stream(self, caf):
        measurement = run(CaFMOT)

        assert measurement.status == "completed", measurement.attrs["error"]
        assert caf.motmaster.fields() == [1.0]
        assert caf.camera.streams == [None]
        assert caf.camera.broadcasts == []
        # Three frames of 2 x 3 pixels at 10 + 5 * field counts each.
        assert measurement.results["counts"][0] == pytest.approx(6 * 15.0)

    def test_it_restores_the_lasers_it_moved(self, caf):
        class Moving(CaFMOT):
            def run_point(self):
                for name, value in self.measure(v0=494.44).items():
                    self.record(name, value)

        measurement = run(Moving)

        assert measurement.status == "completed", measurement.attrs["error"]
        assert caf.motmaster.set_points == START


def test_the_cloud_fit_takes_no_shot_of_its_own(caf):
    measurement = run(CaFFluorescence, Scan(repetitions=2))

    assert measurement.status == "completed", measurement.attrs["error"]
    assert len(caf.motmaster.goes()) == 3


def test_runs_against_the_simulated_motmaster_and_camera():
    measurement = run_local(CaFFluorescence, simulate=True)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert np.isfinite(measurement.results["counts"][0])
    assert set(measurement.constants["start_set_points"]) == set(LASERS)
