from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.experiment import Scan, run_local
from pytweezer.experiment.camera_experiment import CameraShots
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.experiments.caf_fluorescence import LASERS, CaFFluorescence

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
        self.frame = lambda field, set_points, iteration: np.full(
            (2, 3), 10 + 5 * field
        )

    def close_rpc(self):
        pass

    def is_live(self):
        return False

    def stop_live(self):
        pass

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

    def acquire_n_frames(self, nframes, broadcast=False, header=None):
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
        self.camera.catch(
            self.iterations, parameters["MOTCoilsCurrentValue"], dict(self.set_points)
        )

    def goes(self):
        return [entry[1] for entry in self.log if entry[0] == "go"]

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
    measurement = run(CaFFluorescence)
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
    fields = [go["MOTCoilsCurrentValue"] for go in caf.motmaster.goes()]
    assert fields == [0.0, 1.0, 1.0]


def test_runs_against_the_simulated_motmaster_and_camera():
    measurement = run_local(CaFFluorescence, simulate=True)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert np.isfinite(measurement.results["counts"][0])
    assert set(measurement.constants["start_set_points"]) == set(LASERS)
