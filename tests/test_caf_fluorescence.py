from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.experiment import Scan, run_local
from pytweezer.experiments.caf_fluorescence import CaFFluorescence
from pytweezer.experiments.caf_mot import LASERS, CaFMOT
from tests.helpers import motmaster_record

START = {
    "v0": 494.431874,
    "BX": 564.582340,
    "v1": 476.958908,
    "v2": 477.299380,
    "v3": 477.627302,
    "TCool": 494.431874,
}


class FakeCaFMOTCam:
    """Catches one frame per MOTMaster iteration, ``frame(field, set_points, iteration)``.

    A ``frame`` returning ``None`` is a frame the camera missed.
    """

    def __init__(self):
        self.frames = []
        self.headers = []
        self.calls = []
        self.frame = lambda field, set_points, iteration: np.full(
            (2, 3), 10 + 5 * field
        )

    def close_rpc(self):
        pass

    def stop_live(self):
        self.calls.append("stop_live")

    def set_trigger_source(self, source):
        self.calls.append(f"trigger {source}")

    def start_live(self):
        self.calls.append("start_live")

    def end_shots(self):
        self.calls.append("end_shots")

    def frame_geometry(self):
        return 1, (0, 0)

    def start_shot(self, header):
        self.calls.append("start_shot")
        self.headers.append(header)
        return len(self.frames)

    def catch(self, iterations, field, set_points):
        for iteration in range(iterations):
            image = self.frame(field, set_points, iteration)
            if image is not None:
                self.frames.append(np.asarray(image, dtype=np.uint16))

    def wait_frames(self, mark, n):
        if len(self.frames) < mark + n:
            raise TimeoutError(f"{len(self.frames) - mark} of {n} frames arrived")
        return np.stack(self.frames[mark : mark + n])


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

    def set_linked(self, value): ...
    def set_motmaster_experiment(self, script): ...
    def set_run_until_stopped(self, value): ...
    def set_save_toggle(self, value): ...
    def set_trigger_mode(self, value): ...

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

    def start_motmaster_experiment_recorded(self, parameters):
        self.log.append(("go", dict(parameters)))
        if len(self.goes()) == self.fail_on_go:
            raise self.failure
        field = parameters.get(
            "MOTCoilsCurrentValue", self.get_params()["MOTCoilsCurrentValue"]
        )
        self.camera.catch(self.iterations, field, dict(self.set_points))
        return motmaster_record(parameters)

    def goes(self):
        return [entry[1] for entry in self.log if entry[0] == "go"]

    def fields(self):
        """The MOT field of each run: the one sent, else the script's default."""
        default = self.get_params()["MOTCoilsCurrentValue"]
        return [go.get("MOTCoilsCurrentValue", default) for go in self.goes()]

    def ramps(self):
        return [entry for entry in self.log if entry[0] == "ramp"]

    def ramps_before_last_go(self):
        last_go = max(i for i, entry in enumerate(self.log) if entry[0] == "go")
        return [entry for entry in self.log[:last_go] if entry[0] == "ramp"]


@pytest.fixture
def caf(monkeypatch):
    camera = FakeCaFMOTCam()
    motmaster = FakeCaFMotMaster(camera)
    fakes = {"CaF MotMaster": motmaster, "CaFMOT Cam": camera}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: fakes[name],
    )
    return SimpleNamespace(motmaster=motmaster, camera=camera)


def measuring(**set_points):
    """A CaFFluorescence whose points measure at ``set_points``, as a StagedScan would."""

    class Measuring(CaFFluorescence):
        def run_point(self):
            for name, value in self.measure(**set_points).items():
                self.record(name, value)

    return Measuring


def test_counts_are_background_subtracted_frame_sums(caf):
    signal = [1.0, 2.0, 6.0]

    def frame(field, set_points, iteration):
        if field == 0.0:
            return np.full((1, 2), 10.0 + iteration)
        return np.full((1, 2), 11.0 + signal[iteration])

    caf.camera.frame = frame
    measurement = run_local(CaFFluorescence, iterations=3)
    assert measurement.status == "completed", measurement.attrs["error"]
    # Background mean is 11 per pixel, so the frame sums are 2 * signal = [2, 4, 12].
    assert measurement.results["counts"][0] == pytest.approx(6.0)
    assert measurement.results["counts_error"][0] == pytest.approx(np.sqrt(28 / 3))


def test_cafmot_sums_the_frames_as_the_camera_took_them(caf):
    measurement = run_local(CaFMOT)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.fields() == [1.0]
    # One frame of 2 x 3 pixels at 10 + 5 * field counts each.
    assert measurement.results["counts"][0] == pytest.approx(6 * 15.0)
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
    measurement = run_local(measuring(**set_points))
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
    measurement = run_local(measuring(v0=494.44, BX=564.59, TCool=494.42))
    assert measurement.status == status
    assert caf.motmaster.set_points == START
    assert caf.camera.calls[-2:] == ["start_live", "end_shots"]


def test_finish_leaves_a_laser_it_never_moved_alone(caf):
    class HandTuned(CaFFluorescence):
        def run_point(self):
            self.caf.ramp_laser_set_point_wml("v1", 476.96)
            for name, value in self.measure(v0=494.44).items():
                self.record(name, value)

    measurement = run_local(HandTuned)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.ramps() == [
        ("ramp", "v1", 476.96),
        ("ramp", "v0", 494.44),
        ("ramp", "v0", START["v0"]),
    ]


def test_a_laser_that_cannot_return_does_not_strand_the_others(caf):
    caf.motmaster.refused_ramps = {("v0", START["v0"])}
    measurement = run_local(measuring(v0=494.44, BX=564.59))
    assert measurement.status == "failed"
    assert f"{{'v0': {START['v0']}}}" in measurement.attrs["error"]
    assert caf.motmaster.set_points == {**START, "v0": 494.44}


def test_the_background_shot_has_the_field_off(caf):
    measurement = run_local(CaFFluorescence, Scan(repetitions=2))
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.fields() == [0.0, 1.0, 1.0]
    assert all("MOTCoilsCurrentValue" not in go for go in caf.motmaster.goes()[1:])


def test_iterations_set_the_script_iterations_and_the_frames_read(caf):
    measurement = run_local(CaFMOT, iterations=4)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert caf.motmaster.iterations == 4
    assert measurement.results["counts"][0] == pytest.approx(6 * 15.0)
    assert np.isfinite(measurement.results["counts_error"][0])


def test_the_camera_is_on_its_external_trigger_and_live_before_the_first_shot(caf):
    measurement = run_local(CaFFluorescence)
    assert measurement.status == "completed", measurement.attrs["error"]
    first_shot = caf.camera.calls.index("start_shot")
    assert caf.camera.calls[:first_shot] == ["stop_live", "trigger ext", "start_live"]


def gaussian_cloud_frame(field, set_points, iteration):
    """A 100-count cloud on an offset of 20 when the field is on; the offset alone when off."""
    y, x = np.mgrid[:60, :80]
    cloud = 100 * np.exp(-((x - 40) ** 2 / (2 * 6**2) + (y - 30) ** 2 / (2 * 4**2)))
    return 20 + (cloud if field else 0 * cloud)


def test_each_point_records_the_background_subtracted_cloud_fit_of_its_shot(caf):
    caf.camera.frame = gaussian_cloud_frame
    measurement = run_local(CaFFluorescence)

    assert measurement.status == "completed", measurement.attrs["error"]
    results = measurement.results
    assert results["cafmot_sigma_x"][0] == pytest.approx(6, rel=0.03)
    assert results["cafmot_sigma_y"][0] == pytest.approx(4, rel=0.03)
    # The background shot has the offset, so none is left in the fit.
    assert abs(results["cafmot_offset"][0]) < 2
    assert results["counts"][0] == pytest.approx(
        results["cafmot_mean_image"][0].sum(), rel=1e-4
    )
    assert measurement.constants["cloud_mass_amu"] == 59.0


def test_runs_against_the_simulated_motmaster_and_camera():
    measurement = run_local(CaFFluorescence, simulate=True)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert np.isfinite(measurement.results["counts"][0])
    assert set(measurement.constants["start_set_points"]) == set(LASERS)
