import numpy as np
import pytest

from pytweezer.drivers.motmaster import SIMULATED_SCRIPT_PARAMETERS
from pytweezer.experiment import run_local
from pytweezer.experiments import tweezer_imaging
from pytweezer.experiments.tweezer_imaging import (
    SAVED_SEQUENCE_PARAMETERS,
    OptimiseTweezerImaging,
    TweezerImaging,
)

SITES = [(25 + 10 * row, 25 + 10 * column) for row in range(8) for column in range(8)]
LOADING = 0.6
SURVIVAL = 0.9
SAVED = {
    key: 1.0 + index for index, key in enumerate(SAVED_SEQUENCE_PARAMETERS.values())
}
FAST = {"calibration_shots": 30, "shots": 10}


class FakeSequencer:
    def __init__(self):
        self.goes = []

    def close_rpc(self):
        pass

    def set_linked(self, value): ...
    def set_motmaster_experiment(self, script): ...
    def set_run_until_stopped(self, value): ...
    def set_iterations(self, value): ...
    def set_save_toggle(self, value): ...
    def set_trigger_mode(self, value): ...

    def get_params(self):
        return dict(SIMULATED_SCRIPT_PARAMETERS["RbTweezerBasic2026_2"])

    def start_motmaster_experiment(self, parameters):
        self.goes.append(parameters)


class FakeHamCam:
    """Images the array twice per shot; a background drop leaves the second frame empty."""

    def __init__(self, sequencer):
        self.sequencer = sequencer
        self.rng = np.random.default_rng(1)
        y, x = np.mgrid[:128, :128]
        self.spots = np.array(
            [2000 * np.exp(-((y - sy) ** 2 + (x - sx) ** 2) / 4.5) for sy, sx in SITES]
        )

    def close_rpc(self):
        pass

    def set_roi(self, *args): ...
    def set_external_exposure_mode(self): ...
    def enable_em_gain(self, on): ...
    def enable_direct_em_gain(self, on): ...
    def set_sensitivity(self, value): ...
    def setup_acquisition(self, mode, frames): ...
    def start_acquisition(self): ...
    def stop_acquisition(self): ...

    def _frame(self, occupied):
        atoms = np.tensordot(occupied, self.spots, axes=1)
        return self.rng.poisson(100 + atoms).astype(np.uint16)

    def acquire_n_frames(self, n, broadcast=False):
        loaded = self.rng.random(len(SITES)) < LOADING
        if self.sequencer.goes[-1]["BackgroundDrop"]:
            kept = np.zeros(len(SITES), dtype=bool)
        else:
            kept = loaded & (self.rng.random(len(SITES)) < SURVIVAL)
        return self._frame(loaded), self._frame(kept)


@pytest.fixture
def rig(monkeypatch):
    sequencer = FakeSequencer()
    devices = {"Rb MotMaster": sequencer, "Rb HamCam": FakeHamCam(sequencer)}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: devices[name],
    )
    monkeypatch.setattr(tweezer_imaging, "get_saved_parameters", lambda: SAVED)
    return sequencer


def test_calibration_finds_every_trap_site(rig):
    measurement = run_local(TweezerImaging, **FAST)
    assert measurement.status == "completed", measurement.attrs["error"]
    found = measurement.constants["trap_sites"]
    offsets = np.abs(found[None, :, :] - np.array(SITES)[:, None, :]).max(axis=2)
    assert len(found) == len(SITES)
    assert offsets.min(axis=1).max() <= 1


def test_a_point_measures_the_loading_and_survival_put_in(rig):
    measurement = run_local(TweezerImaging, **FAST)
    assert measurement.results["loading"][0] == pytest.approx(LOADING, abs=0.1)
    assert measurement.results["survival"][0] == pytest.approx(SURVIVAL, abs=0.05)
    assert measurement.results["fidelity"][0] > 0.99


def test_every_shot_runs_at_the_saved_parameters_unless_the_form_sets_one(rig):
    run_local(TweezerImaging, **FAST, **{"rb.tMolCoolVCO1": 7.5})
    expected = {
        parameter: SAVED[key] for parameter, key in SAVED_SEQUENCE_PARAMETERS.items()
    } | {"tMolCoolVCO1": 7.5}
    for go in rig.goes:
        assert {name: go[name] for name in expected} == expected


def test_calibration_drops_the_atoms_before_the_second_image_and_points_do_not(rig):
    run_local(TweezerImaging, **FAST)
    drops = [go["BackgroundDrop"] for go in rig.goes]
    assert drops == [1] * FAST["calibration_shots"] + [0] * FAST["shots"]


def test_each_evaluation_images_both_times_at_its_settings(rig):
    measurement = run_local(OptimiseTweezerImaging, **FAST, n_calls=2, n_initial=2)
    assert measurement.status == "completed", measurement.attrs["error"]
    evaluations = rig.goes[FAST["calibration_shots"] :]
    for point in range(2):
        vco = measurement.results["imaging_vco"][point]
        vva = measurement.results["imaging_vva"][point]
        for go in evaluations[point * FAST["shots"] : (point + 1) * FAST["shots"]]:
            assert (go["tImgCoolVCO1"], go["tImgCoolVCO2"]) == (vco, vco)
            assert (go["tImgCoolVVA1"], go["tImgCoolVVA2"]) == (vva, vva)


@pytest.mark.parametrize(
    ("fidelity_floor", "expected"),
    [(0.9, "survival"), (1.0, "0.8 fidelity")],
)
def test_the_score_is_survival_only_above_the_fidelity_floor(
    rig, fidelity_floor, expected
):
    measurement = run_local(
        OptimiseTweezerImaging,
        **FAST,
        n_calls=1,
        n_initial=1,
        fidelity_floor=fidelity_floor,
    )
    results = measurement.results
    score = {
        "survival": results["survival"][0],
        "0.8 fidelity": 0.8 * results["fidelity"][0],
    }
    assert results["objective"][0] == pytest.approx(score[expected])
