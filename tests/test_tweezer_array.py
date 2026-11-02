import numpy as np
import pytest

from pytweezer.drivers.motmaster import SIMULATED_SCRIPT_PARAMETERS
from pytweezer.experiment import optimisers, run_local
from pytweezer.experiments import tweezer_array
from pytweezer.experiments.tweezer_array import (
    SAVED_SEQUENCE_PARAMETERS,
    MolassesRecapture,
    OptimiseMolassesShims,
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
    """Images the array twice per shot; turning the tweezers off empties the second frame."""

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
        if self.sequencer.goes[-1].get("tTweezerSetVVA0") == 0.0:
            kept = np.zeros(len(SITES), dtype=bool)
        else:
            kept = loaded & (self.rng.random(len(SITES)) < SURVIVAL)
        return self._frame(loaded), self._frame(kept)


class FakeRig:
    """Knows no array unless a test gives it one."""

    def __init__(self):
        self.array = None

    def close_rpc(self):
        pass

    def get_array(self):
        return self.array


@pytest.fixture
def rig(monkeypatch):
    sequencer = FakeSequencer()
    rearrangement_rig = FakeRig()
    sequencer.rearrangement_rig = rearrangement_rig
    devices = {
        "Rb MotMaster": sequencer,
        "Rb HamCam": FakeHamCam(sequencer),
        "Rb Rearrangement Rig": rearrangement_rig,
    }
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: devices[name],
    )
    monkeypatch.setattr(tweezer_array, "get_saved_parameters", lambda: SAVED)
    monkeypatch.setattr(optimisers, "get_saved_parameters", lambda: SAVED)
    return sequencer


def test_a_point_measures_the_loading_and_survival_put_in(rig):
    # Also checks the calibration: the wrong sites or threshold would miss these.
    measurement = run_local(TweezerImaging, **FAST)
    assert measurement.results["loading"][0] == pytest.approx(LOADING, abs=0.1)
    assert measurement.results["survival"][0] == pytest.approx(SURVIVAL, abs=0.05)
    assert measurement.results["fidelity"][0] > 0.99


@pytest.mark.parametrize(
    ("array_on_rig", "sites_from"),
    [({"sites": len(SITES)}, "rig"), (None, "argument")],
    ids=["rig knows", "rig does not know"],
)
def test_calibration_finds_as_many_sites_as_the_rig_says_are_on_the_slm(
    rig, array_on_rig, sites_from
):
    rig.rearrangement_rig.array = array_on_rig
    sites_argument = len(SITES) if array_on_rig is None else 10

    measurement = run_local(TweezerImaging, **FAST, sites=sites_argument)

    assert measurement.status == "completed", measurement.attrs.get("error")
    assert len(measurement.constants["trap_sites"]) == len(SITES)
    assert measurement.constants["sites_from"] == sites_from


def test_every_shot_runs_at_the_saved_parameters_unless_the_form_sets_one(rig):
    run_local(TweezerImaging, **FAST, **{"rb.tMolCoolVCO1": 7.5})
    expected = {
        parameter: SAVED[key] for parameter, key in SAVED_SEQUENCE_PARAMETERS.items()
    } | {"tMolCoolVCO1": 7.5}
    for go in rig.goes:
        assert {name: go[name] for name in expected} == expected


@pytest.mark.parametrize(
    ("experiment", "point_drop"),
    [(TweezerImaging, (0, None)), (MolassesRecapture, (1, 3))],
)
def test_only_calibration_turns_the_tweezers_off_and_each_point_drops_as_its_measurement_needs(
    rig, experiment, point_drop
):
    run_local(experiment, **FAST)
    calibration, points = (
        rig.goes[: FAST["calibration_shots"]],
        rig.goes[FAST["calibration_shots"] :],
    )
    assert all(go["BackgroundDrop"] == 1 for go in calibration)
    assert all(
        go["tTweezerSetVVA0"] == go["tTweezerSetVVA1"] == 0.0 for go in calibration
    )
    assert len(points) == FAST["shots"]
    for go in points:
        assert "tTweezerSetVVA0" not in go
        assert (go["BackgroundDrop"], go.get("tDropDuration")) == point_drop


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


def test_each_molasses_evaluation_runs_at_its_shims(rig):
    measurement = run_local(OptimiseMolassesShims, **FAST, n_calls=2, n_initial=2)
    assert measurement.status == "completed", measurement.attrs["error"]
    evaluations = rig.goes[FAST["calibration_shots"] :]
    for point in range(2):
        shims = tuple(
            measurement.results[setting][point]
            for setting in ("shim_x", "shim_y", "shim_z")
        )
        for go in evaluations[point * FAST["shots"] : (point + 1) * FAST["shots"]]:
            sent = (
                go["tMolassesShimXccValue"],
                go["tMolassesShimYccValue"],
                go["tMolassesShimZccValue"],
            )
            assert sent == shims


@pytest.mark.parametrize(
    ("loading_floor", "expected"),
    [(0.5, "floor plus recapture"), (0.9, "loading")],
)
def test_the_molasses_score_counts_recapture_only_once_the_array_loads(
    rig, loading_floor, expected
):
    measurement = run_local(
        OptimiseMolassesShims,
        **FAST,
        n_calls=1,
        n_initial=1,
        loading_floor=loading_floor,
    )
    results = measurement.results
    score = {
        "floor plus recapture": loading_floor + results["survival"][0],
        "loading": results["loading"][0],
    }
    assert results["objective"][0] == pytest.approx(score[expected])
