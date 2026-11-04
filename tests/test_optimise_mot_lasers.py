from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from pytweezer.experiment import run_local
from pytweezer.experiment.catalogue import Catalogue
from pytweezer.experiment.introspect import describe
from pytweezer.experiments.caf_fluorescence import LASERS
from pytweezer.optimisers.optimise_mot_lasers import (
    LASER_KEYS,
    OptimiseMOTLasers,
    create_saved_set_points,
)
from tests.helpers import in_process_client
from tests.test_caf_fluorescence import START, FakeCaFMotMaster, FakeCamera

MODULE = "pytweezer.optimisers.optimise_mot_lasers"
SPANS = {laser: OptimiseMOTLasers.stages[i].span for i, laser in enumerate(LASERS)}
#: Where each laser's fluorescence peaks: a little off its saved set point, between grid points.
PEAKS = {laser: START[laser] + 0.3 * SPANS[laser] for laser in LASERS}


def fluorescence(field, set_points, iteration):
    """Bright when every laser is at its peak, falling off quadratically in its span."""
    if field == 0.0:
        return np.zeros((2, 3))
    penalty = sum(
        ((set_points[laser] - PEAKS[laser]) / SPANS[laser]) ** 2 for laser in LASERS
    )
    return np.full((2, 3), 60000 - 20000 * penalty)


@pytest.fixture
def caf(monkeypatch):
    camera = FakeCamera()
    camera.frame = fluorescence
    motmaster = FakeCaFMotMaster(camera)
    fakes = {"CaF MotMaster": motmaster, "CaF ThorCam": camera}
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device",
        lambda name, timeout: fakes[name],
    )
    monkeypatch.setattr(f"{MODULE}.get_device", lambda name: fakes[name])
    return SimpleNamespace(motmaster=motmaster, camera=camera)


@pytest.fixture
def saved_set_points(monkeypatch):
    monkeypatch.setattr(
        "pytweezer.experiment.client.get_saved_parameters",
        lambda: {LASER_KEYS[laser]: START[laser] for laser in LASERS},
    )


def test_the_stages_and_defaults_match_the_notebook():
    stages = OptimiseMOTLasers.stages
    assert [stage.setting for stage in stages] == list(LASERS)
    assert [stage.span for stage in stages] == [
        10e-6,
        30e-6,
        40e-6,
        150e-6,
        150e-6,
        40e-6,
    ]
    assert [stage.points for stage in stages] == [30, 30, 20, 30, 30, 30]
    assert [stage.save_as for stage in stages] == [
        f"caf_laser_{laser}" for laser in LASERS
    ]
    assert {stage.setting for stage in stages if stage.run} == {"v0", "v1", "TCool"}


class Spying(OptimiseMOTLasers):
    calls: ClassVar[list[dict]] = []

    def measure(self, **set_points):
        Spying.calls.append(set_points)
        return super().measure(**set_points)


def test_each_enabled_laser_is_scanned_around_its_saved_set_point_and_best_values_chain(
    caf, saved_set_points
):
    Spying.calls.clear()
    measurement = run_local(Spying)
    assert measurement.status == "completed", measurement.attrs["error"]
    for laser in ("v0", "v1", "TCool"):
        assert measurement.constants[f"proposed.{LASER_KEYS[laser]}"] == pytest.approx(
            PEAKS[laser], abs=SPANS[laser] / 20
        )
    assert not any(
        f"proposed.{LASER_KEYS[laser]}" in measurement.constants
        for laser in ("BX", "v2", "v3")
    )
    assert "thorcam_sigma_x" in measurement.results
    first_v1_call = next(call for call in Spying.calls if "v1" in call)
    assert first_v1_call["v0"] == pytest.approx(PEAKS["v0"], abs=SPANS["v0"] / 20)
    first_tcool_call = next(call for call in Spying.calls if "TCool" in call)
    assert first_tcool_call["v1"] == pytest.approx(PEAKS["v1"], abs=SPANS["v1"] / 20)


def test_a_laser_with_no_saved_set_point_names_the_key(caf, monkeypatch):
    monkeypatch.setattr(
        "pytweezer.experiment.client.get_saved_parameters",
        lambda: {LASER_KEYS[laser]: START[laser] for laser in LASERS[1:]},
    )
    with pytest.raises(KeyError, match="caf_laser_v0"):
        run_local(OptimiseMOTLasers)
    assert caf.motmaster.log == []


def test_apply_ramps_only_the_accepted_lasers(caf):
    OptimiseMOTLasers.apply({LASER_KEYS["v0"]: 494.4319})
    assert caf.motmaster.log == [("ramp", "v0", 494.4319)]


def test_apply_moves_nothing_if_a_key_is_not_a_laser(caf):
    with pytest.raises(KeyError, match="demo_power"):
        OptimiseMOTLasers.apply({LASER_KEYS["v0"]: 494.4319, "demo_power": 1.0})
    assert caf.motmaster.log == []


def test_apply_tries_every_laser_and_names_the_ones_it_missed(caf):
    caf.motmaster.refused_ramps = {("v0", 494.4319)}
    with pytest.raises(RuntimeError, match=r"\{'v0': 494\.4319\}"):
        OptimiseMOTLasers.apply(
            {LASER_KEYS["v0"]: 494.4319, LASER_KEYS["BX"]: 564.5824}
        )
    assert caf.motmaster.log == [("ramp", "BX", 564.5824)]


@pytest.fixture
def manager_client(make_manager, monkeypatch):
    client = in_process_client(make_manager())
    monkeypatch.setattr(
        "pytweezer.experiment.client.ExperimentManagerClient", lambda: client
    )
    return client


def test_create_saved_set_points_creates_only_the_missing_ones(manager_client):
    manager_client.set_saved_parameters(
        {LASER_KEYS["v0"]: 1.0}, "earlier", "me", create=True
    )
    created = create_saved_set_points(START)
    assert created == [LASER_KEYS[laser] for laser in LASERS[1:]]
    saved = {p.key: p for p in manager_client.saved_parameters()}
    assert saved[LASER_KEYS["v0"]].value == 1.0
    assert saved[LASER_KEYS["BX"]].value == START["BX"]
    assert (saved[LASER_KEYS["BX"]].group, saved[LASER_KEYS["BX"]].unit) == (
        "CaF lasers",
        "THz",
    )
    assert create_saved_set_points(START) == []


def test_create_saved_set_points_reads_the_lock_by_default(caf, manager_client):
    assert create_saved_set_points() == list(LASER_KEYS.values())
    saved = {p.key: p.value for p in manager_client.saved_parameters()}
    assert saved == {LASER_KEYS[laser]: START[laser] for laser in LASERS}


def test_create_saved_set_points_refuses_an_unknown_laser(manager_client):
    with pytest.raises(KeyError, match="v9"):
        create_saved_set_points({"v9": 1.0})
    assert manager_client.saved_parameters() == []


def test_the_optimiser_is_in_the_catalogue_and_imports_without_hardware():
    assert MODULE in Catalogue("pytweezer.optimisers").modules()
    description = describe(MODULE)
    assert description["error"] is None
    [optimiser] = description["classes"]
    assert optimiser["class_name"] == "OptimiseMOTLasers"
    assert optimiser["optimiser"]["applies"] is True
    assert optimiser["optimiser"]["proposes"] == list(LASER_KEYS.values())
    assert optimiser["optimiser"]["group"] == "CaF lasers"
