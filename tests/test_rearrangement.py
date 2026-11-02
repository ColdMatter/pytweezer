"""The rearrangement coordinator as the device server builds it from config."""

import numpy as np
import pytest

from pytweezer.coordinators import rearrangement
from pytweezer.helper_scripts import array_hologram
from pytweezer.servers import device_server

SIM_RIG = {
    "simulate": True,
    "devices": {
        "Rb Rearrangement Cam": {
            "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
            "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
            "role": "camera",
            "stream_name": None,
            "timeout": 2.0,
        },
        "Rb SLM": {
            "class": "pytweezer.drivers.slm:SLM",
            "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
            "teardown": "close",
            "role": "slm",
        },
    },
    "coordinator": "pytweezer.coordinators.rearrangement:Rearrangement",
}


def test_simulated_rig_builds_and_resolves_roles():
    spec = device_server.build_spec("Rb Rearrangement Rig", conf=SIM_RIG)
    try:
        assert set(spec.targets) == {"coordinator", "rbrearrangementcam", "rbslm"}
        coord = spec.targets["coordinator"]
        # The coordinator holds the sub-device objects themselves, so it drives
        # them in-process rather than over RPC.
        assert coord.camera is spec.targets["rbrearrangementcam"]
        assert coord.slm is spec.targets["rbslm"]
        assert coord.status()["roles"] == ["camera", "slm"]
    finally:
        spec.teardown()


@pytest.fixture
def rig():
    spec = device_server.build_spec("Rb Rearrangement Rig", conf=SIM_RIG)
    yield spec.targets["coordinator"]
    spec.teardown()


@pytest.fixture
def fake_gpu(monkeypatch):
    """Stands in for the GPU hologram: a mask filled with the array's row count."""

    def hologram(generator, *, rows, columns, **_):
        y_um, x_um = np.mgrid[0:rows, 0:columns] * 5.0
        return {
            "mask": np.full((1024, 1024), rows, dtype=np.uint8),
            "x_um": x_um.ravel(),
            "y_um": y_um.ravel(),
            "weights": np.ones(rows * columns),
        }

    monkeypatch.setattr(rearrangement, "_HAS_CUPY", True)
    monkeypatch.setattr(array_hologram, "make_generator", object)
    monkeypatch.setattr(array_hologram, "array_hologram", hologram)


ARRAY = {"rows": 3, "columns": 4, "spacing_um": 5.0, "angle_deg": 2.5}


def test_rig_reports_the_array_it_put_on_the_slm(rig, fake_gpu):
    hologram = rig.make_array(**ARRAY)

    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    array = rig.get_array()
    assert array["sites"] == 12
    assert {key: array[key] for key in ARRAY} == ARRAY
    assert len(array["x_um"]) == len(array["y_um"]) == 12


def test_rig_does_not_know_the_array_once_something_else_writes_the_slm(rig, fake_gpu):
    rig.make_array(**ARRAY)

    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert rig.get_array() is None


def test_rig_knows_its_array_again_when_its_mask_is_put_back(rig, fake_gpu):
    hologram = rig.make_array(**ARRAY)
    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    rig.slm.update_mask(hologram["mask"].copy())

    assert rig.get_array()["sites"] == 12


def test_rig_knows_no_array_before_making_one(rig):
    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert rig.get_array() is None


def test_rig_puts_the_computed_hologram_on_the_slm(rig):
    pytest.importorskip("cupy")

    hologram = rig.make_array(
        rows=2, columns=3, spacing_um=5.0, angle_deg=0.0, iterations=5
    )

    assert hologram["mask"].dtype == np.uint8 and hologram["mask"].shape == (1024, 1024)
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    assert rig.get_array()["sites"] == 6
