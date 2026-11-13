"""The tweezer array the SLM device puts on itself, as the device server builds it from config."""

import numpy as np
import pytest

from pytweezer.helper_scripts import array_hologram, rearrangement
from pytweezer.servers import device_server

SIM_SLM = {
    "class": "pytweezer.drivers.slm:SLM",
    "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
    "teardown": "close",
    "simulate": True,
}


@pytest.fixture
def slm():
    spec = device_server.build_spec("Rb SLM", conf=SIM_SLM)
    yield next(iter(spec.targets.values()))
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
    monkeypatch.setattr(array_hologram, "make_generator", lambda settings: object())
    monkeypatch.setattr(array_hologram, "array_hologram", hologram)


ARRAY = {"rows": 3, "columns": 4, "spacing_um": 5.0, "angle_deg": 2.5}
OPTICS = {
    "generator_settings": {
        "wavelength_um": 0.852,
        "focal_length_mm": 17.3,
        "slm_pitch_um": 17.0,
        "input_beam_waist_mm": 16.0,
        "slm_res": (1024, 1024),
        "zernike_coeff_dict": {5: 1.195},
    },
    "envelope": None,
    "blaze_um": (48.0, 4.0),
    "fresnel_focal_length_mm": 1125.0,
    "zernike": False,
    "pupil_radius_um": 8704.0,
    "pupil_dump_um": (100.0, 0.0),
    "iterations": 5,
    "damping": 0.5,
    "phase_randomness": 1.0,
}


def test_slm_reports_the_array_it_was_given(slm, fake_gpu):
    hologram = slm.make_array(**ARRAY, **OPTICS)

    np.testing.assert_array_equal(slm.last_mask, hologram["mask"])
    array = slm.get_array()
    assert array["sites"] == 12
    assert {key: array[key] for key in ARRAY} == ARRAY
    assert len(array["x_um"]) == len(array["y_um"]) == 12


def test_slm_does_not_know_the_array_once_another_mask_is_written(slm, fake_gpu):
    slm.make_array(**ARRAY, **OPTICS)

    slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert slm.get_array() is None


def test_slm_knows_its_array_again_when_its_mask_is_put_back(slm, fake_gpu):
    hologram = slm.make_array(**ARRAY, **OPTICS)
    slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    slm.update_mask(hologram["mask"].copy())

    assert slm.get_array()["sites"] == 12


def test_slm_knows_no_array_before_making_one(slm):
    slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert slm.get_array() is None


def test_slm_displays_the_computed_hologram(slm):
    if not rearrangement._HAS_CUPY:
        pytest.skip("needs a CUDA GPU")

    hologram = slm.make_array(
        rows=2, columns=3, spacing_um=5.0, angle_deg=0.0, **OPTICS
    )

    assert hologram["mask"].dtype == np.uint8 and hologram["mask"].shape == (1024, 1024)
    np.testing.assert_array_equal(slm.last_mask, hologram["mask"])
    assert slm.get_array()["sites"] == 6


def test_slm_builds_a_new_generator_only_when_the_optics_change(
    slm, fake_gpu, monkeypatch
):
    built = []
    monkeypatch.setattr(
        array_hologram, "make_generator", lambda settings: built.append(settings) or 1
    )

    slm.make_array(**ARRAY, **OPTICS)
    slm.make_array(**ARRAY, **OPTICS)
    slm.make_array(**ARRAY, **(OPTICS | {"blaze_um": (50.0, 4.0)}))

    assert [settings["blaze_dx_dy_um"] for settings in built] == [
        (48.0, 4.0),
        (50.0, 4.0),
    ]
