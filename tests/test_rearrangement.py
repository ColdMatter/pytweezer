"""The rearrangement coordinator as the device server builds it from config."""

import numpy as np
import pytest

from pytweezer.coordinators import rearrangement
from pytweezer.experiment import layouts
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
    """Stands in for the GPU hologram: a mask filled with the number of traps."""

    def hologram(generator, x_um, y_um, **_):
        return {
            "mask": np.full((1024, 1024), len(x_um), dtype=np.uint8),
            "x_um": np.asarray(x_um, dtype=float),
            "y_um": np.asarray(y_um, dtype=float),
            "weights": np.ones(len(x_um)),
        }

    monkeypatch.setattr(rearrangement, "_HAS_CUPY", True)
    monkeypatch.setattr(array_hologram, "make_generator", lambda settings: object())
    monkeypatch.setattr(array_hologram, "layout_hologram", hologram)


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


def test_rig_reports_the_array_it_put_on_the_slm(rig, fake_gpu):
    hologram = rig.make_array(**ARRAY, **OPTICS)

    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    array = rig.get_array()
    assert array["sites"] == 12
    assert {key: array[key] for key in ARRAY} == ARRAY
    assert len(array["x_um"]) == len(array["y_um"]) == 12


def test_rig_does_not_know_the_array_once_something_else_writes_the_slm(rig, fake_gpu):
    rig.make_array(**ARRAY, **OPTICS)

    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert rig.get_array() is None


def test_rig_knows_its_array_again_when_its_mask_is_put_back(rig, fake_gpu):
    hologram = rig.make_array(**ARRAY, **OPTICS)
    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    rig.slm.update_mask(hologram["mask"].copy())

    assert rig.get_array()["sites"] == 12


def test_rig_puts_a_named_layout_on_the_slm_and_reports_it(rig, fake_gpu):
    hologram = rig.make_layout([0.0, 5.0, 0.0], [0.0, 0.0, 5.0], name="ell", **OPTICS)

    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    array = rig.get_array()
    assert array["sites"] == 3 and array["layout"] == "ell"
    assert array["x_um"] == [0.0, 5.0, 0.0] and "rows" not in array


def test_rig_array_sites_are_the_rectangle_preset_positions(rig, fake_gpu):
    rig.make_array(**ARRAY, **OPTICS)

    expected = layouts.rectangle(3, 4, 5.0, 2.5)
    np.testing.assert_allclose(rig.get_array()["x_um"], [s.x_um for s in expected])
    np.testing.assert_allclose(rig.get_array()["y_um"], [s.y_um for s in expected])


def test_focal_margin_is_unchanged_for_a_rectangle_and_capped_for_sparse_sites():
    x, y = np.meshgrid(np.arange(3) * 5.0, np.arange(3) * 5.0)
    margin = array_hologram.focal_margin_um
    assert margin(x.ravel(), y.ravel(), 0.9) == pytest.approx(2 * 5.0 + 2.7)
    assert margin(np.array([0.0, 300.0]), np.zeros(2), 0.9) == pytest.approx(22.7)
    assert margin(np.array([1.0]), np.array([1.0]), 0.9) == pytest.approx(2.7)


def test_focal_margin_of_a_large_array_uses_little_memory():
    import tracemalloc

    x, y = np.meshgrid(np.arange(45) * 5.0, np.arange(45) * 5.0)

    tracemalloc.start()
    margin = array_hologram.focal_margin_um(x.ravel(), y.ravel(), 0.9)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert margin == pytest.approx(12.7)
    assert peak < 5e6, "an all-pairs distance matrix would take ~65 MB here"


def test_reachable_field_is_lambda_f_over_pitch_centred_on_the_zeroth_order():
    saved = {
        "slm_wavelength": 0.852e-6,
        "slm_focal_length": 17.3e-3,
        "slm_pixel_pitch": 17e-6,
        "slm_blaze_x": 48e-6,
        "slm_blaze_y": 4e-6,
    }
    x0, y0, width, height = array_hologram.reachable_field_um(saved)

    side = 0.852 * 17300 / 17
    assert (width, height) == pytest.approx((side, side))
    assert (x0 + width / 2, y0 + height / 2) == pytest.approx((-48.0, -4.0))


def test_rig_knows_no_array_before_making_one(rig):
    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    assert rig.get_array() is None


def test_rig_puts_the_computed_hologram_on_the_slm(rig):
    if not rearrangement._HAS_CUPY:
        pytest.skip("needs a CUDA GPU")

    hologram = rig.make_array(
        rows=2, columns=3, spacing_um=5.0, angle_deg=0.0, **OPTICS
    )

    assert hologram["mask"].dtype == np.uint8 and hologram["mask"].shape == (1024, 1024)
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    assert rig.get_array()["sites"] == 6


def test_rig_builds_a_new_generator_only_when_the_optics_change(
    rig, fake_gpu, monkeypatch
):
    built = []
    monkeypatch.setattr(
        array_hologram, "make_generator", lambda settings: built.append(settings) or 1
    )

    rig.make_array(**ARRAY, **OPTICS)
    rig.make_array(**ARRAY, **OPTICS)
    rig.make_array(**ARRAY, **(OPTICS | {"blaze_um": (50.0, 4.0)}))

    assert [settings["blaze_dx_dy_um"] for settings in built] == [
        (48.0, 4.0),
        (50.0, 4.0),
    ]
