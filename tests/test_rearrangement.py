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
            "phases": np.zeros(rows * columns),
            "optimised_weights": np.ones(rows * columns),
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


@pytest.mark.parametrize(
    "generator_class",
    [
        "OptimisationBasedPhasemaskGeneratorGPU",
        "OptimisationBasedPhasemaskGeneratorGPUSeidel",
    ],
)
@pytest.mark.parametrize("sequence", ["planned", "generated"])
def test_a_rearrangement_leaves_float32_initial_terms_untouched(
    generator_class, sequence
):
    # The next shot plans from the same terms, so walking them would start it wrong.
    if not rearrangement._HAS_CUPY:
        pytest.skip("needs a CUDA GPU")
    import cupy as cp

    from pytweezer.helper_scripts import phasemask

    generator = getattr(phasemask, generator_class)(slm_res=(64, 64))
    y_um, x_um = np.mgrid[0:2, 0:3] * 5.0
    terms1 = [
        cp.asarray(values, dtype=cp.float32)
        for values in (np.ones(6), np.zeros(6), x_um.ravel(), y_um.ravel())
    ]
    terms2 = [
        cp.asarray(values, dtype=cp.float32)
        for values in (np.ones(2), np.zeros(2), [20.0, 25.0], [20.0, 20.0])
    ]
    before = [term.get() for term in terms1]
    occupied = np.array([True, False, True, True, False, False])

    if sequence == "planned":
        plan = generator.plan_rearrangement(terms1, terms2, occupied, d0=0.5)
        for _ in generator.iter_rearrangement_sequence_timings(plan):
            pass
    else:
        generator.generate_rearrangement_sequence(terms1, terms2, occupied, d0=0.5)

    for term, original in zip(terms1, before, strict=True):
        np.testing.assert_array_equal(term.get(), original)
