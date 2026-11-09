"""The rearrangement coordinator as the device server builds it from config."""

import numpy as np
import pytest

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
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


class FakeGenerator:
    """Records the plans asked of it and plays ``n_frames`` blank frames for each."""

    def __init__(self):
        self.plans = []
        self.moving_idx, self.final_idx = [], []
        self.n_frames = 3
        self.pupil = None

    def plan_rearrangement(
        self, terms1, terms2, occ_mask, d0=0.5, profile="minimum_jerk", background=None
    ):
        self.plans.append(
            {
                "terms1": terms1,
                "terms2": terms2,
                "occ_mask": occ_mask,
                "d0": d0,
                "profile": profile,
                "background": background,
            }
        )
        return {
            "moving_idx": np.array(self.moving_idx),
            "final_idx": np.array(self.final_idx),
        }

    def iter_planned_sequence(self, plan, *, to_host=False, pupil=None):
        self.pupil = pupil
        for _ in range(self.n_frames):
            yield np.zeros((1024, 1024), dtype=np.uint8)


def three_site_target(generator, **_):
    return {
        "weights": np.ones(3),
        "optimised_weights": np.ones(3),
        "phases": np.zeros(3),
        "x_um": np.array([0.0, 5.0, 10.0]),
        "y_um": np.zeros(3),
        "uniformity": 0.99,
        "min_max_ratio": 0.98,
        "mse": 1e-3,
    }


@pytest.fixture
def fake_gpu(monkeypatch):
    """Stands in for the GPU hologram: a mask filled with the array's row count.

    Returns the :class:`FakeGenerator` the rig builds its arrays and moves with.
    """

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

    generator = FakeGenerator()
    monkeypatch.setattr(rearrangement, "_HAS_CUPY", True)
    monkeypatch.setattr(array_hologram, "make_generator", lambda settings: generator)
    monkeypatch.setattr(array_hologram, "array_hologram", hologram)
    monkeypatch.setattr(array_hologram, "target_terms", three_site_target)
    monkeypatch.setattr(
        array_hologram, "background_phase", lambda generator, **layers: layers
    )
    monkeypatch.setattr(
        array_hologram,
        "pupil_arrays",
        lambda generator, radius_um, dump_dx_um, dump_dy_um: (
            radius_um,
            dump_dx_um,
            dump_dy_um,
        ),
    )
    return generator


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


#: Where the 2x2 shot array's sites image, ``(y, x)`` in the HamCam ROI.
SHOT_SITES = np.array([(16, 16), (16, 48), (48, 16), (48, 48)])
SHOT_ARRAY = {"rows": 2, "columns": 2, "spacing_um": 5.0, "angle_deg": 0.0}
TARGET = {
    "kind": "square",
    "rows": 1,
    "columns": 3,
    "sites": 0,
    "spacing_um": 5.0,
    "angle_deg": 0.0,
    "iterations": 5,
    "damping": 0.5,
    "phase_randomness": 1.0,
}
SCORING = {"window_size": 5, "feature_size": 10}
HEADER = {"rid": 7}


def shot_frame(lit):
    y, x = np.mgrid[:64, :64]
    spots = np.array(
        [np.exp(-((y - sy) ** 2 + (x - sx) ** 2) / 4.5) for sy, sx in SHOT_SITES]
    )
    return (100 + 2000 * np.tensordot(lit, spots, axes=1)).astype(np.uint16)


def shot_threshold():
    """Halfway between the photons of an empty and a full site."""
    sites = dict(enumerate(map(tuple, SHOT_SITES)))

    def photons(lit):
        frame = an.morphological_tophat_high_pass(
            shot_frame(lit).astype(float), feature_size=SCORING["feature_size"]
        )
        return photons_at_sites([frame], sites, SCORING["window_size"]).mean()

    return (photons([1, 1, 1, 1]) + photons([0, 0, 0, 0])) / 2


class ShotCamera:
    """Hands out a shot's three frames and records the calls; can fail one read."""

    def __init__(self, frames, fail_at_frame=None):
        self.frames = np.asarray(frames)
        self.fail_at_frame = fail_at_frame
        self.calls = []

    def start_acquisition(self):
        self.calls.append("start")

    def stop_acquisition(self):
        self.calls.append("stop")

    def acquire_n_frames(self, n, start_frame=0, broadcast=False, header=None):
        self.calls.append(("read", n, start_frame, broadcast, header))
        if start_frame == self.fail_at_frame:
            raise TimeoutError("camera timed out")
        return self.frames[start_frame : start_frame + n]

    def reads(self):
        return [call[1:] for call in self.calls if call[0] == "read"]


def prepare_shot(rig, lit, fail_at_frame=None, optics=OPTICS):
    """Put the 2x2 array on the SLM, prepare a move and give the rig a HamCam showing ``lit``."""
    hologram = rig.make_array(**SHOT_ARRAY, **optics)
    rig.prepare_rearrangement(
        **TARGET,
        occupancy_sites=SHOT_SITES,
        threshold_photons=shot_threshold(),
        **SCORING,
    )
    rig.camera = ShotCamera(
        [shot_frame(lit), shot_frame([0, 0, 0, 0]), shot_frame([0, 0, 0, 0])],
        fail_at_frame,
    )
    return hologram


def test_a_shot_moves_the_occupied_atoms_and_returns_three_frames(rig, fake_gpu):
    hologram = prepare_shot(rig, lit=[1, 0, 1, 0])
    fake_gpu.moving_idx, fake_gpu.final_idx = [0, 2], [1, 0]

    shot = rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    np.testing.assert_array_equal(shot["occupied"], [True, False, True, False])
    np.testing.assert_array_equal(shot["destination"], [1, -1, 0, -1])
    assert shot["frames"].shape[0] == 3
    assert rig.camera.reads() == [(1, 0, True, HEADER), (2, 1, True, HEADER)]
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])


def test_a_shot_with_no_atoms_plans_nothing_and_still_reads_the_follow_up_frames(
    rig, fake_gpu
):
    prepare_shot(rig, lit=[0, 0, 0, 0])

    shot = rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    assert shot["n_frames"] == 0
    assert fake_gpu.plans == []
    assert rig.camera.reads() == [(1, 0, True, HEADER), (2, 1, True, HEADER)]


def test_destinations_send_surplus_atoms_nowhere():
    destination = rearrangement.destinations(5, [0, 3], [1, 0])

    np.testing.assert_array_equal(destination, [1, -1, -1, 0, -1])
    targets = destination[destination >= 0]
    assert len(set(targets)) == len(targets)


def test_a_failed_occupancy_read_names_arm_delay_and_stops_the_camera(rig, fake_gpu):
    hologram = prepare_shot(rig, lit=[1, 0, 1, 0], fail_at_frame=0)

    with pytest.raises(RuntimeError, match="arm_delay"):
        rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    assert rig.camera.calls[-1] == "stop"
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])


def test_a_failed_follow_up_read_still_stops_the_camera_and_restores_the_mask(
    rig, fake_gpu
):
    hologram = prepare_shot(rig, lit=[1, 0, 1, 0], fail_at_frame=1)
    fake_gpu.moving_idx, fake_gpu.final_idx = [0, 2], [1, 0]

    with pytest.raises(TimeoutError):
        rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    assert rig.camera.calls[-1] == "stop"
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])


def test_a_shot_refuses_once_something_else_has_written_the_slm(rig, fake_gpu):
    prepare_shot(rig, lit=[1, 0, 1, 0])
    rig.slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))

    with pytest.raises(RuntimeError):
        rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    assert rig.camera.calls == []


def test_prepare_rearrangement_refuses_without_an_array_on_the_slm(rig, fake_gpu):
    with pytest.raises(RuntimeError):
        rig.prepare_rearrangement(
            **TARGET,
            occupancy_sites=SHOT_SITES,
            threshold_photons=0.0,
            **SCORING,
        )


def test_prepare_rearrangement_refuses_without_a_gpu(rig, monkeypatch):
    monkeypatch.setattr(rearrangement, "_HAS_CUPY", False)

    with pytest.raises(RuntimeError, match="needs cupy/lap and a CUDA GPU"):
        rig.prepare_rearrangement(
            **TARGET,
            occupancy_sites=SHOT_SITES,
            threshold_photons=0.0,
            **SCORING,
        )


def test_plan_is_built_on_the_layers_make_array_used(rig, fake_gpu):
    optics = OPTICS | {"zernike": True}
    prepare_shot(rig, lit=[1, 0, 0, 0], optics=optics)
    fake_gpu.moving_idx, fake_gpu.final_idx = [0], [1]

    rig.rearrange(d0_um=0.5, profile="minimum_jerk", header=HEADER)

    assert fake_gpu.plans[0]["background"] == {
        "blaze_um": (48.0, 4.0),
        "fresnel_focal_length_mm": 1125.0,
        "zernike": True,
    }
    assert fake_gpu.pupil == (8704.0, 100.0, 0.0)


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
