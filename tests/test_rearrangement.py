"""Unit tests for the rearrangement coordinator: construction, role resolution,
status/test, and the no-GPU fallback. The real ARM pipeline needs a CUDA GPU and
is not exercised here; these tests cover the wiring and the graceful-degradation
paths that must work on any machine.
"""

import numpy as np
import pytest

from pytweezer.coordinators import rearrangement as rearr
from pytweezer.coordinators.rearrangement import Rearrangement
from pytweezer.drivers.slm import SimulatedSLM
from pytweezer.servers import device_server


class FakeCamera:
    def __init__(self):
        self.stopped = 0
        self.started = 0

    def is_connected(self):
        return True

    def start_acquisition(self):
        self.started += 1

    def acquire_n_frames(self, n):
        return np.zeros((n, 8, 8), dtype=np.uint16)

    def stop_acquisition(self):
        self.stopped += 1


def _coord():
    return Rearrangement({"camera": FakeCamera(), "slm": SimulatedSLM()}, {})


# --------------------------------------------------------------------------- #
# Construction / roles
# --------------------------------------------------------------------------- #

def test_requires_camera_and_slm_roles():
    coord = _coord()
    assert coord.camera is not None
    assert coord.slm is not None


def test_missing_slm_role_names_available():
    with pytest.raises(KeyError, match="needs a sub-device with role"):
        Rearrangement({"camera": FakeCamera()}, {})


def test_phasemask_kwargs_merge_config_over_defaults():
    coord = Rearrangement(
        {"camera": FakeCamera(), "slm": SimulatedSLM()},
        {"phasemask": {"wavelength_um": 0.780}},
    )
    assert coord.phasemask_kwargs["wavelength_um"] == 0.780
    # untouched defaults survive the merge
    assert coord.phasemask_kwargs["slm_res"] == (1024, 1024)


# --------------------------------------------------------------------------- #
# status / test (no GPU needed)
# --------------------------------------------------------------------------- #

def test_status_shape():
    status = _coord().status()
    assert set(status) == {
        "gpu_available", "initialised", "roles", "camera_connected", "slm_connected"
    }
    assert status["initialised"] is False
    assert status["roles"] == ["camera", "slm"]
    assert status["slm_connected"] is True


def test_test_returns_two_images():
    img0, img1 = _coord().test()
    assert img0.shape == (10, 10)
    assert img1.shape == (10, 10)


def test_get_slm_temperature_proxies_to_slm():
    assert isinstance(_coord().get_slm_temperature(), float)


# --------------------------------------------------------------------------- #
# No-GPU fallback
# --------------------------------------------------------------------------- #

def test_initialise_without_cupy_raises_clearly(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", False)
    coord = _coord()
    with pytest.raises(RuntimeError, match="needs cupy/lap and a CUDA GPU"):
        coord.initialise(
            data1=np.zeros((4, 2)), data2=np.zeros((4, 2)),
            array_shape1=(1, 2), array_shape2=(1, 2),
            d0=0.5, fps=30, threshold=1.0, grid_positions=None,
        )


def test_arm_without_cupy_raises_clearly(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", False)
    with pytest.raises(RuntimeError, match="needs cupy/lap and a CUDA GPU"):
        _coord().arm_rearrangement()


def test_status_still_works_without_cupy(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", False)
    status = _coord().status()
    assert status["gpu_available"] is False


def test_arm_before_initialise_raises(monkeypatch):
    # With the GPU flag on, arm should still refuse until initialise() has run.
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    with pytest.raises(RuntimeError, match="not initialised"):
        _coord().arm_rearrangement()


# --------------------------------------------------------------------------- #
# ARM pipeline: iter_rearrangement_sequence streamed to the SLM
# --------------------------------------------------------------------------- #

class FakePM:
    """Stands in for the GPU phasemask generator's rearrangement entry point.

    Mirrors the real streaming signature: a generator yielding one frame at a time.
    """

    def __init__(self, n_frames=5):
        self.n_frames = n_frames
        self.called_with = None

    def iter_rearrangement_sequence(self, terms1, terms2, occ_mask, d0,
                                    profile="minimum_jerk", to_host=True):
        self.called_with = dict(
            terms1=terms1, terms2=terms2, occ_mask=occ_mask, d0=d0,
            profile=profile, to_host=to_host,
        )
        for _ in range(self.n_frames):
            yield np.zeros((8, 8), dtype=np.uint8)


def _patch_occupancy(monkeypatch):
    """Fake occupancy extraction so no real analysis/GPU/C++ is needed."""
    from pytweezer.analysis import analysis as an
    # Force the numpy branch so the patched sum_pixel_values is the one used.
    monkeypatch.setattr(rearr, "USE_SUM_CPP", False)
    monkeypatch.setattr(an, "morphological_tophat_high_pass", lambda img, feature_size: img)
    # np.fliplr is applied to this before thresholding, so [[5, 0]] -> [0, 5].
    monkeypatch.setattr(an, "sum_pixel_values",
                        lambda img, grid, shape, window_size: np.array([[5.0, 0.0]]))


def _armed_coord(pm, slm=None, camera=None):
    camera = camera or FakeCamera()
    slm = slm or SimulatedSLM()
    coord = Rearrangement({"camera": camera, "slm": slm}, {})
    coord._initialised = True
    coord._state = dict(
        PM=pm,
        terms1=("w1", "phi1", "x1", "y1", (1, 2)),
        terms2=("w2", "phi2", "x2", "y2", (1, 2)),
        pm_init_uint8=np.zeros((1024, 1024), dtype=np.uint8),
        d0=0.5, fps=1000.0, threshold=1.0, grid_positions=None, roi=None,
        profile="minimum_jerk",
    )
    return coord


def test_arm_streams_sequence_to_slm(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    _patch_occupancy(monkeypatch)

    slm = SimulatedSLM()
    pm = FakePM(n_frames=5)
    coord = _armed_coord(pm, slm=slm)

    img0, img1 = coord.arm_rearrangement()

    # Sequence generated from the stored terms, thresholded occupancy passed through.
    assert pm.called_with["terms1"] == coord._state["terms1"]
    assert pm.called_with["d0"] == 0.5
    np.testing.assert_array_equal(pm.called_with["occ_mask"], np.array([False, True]))
    # Frames are consumed on the writer thread, so the copy is left to it.
    assert pm.called_with["to_host"] is False
    # SLM saw the initial mask (1) plus every streamed frame (5), in-process.
    assert slm.frames_written == 1 + 5
    assert img0.shape == (8, 8) and img1.shape == (8, 8)


def test_arm_passes_configured_profile(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    _patch_occupancy(monkeypatch)

    pm = FakePM(n_frames=2)
    coord = _armed_coord(pm)
    coord._state["profile"] = "linear"
    coord.arm_rearrangement()

    assert pm.called_with["profile"] == "linear"


def test_arm_reraises_slm_upload_error(monkeypatch):
    """An upload failure must surface, not be swallowed by the writer thread."""
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    _patch_occupancy(monkeypatch)

    class ExplodingSLM(SimulatedSLM):
        def update_mask(self, mask_array):
            if getattr(self, "_armed", False):
                raise RuntimeError("DMA failed")
            self._armed = True
            super().update_mask(mask_array)

    coord = _armed_coord(FakePM(n_frames=4), slm=ExplodingSLM())
    with pytest.raises(RuntimeError, match="DMA failed"):
        coord.arm_rearrangement()


# --------------------------------------------------------------------------- #
# Teardown
# --------------------------------------------------------------------------- #

def test_shutdown_is_safe_before_initialise():
    coord = _coord()
    coord.shutdown()  # must not touch the camera when never initialised
    assert coord.camera.stopped == 0


def test_shutdown_stops_camera_when_initialised():
    coord = _coord()
    coord._initialised = True  # simulate a prior initialise() without the GPU
    coord.shutdown()
    assert coord.camera.stopped == 1
    assert not coord._initialised


# --------------------------------------------------------------------------- #
# End-to-end wiring through build_spec (simulated rig)
# --------------------------------------------------------------------------- #

SIM_RIG = {
    "simulate": True,
    "devices": {
        "Rb Rearrangement Cam": {
            "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
            "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
            "role": "camera", "stream_name": None, "timeout": 2.0,
        },
        "Rb SLM": {
            "class": "pytweezer.drivers.slm:SLM",
            "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
            "teardown": "close", "role": "slm",
        },
    },
    "coordinator": "pytweezer.coordinators.rearrangement:Rearrangement",
}


def test_simulated_rig_builds_and_resolves_roles():
    spec = device_server.build_spec("Rb Rearrangement Rig", conf=SIM_RIG)
    try:
        assert set(spec.targets) == {"coordinator", "rbrearrangementcam", "rbslm"}
        coord = spec.targets["coordinator"]
        # The coordinator found its backends by role, and they are the sub-devices.
        assert coord.camera is spec.targets["rbrearrangementcam"]
        assert coord.slm is spec.targets["rbslm"]
        assert coord.status()["roles"] == ["camera", "slm"]
    finally:
        spec.teardown()


def test_simulated_rig_slm_addressable_and_driven_in_process():
    spec = device_server.build_spec("Rb Rearrangement Rig", conf=SIM_RIG)
    try:
        slm = spec.targets["rbslm"]
        slm.update_mask(np.zeros((1024, 1024), dtype=np.uint8))
        assert slm.frames_written == 1
    finally:
        spec.teardown()
