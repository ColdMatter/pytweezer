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
# Simplified ARM pipeline: generate_rearrangement_sequence + slm.run_sequence
# --------------------------------------------------------------------------- #

class FakePM:
    """Stands in for the GPU phasemask generator's rearrangement entry point."""

    def __init__(self, n_frames=5):
        self.n_frames = n_frames
        self.called_with = None

    def generate_rearrangement_sequence(self, terms1, terms2, occ_mask, d0):
        self.called_with = dict(terms1=terms1, terms2=terms2, occ_mask=occ_mask, d0=d0)
        seq = np.zeros((self.n_frames, 8, 8), dtype=np.uint8)
        return seq, "debug"


def test_arm_generates_sequence_and_plays_it(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    # Occupancy extraction is faked so no real analysis/GPU is needed.
    from pytweezer.analysis import analysis as an
    monkeypatch.setattr(an, "morphological_tophat_high_pass", lambda img, feature_size: img)
    # np.fliplr is applied to this before thresholding, so [[5, 0]] -> [0, 5].
    monkeypatch.setattr(an, "sum_pixel_values",
                        lambda img, grid, shape, window_size: np.array([[5.0, 0.0]]))

    camera, slm = FakeCamera(), SimulatedSLM()
    coord = Rearrangement({"camera": camera, "slm": slm}, {})
    pm = FakePM(n_frames=5)
    coord._initialised = True
    coord._state = dict(
        PM=pm,
        terms1=("w1", "phi1", "x1", "y1", (1, 2)),
        terms2=("w2", "phi2", "x2", "y2", (1, 2)),
        pm_init_uint8=np.zeros((1024, 1024), dtype=np.uint8),
        d0=0.5, fps=1000.0, threshold=1.0, grid_positions=None, roi=None,
    )

    img0, img1 = coord.arm_rearrangement()

    # Sequence generated from the stored terms, thresholded occupancy passed through.
    assert pm.called_with["terms1"] == coord._state["terms1"]
    assert pm.called_with["d0"] == 0.5
    np.testing.assert_array_equal(pm.called_with["occ_mask"], np.array([False, True]))
    # SLM saw the initial mask (1) plus every sequence frame (5) played in-process.
    assert slm.frames_written == 1 + 5
    assert img0.shape == (8, 8) and img1.shape == (8, 8)


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
    "driver": "composite",
    "simulate": True,
    "devices": {
        "Rb Rearrangement Cam": {
            "driver": "imagemx2", "role": "camera", "stream_name": None, "timeout": 2.0,
        },
        "Rb SLM": {"driver": "slm", "role": "slm"},
    },
    "coordinator": "rearrangement",
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
