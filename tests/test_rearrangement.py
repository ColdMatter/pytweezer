"""Rearrangement coordinator wiring and no-GPU degradation. The real GPU
pipeline is replaced by a fake phasemask generator."""

import numpy as np
import pytest

from pytweezer.coordinators import rearrangement as rearr
from pytweezer.coordinators.rearrangement import Rearrangement
from pytweezer.drivers.slm import SimulatedSLM
from pytweezer.servers import device_server


class FakeCamera:
    stopped = 0

    def is_connected(self):
        return True

    def start_acquisition(self):
        pass

    def acquire_n_frames(self, n):
        return np.zeros((n, 8, 8), dtype=np.uint16)

    def stop_acquisition(self):
        self.stopped += 1


def _coord():
    return Rearrangement({"camera": FakeCamera(), "slm": SimulatedSLM()}, {})


def test_missing_slm_role_names_available():
    with pytest.raises(KeyError, match="needs a sub-device with role"):
        Rearrangement({"camera": FakeCamera()}, {})


def test_phasemask_kwargs_merge_config_over_defaults():
    coord = Rearrangement(
        {"camera": FakeCamera(), "slm": SimulatedSLM()},
        {"phasemask": {"wavelength_um": 0.780}},
    )
    assert coord.phasemask_kwargs["wavelength_um"] == 0.780
    assert coord.phasemask_kwargs["slm_res"] == (1024, 1024)


def test_status_without_gpu(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", False)
    assert _coord().status() == {
        "gpu_available": False,
        "initialised": False,
        "roles": ["camera", "slm"],
        "camera_connected": True,
        "slm_connected": True,
    }


def test_gpu_operations_without_cupy_raise_clearly(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", False)
    coord = _coord()
    with pytest.raises(RuntimeError, match="needs cupy/lap and a CUDA GPU"):
        coord.initialise(
            data1=np.zeros((4, 2)),
            data2=np.zeros((4, 2)),
            d0=0.5,
            fps=30,
            threshold=1.0,
            grid_positions=None,
        )
    with pytest.raises(RuntimeError, match="needs cupy/lap and a CUDA GPU"):
        coord.arm_rearrangement()


def test_arm_before_initialise_raises(monkeypatch):
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    with pytest.raises(RuntimeError, match="not initialised"):
        _coord().arm_rearrangement()


class FakePM:
    """The GPU phasemask generator's streaming entry point: yields one frame at a time."""

    def __init__(self, n_frames=5):
        self.n_frames = n_frames
        self.called_with = None

    def iter_rearrangement_sequence(
        self, terms1, terms2, occ_mask, d0, profile="minimum_jerk", to_host=True
    ):
        self.called_with = {
            "terms1": terms1,
            "terms2": terms2,
            "occ_mask": occ_mask,
            "d0": d0,
            "profile": profile,
            "to_host": to_host,
        }
        for _ in range(self.n_frames):
            yield np.zeros((8, 8), dtype=np.uint8)


def _patch_occupancy(monkeypatch):
    from pytweezer.analysis import analysis as an

    # Force the numpy branch so the patched sum_pixel_values is the one used.
    monkeypatch.setattr(rearr, "USE_SUM_CPP", False)
    monkeypatch.setattr(
        an, "morphological_tophat_high_pass", lambda img, feature_size: img
    )
    monkeypatch.setattr(
        an,
        "sum_pixel_values",
        lambda img, grid, window_size: np.array([5.0, 0.0]),
    )


def _armed_coord(monkeypatch, pm, slm):
    monkeypatch.setattr(rearr, "_HAS_CUPY", True)
    _patch_occupancy(monkeypatch)
    coord = Rearrangement({"camera": FakeCamera(), "slm": slm}, {})
    coord._initialised = True
    coord._state = {
        "PM": pm,
        "terms1": ("w1", "phi1", "x1", "y1"),
        "terms2": ("w2", "phi2", "x2", "y2"),
        "pm_init_uint8": np.zeros((1024, 1024), dtype=np.uint8),
        "d0": 0.5,
        "fps": 1000.0,
        "threshold": 1.0,
        "grid_positions": None,
        "window_size": 3,
        "feature_size": 10,
        "detector": None,
        "num_images": 1,
        "profile": "linear",
    }
    return coord


def test_arm_streams_sequence_to_slm(monkeypatch):
    slm = SimulatedSLM()
    pm = FakePM(n_frames=5)
    coord = _armed_coord(monkeypatch, pm, slm)

    img0, imgs1, timings = coord.arm_rearrangement()

    assert pm.called_with["terms1"] == coord._state["terms1"]
    assert pm.called_with["d0"] == 0.5
    assert pm.called_with["profile"] == "linear"
    np.testing.assert_array_equal(pm.called_with["occ_mask"], np.array([True, False]))
    # Frames are consumed on the writer thread, so the copy is left to it.
    assert pm.called_with["to_host"] is False
    # The initial mask plus every streamed frame.
    assert slm.frames_written == 1 + 5
    assert img0.shape == (8, 8) and imgs1.shape == (1, 8, 8)
    assert timings["n_frames"] == 5


def test_arm_reraises_slm_upload_error(monkeypatch):
    """An upload failure must surface, not be swallowed by the writer thread."""

    class ExplodingSLM(SimulatedSLM):
        def update_mask(self, mask_array):
            if getattr(self, "_armed", False):
                raise RuntimeError("DMA failed")
            self._armed = True
            super().update_mask(mask_array)

    coord = _armed_coord(monkeypatch, FakePM(n_frames=4), ExplodingSLM())
    with pytest.raises(RuntimeError, match="DMA failed"):
        coord.arm_rearrangement()


def test_shutdown_stops_the_camera_only_if_initialised():
    coord = _coord()
    coord.shutdown()
    assert coord.camera.stopped == 0

    coord._initialised = True
    coord.shutdown()
    assert coord.camera.stopped == 1
    assert not coord._initialised


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
