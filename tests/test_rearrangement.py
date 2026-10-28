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
