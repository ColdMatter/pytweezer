"""Unit tests for the coordinator layer: the ``CameraDACFeedback`` control law and
arming, against fake backends, plus one end-to-end build of a fully simulated
composite (simulated camera + simulated DAC + real coordinator). No RPC, no ZMQ,
no hardware — the camera is built with ``stream_name=None`` so it never opens an
``ImageClient``.
"""

import numpy as np
import pytest

from pytweezer.coordinators.camera_dac_feedback import CameraDACFeedback
from pytweezer.drivers.ni_dac import SimulatedNIDAC
from pytweezer.servers import device_server


class FakeCamera:
    """Records arming calls and hands out frames of a known mean."""

    def __init__(self, mean=130.0):
        self.mean = mean
        self.setup_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.grabs = 0

    def setup_acquisition(self, acq_mode, nframes):
        self.setup_calls += 1

    def start_acquisition(self):
        self.start_calls += 1

    def stop_acquisition(self):
        self.stop_calls += 1

    def acquire_single_frame(self):
        self.grabs += 1
        return np.full((4, 4), self.mean, dtype=np.float32)


@pytest.fixture
def rig():
    camera = FakeCamera()
    dac = SimulatedNIDAC(["Dev1/ao0"])
    coordinator = CameraDACFeedback({"camera": camera, "dac": dac}, {})
    return coordinator, camera, dac


# --------------------------------------------------------------------------- #
# Arming
# --------------------------------------------------------------------------- #

def test_arm_is_idempotent(rig):
    coordinator, camera, _ = rig
    coordinator.arm()
    coordinator.arm()
    assert (camera.setup_calls, camera.start_calls) == (1, 1)
    assert coordinator.is_armed()


def test_disarm_is_idempotent(rig):
    coordinator, camera, _ = rig
    coordinator.arm()
    coordinator.disarm()
    coordinator.disarm()
    assert camera.stop_calls == 1
    assert not coordinator.is_armed()


def test_disarm_without_arm_does_nothing(rig):
    coordinator, camera, _ = rig
    coordinator.disarm()
    assert camera.stop_calls == 0


def test_shutdown_disarms(rig):
    coordinator, camera, _ = rig
    coordinator.arm()
    coordinator.shutdown()
    assert camera.stop_calls == 1


# --------------------------------------------------------------------------- #
# Control law
# --------------------------------------------------------------------------- #

def test_control_is_proportional(rig):
    coordinator, _, _ = rig
    assert coordinator.control(measured=100.0, setpoint=130.0, gain=0.1) == pytest.approx(3.0)


def test_control_clamps_to_limits(rig):
    coordinator, _, _ = rig
    assert coordinator.control(100.0, 1e6, gain=1.0) == pytest.approx(10.0)
    assert coordinator.control(1e6, 100.0, gain=1.0) == pytest.approx(-10.0)


def test_control_respects_custom_limits(rig):
    coordinator, _, _ = rig
    assert coordinator.control(0.0, 100.0, gain=1.0, limits=(-1.0, 1.0)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# One-shot and batch
# --------------------------------------------------------------------------- #

def test_image_to_dac_arms_grabs_and_writes(rig):
    coordinator, camera, dac = rig
    result = coordinator.image_to_dac(setpoint=140.0, gain=0.1, channel="Dev1/ao0")

    assert camera.grabs == 1
    assert result["mean"] == pytest.approx(130.0)
    assert result["voltage"] == pytest.approx(1.0)  # 0.1 * (140 - 130)
    assert dac.get_last_values()["Dev1/ao0"] == pytest.approx(1.0)


def test_image_to_dac_does_not_return_the_frame(rig):
    coordinator, _, _ = rig
    result = coordinator.image_to_dac(setpoint=130.0, gain=0.01, channel="Dev1/ao0")
    assert set(result) == {"mean", "voltage", "t"}
    assert not any(isinstance(v, np.ndarray) for v in result.values())


def test_run_n_iterates_in_process(rig):
    coordinator, camera, _ = rig
    results = coordinator.run_n(3, setpoint=140.0, gain=0.1, channel="Dev1/ao0")
    assert len(results) == 3
    assert camera.grabs == 3
    # Armed once for the whole batch, not once per iteration.
    assert camera.start_calls == 1


def test_run_n_rejects_zero_iterations(rig):
    coordinator, _, _ = rig
    with pytest.raises(ValueError, match="n must be >= 1"):
        coordinator.run_n(0, setpoint=130.0, gain=0.01, channel="Dev1/ao0")


def test_write_to_unconfigured_channel_raises(rig):
    coordinator, _, _ = rig
    with pytest.raises(KeyError, match="not configured"):
        coordinator.image_to_dac(setpoint=130.0, gain=0.01, channel="Dev1/ao7")


def test_missing_role_names_what_is_available():
    with pytest.raises(KeyError, match="needs a sub-device with role"):
        CameraDACFeedback({"camera": FakeCamera()}, {})


# --------------------------------------------------------------------------- #
# End-to-end: a real simulated composite built through build_spec
# --------------------------------------------------------------------------- #

SIM_COMPOSITE = {
    "driver": "composite",
    "simulate": True,
    "devices": {
        # stream_name=None keeps the camera from opening a ZMQ ImageClient.
        "Rb Feedback Cam": {
            "driver": "imagemx2", "role": "camera", "stream_name": None, "timeout": 5.0,
        },
        "Rb Feedback DAC": {"driver": "nidac", "role": "dac", "channels": ["Dev1/ao0"]},
    },
    "coordinator": "camera_dac_feedback",
}

CAM_TARGET = "rbfeedbackcam"
DAC_TARGET = "rbfeedbackdac"


def test_simulated_composite_closes_the_loop():
    spec = device_server.build_spec("Rb Feedback Rig", conf=SIM_COMPOSITE)
    assert set(spec.targets) == {CAM_TARGET, DAC_TARGET, "coordinator"}

    try:
        results = spec.targets["coordinator"].run_n(
            2, setpoint=130.0, gain=0.01, channel="Dev1/ao0"
        )
        assert len(results) == 2
        assert all(r["mean"] > 0 for r in results)
        # The DAC actually got driven, by direct in-process call.
        assert spec.targets[DAC_TARGET].get_last_values()["Dev1/ao0"] == pytest.approx(
            results[-1]["voltage"]
        )
    finally:
        spec.teardown()

    # Teardown disarmed the camera and closed the DAC.
    assert not spec.targets["coordinator"].is_armed()
    with pytest.raises(RuntimeError, match="has been closed"):
        spec.targets[DAC_TARGET].set_voltage("Dev1/ao0", 0.0)


def test_simulated_composite_camera_is_still_addressable_alone():
    """The point of a composite: the camera stays a first-class device."""
    spec = device_server.build_spec("Rb Feedback Rig", conf=SIM_COMPOSITE)
    try:
        camera = spec.targets[CAM_TARGET]
        assert camera.is_connected()
        camera.setup_acquisition("sequence", 1)
        camera.start_acquisition()
        frame = camera.acquire_single_frame()
        assert frame.ndim == 2
    finally:
        spec.teardown()
