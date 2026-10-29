"""SLM driver's simulated backend, device factory and SDK lifetime; no hardware."""

import numpy as np
import pytest

from pytweezer.drivers import slm as slm_module
from pytweezer.drivers.slm import SLM, SimulatedSLM
from pytweezer.servers import device_server


class FakeBlinkSDK:
    """Stands in for Blink_C_wrapper: one 1024x1024 board; tracks whether the SDK is alive."""

    def __init__(self, constructs=True, accepts_lut=True):
        self.constructs = constructs
        self.accepts_lut = accepts_lut
        self.sdk_alive = False

    def Create_SDK(self, n_boards_found, constructed_ok):
        n_boards_found._obj.value = 1
        constructed_ok._obj.value = int(self.constructs)
        self.sdk_alive = True

    def Delete_SDK(self):
        self.sdk_alive = False

    def Get_last_error_message(self):
        if self.constructs:
            return None
        return b"No PCIe hardware found, running in simulation mode"

    def Load_LUT_file(self, board, path):
        return int(self.accepts_lut)

    def Get_image_width(self, board):
        return 1024

    def Get_image_height(self, board):
        return 1024

    def Get_image_depth(self, board):
        return 8

    def Read_Serial_Number(self, board):
        return 1234

    def Read_SLM_temperature(self, board):
        return 35.0

    def SetWaitForTrigger(self, board, enabled):
        return 1

    def SetFlipImmediate(self, board, enabled):
        return 1

    def SetOutputPulse(self, board, enabled):
        return 1

    def Write_image(self, board, image, timeout_ms):
        return 1

    def ImageWriteComplete(self, board, timeout_ms):
        return 1


@pytest.fixture
def use_fake_sdk(monkeypatch):
    monkeypatch.setattr(SLM, "_sdk_owner", None)

    def install(**behaviour):
        fake = FakeBlinkSDK(**behaviour)
        monkeypatch.setattr(slm_module, "_load_blink_sdk", lambda sdk_dll: fake)
        return fake

    return install


@pytest.mark.parametrize(
    "behaviour", [{"constructs": False}, {"accepts_lut": False}], ids=str
)
def test_failed_connect_releases_the_sdk(use_fake_sdk, behaviour):
    sdk = use_fake_sdk(**behaviour)
    with pytest.raises(RuntimeError):
        SLM()
    assert not sdk.sdk_alive


def test_construct_failure_reports_the_sdk_reason(use_fake_sdk):
    use_fake_sdk(constructs=False)
    with pytest.raises(RuntimeError, match="No PCIe hardware found"):
        SLM()


def test_only_one_slm_may_be_open_per_process(use_fake_sdk):
    sdk = use_fake_sdk()
    first = SLM()
    with pytest.raises(RuntimeError, match="already open"):
        SLM()
    assert first.is_connected() and sdk.sdk_alive

    first.close()
    SLM().close()


def test_simulated_slm_records_what_was_written():
    slm = SimulatedSLM(width=512, height=512)
    assert slm.get_dimensions() == {"width": 512, "height": 512, "depth": 8}
    assert isinstance(slm.get_temperature(), float)

    mask = np.arange(512 * 512, dtype=np.uint8).reshape(512, 512)
    assert slm.update_mask(mask) is None  # imperative: no ZMQ-style status wrapper
    np.testing.assert_array_equal(slm.last_mask, mask)

    assert slm.run_sequence(np.zeros((7, 8, 8), dtype=np.uint8), fps=1000) is None
    assert slm.frames_written == 1 + 7


_SLM_CONF = {
    "class": "pytweezer.drivers.slm:SLM",
    "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
    "teardown": "close",
}


def test_device_teardown_releases_the_real_sdk(use_fake_sdk):
    sdk = use_fake_sdk()
    spec = device_server.build_spec("SLM", conf={**_SLM_CONF, "simulate": False})
    (slm,) = spec.targets.values()
    assert isinstance(slm, SLM)  # the real path, not a silent fallback to simulation
    spec.teardown()
    assert not sdk.sdk_alive
