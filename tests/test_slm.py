"""Unit tests for the SLM driver (simulated backend) and its device factory.
No Blink SDK, no hardware.
"""

import numpy as np
import pytest

from pytweezer.drivers.slm import SLM, SimulatedSLM
from pytweezer.servers import device_server


def test_simulated_slm_remembers_last_mask():
    slm = SimulatedSLM()
    mask = np.arange(1024 * 1024, dtype=np.uint8).reshape(1024, 1024)
    assert slm.update_mask(mask) is None  # imperative: no ZMQ-style status wrapper
    assert slm.frames_written == 1
    np.testing.assert_array_equal(slm.last_mask, mask)


def test_simulated_slm_run_sequence_counts_frames():
    slm = SimulatedSLM()
    seq = np.zeros((7, 8, 8), dtype=np.uint8)
    assert slm.run_sequence(seq, fps=1000) is None
    assert slm.frames_written == 7


def test_simulated_slm_dimensions_and_temperature():
    slm = SimulatedSLM(width=512, height=512)
    assert slm.get_dimensions() == {"width": 512, "height": 512, "depth": 8}
    assert isinstance(slm.get_temperature(), float)


def test_simulated_slm_is_interface_complete_with_real():
    # simulate() should have stubbed every public method of SLM the hand-written
    # class doesn't define, so the simulated surface never drifts from the real one.
    real_methods = {m for m in dir(SLM) if not m.startswith("_")}
    sim_methods = {m for m in dir(SimulatedSLM) if not m.startswith("_")}
    assert real_methods <= sim_methods


def test_simulated_slm_close_marks_disconnected():
    slm = SimulatedSLM()
    assert slm.is_connected()
    slm.close()
    assert not slm.is_connected()


# --------------------------------------------------------------------------- #
# Device factory
# --------------------------------------------------------------------------- #

def test_make_slm_simulated():
    spec = device_server.build_spec("SLM", conf={"driver": "slm", "simulate": True})
    assert list(spec.targets) == ["slm"]
    assert isinstance(spec.targets["slm"], SimulatedSLM)


def test_make_slm_teardown_closes():
    spec = device_server.build_spec("SLM", conf={"driver": "slm", "simulate": True})
    slm = spec.targets["slm"]
    spec.teardown()
    assert not slm.is_connected()


def test_make_slm_real_requires_sdk(monkeypatch):
    # The real branch tries to load the Blink DLL in SLM._connect; without the SDK
    # that raises. Assert the factory attempts the real path (not silently sim).
    with pytest.raises(Exception):
        device_server.build_spec("SLM", conf={"driver": "slm", "simulate": False})
