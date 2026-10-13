"""SLM driver's simulated backend and device factory; no Blink SDK, no hardware."""

import numpy as np
import pytest

from pytweezer.drivers.slm import SimulatedSLM
from pytweezer.servers import device_server


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


def test_build_simulated_slm_and_teardown_closes_it():
    spec = device_server.build_spec("SLM", conf={**_SLM_CONF, "simulate": True})
    (slm,) = spec.targets.values()
    assert isinstance(slm, SimulatedSLM)
    assert slm.is_connected()
    spec.teardown()
    assert not slm.is_connected()


def test_build_real_slm_attempts_the_sdk():
    # Without the Blink DLL the real path raises; it must not silently simulate.
    with pytest.raises(Exception):
        device_server.build_spec("SLM", conf={**_SLM_CONF, "simulate": False})
