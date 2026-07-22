"""Smoke tests for the generic simulated-device mechanism
(pytweezer/servers/simulated_device.py).

For each driver with simulate support, build its simulated backend via the
same factory device_server.py uses in production, then best-effort call
every public method and PYON-round-trip the result -- catching a
non-serializable stub return value here, at test time, instead of at first
live RPC call.
"""

import pytest
from sipyco import pyon

from pytweezer.servers import device_server
from pytweezer.servers.simulated_device import public_methods

#: driver key -> minimal simulate=True conf, built through the same
#: device_server.build_spec path used in production.
SIM_CONFS = {
    # No hand-written sim class: exercises the auto-generated stand-in built
    # from the real driver's method surface.
    "motmaster": {
        "class": "pytweezer.drivers.motmaster:MotMasterInterface",
        "simulate": True,
    },
    "imagemx2": {
        "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
        "simulate": True,
    },
}


@pytest.mark.parametrize("driver", sorted(SIM_CONFS))
def test_simulated_backend_methods_are_pyon_safe(driver):
    spec = device_server.build_spec(driver, conf=SIM_CONFS[driver])
    target = spec.target

    try:
        for name in public_methods(type(target)):
            method = getattr(target, name)
            try:
                result = method()
            except TypeError:
                # Needs real arguments we can't guess here; exercised by the
                # manual verification steps in docs/device_framework.md instead.
                continue
            except RuntimeError:
                # Expected state-machine guard (e.g. "camera is not running").
                # sipyco marshals this as a normal remote exception over RPC;
                # it's not a PYON-serialization problem.
                continue
            pyon.encode({"ret": result})
    finally:
        if spec.teardown is not None:
            spec.teardown()
