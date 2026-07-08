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

from pytweezer.servers.device_server import _make_imagemx2, _make_motmaster
from pytweezer.servers.simulated_device import public_methods

#: driver key -> (factory, minimal simulate=True conf). "blackfly" is
#: excluded: its factory has no simulate support today.
FACTORIES = {
    "motmaster": (_make_motmaster, {"simulate": True, "config_file": "unused.json"}),
    "imagemx2": (_make_imagemx2, {"simulate": True}),
}


@pytest.mark.parametrize("driver", sorted(FACTORIES))
def test_simulated_backend_methods_are_pyon_safe(driver):
    factory, conf = FACTORIES[driver]
    spec = factory(driver, conf)
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
