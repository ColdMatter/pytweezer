"""Simulated devices built inside the experiment's own process.

In simulation an experiment gets each device's simulated backend directly,
built by the same :func:`~pytweezer.servers.device_server.build_spec` a device
server uses, instead of an RPC client. No device servers or lab network are
needed, so experiments run anywhere (e.g. on a laptop with ``SIMULATING``).
Calls are plain Python, so values are not round-tripped through PYON as they
would be over RPC.
"""

from typing import Any

from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_server import build_spec, resolve_address

logger = get_logger("pytweezer.experiment.simulation")


def _simulated_conf(conf: dict) -> dict:
    """``conf`` with simulation forced on, including every composite sub-device."""
    simulated = dict(conf, simulate=True)
    if "devices" in conf:
        simulated["devices"] = {
            name: dict(sub, simulate=True) for name, sub in conf["devices"].items()
        }
    return simulated


class SimulatedDevices:
    """Builds simulated backends on demand; one per owning config entry, so
    sub-devices of a composite share one simulated rig as they would for real.
    """

    def __init__(self) -> None:
        self._specs: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        address = resolve_address(name)
        spec = self._specs.get(address.owner_name)
        if spec is None:
            logger.warning("Simulating device %r in-process", address.owner_name)
            spec = build_spec(address.owner_name, _simulated_conf(address.owner_conf))
            self._specs[address.owner_name] = spec
        if address.target_name is None:
            return next(iter(spec.targets.values()))
        try:
            return spec.targets[address.target_name]
        except KeyError:
            raise KeyError(
                f"simulated {address.owner_name!r} has no target for {name!r}"
                + (
                    f" (failed to build: {', '.join(spec.failed)})"
                    if spec.failed
                    else ""
                )
            ) from None

    def close(self) -> None:
        while self._specs:
            _name, spec = self._specs.popitem()
            if spec.teardown is not None:
                try:
                    spec.teardown()
                except Exception:
                    logger.warning(
                        "Error tearing down a simulated device", exc_info=True
                    )
