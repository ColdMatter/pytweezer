"""Generic device RPC-server launcher.

Every device in ``CONFIG["Devices"]`` runs a sipyco RPC server. Rather than give
each driver module its own ``argparse`` + config-reading + ``simple_server_loop``
boilerplate, this module provides a single launcher:

* ``pytweezer-device <device_name>`` (console script) — start the server for the
  named device from the command line.
* The **device manager** launches the same thing: each device's ``script`` in the
  config points here, so ``ProcessTile`` runs ``python device_server.py <name>``.

Adding a *new device instance* means only editing ``config.py``. Adding a new
*driver type* means only adding one factory to :data:`DRIVER_REGISTRY` — the
driver module itself needs no launch code, just its backend class.

Each factory turns a device's config dict into a :class:`DeviceServerSpec`
(the sipyco target object, its target name, a description, and an optional
teardown callable). Backend modules are imported lazily inside their factory so
that importing this launcher never pulls in a hardware library that may be
absent (e.g. ``rotpy`` for the Blackfly, ``pylablib`` for the ImagEM).
"""

import argparse
import signal
from dataclasses import dataclass
from typing import Callable, Optional

from sipyco.pc_rpc import simple_server_loop

from pytweezer.servers.configreader import ConfigReader

from pytweezer.logging_utils import get_logger

logger = get_logger("device server")


@dataclass
class DeviceServerSpec:
    """Everything :func:`run_device_server` needs to serve one device.

    ``target``/``target_name`` are passed straight to ``simple_server_loop`` as
    ``{target_name: target}``. ``teardown`` (if given) is called in a ``finally``
    after the loop ends, for backends that need an explicit disconnect.
    """

    target_name: str
    target: object
    description: str
    teardown: Optional[Callable[[], None]] = None


# --------------------------------------------------------------------------- #
# Driver factories: config dict -> DeviceServerSpec
# --------------------------------------------------------------------------- #

def _make_motmaster(name, conf):
    from pytweezer.experiment.motmaster_server import (
        SimulatedMotMasterInterface,
        MotMasterInterface,
        _ensure_motmaster_running,
    )
    from pytweezer.servers import tweezerpath

    simulate = conf.get("simulate", False)
    interval = conf.get("interval") or 0.1
    config_file = tweezerpath + "/pytweezer/configuration/" + conf["config_file"]

    if simulate:
        interface = SimulatedMotMasterInterface(interval=interval)
    else:
        # Only touch real hardware when not simulating.
        _ensure_motmaster_running(config_file)
        interface = MotMasterInterface(config_file, interval=interval)
    interface.connect()
    if (not simulate) and interface.motmaster is None:
        raise RuntimeError("Failed to connect to MotMaster.")

    return DeviceServerSpec(
        target_name="motmaster",
        target=interface,
        description="MotMaster command server",
        teardown=lambda: _safe(interface.disconnect),
    )


def _make_imagemx2(name, conf):
    from pytweezer.drivers.imagemX2 import ImagEMX2Camera, SimulatedImagEMX2Camera

    simulate = conf.get("simulate", False)
    timeout = conf.get("timeout", 5.0)
    image_dir = conf.get("image_dir")
    stream_name = conf.get("stream_name", "imagemx2")

    if simulate:
        logger.warning("Running ImagEM X2 camera %r in SIMULATION MODE", name)
        camera = SimulatedImagEMX2Camera(
            stream_name=stream_name, image_dir=image_dir, timeout=timeout
        )
    else:
        camera = ImagEMX2Camera(
            stream_name=stream_name, image_dir=image_dir, timeout=timeout
        )

    return DeviceServerSpec(
        target_name="camera",
        target=camera,
        description="ImagEM X2 RPC server",
    )


def _make_blackfly(name, conf):
    from rotpy.system import SpinSystem

    from pytweezer.drivers.bfly2 import Blackfly

    camera = Blackfly(serial=12345678, system=SpinSystem())
    return DeviceServerSpec(
        target_name="camera",
        target=camera,
        description="Blackfly RPC server",
    )


#: Maps a device's ``"driver"`` config key to the factory that builds its server.
DRIVER_REGISTRY = {
    "motmaster": _make_motmaster,
    "imagemx2": _make_imagemx2,
    "blackfly": _make_blackfly,
}


def _safe(fn):
    """Run a teardown callable, logging (not raising) any error."""
    try:
        fn()
    except Exception:
        logger.exception("Error during device-server teardown")


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #

def _normalize(s):
    """Collapse whitespace and lowercase, for lenient name matching."""
    return "".join(s.split()).lower()


def resolve_device(name):
    """Return ``(canonical_name, conf)`` for ``name`` in ``CONFIG["Devices"]``.

    Matches the config key exactly first, then falls back to a
    whitespace-/case-insensitive match so command-line callers can pass e.g.
    ``RbHamCam`` or ``rb hamcam`` for the config key ``"Rb HamCam"``. Raises
    ``KeyError`` listing the available devices if nothing matches.
    """
    devices = ConfigReader.getConfiguration().get("Devices", {})
    if name in devices:
        return name, devices[name]

    target = _normalize(name)
    for key, conf in devices.items():
        if _normalize(key) == target:
            logger.debug("Resolved device name %r -> %r", name, key)
            return key, conf

    available = ", ".join(sorted(devices)) or "(none)"
    raise KeyError(
        f"Device {name!r} not found in config. Available devices: {available}"
    )


def build_spec(name, conf=None):
    """Return the :class:`DeviceServerSpec` for the device named ``name``.

    ``conf`` defaults to that device's ``CONFIG["Devices"]`` entry; pass an
    explicit dict to override. The device's ``"driver"`` key selects the factory.
    """
    if conf is None:
        name, conf = resolve_device(name)

    driver = conf.get("driver")
    if driver is None:
        raise KeyError(
            f"Device {name!r} has no 'driver' key; expected one of "
            f"{sorted(DRIVER_REGISTRY)}"
        )
    try:
        factory = DRIVER_REGISTRY[driver]
    except KeyError:
        raise KeyError(
            f"Unknown driver {driver!r} for device {name!r}; "
            f"registered drivers: {sorted(DRIVER_REGISTRY)}"
        ) from None

    return factory(name, conf)


def run_device_server(name, host=None, port=None):
    """Build and serve the RPC server for the device named ``name`` (blocks)."""
    name, conf = resolve_device(name)
    host = host or conf.get("host", "127.0.0.1")
    if port is None:
        port = conf.get("port")
    if port is None:
        raise ValueError(f"Device {name!r} has no 'port' configured and none was given")
    port = int(port)

    spec = build_spec(name, conf)
    logger.info(
        "Serving device %r (%s target) on tcp://%s:%s",
        name,
        spec.target_name,
        host,
        port,
    )
    try:
        simple_server_loop(
            {spec.target_name: spec.target},
            host=host,
            port=port,
            description=spec.description,
        )
    finally:
        if spec.teardown is not None:
            spec.teardown()


def main():
    parser = argparse.ArgumentParser(
        description="Start the sipyco RPC server for a configured device"
    )
    parser.add_argument("name", help="device name (key in CONFIG['Devices'])")
    parser.add_argument("--host", default=None, help="override RPC bind host")
    parser.add_argument("--port", type=int, default=None, help="override RPC bind port")
    args, _unknown = parser.parse_known_args()

    def _stop(_signo, _frame):
        # Let simple_server_loop unwind (KeyboardInterrupt) so teardown runs.
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass

    run_device_server(args.name, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
