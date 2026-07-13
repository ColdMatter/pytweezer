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

The ``"composite"`` driver (:func:`_make_composite`) serves several devices from one
process and one port, one RPC target each, optionally alongside a *coordinator*
target (:data:`COORDINATOR_REGISTRY`) that drives those backends through direct
Python calls rather than RPC. That is how a camera-to-DAC feedback step avoids
serializing a frame. Sub-devices stay individually addressable:
``get_device("Rb Feedback Rig", target_name="camera")``. See ``docs/device_framework.md``.
"""

import argparse
import signal
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from sipyco.pc_rpc import simple_server_loop

from pytweezer.servers.configreader import ConfigReader

from pytweezer.logging_utils import get_logger

logger = get_logger("device server")


@dataclass
class DeviceServerSpec:
    """Everything :func:`run_device_server` needs to serve one device.

    A spec carries either a single target (``target_name``/``target``) or several
    (``targets``, a ``{target_name: target}`` dict — see :func:`_make_composite`),
    never both. Either way :attr:`targets` is the normalized form passed to
    ``simple_server_loop``. ``teardown`` (if given) is called in a ``finally``
    after the loop ends, for backends that need an explicit disconnect.
    """

    target_name: Optional[str] = None
    target: Optional[object] = None
    description: str = ""
    teardown: Optional[Callable[[], None]] = None
    targets: Optional[Dict[str, object]] = None

    def __post_init__(self):
        if self.targets is None:
            if self.target_name is None:
                raise ValueError(
                    "DeviceServerSpec needs either target_name/target or targets"
                )
            self.targets = {self.target_name: self.target}
        elif self.target_name is not None:
            raise ValueError(
                "DeviceServerSpec takes target_name/target or targets, not both"
            )


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



def _make_slm(name, conf):
    simulate = conf.get("simulate", False)
    if simulate:
        from pytweezer.drivers.slm import SimulatedSLM

        logger.warning("Running SLM %r in SIMULATION MODE", name)
        slm = SimulatedSLM(
            width=conf.get("width", 1024),
            height=conf.get("height", 1024),
            depth=conf.get("depth", 8),
        )
    else:
        from pytweezer.drivers.slm import SLM, DEFAULT_SDK_DLL, DEFAULT_LUT_FILE

        slm = SLM(
            sdk_dll=conf.get("sdk_dll", DEFAULT_SDK_DLL),
            lut_file=conf.get("lut_file", DEFAULT_LUT_FILE),
            board_number=conf.get("board_number", 1),
            timeout_ms=conf.get("timeout_ms", 5000),
            wait_for_trigger=conf.get("wait_for_trigger", False),
            flip_immediate=conf.get("flip_immediate", False),
            output_pulse=conf.get("output_pulse", False),
        )
    return DeviceServerSpec(
        target_name="slm",
        target=slm,
        description="SLM RPC server",
        teardown=lambda: _safe(slm.close),
    )


def _make_composite(name, conf):
    """Build one server exposing several targets from one config entry.

    ``conf["devices"]`` maps a *device name* to an ordinary per-driver config dict;
    each is dispatched through :data:`DRIVER_REGISTRY` as usual. Sub-devices are
    named exactly like top-level devices and are reached the same way —
    ``get_device("Rb Feedback Cam")`` — because :func:`device_index` flattens them
    into the same namespace. Their RPC target name is :func:`composite_target_name`
    of that device name. Sub-configs inherit the composite's ``simulate`` flag
    unless they set their own.

    ``conf["coordinator"]`` optionally names a :data:`COORDINATOR_REGISTRY` entry; the
    coordinator is constructed with direct references to the backend objects, so its
    methods drive them with plain Python calls rather than RPC. It receives them keyed
    by each sub-config's ``"role"`` (defaulting to the device name), so a coordinator
    asks for ``targets["camera"]`` regardless of what that camera is called in config.
    """
    sub_confs = conf.get("devices")
    if not sub_confs:
        raise KeyError(
            f"Composite device {name!r} needs a non-empty 'devices' dict mapping "
            "device name -> per-driver config"
        )

    coordinator_name = _coordinator_target_name(conf)
    target_names = {}
    for device_name in sub_confs:
        target_name = _check_target_name(name, device_name)
        if target_name in target_names:
            raise ValueError(
                f"Composite device {name!r}: sub-devices {target_names[target_name]!r} "
                f"and {device_name!r} both map to RPC target {target_name!r}"
            )
        target_names[target_name] = device_name

    if conf.get("coordinator") is not None and coordinator_name in target_names:
        raise ValueError(
            f"Composite device {name!r}: coordinator target name {coordinator_name!r} "
            f"collides with sub-device {target_names[coordinator_name]!r}"
        )

    targets = {}
    roles = {}
    teardowns = []

    def teardown_all():
        # Reversed: the coordinator is appended last and must release the backends
        # before they are closed underneath it.
        for teardown in reversed(teardowns):
            _safe(teardown)

    try:
        for device_name, sub_conf in sub_confs.items():
            if sub_conf.get("driver") == "composite":
                raise ValueError(
                    f"Composite device {name!r}: sub-device {device_name!r} may not "
                    "itself be a composite"
                )
            sub_conf = dict(sub_conf)
            sub_conf.setdefault("simulate", conf.get("simulate", False))

            sub_spec = build_spec(device_name, conf=sub_conf)
            if len(sub_spec.targets) != 1:
                raise ValueError(
                    f"Composite device {name!r}: sub-device {device_name!r} exposes "
                    f"{len(sub_spec.targets)} targets; exactly one is required"
                )
            target = next(iter(sub_spec.targets.values()))
            _check_target_callable(name, device_name, target)
            targets[composite_target_name(device_name)] = target
            roles[sub_conf.get("role", device_name)] = target
            if sub_spec.teardown is not None:
                teardowns.append(sub_spec.teardown)

        coordinator_key = conf.get("coordinator")
        if coordinator_key is not None:
            coordinator = _build_coordinator(coordinator_key, name, roles, conf)
            _check_target_callable(name, coordinator_name, coordinator)
            targets[coordinator_name] = coordinator
            teardowns.append(coordinator.shutdown)
    except Exception:
        # Release whatever already opened, or a half-built rig leaks hardware handles.
        teardown_all()
        raise

    return DeviceServerSpec(
        targets=targets,
        description=conf.get("description", f"composite device server {name!r}"),
        teardown=teardown_all,
    )


#: Maps a device's ``"driver"`` config key to the factory that builds its server.
DRIVER_REGISTRY = {
    "motmaster": _make_motmaster,
    "imagemx2": _make_imagemx2,
    "blackfly": _make_blackfly,
    "nidac": _make_nidac,
    "slm": _make_slm,
    "composite": _make_composite,
}


def _make_rearrangement(targets, conf):
    from pytweezer.coordinators.rearrangement import Rearrangement

    return Rearrangement(targets, conf)


#: Maps a composite device's ``"coordinator"`` config key to its factory.
COORDINATOR_REGISTRY = {
    "rearrangement": _make_rearrangement,
}


def _build_coordinator(key, device_name, targets, conf):
    try:
        factory = COORDINATOR_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown coordinator {key!r} for device {device_name!r}; "
            f"registered coordinators: {sorted(COORDINATOR_REGISTRY)}"
        ) from None
    return factory(targets, conf)


def _check_target_name(composite_name, device_name):
    """Return the RPC target name for a sub-device, or raise if it can't have one.

    ``sipyco.pc_rpc.Server`` refuses target names containing whitespace, so a
    sub-device's *display* name is folded to a wire-safe one by
    :func:`composite_target_name`. Checking here fails a config typo before any
    hardware is opened.
    """
    if not isinstance(device_name, str) or not composite_target_name(device_name):
        raise ValueError(
            f"Composite device {composite_name!r}: sub-device name {device_name!r} "
            "must be a non-empty string"
        )
    return composite_target_name(device_name)


def _check_target_callable(device_name, target_name, target):
    """Reject callable targets: sipyco *invokes* them instead of serving them.

    ``Server._handle_connection_cr`` does ``if callable(target): target = target()``,
    treating a callable target as a per-connection factory.
    """
    if callable(target):
        raise TypeError(
            f"Composite device {device_name!r}: target {target_name!r} is callable "
            f"({type(target).__name__} defines __call__); sipyco would call it instead "
            "of serving it"
        )


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


def composite_target_name(device_name):
    """RPC target name a composite serves a sub-device under.

    Sub-devices are named like any other device (``"Rb Feedback Cam"``), but sipyco
    target names cannot contain whitespace, so the display name is folded down.
    Clients never type this — :func:`resolve_address` supplies it.
    """
    return _normalize(device_name)


def _coordinator_target_name(conf):
    return _normalize(conf.get("coordinator_target_name", "coordinator"))


@dataclass(frozen=True)
class DeviceAddress:
    """Where a named device lives and which RPC target serves it.

    For a plain device the device *is* its own server, so ``owner_conf is conf`` and
    ``target_name`` is ``None`` (the server has one target; ``AutoTarget`` finds it).
    For a composite's sub-device, ``owner_conf`` is the composite's entry — that is
    where ``host``/``port`` live — and ``target_name`` selects the sub-device.
    """

    name: str
    conf: dict
    owner_name: str
    owner_conf: dict
    target_name: Optional[str] = None

    @property
    def is_sub_device(self):
        return self.owner_name != self.name


def device_index():
    """Return ``{normalized_name: DeviceAddress}`` over every addressable device.

    Composite sub-devices are flattened into the same namespace as top-level
    devices, which is what lets ``get_device("Rb Feedback Cam")`` work without the
    caller knowing that camera happens to share a process with a DAC. A composite's
    own name resolves to its coordinator target.

    Raises ``KeyError`` if two devices share a name (case- and whitespace-insensitively),
    since such a name could not be resolved unambiguously.
    """
    devices = ConfigReader.getConfiguration().get("Devices", {})
    index = {}

    def add(address):
        key = _normalize(address.name)
        if not key:
            raise ValueError(f"Device name {address.name!r} is empty")
        if key in index:
            raise KeyError(
                f"Duplicate device name {address.name!r} in CONFIG['Devices'] "
                f"(collides with {index[key].name!r}); device names must be unique "
                "across composites too"
            )
        index[key] = address

    for name, conf in devices.items():
        if conf.get("driver") == "composite":
            coordinator = (
                _coordinator_target_name(conf) if conf.get("coordinator") else None
            )
            add(DeviceAddress(name, conf, name, conf, coordinator))
            for sub_name, sub_conf in (conf.get("devices") or {}).items():
                add(
                    DeviceAddress(
                        sub_name, sub_conf, name, conf, composite_target_name(sub_name)
                    )
                )
        else:
            add(DeviceAddress(name, conf, name, conf, None))
    return index


def resolve_address(name):
    """Return the :class:`DeviceAddress` for ``name``, matched leniently.

    Accepts any whitespace-/case-insensitive spelling of a top-level device, a
    composite, or a composite's sub-device. Raises ``KeyError`` listing every
    addressable device if nothing matches.
    """
    index = device_index()
    try:
        return index[_normalize(name)]
    except KeyError:
        available = ", ".join(sorted(a.name for a in index.values())) or "(none)"
        raise KeyError(
            f"Device {name!r} not found in config. Available devices: {available}"
        ) from None


def resolve_device(name):
    """Return ``(canonical_name, conf)`` for a *launchable* device.

    Only top-level ``CONFIG["Devices"]`` entries are launchable — a composite's
    sub-device has no server of its own. Matches the config key exactly first, then
    falls back to a whitespace-/case-insensitive match so command-line callers can
    pass ``RbHamCam`` or ``rb hamcam`` for ``"Rb HamCam"``.
    """
    devices = ConfigReader.getConfiguration().get("Devices", {})
    if name in devices:
        return name, devices[name]

    target = _normalize(name)
    for key, conf in devices.items():
        if _normalize(key) == target:
            logger.debug("Resolved device name %r -> %r", name, key)
            return key, conf

    # A sub-device name is addressable by clients but not launchable; say so rather
    # than claiming the name doesn't exist.
    address = device_index().get(target)
    if address is not None and address.is_sub_device:
        raise KeyError(
            f"Device {name!r} is a sub-device of composite {address.owner_name!r} and "
            f"has no server of its own; launch {address.owner_name!r} instead"
        )

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


def run_device_server(name, host=None, port=None, allow_parallel=None):
    """Build and serve the RPC server for the device named ``name`` (blocks).

    ``allow_parallel`` (config key of the same name, default ``False``) drops the
    lock sipyco holds across each RPC call. **It has no effect while every target
    method is a plain ``def``**: ``Server._process_action`` awaits a method's result
    only when it is a coroutine, and an uncontended ``asyncio.Lock.acquire()`` does
    not suspend — so with synchronous methods there is no suspension point between
    acquiring and releasing that lock, and nothing can contend for it. What
    serializes calls today is the single-threaded event loop, not the lock. The flag
    becomes meaningful only once a target method is ``async def`` (which then also
    wants ``await asyncio.to_thread(...)`` for its blocking work, plus its own
    per-backend lock, since the sipyco lock currently supplies mutual exclusion for
    free).
    """
    name, conf = resolve_device(name)
    device_index()  # fail fast on a duplicate device name anywhere in the config
    host = host or conf.get("host", "127.0.0.1")
    if port is None:
        port = conf.get("port")
    if port is None:
        raise ValueError(f"Device {name!r} has no 'port' configured and none was given")
    port = int(port)
    if allow_parallel is None:
        allow_parallel = bool(conf.get("allow_parallel", False))

    spec = build_spec(name, conf)
    logger.info(
        "Serving device %r (targets: %s) on tcp://%s:%s",
        name,
        ", ".join(sorted(spec.targets)),
        host,
        port,
    )
    try:
        simple_server_loop(
            spec.targets,
            host=host,
            port=port,
            description=spec.description,
            allow_parallel=allow_parallel,
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
    parser.add_argument(
        "--allow-parallel",
        action="store_true",
        default=None,
        help="allow concurrent asyncio RPC calls (inert unless a target method is "
        "async def; see run_device_server)",
    )
    args, _unknown = parser.parse_known_args()

    def _stop(_signo, _frame):
        # Let simple_server_loop unwind (KeyboardInterrupt) so teardown runs.
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass

    run_device_server(
        args.name, host=args.host, port=args.port, allow_parallel=args.allow_parallel
    )


if __name__ == "__main__":
    main()
