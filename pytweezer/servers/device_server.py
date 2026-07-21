"""Generic device RPC-server launcher.

Every device in ``CONFIG["Devices"]`` runs a sipyco RPC server. Rather than give
each driver module its own ``argparse`` + config-reading + ``simple_server_loop``
boilerplate, this module provides a single launcher:

* ``pytweezer-device <device_name>`` (console script) — start the server for the
  named device from the command line.
* The **device manager** launches the same thing: each device's ``script`` in the
  config points here, so ``ProcessTile`` runs ``python device_server.py <name>``.

A device's config entry points directly at its backend class, so adding a device
or a whole new driver type means editing only ``config.py``. A plain device entry
carries:

* ``"class"``: the real backend, as a ``"module.path:ClassName"`` string.
* ``"sim_class"`` (optional): the simulated/dummy backend, same form. When the
  entry sets ``"simulate": True`` this class is used instead of ``"class"``. If it
  is omitted, a hardware-free stand-in is generated from ``"class"`` automatically
  (see :func:`~pytweezer.servers.simulated_device.default_simulated`), so a device
  can always be simulated; supply ``"sim_class"`` only for an interesting fake.
* ``"teardown"`` (optional): the name of a zero-argument method to call when the
  server stops (e.g. ``"close"``, ``"disconnect"``).
* driver-specific keyword arguments (``stream_name``, ``sdk_dll``, …).

:func:`build_spec` imports the chosen class and constructs it **automatically**: it
reads ``__init__``'s signature and passes the config entries whose keys match
parameter names, so no per-driver "unpack the config into the constructor" glue is
needed. Anything a backend needs beyond receiving those values — resolving a path,
starting a helper process, connecting to hardware — it does in its own ``__init__``
from the arguments it is given (e.g. the MotMaster interface takes a config-file
name, resolves it, ensures the app is running, and connects). Classes are named as
strings and imported lazily — only when actually built — so importing this launcher
never pulls in a hardware library that may be absent (e.g. ``pylablib`` for the
ImagEM).

A **composite** device (any entry with a ``"devices"`` sub-dict) serves several
devices from one process and one port, one RPC target each, optionally alongside a
*coordinator* target — a class named by ``"coordinator"`` (again ``"module:Class"``)
that drives those backends through direct Python calls rather than RPC. That is how
a camera-to-SLM step avoids serializing a frame. Sub-devices stay
individually addressable: ``get_device("RbHamCam")``. See
``docs/device_framework.md``.
"""

import argparse
import importlib
import inspect
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
    ``failed`` names the composite sub-devices that could not be built and are
    therefore absent from :attr:`targets`.
    """

    target_name: Optional[str] = None
    target: Optional[object] = None
    description: str = ""
    teardown: Optional[Callable[[], None]] = None
    targets: Optional[Dict[str, object]] = None
    failed: tuple = ()

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
# Backend construction from config
# --------------------------------------------------------------------------- #

def _backend_class(name, conf):
    """Return the backend class ``conf`` selects.

    Normally this is the class named by ``"class"``. In simulation mode
    (``conf["simulate"]``) it is the class named by ``"sim_class"`` if given, else a
    hardware-free stand-in generated from the real class by
    :func:`~pytweezer.servers.simulated_device.default_simulated` — so a device can
    always be simulated whether or not it ships a hand-written simulated class.
    """
    if conf.get("simulate", False):
        sim_path = conf.get("sim_class")
        if sim_path:
            return _load(sim_path)
        # No hand-written simulated class: generate a stand-in from the real one.
        from pytweezer.servers.simulated_device import default_simulated

        return default_simulated(_load(_require_class(name, conf)))
    return _load(_require_class(name, conf))


def _require_class(name, conf):
    path = conf.get("class")
    if not path:
        raise KeyError(
            f"Device {name!r} has no 'class'; expected a 'module.path:ClassName' "
            "string naming its backend"
        )
    return path


def _load(path):
    """Import and return the object named by a ``"module.path:attr"`` string."""
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"Backend path {path!r} must be 'module.path:ClassName'")
    return getattr(importlib.import_module(module_name), attr)


def _config_kwargs(cls, conf):
    """Config entries whose keys name a parameter of ``cls.__init__``.

    This is the automatic "unpack the config into the constructor" step: framework
    keys like ``class``/``sim_class``/``simulate``/``host``/``port``/``teardown`` are
    dropped simply because they are not constructor parameters, and every backend
    keeps ownership of its own defaults (an absent key just isn't passed).
    """
    params = set(inspect.signature(cls).parameters)
    return {key: value for key, value in conf.items() if key in params}


def _make_composite(name, conf):
    """Build one server exposing several targets from one config entry.

    ``conf["devices"]`` maps a *device name* to an ordinary device config dict (its
    own ``"class"``/``"sim_class"`` etc.); each is built through :func:`build_spec`
    as usual. Sub-devices are named exactly like top-level devices and are reached
    the same way — ``get_device("Rb Feedback Cam")`` — because :func:`device_index`
    flattens them into the same namespace. Their RPC target name is
    :func:`composite_target_name` of that device name. Sub-configs inherit the
    composite's ``simulate`` flag unless they set their own.

    ``conf["coordinator"]`` optionally names a coordinator class as a
    ``"module.path:ClassName"`` string; it is constructed as ``cls(roles, conf)``
    with direct references to the backend objects, so its methods drive them with
    plain Python calls rather than RPC. It receives them keyed by each sub-config's
    ``"role"`` (defaulting to the device name), so a coordinator asks for
    ``targets["camera"]`` regardless of what that camera is called in config.

    **A sub-device that fails to build does not stop the rig.** Its exception is
    logged, its name is recorded in :attr:`DeviceServerSpec.failed`, and the server
    comes up serving whatever else built — so an SLM that won't connect still leaves
    its camera usable. The coordinator is the exception: it is only constructed when
    *every* sub-device built, since it drives all of them, so an incomplete rig
    leaves the composite's own name unserved rather than half-working.
    """
    sub_confs = conf.get("devices")
    if not sub_confs:
        raise KeyError(
            f"Composite device {name!r} needs a non-empty 'devices' dict mapping "
            "device name -> per-driver config"
        )

    coordinator_name = coordinator_target_name(conf)
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

    for device_name, sub_conf in sub_confs.items():
        if "devices" in sub_conf:
            raise ValueError(
                f"Composite device {name!r}: sub-device {device_name!r} may not "
                "itself be a composite"
            )

    targets = {}
    roles = {}
    teardowns = []
    failed = []

    def teardown_all():
        # Reversed: the coordinator is appended last and must release the backends
        # before they are closed underneath it.
        for teardown in reversed(teardowns):
            _safe(teardown)

    try:
        for device_name, sub_conf in sub_confs.items():
            sub_conf = dict(sub_conf)
            sub_conf.setdefault("simulate", conf.get("simulate", False))

            try:
                sub_spec = build_spec(device_name, conf=sub_conf)
                if len(sub_spec.targets) != 1:
                    raise ValueError(
                        f"Composite device {name!r}: sub-device {device_name!r} exposes "
                        f"{len(sub_spec.targets)} targets; exactly one is required"
                    )
                target = next(iter(sub_spec.targets.values()))
                _check_target_callable(name, device_name, target)
            except Exception:
                # One unavailable device must not cost the whole rig its server.
                logger.exception(
                    "Composite device %r: sub-device %r failed to start; serving the "
                    "rest of the rig without it",
                    name,
                    device_name,
                )
                failed.append(device_name)
                continue

            targets[composite_target_name(device_name)] = target
            roles[sub_conf.get("role", device_name)] = target
            if sub_spec.teardown is not None:
                teardowns.append(sub_spec.teardown)

        coordinator_path = conf.get("coordinator")
        if coordinator_path is not None:
            _add_coordinator(
                name, conf, coordinator_path, coordinator_name, roles, failed,
                targets, teardowns,
            )
    except Exception:
        # Release whatever already opened, or a half-built rig leaks hardware handles.
        teardown_all()
        raise

    if not targets:
        logger.error(
            "Composite device %r: nothing could be started (%s); the server stays up "
            "with no targets so its state is visible, but every client call will fail",
            name,
            ", ".join(sorted(failed)) or "no sub-devices configured",
        )

    return DeviceServerSpec(
        targets=targets,
        description=conf.get("description", f"composite device server {name!r}"),
        teardown=teardown_all,
        failed=tuple(failed),
    )


def _add_coordinator(name, conf, path, target_name, roles, failed, targets, teardowns):
    """Construct the composite's coordinator into ``targets``, if it can run.

    A coordinator drives every backend in the rig, so a partial rig gets none: when
    any sub-device failed it is skipped, leaving the composite's own name unserved
    (``get_device(<composite>)`` then raises) rather than serving a coordinator whose
    ``require_role`` would fail mid-experiment. A coordinator that raises while
    constructing is skipped the same way — the sub-devices that did build stay
    individually addressable.
    """
    if failed:
        logger.error(
            "Composite device %r: coordinator not started because sub-device(s) %s are "
            "unavailable; the rest of the rig is still addressable by device name",
            name,
            ", ".join(sorted(failed)),
        )
        return
    try:
        coordinator = _load(path)(roles, conf)
        _check_target_callable(name, target_name, coordinator)
    except Exception:
        logger.exception(
            "Composite device %r: coordinator %s failed to start; serving its "
            "sub-devices without it",
            name,
            path,
        )
        return
    targets[target_name] = coordinator
    teardowns.append(coordinator.shutdown)


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


def coordinator_target_name(conf):
    """RPC target name a composite serves its coordinator under.

    Defaults to ``"coordinator"``; override per-rig with the config key of the same
    name. This is the target :func:`get_device` binds when asked for the composite's
    own name, so its absence from a running server means the coordinator stood down.
    """
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
        if "devices" in conf:
            coordinator = (
                coordinator_target_name(conf) if conf.get("coordinator") else None
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
    explicit dict to override. An entry with a ``"devices"`` sub-dict is a composite
    (see :func:`_make_composite`); otherwise the backend named by the config's
    ``"class"``/``"sim_class"`` is imported and constructed automatically.
    """
    if conf is None:
        name, conf = resolve_device(name)

    if "devices" in conf:
        return _make_composite(name, conf)

    cls = _backend_class(name, conf)
    if conf.get("simulate", False):
        logger.warning("Running device %r in SIMULATION MODE", name)

    backend = cls(**_config_kwargs(cls, conf))

    teardown = None
    teardown_method = conf.get("teardown")
    if teardown_method is not None:
        method = getattr(backend, teardown_method)
        teardown = lambda: _safe(method)  # noqa: E731 — small local teardown wrapper

    return DeviceServerSpec(
        target_name=conf.get("target_name", "device"),
        target=backend,
        description=conf.get("description", f"{name} RPC server"),
        teardown=teardown,
    )


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
        ", ".join(sorted(spec.targets)) or "(none)",
        host,
        port,
    )
    if spec.failed:
        logger.warning(
            "Device %r started degraded: %s unavailable",
            name,
            ", ".join(sorted(spec.failed)),
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
