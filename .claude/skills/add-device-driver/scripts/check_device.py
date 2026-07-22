"""Static check for a device entry in CONFIG["Devices"].

Builds the device the same way ``device_server.py`` does in production, but
forced into simulation so no hardware is touched, and reports the problems that
are otherwise silent:

* config keys that never reach the constructor (a typo'd key is not an error,
  it is simply ignored, so the device runs with a default nobody intended)
* constructor parameters with no default that config doesn't supply
* abstract hooks a Camera subclass forgot, or implemented with a signature that
  drifts from the base class
* a callable target, which sipyco would invoke instead of serving

Usage:
    poetry run python check_device.py "Rb ThorLabs Camera"
    poetry run python check_device.py --all
"""

import argparse
import inspect
import sys

from pytweezer.servers import device_server
from pytweezer.servers.configreader import ConfigReader

# Keys the framework consumes itself; they are expected not to match a
# constructor parameter, so they are never reported as unused.
FRAMEWORK_KEYS = {
    "active", "class", "sim_class", "simulate", "host", "port", "teardown",
    "description", "target_name", "allow_parallel", "script", "devices",
    "coordinator", "coordinator_target_name", "role",
}

problems = []
notes = []


def problem(msg):
    problems.append(msg)
    print(f"  FAIL  {msg}")


def note(msg):
    notes.append(msg)
    print(f"  warn  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def check_config_keys(name, conf, cls):
    """Report config keys that don't map onto the constructor."""
    params = inspect.signature(cls).parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    unused = [
        key for key in conf
        if key not in FRAMEWORK_KEYS and key not in params
    ]
    if unused and not accepts_kwargs:
        for key in unused:
            problem(
                f"config key {key!r} does not match any parameter of "
                f"{cls.__name__}.__init__ - it is silently ignored "
                f"(parameters: {', '.join(k for k in params if k != 'self')})"
            )
    else:
        ok("every config key maps to a constructor parameter")

    missing = [
        pname for pname, p in params.items()
        if pname != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.VAR_KEYWORD)
        and pname not in conf
    ]
    for pname in missing:
        problem(
            f"{cls.__name__}.__init__ requires {pname!r} but config supplies no "
            f"such key - the server will fail to start"
        )


def check_camera_hooks(cls):
    """Verify Camera subclasses implement the hooks with matching signatures."""
    try:
        from pytweezer.drivers.camera_base import Camera
    except Exception as exc:  # pragma: no cover - import guard
        note(f"could not import Camera base to check hooks: {exc}")
        return
    if not (isinstance(cls, type) and issubclass(cls, Camera)):
        return

    unimplemented = getattr(cls, "__abstractmethods__", frozenset())
    if unimplemented:
        problem(
            f"{cls.__name__} does not implement abstract hook(s): "
            f"{', '.join(sorted(unimplemented))}"
        )
        return

    drifted = False
    for hook, base_func in vars(Camera).items():
        if not getattr(base_func, "__isabstractmethod__", False):
            continue
        impl = getattr(cls, hook, None)
        if impl is None:
            continue
        base_params = list(inspect.signature(base_func).parameters)
        impl_params = list(inspect.signature(impl).parameters)
        if base_params != impl_params:
            drifted = True
            problem(
                f"{cls.__name__}.{hook}{tuple(impl_params)} does not match the "
                f"Camera base signature {tuple(base_params)} - callers using the "
                f"documented interface will TypeError"
            )
    if not drifted:
        ok(f"{cls.__name__} implements the Camera hooks with matching signatures")


def check_device(name):
    print(f"\n=== {name} ===")
    before = len(problems)

    canonical, conf = device_server.resolve_device(name)
    if canonical != name:
        note(f"config key is actually {canonical!r}")

    if not conf.get("class") and "devices" not in conf:
        problem("entry has no 'class' (expected a 'module.path:ClassName' string)")
        return

    if "devices" in conf:
        note("composite device - see docs/device_framework.md; checking sub-devices")

    # Check the real class's wiring, without constructing it (that would open
    # hardware). Construction is exercised below against the simulated class.
    if conf.get("class"):
        try:
            real_cls = device_server._load(conf["class"])
        except Exception as exc:
            problem(f"cannot import {conf['class']!r}: {exc}")
            return
        ok(f"real class imports: {real_cls.__module__}:{real_cls.__name__}")
        check_camera_hooks(real_cls)
        check_config_keys(canonical, conf, real_cls)

    # Build in simulation: same code path as production, no hardware.
    sim_conf = dict(conf, simulate=True)
    try:
        spec = device_server.build_spec(canonical, conf=sim_conf)
    except Exception as exc:
        problem(f"build_spec failed in simulation: {type(exc).__name__}: {exc}")
        return
    ok(f"builds in simulation; RPC targets: {', '.join(sorted(spec.targets))}")

    for target_name, target in spec.targets.items():
        if callable(target):
            problem(
                f"target {target_name!r} is callable ({type(target).__name__} "
                f"defines __call__); sipyco would call it instead of serving it"
            )
        methods = [
            m for m in dir(target)
            if not m.startswith("_") and callable(getattr(target, m, None))
        ]
        ok(f"target {target_name!r} exposes {len(methods)} RPC methods")

    if conf.get("teardown"):
        if spec.teardown is None:
            problem(f"teardown {conf['teardown']!r} did not resolve to a method")
        else:
            ok(f"teardown {conf['teardown']!r} resolves")

    if spec.teardown is not None:
        try:
            spec.teardown()
        except Exception as exc:
            note(f"simulated teardown raised {type(exc).__name__}: {exc}")

    if len(problems) == before:
        print("  --> no problems found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="device name (key in CONFIG['Devices'])")
    parser.add_argument("--all", action="store_true", help="check every device")
    args = parser.parse_args()

    if args.all:
        names = list(ConfigReader.getConfiguration().get("Devices", {}))
    elif args.name:
        names = [args.name]
    else:
        parser.error("give a device name or --all")

    for name in names:
        try:
            check_device(name)
        except KeyError as exc:
            problem(f"{name}: {exc}")
        except Exception as exc:
            problem(f"{name}: unexpected {type(exc).__name__}: {exc}")

    print(f"\n{len(problems)} problem(s), {len(notes)} warning(s)")
    sys.stdout.flush()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
