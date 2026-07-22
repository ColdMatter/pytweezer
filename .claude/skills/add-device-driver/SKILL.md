---
name: add-device-driver
description: Add a new hardware device driver to pytweezer and wire it into CONFIG["Devices"] so it runs under the generic device server. Use this whenever the user wants to add, create, write, or register a device driver, camera driver, or new piece of lab hardware (a camera, DAC, SLM, laser, sequencer, power meter, translation stage...), asks how to make a device reachable via get_device()/pytweezer-device, or asks to add a device entry to config.py — even if they only describe the hardware and don't use the words "driver" or "config".
---

# Adding a device driver to pytweezer

A device in pytweezer is two things: a **backend class** in `pytweezer/drivers/`
that talks to the vendor SDK, and an **entry in `CONFIG["Devices"]`** naming that
class. There is no registry to edit and no per-driver server script to write —
`pytweezer/servers/device_server.py` imports the class named in config and
constructs it automatically. Getting a device working means writing one module
and one config entry correctly.

## How the framework builds your device

Understanding this makes the rest of the skill obvious, so read it before writing
code:

1. `build_spec(name)` reads the config entry and imports the class named by
   `"class"` (a `"module.path:ClassName"` string), or by `"sim_class"` when
   `"simulate": True`.
2. It inspects that class's `__init__` signature and passes **only the config
   keys whose names match a parameter**. Everything else in the entry
   (`host`, `port`, `class`, …) is dropped simply by not matching.
3. It calls `cls(**those_kwargs)` — immediately, at server startup.
4. `simple_server_loop` serves the resulting object; every public method becomes
   an RPC call.

Step 2 is where most mistakes land, and they are silent: a config key that
doesn't match a constructor parameter is not an error, it just does nothing. The
bundled checker (below) exists to catch exactly this.

Step 3 means the constructor runs on the machine hosting the device. A camera's
`__init__` opens the hardware, so building a real (non-simulated) device on a
dev machine without that hardware attached will fail — that is expected, not a
bug in your driver. Verify in simulation instead.

## Decide which path you're on

**Cameras** subclass `Camera` from `pytweezer/drivers/camera_base.py`. That base
already owns image-stream broadcasting, TIFF autosave, the acquisition loop, and
the connect/relinquish lifecycle. Your driver is a thin shim implementing the
abstract hooks. Follow `pytweezer/drivers/thorcam.py` as the shortest example.

**Everything else** is a plain class with no required base. Public methods become
RPC methods; that's the whole contract. Follow `pytweezer/drivers/slm.py`.

If the hardware produces 2D frames on a stream, it's a camera even if the vendor
doesn't call it one — the base class's broadcast/autosave machinery is worth far
more than the abstraction costs.

## Writing a camera driver

Implement each abstract hook from `Camera`

`_connect` opens the hardware and stores the handle on `self._backend`. On
failure set `self._backend = None` and raise — leaving a half-open handle makes
`is_connected()` lie. `_read_frames(nframes, start_frame)` returns frames stacked
as `(nframes, H, W)`.

Decorate any method that touches `self._backend` with `@requires_camera`, so a
call after `relinquish_camera()` raises a clear error rather than an
`AttributeError` on `None`.

**Match the base class signatures exactly.** `setup_acquisition(self, acq_mode,
nframes)` takes both arguments. A shim that narrows this to `(self, nframes)`
will `TypeError` the moment a caller uses the documented interface — note that
`thorcam.py` currently has this drift, so copy its structure but not that
signature.

## Writing a generic driver

No base class required. Keep methods' arguments and return values
PYON-serializable (numbers, strings, dicts, lists, numpy arrays) — sipyco
marshals them over the wire, and a returned SDK handle object will fail there.

The object must not define `__call__`. sipyco treats a callable target as a
per-connection factory and would invoke it instead of serving it; the framework
rejects this for composites but a plain device would fail confusingly at runtime.

## The simulated twin

Every device must be usable with `"simulate": True`, since that's how the whole
system runs on a dev machine. You almost never hand-write this:

- **Camera:** one line at module scope —
  `SimulatedMyCamera = simulated_camera_for(MyCamera)`. You get synthetic frames
  plus auto-stubbed no-ops for your driver's extra methods.
- **Generic:** omit `"sim_class"` entirely. `default_simulated()` generates a
  stand-in that accepts any constructor arguments and stubs every public method.

Hand-write a `Simulated*` class only when the fake needs *interesting* behavior —
remembering a mask, returning a plausible reading. Decorate it `@simulate(RealCls)`
so methods you didn't bother to fake are still stubbed in, and it can't drift out
of sync as the real class grows. Stubs must return PYON-safe values, never a
`Mock`.

## The config entry

Add to `CONFIG["Devices"]` in `pytweezer/configuration/config.py`:

```python
"Rb ThorLabs Camera": {
    "active": True,
    "class": "pytweezer.drivers.thorcam:ThorLabsCamera",
    "sim_class": "pytweezer.drivers.thorcam:SimulatedThorLabsCamera",
    "teardown": "close",
    "host": SERVER_HOST,
    "port": get_next_port(),
    "simulate": SIMULATING,
    "stream_name": "rb_thorcam",
    "timeout": 5.0,
},
```

- **`host`** decides which PC actually runs the server, independent of where the
  GUI was launched. Use a `HOSTS[...]` entry or `SERVER_HOST`.
- **`port`** is always `get_next_port()`, never a literal. Ports come from a
  sequential iterator consumed in file order, so **inserting an entry mid-dict
  renumbers every device below it**. Append new devices at the end of the
  `"Devices"` block unless you have a reason not to, and expect to restart
  servers on other PCs if you do renumber.
- **`simulate`** is `SIMULATING` (the module-level flag), not a hardcoded bool.
- **`teardown`** names a zero-argument method as a *string* (`"close"`,
  `"disconnect"`), called when the server stops. Omit it if the device needs no
  explicit release.
- **Device names must be unique** across the whole category, including inside
  composite `"devices"` blocks — they share one flat namespace so
  `get_device("Rb SLM")` is unambiguous.
- Remaining keys are your constructor's arguments, and must spell its parameter
  names exactly.

For multi-device rigs sharing one process and a coordinator, see
`docs/device_framework.md` — that's a different shape than covered here.

## Verify it

Run the bundled checker, which resolves the device, flags config keys that don't
reach the constructor, builds the backend in simulation, and reports the RPC
surface:

```bash
poetry run python .claude/skills/add-device-driver/scripts/check_device.py "Rb ThorLabs Camera"
```

Then add the device to `SIM_CONFS` in `tests/test_simulated_devices.py` — that
test PYON-round-trips every simulated method's return value, catching an
unserializable stub at test time rather than at the first live RPC call. Run:

```bash
poetry run pytest tests/ -q
```

Static checks are the goal here; don't launch real servers or hardware to prove
the work. If the user wants a live check, `poetry run pytweezer-device "<name>"`
starts it standalone and `get_device("<name>")` connects from a notebook.

Report honestly what the checker found. A driver that imports and builds in
simulation is verified only that far — connecting to real hardware is still
untested, and saying so is more useful than implying it's done.
