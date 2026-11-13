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
4. `serve()` (sipyco's `simple_server_loop`, logging every call that raises)
   serves the resulting object; every public method becomes an RPC call.

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
the connect/close lifecycle. Your driver is a thin shim implementing the
abstract hooks. Follow `pytweezer/drivers/orca.py` as the shortest example.

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

The base also gives every camera a frame-index protocol, which notebooks'
`PycafCamera` relies on (see `notebook-rig`): `arm(buffer_frames)` starts a continuous
ring-buffer acquisition, `frames_acquired()` counts frames since it started,
`read_frames_from(start)` returns every frame since `start`, `settings()` is
saved beside frames in zips. Live mode is something a camera is put in and
taken out of at run time, never a constructor option: `start_live(buffer_frames)`
arms it and broadcasts every new frame, `stop_live()` stops the broadcast and
the acquisition, `is_live()` says which. An experiment using a camera takes a
live one out of live mode in `prepare` and puts it back in `finish` (see
`LoadTweezers`). The defaults assume
a pylablib backend; otherwise implement `_acquisition_in_progress`,
`_count_frames` and `_read_range`. Take `self._lock` around backend calls the
live thread could race. Paths such as `image_dir` come from the CONFIG entry;
a driver holds no default for them.

`set_roi(x0, width, y0, height)` passes pylablib exclusive ends
(`x0 + width`, not `x0 + width - 1`) and the current binning from
`get_roi()` (pylablib's `set_roi` resets binning to 1 otherwise), stores the corner the camera actually
applied in `self.roi_origin` and returns it: `broadcast_image` puts it in every
frame's header as `roi_x0`/`roi_y0`, which is how analyses place trap sites on
a cropped frame. `set_roi` and `set_binning` also keep the binning in
`self.binning`, broadcast as `binning`, so a viewer can turn frame pixels back
into sensor pixels.
`set_binning` and `set_gain` are optional (the base raises
`NotImplementedError`); only experiments written for a camera that has them
call them.

A camera declares the settings it offers in `CONTROLS`, a tuple of `Control`
(`camera_base.py`; shared ones `EXPOSURE`, `BINNING`, `ROI`, `TRIGGER_POLARITY`,
and the `trigger_source()` and `gain()` factories). `controls()`,
`read_controls()` and `apply_controls()` work from it, so a driver implements
the setters, calls `self._keep(name=value)` with the SI value the camera
actually took, takes `self._lock` around each, and overrides
`_read_control_values()` to read every control back. A control whose setter is
not `set_<name>` is named in `CONTROL_SETTERS`; do not override
`_apply_control`, which the simulated twin would not inherit. `controls()`
gives the ROI controls the sensor size as their `maximum`, read once connected
(and on each `read_controls()`) by `_sensor_shape()`, which calls the
backend's `get_detector_size()` (pylablib cameras have it); a backend without
one overrides `_sensor_shape()`. A driver that can
read each frame's camera-clock time overrides `_read_range_timed`; the live
broadcast then adds it as `camera_timestamp`. `snap()` takes one frame from a
stopped camera. A driver that watches the sensor temperature sets
`self._temperature_monitor` to a `TemperatureMonitor`; `sensor_temperature()`
then serves its last reading without touching the hardware or taking `_lock`.

The Thorlabs cameras are `thorcam.ThorCam`; `tweezermonitorcam.ThorCam` is it with
another default serial number. The Hamamatsu DCAM cameras (ImagEM X2, ORCA) subclass `HamamatsuCamera`
(`hamamatsu.py`), which holds everything they share: connecting, the common
setters, the ROI and trigger controls, timestamps and the temperature monitor.
A new one sets `model_prefix`, `image_prefix`, `default_timeout` and `CONTROLS`,
and adds only its own setters and the extra keys in `_read_control_values()`.

**Match the base class signatures exactly.** `setup_acquisition(self, nframes,
mode="sequence")` takes both, and the base `arm()` calls it; a camera with no
acquisition modes (the Thorlabs ones) accepts `mode` and ignores it. A shim
that narrows a signature will `TypeError` the moment a caller uses the
documented interface; the checker flags it.

## Writing a generic driver

No base class required. Keep methods' arguments and return values
PYON-serializable (numbers, strings, dicts, lists, numpy arrays) — sipyco
marshals them over the wire, and a returned SDK handle object will fail there.

The object must not define `__call__`. sipyco treats a callable target as a
per-connection factory and would invoke it instead of serving it; the framework
rejects this for composites but a plain device would fail confusingly at runtime.

`pytweezer/drivers/ni_cards.py` is the example of hardware shared with other
programs. It enforces its `reserved` and `protected` channels itself, not only
in its GUI tab, and takes them as parameters with no default, so a misspelt
config key fails construction instead of leaving channels unguarded. It names
the MotMaster it defers to by its CONFIG key (`"motmaster"`) and imports
`pytweezer.drivers.motmaster` only on first use (`ni_cards_sources.py`), since
importing that module without .NET prints pythonnet's errors in every process
that loads the driver. Errors that come back through .NET reflection arrive
wrapped in `TargetInvocationException`; unwrap them (`_unwrapped`) before
matching on type.

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
"ORCA": {
    "active": True,
    "class": "pytweezer.drivers.orca:ORCACamera",
    "sim_class": "pytweezer.drivers.orca:SimulatedORCACamera",
    "teardown": "close",
    "host": HOSTS["ph-bonesaw"],
    "port": get_next_port(),
    "simulate": SIMULATING,
    "serial_number": "280187",
    "stream_name": "orca",
    "timeout": 5.0,
},
```

- **`host`** decides which PC actually runs the server, independent of where the
  GUI was launched. Name the PC the hardware is plugged into with its `HOSTS[...]`
  entry, even the server PC: never `SERVER_HOST`, which only happens to be
  the same machine (and becomes localhost when simulating).
- **`port`** is always `get_next_port()`, never a literal. Ports come from a
  sequential iterator consumed in file order, so **adding an entry renumbers
  every entry after it**. Put it where it belongs, then update and restart
  every PC together; a PC on the old config dials the old ports.
- **`simulate`** is `SIMULATING` (the module-level flag), not a hardcoded bool.
- **`teardown`** names a zero-argument method as a *string* (`"close"`,
  `"disconnect"`), called when the server stops. Omit it if the device needs no
  explicit release.
- **Device names must be unique** across the whole category, including inside
  composite `"devices"` blocks — they share one flat namespace so
  `get_device("Rb SLM")` is unambiguous.
- Remaining keys are your constructor's arguments, and must spell its parameter
  names exactly.

For multi-device rigs sharing one process (composites, and the currently
unused coordinators), load the `pytweezer-device-framework` skill — that's a
different shape than covered here.

## Verify it

Run the bundled checker, which resolves the device, flags config keys that don't
reach the constructor, builds the backend in simulation, and reports the RPC
surface:

```bash
uv run python .claude/skills/add-device-driver/scripts/check_device.py "ORCA"
```

`--all` checks every entry. If you hand-wrote a `Simulated*` class, add the
`(Real, Simulated)` pair to `test_simulated_backend_covers_real_interface` in
`tests/test_simulate_decorator.py`; for anything else worth testing, see the
`write-tests` skill. Run:

```bash
uv run pytest tests/ -q
```

Static checks are the goal here; don't launch real servers or hardware to prove
the work. If the user wants a live check, `uv run pytweezer-device "<name>"`
starts it standalone and `get_device("<name>")` connects from a notebook.

Report honestly what the checker found. A driver that imports and builds in
simulation is verified only that far — connecting to real hardware is still
untested, and saying so is more useful than implying it's done.
