# pytweezer device control framework

How device RPC servers and clients are defined, launched, and connected to.
Complements [`gui_architecture.md`](gui_architecture.md), which covers the GUIs
that embed the **Devices** tab and the cross-PC **Device Status** feed described
here only in passing.

## The problem this solves

Every physical device (a MotMaster experiment sequencer, a camera) is driven by
a small [sipyco](https://github.com/m-labs/sipyco) RPC server process running on
whichever PC the hardware is attached to. Before this framework, each driver
module (`motmaster_server.py`, `imagemX2.py`, `bfly2.py`) duplicated its own
`argparse` parser, config-reading, backend construction, and
`simple_server_loop(...)` call. Adding a device meant copy-pasting that
boilerplate; the client side had the same duplication (`MotMasterClient`, one
per device type).

Now there is **one generic server launcher** and **one generic client factory**.
A device's config entry points directly at its backend class, so both adding a new
*instance* and adding a new *driver type* are config-only changes.

## Component map

```
pytweezer/configuration/config.py
  CONFIG["Devices"] = { name: {"class": "module:Class", "host":..., "port":..., ...}, ... }

pytweezer/servers/device_server.py      (generic server launcher)
├── build_spec(name)       config -> DeviceServerSpec (imports "class"/"sim_class", constructs it)
├── _make_composite   several devices in one process, one target each
├── device_index()         flat {name -> DeviceAddress} incl. composite sub-devices
├── resolve_address(name)  any addressable device -> DeviceAddress
├── resolve_device(name)   launchable (top-level) device only
├── run_device_server(name, host=None, port=None, allow_parallel=None)  blocks
└── main()                 CLI entry: pytweezer-device <name>

pytweezer/coordinators/                  (in-process device coordination)
├── base.py           Coordinator: holds direct refs to a composite's backends
├── camera_dac_feedback.py   CameraDACFeedback: worked example (image -> DAC)
└── rearrangement.py         Rearrangement: camera + SLM atom-rearrangement loop

pytweezer/servers/device_client.py       (generic client factory)
├── get_device_config(name)   lenient config-key lookup (via resolve_device)
├── get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None)
│     -> sipyco.pc_rpc.Client (transparent RPC proxy)
└── get_device_async(name, host=None, port=None, target_name=AutoTarget)
      -> sipyco.pc_rpc.AsyncioClient (coroutine methods; concurrent servers)

pytweezer/servers/simulated_device.py    (generic simulated-backend mechanism)
├── public_methods(real_cls, exclude=())   introspects real_cls's public methods
├── simulate(real_cls, exclude=(), defaults=None)   class decorator: auto-stub
│     any public method of real_cls a decorated class doesn't already define
└── default_simulated(real_cls)   generate a full no-op stand-in from real_cls
      (used when a device config has no "sim_class")

bin/process_manager.py            DeviceManager (host-filtered start/stop tiles)
bin/process_tile_base.py          ProcessTile (subprocess: python <script> <name>)
pytweezer/servers/device_status.py DeviceStatusServer/-Client (cross-PC up/down feed)
```

## Server side: one launcher for every device

### Config shape

Each entry in `CONFIG["Devices"]` (`pytweezer/configuration/config.py`) names its
backend class with `"class"` (a `"module.path:ClassName"` string), alongside the
usual `host`/`port`/`active`:

```python
"Rb HamCam": {
    "active": True,
    "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
    "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
    "host": SERVER_HOST,
    "port": get_next_port(),
    "simulate": SIMULATING,
    "stream_name": "rb_hamcam",
    "timeout": 5.0,
    "image_dir": "...",
},
```

The config keys `build_spec` reads:

| Key | Meaning |
|---|---|
| `"class"` | Real backend, `"module.path:ClassName"`. Required. |
| `"sim_class"` | Simulated/dummy backend, same form. Used instead of `"class"` when `"simulate": True`; if absent, a hardware-free stand-in is generated from `"class"` (`simulated_device.default_simulated`), so simulation is always available. Supply this only for an interesting fake. |
| `"teardown"` | Name of a zero-argument method run when the server stops (e.g. `"close"`, `"disconnect"`). Optional. |
| `"target_name"` | RPC target label (cosmetic for a single-device server; clients auto-select the sole target). Defaults to `"device"`. |
| `"description"` | sipyco server description. Defaults to `"<name> RPC server"`. |
| anything else | Passed to the backend constructor if its name matches an `__init__` parameter (e.g. `stream_name`, `sdk_dll`). |

Every device is launched by the same file,
`pytweezer/servers/device_server.py`. Because device entries omit `"script"`
while every other category names one, that path lives in exactly one place — the
`DEVICE_SERVER_SCRIPT` constant in `pytweezer/servers/configreader.py`. Device
spawn sites (`DevicesPanel`, the legacy `DeviceManager`, and the generic
`process_cleanup` loop) reference it directly; Servers/Loggers/GUI spawn sites
read their own `params["script"]`.

**Device names must be unique across the whole category**, composite sub-devices
included, because `get_device` addresses every device by name (see below).

### `build_spec`: config → backend → `DeviceServerSpec`

`build_spec(name)` imports the class named by `"class"` (or `"sim_class"` when
simulating) and constructs it **automatically**: it reads the constructor's
signature and passes the config entries whose keys match parameter names. No
per-driver glue function is involved. Anything a backend needs beyond receiving
those values — resolving a path, starting a helper process, connecting to
hardware — it does in its own `__init__` from the arguments it is given. For
example `MotMasterInterface(config_file, interval)` resolves `config_file` against
the package `configuration/` dir, ensures `MOTMaster.exe` is running, and connects,
all in `__init__`; the config just passes `"config_file": "rb_mm_config.json"`.

The result is a `DeviceServerSpec(target_name, target, description,
teardown=None)`, or `DeviceServerSpec(targets={...}, ...)` for the composite
multi-target form. Either way `spec.targets` is the normalized
`{target_name: target}` dict that gets served, and `run_device_server` does the
generic part:

```python
spec = build_spec(name)                  # config -> DeviceServerSpec
simple_server_loop(spec.targets, host=host, port=port,
                    description=spec.description, allow_parallel=allow_parallel)
# ... then, in finally: spec.teardown() if provided (e.g. MotMaster disconnect)
```

**Backends are imported lazily** — the class string is resolved only when that
device is actually built. This matters because a driver like the ImagEM pulls in
`pylablib`, a hardware library that may not be installed on every machine —
importing `device_server.py` itself must never fail just because one driver's
hardware library is absent. Only launching *that specific* device pulls in its
library.

**Simulate-mode is a config choice, not launcher logic.** `"simulate": True`
selects `"sim_class"` (or an auto-generated stand-in if there is none), so a
simulated device never touches real hardware — the simulated MotMaster doesn't
launch `MOTMaster.exe`, and the simulated camera synthesizes frames.

### `simulate()`: one mechanism for every driver's simulated backend

`pytweezer/servers/simulated_device.py` provides a class decorator,
`simulate(real_cls, *, exclude=(), defaults=None)`, used to build both
`SimulatedMotMasterInterface` (`pytweezer/experiment/motmaster_server.py`) and
`SimulatedImagEMX2Camera` (`pytweezer/drivers/imagemX2.py`). Rather than
hand-copying every method of the real backend class into a parallel
`Simulated*` class — which drifts out of sync silently as the real class
changes — `simulate()` inspects `real_cls`'s public methods at class
*definition* time (never an instance, so it never touches hardware) and
injects a safe, logging, no-op stub for every one the decorated class doesn't
already define itself:

```python
@simulate(ImagEMX2Camera)
class SimulatedImagEMX2Camera:
    def __init__(self, image_dir=None, timeout=5.0, stream_name=None):
        ...
    def _generate_frame(self) -> np.ndarray:
        ...  # Gaussian-noise background + scattered atom blobs
    def acquire_n_frames(self, nframes, start_frame=0, autosave=False, broadcast=False):
        ...
    # set_ccd_mode, enable_em_gain, set_sensitivity, ... are all auto-stubbed
```

Only methods with genuinely interesting fake behavior need a hand-written
body; everything else stays interface-complete automatically, and stubs
return plain `None`/`dict`/`list` values (never a `unittest.mock.Mock`) so
they remain serializable by `sipyco.pyon` when a client calls them over RPC —
a bare `MagicMock`/`create_autospec` object is **not** PYON-encodable and
breaks every unconfigured RPC call, which is why this repo doesn't use one as
the literal RPC target. `defaults={method_name: value_or_zero_arg_callable}`
overrides the stub's return value for methods where `None` would be a poor
fake (e.g. `dict`/`list` for methods real callers expect to iterate over).

`simulate()` is MRO-aware: it only stubs a method the decorated class does not
already provide *itself or via a base class*. That's what lets a per-driver
simulated class inherit real fake behaviour from a shared base (e.g.
`SimulatedCamera`) and get only the driver's *extra* methods stubbed — see the
`Camera` abstraction above.

### Cameras: the shared `Camera` abstraction

Camera drivers don't each reimplement ROI/trigger/acquisition/broadcast logic.
`pytweezer/drivers/camera_base.py` defines:

- `Camera(ABC)` — owns everything generic to a camera server: image
  broadcasting onto the ZMQ stream, TIFF autosave, the autosave/broadcast
  acquisition loop (`acquire_n_frames`/`acquire_single_frame`), and the
  connect/relinquish/reacquire lifecycle. It declares the small set of
  hardware-specific hooks a concrete driver must implement: `_connect`/
  `_disconnect`, `set_roi`, `set_trigger_source`, `set_exposure_time`,
  `setup_acquisition`, `start_acquisition`/`stop_acquisition`, and
  `_read_frames` (the raw grab). The live hardware handle lives on
  `self._backend` (`None` == relinquished).
- `SimulatedCamera(Camera)` — a single, one-size-fits-all synthetic backend
  that implements those hooks with generated frames, usable for **any** camera
  in simulation mode.
- `simulated_camera_for(real_cls)` — returns a `SimulatedCamera` subclass with
  `real_cls`'s *extra* methods (beyond the `Camera` interface, e.g. the
  ImagEM's EM-gain setters) auto-stubbed via `@simulate`. This is how
  `SimulatedImagEMX2Camera` is defined — one line, no hand-written per-driver
  simulated class.

A concrete camera driver (e.g. `ImagEMX2Camera` in `imagemX2.py`) is therefore
just the thin dcam→hooks translation shim plus its camera-specific extras.
**A new camera type** is: subclass `Camera`, implement the hooks over its SDK,
and set `SimulatedX = simulated_camera_for(X)` — no bespoke simulated class.

### Adding a new driver type

1. Write the backend class as normal (own module, own connect/acquire/etc.
   methods) — no RPC or config code in it. Give its constructor parameter names
   that match the config keys you intend to pass, and do any further setup (path
   resolution, starting a helper process, connecting) inside `__init__` from those
   arguments. If it needs a simulated variant with interesting fake behavior, write
   a small `Simulated<X>` class decorated with `@simulate(<X>)` next to it (omit it
   entirely and a no-op stand-in is generated automatically).
2. Add device instances to `CONFIG["Devices"]` pointing `"class"` (and, if it has
   one, `"sim_class"`) at those classes.

No factory, no new console script, no new argparse, no `simple_server_loop` call.

### Adding a new instance of an existing driver type

Config-only: add an entry to `CONFIG["Devices"]` reusing the same `"class"` (and
`"sim_class"`) and that backend's expected constructor keys (see its `__init__`).
**Append it**, don't insert: ports come
from `get_next_port()` in declaration order, so inserting renumbers every device
below.

### Launching a device server

Two equivalent paths, both ending up at `run_device_server(name)`:

- **CLI:** `pytweezer-device "Rb HamCam"` (console script in `pyproject.toml`,
  → `pytweezer.servers.device_server:main`). Also runnable as
  `python -m pytweezer.servers.device_server "Rb HamCam"` without `poetry
  install`.
- **Device manager GUI:** `DeviceManager` (`bin/process_manager.py`) builds a
  `ProcessTile` per host-matching, active device; the tile's start button runs
  `python <script> <name>` — same module, same `main()`.

`--host`/`--port` CLI flags override the config's values if given.

### Device-name resolution (`device_index` / `resolve_address` / `resolve_device`)

Every device — top-level or a composite's sub-device — lives in one flat namespace
keyed by its config name. `device_index()` builds that map, returning a
`DeviceAddress(name, conf, owner_name, owner_conf, target_name)` per device:
`owner_conf` is the entry of the process that *serves* it (itself, for a plain
device; the composite, for a sub-device) and is therefore where `host`/`port` live.
It raises `KeyError` if two devices share a name, since such a name could not be
resolved unambiguously.

Names match leniently: whitespace-stripped and lowercased, so `RbHamCam`,
`rb hamcam`, and `RB HAMCAM` all resolve to `"Rb HamCam"`. This matters because
config names contain spaces and are awkward as bare CLI arguments.

Two entry points, with different jobs:

- **`resolve_address(name)`** — used by clients. Resolves *any* addressable device.
- **`resolve_device(name)`** — used by the launcher. Resolves only devices with a
  server of their own, i.e. top-level entries. Naming a sub-device raises a
  `KeyError` telling you to launch its composite instead, rather than pretending
  the name doesn't exist.

Both raise `KeyError` listing the valid names on a typo, so a mistake fails fast
instead of surfacing as a confusing downstream error ("no port configured").
`run_device_server` also calls `device_index()` at startup purely to fail fast on a
duplicate name anywhere in the config.

## Client side: one factory for every device

`pytweezer/servers/device_client.py` gives callers (experiment scripts,
notebooks, other servers) a way to get an RPC client without writing a client
class per device:

```python
from pytweezer.servers.device_client import get_device

cam = get_device("Rb HamCam")
cam.acquire()          # remote call, transparently proxied
cam.close_rpc()
```

`get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None)`
resolves `name` through `resolve_address`, connects to whichever process serves it,
and returns a `sipyco.pc_rpc.Client`.

**Every device is addressed by its own name**, whether it has a server to itself or
shares one with other devices. The caller never needs to know which process a device
lives in, nor that sipyco targets exist:

```python
cam = get_device("Rb HamCam")          # a server of its own
cam = get_device("Rb Feedback Cam")    # shares a process with a DAC; same call shape
```

`target_name` therefore rarely needs passing. Its `AutoTarget` default now means
"whatever target the config says serves this device": the server's sole target for a
plain device, or the sub-device's target on a composite. Pass it explicitly only to
reach a target the config doesn't name.

`get_device_config(name)` returns the device's own config entry without connecting.
For a sub-device that entry holds only its driver settings — `host`/`port` belong to
the composite serving it, so use `get_device` or `resolve_address` rather than
reading them off the dict.

### Concurrent devices: `run_parallel`

`get_device(...)` returns a blocking client — the calling script doesn't move
past an RPC call until that call's full remote execution finishes. That's a
problem for something like starting two independent MotMaster sequencers "in
parallel": each is its own device server process, so the hardware doesn't
contend, but a blocking client only *issues* the second call after the first
one's `Go()` has already returned. It's a hard requirement, not just a speed-up,
when one call cannot return until another fires — e.g. a MotMaster armed in
trigger mode, whose `Go()` blocks until a second MotMaster's sequence triggers it.

The ergonomic path is `run_parallel` (in `pytweezer/parallel.py`): it runs
zero-argument callables in one thread each and returns their results in order.
Because every device is its own server process, a `get_device` call is just a
blocking socket round-trip whose `recv` releases the GIL, so the threads overlap
for real — no `async`/`await`, and it works from the PyQt5 GUI (which has no
`qasync`). `after(delay, call)` staggers one call's start to win the arm-before-
trigger race:

```python
from pytweezer.parallel import run_parallel, after
from pytweezer.servers.device_client import get_device

cam = get_device("Rb HamCam")
mm1 = get_device("Rb MotMaster Server")
mm2 = get_device("CaF MotMaster Server")

cam.start_acquisition()          # arm the camera (returns immediately)
mm1.set_trigger_mode(True)       # mm1's Go() will wait for a hardware trigger

frame, _, _ = run_parallel(
    lambda: cam.acquire_n_frames(1),               # blocks reading the frame
    mm1.start_motmaster_experiment,                # armed; waits for trigger
    after(0.05, mm2.start_motmaster_experiment),   # fires 50 ms later
)
```

Each parallel call must use a **different** client — a single sipyco `Client` is
not thread-safe. Pass `timeout=` for an overall deadline (raises `TimeoutError`,
abandoning the still-running daemon threads). A single failing call re-raises its
own exception; several raise an `ExceptionGroup`.

#### Lower-level async alternative: `get_device_async`

If you'd rather drive the servers with `asyncio` directly,
`get_device_async(name, host=None, port=None, target_name=AutoTarget)` is the
same lookup and config resolution, but returns a `sipyco.pc_rpc.AsyncioClient`
whose RPC methods are coroutines. Connect several and drive them with
`asyncio.gather` so the calls are issued together instead of one waiting on
the other's reply:

```python
import asyncio
from pytweezer.servers.device_client import get_device_async

async def main():
    mm1 = await get_device_async("Rb MotMaster Server")
    mm2 = await get_device_async("CaF MotMaster Server")
    try:
        await asyncio.gather(
            mm1.start_motmaster_experiment(),
            mm2.start_motmaster_experiment(),
        )
    finally:
        await mm1.close_rpc()
        await mm2.close_rpc()

asyncio.run(main())
```

No server-side change is needed for this — each device already runs its own
`simple_server_loop`, so the concurrency only needed solving on the client.
Fine to call from scripts/notebooks; the GUI is PyQt5 with no `qasync`
integration, so calling from GUI code would need a worker thread. See
`async_device_comms_notes.md` (repo root) for the fuller investigation
(including why `AsyncioClient` still awaits each reply — the win is running
calls to *different* servers concurrently, not skipping replies).

### Relationship to `MotMasterClient`

`pytweezer/experiment/motmaster_client.py`'s `MotMasterClient` predates this
framework and still exists for its back-compat method aliases
(`set_script`/`start_experiment`/`shutdown_server` → the real
`set_motmaster_experiment`/`start_motmaster_experiment`/`terminate`). New code
should prefer `get_device(...)` directly; `MotMasterClient` has not been
migrated onto it.

## Composite devices and coordinators

Some experiments need two devices coupled tightly: grab a camera frame, reduce it,
and set a DAC voltage from the result. Doing that with two `get_device` clients
costs an RPC round trip per device per step, and the frame gets pyon-encoded onto a
socket and decoded again — far too slow for a control loop.

A **composite device** solves this. One `CONFIG["Devices"]` entry produces one
`device_server.py` process serving several RPC targets on one port: one per
sub-device, plus an optional **coordinator**. The sub-devices are ordinary Python
objects in that one process, so the coordinator holds direct references to them and
drives them with plain method calls. There is no transport between targets to
configure, because there is no transport — a frame handed from camera to coordinator
is an attribute lookup. Only the scalar summary crosses RPC.

Crucially, the camera stays a first-class device. A composite is not a merged
camera+DAC blob, and it is not a namespace you address *through* — sub-devices are
named and reached exactly like any other device.

### Config shape

```python
"Rb Feedback Rig": {
    "active": False,
    "host": SERVER_HOST,
    "port": get_next_port(),
    "simulate": SIMULATING,
    "devices": {
        "Rb Feedback Cam": {"class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
                            "sim_class": "pytweezer.drivers.imagemX2:SimulatedImagEMX2Camera",
                            "role": "camera",
                            "stream_name": "rb_feedback_cam", "timeout": 5.0},
        "Rb Feedback DAC": {"class": "pytweezer.drivers.ni_dac:NIDAC",
                            "sim_class": "pytweezer.drivers.ni_dac:SimulatedNIDAC",
                            "role": "dac", "channels": ["Dev1/ao0"]},
    },
    "coordinator": "pytweezer.coordinators.camera_dac_feedback:CameraDACFeedback",
},
```

An entry is a composite exactly when it has a `"devices"` sub-dict. Each sub-entry
is an ordinary device config (its own `"class"`/`"sim_class"`), built through
`build_spec` like any other, and is **named like a top-level device**. Its RPC target
name is that name folded to be wire-safe (`composite_target_name`: whitespace
stripped, lowercased — sipyco rejects target names containing spaces), but nothing
outside this module needs to know that. Sub-configs inherit the composite's
`simulate` flag unless they set their own.

`"role"` is how the *coordinator* looks a backend up; the device *name* is how RPC
clients address it. Separating the two lets one coordinator class serve rigs whose
devices are named differently. A role defaults to the device name.

`DeviceManager`, `ProcessTile`, and `DeviceStatusServer` need no special handling:
a composite is one config entry, one host:port, one launch command.

### Addressing

Sub-devices are addressed by name, like everything else. The composite's own name
resolves to its coordinator:

```python
cam   = get_device("Rb Feedback Cam")    # camera target on the rig's server
dac   = get_device("Rb Feedback DAC")    # same server, same port, different target
coord = get_device("Rb Feedback Rig")    # the composite -> its coordinator

coord.image_to_dac(setpoint=130.0, gain=0.01, channel="Dev1/ao0")   # one step
coord.run_n(50, setpoint=130.0, gain=0.01, channel="Dev1/ao0")      # 50 steps, one RPC
```

A composite with no coordinator has nothing to serve under its own name, so
`get_device` on it raises a `KeyError` naming its sub-devices instead.

### Writing a coordinator

Subclass `Coordinator` (`pytweezer/coordinators/base.py`), reach the backends
through `self.require_role(<role>)`, and point the composite's `"coordinator"`
config field at the class as a `"module.path:ClassName"` string. `build_spec`
imports it and constructs it as `cls(roles, conf)`.
`pytweezer/coordinators/camera_dac_feedback.py` is the worked example: `arm()`,
`measure()`, `control()`, `image_to_dac()`, and `run_n()`.
`pytweezer/coordinators/rearrangement.py` (`Rearrangement`) is a real one — a
camera + Blink SLM atom-rearrangement loop, with the GPU stack imported lazily so
it degrades gracefully off the GPU host; see `docs/rearrangement_coordinator.md`.

Three constraints, all consequences of how sipyco serves targets:

- **Every public method is a synchronous RPC method and stalls the entire server
  while it runs.** The sipyco server is single-threaded asyncio and calls a plain
  `def` target method inline on the event loop, so no target on that server —
  camera, DAC, or coordinator — answers anything until it returns. This is not new
  (a plain `camera.acquire_n_frames(100)` RPC already does it), but it means a batch
  method must take a bounded iteration count rather than free-run. `run_n(n)` is the
  fast-loop answer: one RPC issues the whole batch, and each iteration is direct
  Python calls with no serialization. It cannot be aborted mid-batch.
- **Return values cross a PYON boundary.** Return plain types, dicts, lists, numpy
  arrays; never exception objects or backend handles. `image_to_dac` deliberately
  returns `{"mean", "voltage", "t"}` and *not* the frame — returning the frame would
  reintroduce exactly the serialization cost the coordinator exists to avoid.
- **Never define `__call__`** on a target. `Server._handle_connection_cr` does
  `if callable(target): target = target()`, treating a callable target as a
  per-connection factory. `_make_composite` rejects callable targets up front, along
  with nameless sub-devices and two sub-devices whose names fold to the same target.

If a loop ever needs to free-run while the server stays answerable, the coordinator
grows `start`/`stop`/`status` backed by a `threading.Thread`; nothing else in this
design moves. A thread — not an asyncio task — because the drivers are synchronous
and an asyncio task's cadence is hostage to any other client's blocking RPC.

### `allow_parallel`

`run_device_server` accepts `allow_parallel` (config key of the same name, or
`--allow-parallel`), which drops the lock sipyco holds across each RPC call.

**It has no effect while every target method is a plain `def`.**
`Server._process_action` awaits a method's result only when that result is a
coroutine, and an uncontended `asyncio.Lock.acquire()` returns without suspending.
With synchronous methods there is therefore no suspension point between acquiring
and releasing that lock, so nothing can ever contend for it. What serializes calls
is the single-threaded event loop, not the lock. Do not reach for this flag as a
speed knob.

It becomes meaningful only once some target method is `async def` — which then also
wants `await asyncio.to_thread(...)` around its blocking work, and its own
per-backend lock, because the sipyco lock is currently supplying mutual exclusion
for free.

## How this fits the rest of the system

- **`DevicesPanel`** (`bin/managed_panel.py`, the live Devices tab; the legacy
  `DeviceManager` in `bin/process_manager.py` is the earlier equivalent) does its
  own host filtering (`check_host`) and start/stop tile management. Every device's
  tile launches the same `device_server.py` via the `DEVICE_SERVER_SCRIPT`
  constant, since device entries carry no `"script"`. A composite is one tile;
  its sub-devices are not separately launchable.
- **`DeviceStatusServer`** (`pytweezer/servers/device_status.py`) independently
  TCP-probes `host:port` for every top-level `CONFIG["Devices"]` entry to publish a
  cross-PC up/down feed — it does not use `get_device`/RPC calls, just raw
  reachability (see `gui_architecture.md` for the full Device Status writeup). A
  composite is probed once, since its sub-devices share its port.
- **Both GUIs' Applets/experiment code** are the intended callers of
  `get_device` going forward, instead of hand-rolled `sipyco.pc_rpc.Client(...)`
  calls with hardcoded host/port.

## File index

| File | Role |
|------|------|
| `pytweezer/servers/device_server.py` | Generic server launcher: `build_spec` (imports the config's `"class"`/`"sim_class"` and constructs it), `resolve_device`, `run_device_server`, `main`. |
| `pytweezer/servers/device_client.py` | Generic client factory: `get_device`, `get_device_async`, `get_device_config`. |
| `pytweezer/servers/simulated_device.py` | Generic simulated-backend mechanism: `simulate` class decorator (MRO-aware), `public_methods`, `default_simulated` (auto-generated stand-in when a device has no `sim_class`). |
| `pytweezer/coordinators/base.py` | `Coordinator` — base for in-process coordination of a composite's backends. |
| `pytweezer/coordinators/camera_dac_feedback.py` | `CameraDACFeedback` — worked example: frame mean → proportional DAC correction. |
| `pytweezer/coordinators/rearrangement.py` | `Rearrangement` — camera + SLM atom-rearrangement loop (lazy GPU stack). See `docs/rearrangement_coordinator.md`. |
| `pytweezer/drivers/ni_dac.py` | `SimulatedNIDAC`; the real NI analog-output driver belongs here. |
| `pytweezer/drivers/slm.py` | `SLM` (Meadowlark Blink SDK) + `SimulatedSLM`. |
| `pytweezer/experiment/motmaster_server.py` | MotMaster backend classes (`MotMasterInterface`, `SimulatedMotMasterInterface`) + legacy standalone `main()`. |
| `pytweezer/drivers/camera_base.py` | Generic camera abstraction: `Camera` (ABC), `SimulatedCamera` (one-size-fits-all sim), `simulated_camera_for`, `requires_camera`. |
| `pytweezer/drivers/imagemX2.py` | ImagEM X2 dcam shim (`ImagEMX2Camera` on `Camera`) + `SimulatedImagEMX2Camera = simulated_camera_for(...)` + legacy standalone `main()`. |
| `pytweezer/drivers/bfly2.py` | Blackfly backend class + legacy standalone `main()`. |
| `pytweezer/experiment/motmaster_client.py` | `MotMasterClient` — pre-framework client with back-compat aliases. |
| `pytweezer/configuration/config.py` | `CONFIG["Devices"]` — `class`/`sim_class`/`host`/`port`/`active`/... per device. |
| `bin/process_manager.py` | `DeviceManager` — host-filtered start/stop tiles (unchanged besides the shared script path). |
| `pyproject.toml` | `pytweezer-device = "pytweezer.servers.device_server:main"` console script. |

**Note:** each driver module's original standalone `main()`/`run_server()` is
still present (useful for ad hoc debugging directly against that module) but is
no longer on the launch path — both the CLI and the device manager go through
`device_server.py`.
