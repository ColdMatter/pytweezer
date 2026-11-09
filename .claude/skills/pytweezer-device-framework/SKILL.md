---
name: pytweezer-device-framework
description: How pytweezer's generic device server and client work — device_server.py building any driver from its CONFIG entry, get_device() addressing, simulation, composite devices and coordinators, and the rules for what blocks what over RPC. Use when calling devices from experiment or notebook code, running calls against several devices concurrently, debugging a device that will not start or resolve, changing device_server.py/device_client.py/simulated_device.py themselves, grouping devices into one process or writing a coordinator, working out who may use a device while a task runs (reservations, restrict_devices, UndeclaredDeviceError, a device that stays off hardware a task holds), or reasoning about RPC latency and blocking between targets. For writing a brand-new hardware driver, use the add-device-driver skill instead.
---

# The device framework

Every device runs the same launcher. `pytweezer/servers/device_server.py` reads
`CONFIG["Devices"][name]["class"]` (a `"module.path:ClassName"` string), imports
it, constructs it from the matching config keys, and serves it with `serve()`
(sipyco's `simple_server_loop`, with the `RPCServer` below). There is no
per-driver server script and no registry — `"class"` is the only thing that
differentiates one device from another.

Imports are lazy and per-device, so a missing hardware library (e.g. `pylablib`
for the ImagEM) never breaks importing the launcher or starting other devices.

**Writing a new driver is a different task** — the class contract, the `Camera`
base, the simulated twin, and the config-key matching rules live in the
`add-device-driver` skill.

## Calling a device

```python
from pytweezer.servers.device_client import get_device

cam = get_device("Rb HamCam")  # transparent sipyco Client; methods are RPC calls
cam.close_rpc()
```

A call that raises comes back to the caller as the same exception, and the
device server also logs it with its traceback (`RPCServer` in
`device_server.py`). Plain sipyco would only send it back. A method name
the server doesn't serve fails in the client (`AttributeError`) and never
reaches the server, so it leaves no server log.

Prefer this over hand-built `Client(host, port, target)` calls anywhere in new
experiment code — it resolves host, port and target from config for you.
`get_device_config(name)` returns an entry without connecting, but for a
sub-device that entry holds only driver settings: `host`/`port` belong to the
composite serving it, so address it through `get_device`, never by reading the
dict.

## Names and addressing

`device_index()` builds `{normalized_name: DeviceAddress}` over **every**
addressable device, flattening composite sub-devices into the same namespace as
top-level ones. Consequences worth holding on to:

- **Device names must be unique across the whole category**, composites
  included. A collision raises `KeyError` at resolution time, and
  `run_device_server` calls `device_index()` at startup purely to fail fast on
  one anywhere in the config.
- `resolve_address(name)` matches case- and whitespace-insensitively, so
  `rbhamcam` and `"Rb HamCam"` both work. Callers never pass a target name.
- `resolve_device(name)` is the *launchable* subset — top-level entries only. A
  composite's sub-device has no server of its own, so naming one raises a
  `KeyError` telling you to launch its composite instead.

## Config keys the launcher itself reads

Everything not in this table is passed to the backend constructor if its name
matches a parameter, and silently ignored if it doesn't.

| Key | Meaning |
| --- | --- |
| `"class"` | Real backend, `"module.path:ClassName"`. Required. |
| `"sim_class"` | Used instead when `"simulate": True`; if absent, a hardware-free stand-in is generated from `"class"`, so simulation always works. Supply one only for an interesting fake. |
| `"teardown"` | Name of a zero-arg method to run when the server stops (`"close"`, `"disconnect"`). |
| `"target_name"` | RPC target label. Cosmetic on a single-device server — clients auto-select the sole target. |
| `"description"` | sipyco server description. |
| `"allow_parallel"` | Drop sipyco's per-call lock; inert for plain `def` methods (see "What blocks what"). |
| `"devices"`, `"coordinator"`, `"coordinator_target_name"`, `"role"` | Composite wiring, below. |

`"active"` only says whether the GUI starts the device at launch; one started
by hand is used, reserved and shown like any other. **Adding an entry renumbers
every port after it** (`get_next_port()` hands them out in declaration order),
so every PC is updated and restarted together; see `pytweezer-architecture`.

## Composite devices and coordinators

An entry with a `"devices"` sub-dict runs several devices in **one process**, one
sipyco target each, plus an optional **coordinator** holding direct Python
references to them. The point is latency: a camera→DAC feedback step through a
coordinator is an attribute lookup, not an RPC round trip, and no frame is ever
serialized — only the scalar summary crosses the wire.

Sub-devices stay first-class. They are addressed exactly like top-level devices
(`get_device("Rb Feedback Cam")`), and the composite's own name resolves to its
coordinator. A composite with no coordinator has nothing to serve under its own
name, so `get_device` on it raises `KeyError` naming its sub-devices.

- A coordinator finds its backends by each sub-config's **`"role"`**, never by
  device name, so one coordinator class can serve rigs whose devices are named
  differently. A role defaults to the device name.
- Sub-configs inherit the composite's `simulate` flag unless they set their own.
- **A native crash while building a sub-device does take every sibling down**, as
  it kills the process and no `except` can catch it. A driver whose library can do
  that must probe it in a child process first and raise on failure, as
  `open_dcam_camera` does (`_probe_dcam_library` in `drivers/hamamatsu.py`).
- **A dead sub-device does not take the server down.** One that fails to build is
  logged, recorded in `DeviceServerSpec.failed`, and skipped; everything else
  still serves. The **coordinator is the exception** — it drives every backend,
  so it is built only when all sub-devices built, and stands down otherwise
  rather than failing halfway through an experiment.
- **A sub-device restarts on its own.** Each composite also serves a `rigcontrol`
  target (`_RigControl`, over `_CompositeRig`, both in `device_server.py`) with
  `status`/`start`/`stop`/`restart(device_name)`; `get_rig_control(name)` returns its
  client, and the Devices tab's sub-device rows use it. The process and the other
  sub-devices keep serving. The coordinator holds references to every backend, so
  it is dropped and rebuilt around any change (only when every sub-device is up),
  and a client connected to the old one must reconnect.
- If every sub-device fails, the server still binds its port with no targets.
  That is deliberate: the rig reads as "running, all parts failed" in the Devices
  tab, which is diagnosable, where an exited process looks like one never
  started.
- Config errors still fail loudly and up front — a nested composite, two
  sub-device names folding to the same target, a coordinator name colliding with
  a sub-device. Degradation covers hardware that is *unavailable*, not configs
  that are *wrong*.

### Writing a coordinator

Subclass `Coordinator` (`pytweezer/coordinators/base.py`), reach backends through
`self.require_role(<role>)`, and point the composite's `"coordinator"` key at the
class. `build_spec` constructs it as `cls(roles, conf)`; override `shutdown()`
to release the backends before they are torn down.
`pytweezer/coordinators/rearrangement.py` is the real worked example (the
`Rb Rearrangement Rig`); `transport.py` is an unconfigured variant of it.

The rig's queued path is `prepare_rearrangement(...)` (once, builds the target
array on the one `make_array` made) and `rearrange(d0_um=, profile=, header=)`
(one shot). `rearrange` is one synchronous call that blocks the rig server
while it waits for the HamCam's occupancy frame (up to the `frame_timeout_s`,
20 s by default, that `prepare_rearrangement` sets as the camera's `timeout`),
so an experiment runs it in
parallel with MOTMaster's Go (`run_parallel`, with `RearrangeTweezers.arm_delay`
between the two) rather than before it.

Three constraints, all consequences of how sipyco serves targets:

- **Every public method is a synchronous RPC method that stalls the entire
  server while it runs.** So a batch method must take a bounded iteration count
  rather than free-run — e.g. a `run_n(n)` that issues the whole batch in one
  RPC, with each iteration a direct Python call. It cannot be aborted mid-batch.
- **Return values cross a PYON boundary.** Plain types, dicts, lists, numpy
  arrays; never exception objects or backend handles. Returning the frame from a
  feedback step would reintroduce exactly the serialization cost the coordinator
  exists to avoid.
- **Never define `__call__` on a target.** sipyco treats a callable target as a
  per-connection factory and would invoke it instead of serving it.

Sub-devices stay reachable on their own, so other clients can change the
hardware behind the coordinator's back. State a coordinator reports about its
backends should be checked against them, not just remembered: the
rearrangement rig owns the tweezer array (`make_array`), and `get_array()`
returns it only while `slm.last_mask` is still the mask it wrote, `None` once
anything else has written the SLM. Gate GPU-only methods on a CUDA device
count, not on `import cupy` succeeding — cupy imports on machines with no
usable driver.

If a loop ever needs to free-run while the server stays answerable, give the
coordinator `start`/`stop`/`status` backed by a `threading.Thread` — a thread,
not an asyncio task, because the drivers are synchronous and a task's cadence
would be hostage to any other client's blocking RPC.

## Who may use a device

Nothing on a device server knows about experiments: it serves every client,
and the event loop serialises their calls. Ownership is bookkeeping in the
Experiment Manager (detail in `add-experiment`):

- A queued task reserves the devices its experiment declares
  (`pytweezer/experiment/reservations.py`: a composite reserves itself and every
  sub-device, a sub-device only itself, a linked MotMaster also its
  `linked_partner`); tasks whose reservations overlap never run together.
- Its worker calls `restrict_devices(task.devices)`, after which `get_device` /
  `get_device_async` in that process raise `UndeclaredDeviceError` for any
  other device. Notebooks, GUI panels and other programs are not restricted.
- A device that must stay off hardware a task is using reads the manager's
  state itself: the NI Cards driver holds MOTMaster's pattern boards while that
  MotMaster runs or a running task reserves it (`ManagerQueue` and
  `RemotingMotMasterStatus` in `pytweezer/drivers/ni_cards_sources.py`).
- In simulation (the manager's `simulate`, or `run_local(..., simulate=True)`),
  `Experiment.device()` returns backends built in-process by `build_spec`
  (`pytweezer/experiment/simulation.py`), one per owning entry so sub-devices
  share a rig. Calls are plain Python, so a return value that would not survive
  PYON still passes there.

## What blocks what

sipyco is single-threaded asyncio and runs plain `def` methods **inline**. A
synchronous RPC method therefore stalls *every* target on that server for its
duration — the DAC cannot answer while the camera is mid-exposure. This is the
first thing to check when a device "feels unresponsive".

`allow_parallel` does not change this. It drops the lock sipyco holds across each
call, but with synchronous methods there is no suspension point between acquiring
and releasing it, so nothing can ever contend for it — what serializes calls is
the event loop, not the lock. It becomes meaningful only once a target method is
`async def`, which then also wants `await asyncio.to_thread(...)` around its
blocking work and its own per-backend lock. Do not reach for it as a speed knob.

## Running calls against several devices at once

`get_device` returns a **blocking** client, so a script does not issue the next
call until the previous one's full remote execution has finished. That is a
correctness problem, not just a slow one: `MotMasterInterface.start_motmaster_experiment()`
→ `Go()` is a synchronous .NET remoting call that blocks for the **whole
sequence**, so two MotMasters driven one after another run back-to-back, never
together — and a MotMaster armed in trigger mode blocks until something else
fires its trigger.

`run_parallel` (`pytweezer/helper_scripts/parallel.py`) is the ergonomic answer: it runs
zero-argument callables one thread each and returns results in order. The threads
overlap for real because each call is a blocking socket round trip whose `recv`
releases the GIL. `after(delay, call)` staggers a start to win an
arm-before-trigger race:

```python
from pytweezer.helper_scripts.parallel import run_parallel, after

frame, _, _ = run_parallel(
    lambda: cam.acquire_n_frames(1),  # blocks reading the frame
    mm1.start_motmaster_experiment,  # armed; waits for trigger
    after(0.05, mm2.start_motmaster_experiment),  # fires 50 ms later
)
```

Each parallel call must use a **different** client — a single sipyco `Client` is
not thread-safe. `timeout=` raises `TimeoutError`, abandoning the still-running
daemon threads. One failing call re-raises its own exception; several raise an
`ExceptionGroup`.

`get_device_async` is the lower-level alternative, returning an `AsyncioClient`
whose methods are coroutines. It is **not** fire-and-forget — it still awaits
every reply; the win is `asyncio.gather` issuing calls to *different* servers
before awaiting either. Fine from scripts and notebooks; the PyQt6 GUI has no
`qasync`, so GUI code would need a worker thread.

## Verifying a change

`.claude/skills/add-device-driver/scripts/check_device.py` (`--all` for every
entry) builds each device from config in simulation, composites included, and
reports config keys that reach no constructor parameter, required parameters
config doesn't supply, and Camera hook signatures that drift — useful for any
framework change, not just new drivers. The `tests/` suite covers
resolution, composites and spec building; extend it there rather than writing
one-off scripts.
