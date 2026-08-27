---
name: pytweezer-device-framework
description: How pytweezer's generic device server and client work — device_server.py building any driver from its CONFIG entry, get_device() addressing, simulation, composite devices and coordinators, and the rules for what blocks what over RPC. Use when calling devices from experiment or notebook code, running calls against several devices concurrently, debugging a device that will not start or resolve, changing device_server.py/device_client.py/simulated_device.py themselves, grouping devices into one process or writing a coordinator, or reasoning about RPC latency and blocking between targets. For writing a brand-new hardware driver, use the add-device-driver skill instead.
---

# The device framework

Every device runs the same launcher. `pytweezer/servers/device_server.py` reads
`CONFIG["Devices"][name]["class"]` (a `"module.path:ClassName"` string), imports
it, constructs it from the matching config keys, and serves it with
`sipyco.pc_rpc.simple_server_loop`. There is no per-driver server script and no
registry — `"class"` is the only thing that differentiates one device from
another.

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

**Append new entries, never insert them.** Ports come from `get_next_port()` in
declaration order, so inserting an entry renumbers every device below it.

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
- **A dead sub-device does not take the server down.** One that fails to build is
  logged, recorded in `DeviceServerSpec.failed`, and skipped; everything else
  still serves. The **coordinator is the exception** — it drives every backend,
  so it is built only when all sub-devices built, and stands down otherwise
  rather than failing halfway through an experiment.
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
class. `build_spec` constructs it as `cls(roles, conf)`.
`pytweezer/coordinators/rearrangement.py` is the real worked example.

Three constraints, all consequences of how sipyco serves targets:

- **Every public method is a synchronous RPC method that stalls the entire
  server while it runs.** So a batch method must take a bounded iteration count
  rather than free-run — `run_n(n)` issues the whole batch in one RPC, with each
  iteration a direct Python call. It cannot be aborted mid-batch.
- **Return values cross a PYON boundary.** Plain types, dicts, lists, numpy
  arrays; never exception objects or backend handles. Returning the frame from a
  feedback step would reintroduce exactly the serialization cost the coordinator
  exists to avoid.
- **Never define `__call__` on a target.** sipyco treats a callable target as a
  per-connection factory and would invoke it instead of serving it.

If a loop ever needs to free-run while the server stays answerable, give the
coordinator `start`/`stop`/`status` backed by a `threading.Thread` — a thread,
not an asyncio task, because the drivers are synchronous and a task's cadence
would be hostage to any other client's blocking RPC.

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

`run_parallel` (`pytweezer/parallel.py`) is the ergonomic answer: it runs
zero-argument callables one thread each and returns results in order. The threads
overlap for real because each call is a blocking socket round trip whose `recv`
releases the GIL. `after(delay, call)` staggers a start to win an
arm-before-trigger race:

```python
from pytweezer.parallel import run_parallel, after

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
before awaiting either. Fine from scripts and notebooks; the PyQt5 GUI has no
`qasync`, so GUI code would need a worker thread.

## Verifying a change

`.claude/skills/add-device-driver/scripts/check_device.py` builds a device from
config without hardware and reports the constructor kwargs actually matched —
useful for any framework change, not just new drivers. The `tests/` suite covers
resolution, composites and spec building; extend it there rather than writing
one-off scripts.
