---
name: pytweezer-gui-internals
description: How the pytweezer PyQt6 shell is built — TabbedGUI's dock-based tabs, the ManagedRow/ControlPanel/DevicesPanel process tiles, the two different status mechanisms and why they differ, and the deliberate hard-exit teardown. Use when adding or editing a tab or panel in bin/gui.py or bin/managed_panel.py, touching pytweezer/GUI/pytweezerQt.py or device_status.py, debugging a GUI that hangs on exit or loses its window geometry, working out why a row shows the wrong Running/Stopped state, or changing how processes are started and stopped from the GUI. For building a live viewer window, use the add-applet skill instead.
---

# GUI internals

`pytweezer-server` and `pytweezer-client` are the same window built by
`build_gui(server: bool)` in `bin/gui.py`, differing only in whether the
Servers/Loggers tabs are controllable. Tabs: Servers, Devices, Loggers, Streams,
Applets, Analysis, Experiments, Results, Properties. Experiments and Results
(`pytweezer/GUI/experiments/`) are covered by the `add-experiment` skill; they
are built before the list so Results can hand "resubmit" to Experiments.

Each panel is constructed through `_safe_panel`, so one panel that fails to
build leaves a placeholder instead of killing the whole window.

## `TabbedGUI` is a plain `QMainWindow` — deliberately

`pytweezer/GUI/pytweezerQt.py` holds the legacy base classes `BWidget`/`BFrame`/
`BMainWindow`, each of which constructs a `Properties(name)` connection to the
Properties server; with the server down, that waits a few seconds before
falling back to the saved file.

`TabbedGUI` therefore does **not** inherit `BMainWindow`, which keeps startup
fast and independent of the servers being up. Two consequences that look like
bugs and are not:

- `_run()` calls `logging.shutdown(); os._exit(0)` immediately after the Qt event
  loop ends, so no lingering thread or ZMQ socket in any panel can hang
  shutdown. By then every panel's `closeEvent` has already terminated its
  children, so the hard exit is safe. Do not "fix" this into a normal return.
- Because the exit is hard, `closeEvent` must call `settings.sync()` explicitly
  after saving geometry and dock state, or `QSettings`' deferred write never
  reaches disk.

Panels that only want `BWidget`'s layout helpers pass `create_props=False`
(`ControlPanel` and `DevicesPanel` both do) and so avoid the Properties
connection entirely.

**Teardown is the shell's job, cleanup is the panel's.** Qt delivers `closeEvent`
only to top-level widgets, so `TabbedGUI.closeEvent` explicitly calls `.close()`
on every embedded panel before chaining up. If you add a tab that owns
subprocesses or threads, **give it a `closeEvent` that cleans them up** — the
shell will call it, but only if it exists.

## Tabs are docks

Panels are `QDockWidget`s tabified into one area, not a `QTabWidget`, which is
what lets a user drag a panel out into its own floating window. Two rules follow:

- Every dock needs a **stable `objectName`** (`f"dock:{label}"`), or
  `saveState()`/`restoreState()` cannot restore the layout.
- Panels are wrapped in a `QScrollArea` by `_wrap()`, except an already-tabbed
  widget (the stream monitor), which is added as-is.

## Process tiles: `ManagedRow`

`bin/managed_panel.py`'s `ManagedRow` is one row = one process. It spawns
`<sys.executable> <script> <name>` (see the `pytweezer-architecture` skill for that launch
protocol) and owns the Start/Stop toggle and status LED. Rows for entries marked
`"active": True` auto-start when the panel opens.

## Three status sources, on purpose

A row's state can come from three different places, and picking the wrong one is
the usual cause of a row reporting nonsense:

| Panel | Rows | Status from |
| --- | --- | --- |
| `ControlPanel(name, category, controllable=True)` | Servers, Loggers on the server PC | `enable_self_polling()` — authoritative, this PC owns the subprocess |
| `ControlPanel(..., controllable=False)` | the client's view-only Servers tab | client-side TCP probe (`_ProbeWorker`) |
| `DevicesPanel` | every device, on any PC | the server-published `DeviceStatusClient` feed |

They differ because of what each machine can actually see. A client cannot poll
processes that run on the server PC, so it probes reachability instead — which is
what a client operator actually cares about ("can I reach it?"), not full process
health. The probe runs in a `QThread` so a down host never freezes the UI, and
uses `_status_port(params) = params.get("port") or params.get("pub_port")`; pure
subscribers like `Datalogger`/`Imagelogger` bind no port at all and show grey
**n/a**.

## Device Status: published once, centrally

Each PC's `DevicesPanel` is host-filtered for *control*, so no single machine
otherwise has the global picture. `pytweezer/servers/device_status.py` fixes that
from the server PC: `DeviceStatusServer` probes every `CONFIG["Devices"]` entry
every ~2 s and PUBs a snapshot on a dedicated socket (not through a hub);
`DeviceStatusClient` is the Qt-side subscriber.

- It publishes a **full snapshot every cycle**, so a new subscriber syncs within
  one interval — no REP/snapshot endpoint and no PUB slow-joiner problem.
- The snapshot is flat (`{name: {"state", "host", "port", "last_seen"}}`), with a
  composite adding `"children"` and each sub-device carrying `"parent"` — flat
  works precisely because device names are unique across the category.
- `active=False` reads as `"disabled"` rather than down.

**Composites need a second probe.** A composite serves several sub-devices from
one port and comes up with only the ones that opened, so a TCP probe of that port
cannot tell you which. `server_targets(host, port)` does a sipyco handshake with
`target_name=None` — listing the served targets without ever invoking an RPC
method. A sub-device is `up` when its target is present, `failed` when it is not,
and the rig reads `degraded` when only its coordinator target is missing.

That handshake needs the server's event loop, which a long synchronous call
occupies for its whole duration (see the `pytweezer-device-framework` skill), so
`DeviceStatusServer` **caches the last target list per composite** and reuses it
when a handshake times out — otherwise a busy rig would flap all its sub-devices
to `failed` mid-acquisition. A server that has never completed a handshake reads
`unknown`.

`DevicesPanel` renders all of it: every device on every PC, and an indented
view-only row per sub-device (they share the rig's process and start and stop
with it).

## Starting and stopping a device on another PC

A row spawns its process only where `check_host` matches this machine. Every
other device row still has a toggle, which publishes a start/stop request on the
Commandhub (`pytweezer/servers/device_control.py`). Every `DevicesPanel` hears
every request and acts only on its own devices, by calling the same
`startProcess`/`stopProcess` its toggle would, so the host PC always owns the
subprocess. Consequences:

- **The host's GUI must be running**, or nobody acts. There is no
  acknowledgement: the requester shows "Starting"/"Stopping" until the Device
  Status feed agrees, or the grace period runs out and the feed wins.
- A request matching the current state (start of a running device) is ignored,
  so a repeated click never restarts anything.
- `stopProcess` (toggle and remote stop) differs from `terminateProcess` (panel
  close): if the row is running but did not spawn the process (one left behind
  by a GUI that died), it kills it by exact command-line match
  (`process_cleanup.kill_device_process`). Closing a panel never does that, and
  never sends remote stops.

## Shared vs per-machine state

Properties is shared: every `set()` is broadcast to all clients and persisted
centrally, with no local-only write. So anything that is per-operator must go in
local `QSettings("pytweezer", <name>)` instead — window geometry, and the
Applet Launcher's set of *running* applets (the applet catalogue itself is
shared, under the Properties `"Applets"` key). Putting running-state in
Properties would start one PC's applets on every other PC.

## Running it headless

PyQt6 widgets cannot construct without a display — set
`QT_QPA_PLATFORM=offscreen` for any non-interactive run. To actually see a change
rendered, use the `run-pytweezer` skill's screenshot driver rather than launching
the blocking GUI.
