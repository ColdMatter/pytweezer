---
name: pytweezer-gui-internals
description: How the pytweezer PyQt5 shell is built — TabbedGUI's dock-based tabs, the ManagedRow/ControlPanel/DevicesPanel process tiles, the two different status mechanisms and why they differ, and the deliberate hard-exit teardown that the legacy BWidget/Properties stack makes necessary. Use when adding or editing a tab or panel in bin/gui.py or bin/managed_panel.py, touching pytweezer/GUI/pytweezerQt.py or device_status.py, debugging a GUI that hangs on exit or loses its window geometry, working out why a row shows the wrong Running/Stopped state, or changing how processes are started and stopped from the GUI. For building a live viewer window, use the add-applet skill instead.
---

# GUI internals

`pytweezer-server` and `pytweezer-client` are the same window built by
`build_gui(server: bool)` in `bin/gui.py`, differing only in whether the
Servers/Loggers tabs are controllable. Tabs: Servers, Devices, Loggers, Streams,
Applets, Analysis, Properties.

Each panel is constructed through `_safe_panel`, so one panel that fails to
build leaves a placeholder instead of killing the whole window.

## `TabbedGUI` is a plain `QMainWindow` — deliberately

`pytweezer/GUI/pytweezerQt.py` holds the legacy base classes `BWidget`/`BFrame`/
`BMainWindow`, each of which constructs a `Properties(name)` connection to the
Propertyhub. `Properties` spawns **non-daemon** `event_monitor` threads that loop
forever, so any process using it hangs at interpreter shutdown.

`TabbedGUI` therefore does **not** inherit `BMainWindow` — which also keeps
startup fast and independent of the hubs being up. Two consequences that look
like bugs and are not:

- `_run()` calls `logging.shutdown(); os._exit(0)` immediately after the Qt event
  loop ends, to bypass those lingering threads. By then every panel's
  `closeEvent` has already terminated its children, so the hard exit is safe. Do
  not "fix" this into a normal return.
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
`python3 <script> <name>` (see the `pytweezer-architecture` skill for that launch
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

`DevicesPanel` renders all of it: every device on every PC, a controllable row
only where `check_host` matches this machine, and an indented view-only row per
sub-device (they share the rig's process and start and stop with it).

## Shared vs per-machine state

Properties is shared: every `set()` is broadcast to all clients and persisted
centrally, with no local-only write. So anything that is per-operator must go in
local `QSettings("pytweezer", <name>)` instead — window geometry, and the
Applet Launcher's set of *running* applets (the applet catalogue itself is
shared, under the Properties `"Applets"` key). Putting running-state in
Properties would start one PC's applets on every other PC.

## Running it headless

PyQt5 widgets cannot construct without a display — set
`QT_QPA_PLATFORM=offscreen` for any non-interactive run. To actually see a change
rendered, use the `run-pytweezer` skill's screenshot driver rather than launching
the blocking GUI.
