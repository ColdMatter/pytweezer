# pytweezer GUI architecture

Role-based tabbed GUIs for the tweezer control system. There are **two** windows,
one per machine role, sharing a single tabbed shell and differing only in which
panels they show.

## Roles at a glance

The control system spans two kinds of machine:

- **Server PC** — runs the long-lived server processes (message/data/image/property
  hubs, loggers, Analysis/Property managers, MotMaster). May also host devices.
- **Client PC(s)** — do **not** run servers. They launch applets (viewers/plots) and
  the RPC drivers for devices physically attached to that client.

Which machine a device lives on is set per entry by its `host` in
[`pytweezer/configuration/config.py`](../pytweezer/configuration/config.py).

| GUI | Command | `.bat` | Tabs |
|-----|---------|--------|------|
| Server GUI | `pytweezer-server` | `start_servers.bat` | **Servers** · **Devices** · **Device Status** · **Streams** · **Applets** |
| Client GUI | `pytweezer-client` | `start_client.bat` | **Server Status** · **Device Status** · **Devices** · **Applets** · **Streams** |

The main differences between the two: the server GUI has a full start/stop
**Servers** tab; the client GUI replaces it with a read-only **Server Status** tab
(the servers run on a remote machine, so the client can only observe them). Both
share a read-only **Device Status** tab showing the cross-PC device-server feed
(see below).

Two status views, deliberately different mechanisms:

- **Server Status** (client only) — *client-side* TCP probe of the hub/logger
  servers in `CONFIG["Servers"]` (`ServerStatusPanel`). Answers "can this client
  reach the servers?".
- **Device Status** (both) — the *server-published* aggregate of every device
  server in `CONFIG["Devices"]` across all PCs (`DeviceStatusPanel`). The server
  PC does the probing; clients just display the feed.

## Entry points and launch flow

```
pyproject.toml [tool.poetry.scripts]
  pytweezer-server = "bin.gui:server_main"
  pytweezer-client = "bin.gui:client_main"

bin/gui.py
  server_main() -> _run(build_server_gui)
  client_main() -> _run(build_client_gui)
    _run(build): QApplication -> build() -> win.show() -> SIGTERM handler -> exec_()
```

> After changing `pyproject.toml`, run `poetry install` once so the
> `pytweezer-server` / `pytweezer-client` console scripts are (re)generated.

## Component map

Every tab is an existing widget reused as-is; the shell and the status panel are
the only new pieces.

```
bin/gui.py
├── TabbedGUI(QMainWindow)         # shared shell: QTabWidget + geometry + teardown
├── ServerStatusPanel(QWidget)     # client-side reachability probe of CONFIG["Servers"]
│   └── _ProbeWorker(QThread)      #   background TCP reachability poller
├── DeviceStatusPanel(QWidget)     # displays the server-published device-status feed
│   └── DeviceStatusClient(SUB)    #   -> pytweezer/servers/device_status.py
├── build_server_gui() / build_client_gui()
└── server_main() / client_main() / _run()

Reused panels
├── ServerManager / DeviceManager   -> bin/process_manager.py      (ProcessTile grids)
├── AppletLauncher                  -> pytweezer/GUI/applet_launcher.py
└── make_stream_monitor()           -> pytweezer/GUI/streammonitor.py (nested QTabWidget)

Device-status service
├── DeviceStatusServer  -> pytweezer/servers/device_status.py  (server PC: poll + PUB)
├── DeviceStatusClient  -> pytweezer/servers/device_status.py  (SUB QObject)
└── is_reachable()      -> pytweezer/servers/reachability.py   (shared TCP probe)
```

### `TabbedGUI` (the shell)
A plain `QMainWindow` whose central widget is a `QTabWidget`. It:

- **Wraps** grid/tree panels (`ServerManager`, `DeviceManager`, `AppletLauncher`,
  `ServerStatusPanel`, `DeviceStatusPanel`) in a `QScrollArea` so they stay usable
  when embedded. An already-tabbed panel (the stream monitor `QTabWidget`) is added
  **unwrapped**.
- **Persists geometry** via `QSettings("pytweezer", <window name>)`.
- **Tears down children on close** — see below.

It is intentionally **not** a `BMainWindow`: `BMainWindow` creates a `Properties`
object (a Propertyhub/Propertylogger network connection). The shell doesn't need
properties, and avoiding it keeps startup fast and independent of the hubs being
up. (`ProcessManager` avoids `Properties` for the same reason, via
`create_props=False`.)

### Teardown (important)
Qt delivers `closeEvent` only to **top-level** widgets. The embedded panels each
manage child subprocesses/threads that must be cleaned up:

- `ServerManager` / `DeviceManager` → terminate their `ProcessTile` server/device
  subprocesses.
- `AppletLauncher` → stop applet subprocesses.
- `ServerStatusPanel` → stop its probe thread.
- `DeviceStatusPanel` → stop its `DeviceStatusClient` (SUB timer + socket).

So `TabbedGUI.closeEvent` explicitly calls `.close()` on every embedded panel
(dispatching each panel's own `closeEvent` cleanup) before chaining to
`super().closeEvent`. **If you add a new tab that owns processes/threads, its
`closeEvent` must do the cleanup — the shell will call it, but only if it exists.**

After the Qt loop ends, `_run()` calls `logging.shutdown()` + `os._exit(0)`. This
is deliberate: `Properties` (used by `AppletLauncher`) starts **non-daemon** ZMQ
`event_monitor` threads that loop forever, so a normal return would hang interpreter
shutdown. By this point the panels' `closeEvent`s have already terminated the child
subprocesses, so a hard exit is safe. (Standalone panel processes don't need this —
they're killed via `SIGTERM`.)

### Auto-start behavior
Embedding preserves the panels' existing behavior: `ProcessTile` auto-starts any
config entry marked `"active": True`. Therefore:

- Opening the **server GUI** brings up all active `CONFIG["Servers"]`.
- Opening either GUI's **Devices** tab brings up the active `CONFIG["Devices"]`
  whose `host` matches the local machine (`DeviceManager.check_host`,
  case-insensitive). On a machine that hosts no devices, the tab is empty.

## Server Status panel (client, view-only)

`ProcessTile` can only report status for processes it launched locally; on a client
the servers are remote, so status comes from a **network reachability probe**.

- **What it probes:** for each `CONFIG["Servers"]` entry, the port from
  `_status_port(params) = params.get("port") or params.get("pub_port")`.
  - Hubs expose `pub_port`/`sub_port` (they `bind()` real TCP sockets in
    `xsub_xpub.py`), managers/loggers expose `port`.
  - Pure subscribers (`Datalogger`, `Imagelogger`) bind **no** port → shown as grey
    **n/a**.
- **How:** `_ProbeWorker(QThread)` loops every ~3 s calling
  `socket.create_connection((host, port), timeout=0.3)` off the GUI thread (so a
  down host never freezes the UI) and emits `{name: True|False|None}` via a Qt
  signal. The panel updates a colored dot per server: green=reachable,
  red=unreachable, grey=n/a. There are **no** start/stop controls.

Reachability answers "can this client reach the server endpoint?" — which is what a
client operator actually cares about. It is not full process-health.

## Device Status service (server-published, cross-PC)

Unlike Server Status (each client probes for itself), device-server status is
gathered **once, centrally, by the server PC** and published to everyone. This is
because each PC's `DeviceManager` is host-filtered and only sees its own devices —
no single machine otherwise has the global picture.

```
Server PC:  DeviceStatusServer            pytweezer/servers/device_status.py
              every ~2 s: for each CONFIG["Devices"] entry,
                is_reachable(host, port) -> "up" | "down"   (active=False -> "disabled")
              PUB-bind tcp://SERVER_HOST:pub_port, send_json full snapshot
                        │
Client/Server GUI:  DeviceStatusClient (SUB) --status_received--> DeviceStatusPanel
```

- **Pull model:** only the server PC probes; it can reach every device server
  because `CONFIG["Devices"]` lists each one's `host:port`. No client-side plumbing.
- **Transport:** a dedicated ZMQ `PUB` socket (not model_sync, not a shared hub),
  registered as the `Device Status` entry in `CONFIG["Servers"]` and launched like
  any other server from the server GUI's **Servers** tab.
- **Periodic full snapshot** every poll cycle → new subscribers sync within one
  interval (no REP/snapshot endpoint, no PUB slow-joiner problem).
- **Snapshot shape:** `{"type":"device_status","timestamp":ts,
  "devices":{name:{"state","host","port","last_seen"}}}` (JSON).
- **Display:** `DeviceStatusPanel` rebuilds its rows from each snapshot, so it shows
  **all** devices across all PCs (green up / red down / grey disabled), independent
  of local host-filtering. Present on both GUIs (client = primary requirement,
  server = operator convenience).

The reachability probe itself lives in `pytweezer/servers/reachability.py`
(`is_reachable`), shared by both this server and `ServerStatusPanel`.

## Stream monitor

`pytweezer/GUI/streammonitor.py` exposes `make_stream_monitor(name) -> QTabWidget`
building the nested Image/Data/Command/Message/Logs sub-tabs. Both GUIs embed the
result as a single top-level **Streams** tab. `main()` (standalone launch) now just
calls the same factory.

## File index

| File | Role |
|------|------|
| `bin/gui.py` | Shell, `ServerStatusPanel`, `DeviceStatusPanel`, builders, `server_main`/`client_main`. |
| `bin/process_manager.py` | `ProcessManager`/`ServerManager`/`DeviceManager` panels. CLI `main()` retired. |
| `bin/process_tile_base.py` | `ProcessTile` — one start/stop tile + local status polling. |
| `pytweezer/GUI/applet_launcher.py` | `AppletLauncher` panel (applet subprocess manager). See [`applets.md`](applets.md). |
| `pytweezer/GUI/applet.py` | `Applet` base class + `run_applet()` entry helper for viewer/plot applets. |
| `pytweezer/GUI/streammonitor.py` | `make_stream_monitor()` factory + `StreamMonitor`/`LogMonitor`. |
| `pytweezer/servers/device_status.py` | `DeviceStatusServer` (poll + PUB) + `DeviceStatusClient` (SUB). |
| `pytweezer/servers/reachability.py` | `is_reachable()` — shared TCP liveness probe. |
| `pytweezer/configuration/config.py` | `CONFIG["Servers"]` (incl. `Device Status`) / `CONFIG["Devices"]`, `HOSTS`. |
| `start_servers.bat` / `start_client.bat` | Launchers → `pytweezer-server` / `pytweezer-client`. |

**Retired:** `bin/dashboard.py`, `start_device_manager.bat`, `start_dashboard.bat`,
and the `pytweezer-run <server|device|dashboard>` entry point.

## How to extend

- **Add a tab:** append `(label, widget)` to the list in `build_server_gui()` /
  `build_client_gui()`. If the widget owns processes/threads, give it a `closeEvent`
  that cleans them up (the shell calls `.close()` on it).
- **Add a server to the Server-Status view:** nothing to do — `ServerStatusPanel`
  reads `CONFIG["Servers"]` at construction. Give it a `port` or `pub_port` to make
  it probeable.
- **Add a device to the Device-Status view:** nothing to do — `DeviceStatusServer`
  reads `CONFIG["Devices"]` and publishes it; `DeviceStatusPanel` renders whatever
  arrives. Just give the device a reachable `host:port`.
- **Change which machine runs a device:** edit that device's `host` in
  `config.py`; the host-filtered `DeviceManager` shows/controls it on the matching
  machine, while `DeviceStatusPanel` shows it everywhere.
