---
name: pytweezer-architecture
description: How the pytweezer system fits together — the one server PC and its arbitrarily many client PCs, CONFIG as the single source of truth for what runs where, the launch protocol every process shares, and the ZMQ messaging fabric. Use when working out which PC a process runs on or binds to, editing pytweezer/configuration/config.py or paths.py, adding or changing an entry under CONFIG["Servers"]/["Devices"]/["Loggers"]/["GUI"], writing or changing a server script's main(), tracing a pub/sub stream from publisher to subscriber, or getting oriented in an unfamiliar corner of the codebase.
---

# How pytweezer fits together

Control software for an atom-tweezer experiment. Four ideas carry most of it:
one server and many clients, one config file, one launch protocol, one pub/sub
fabric.

## One server, many clients

There is a **single server PC** (`PH-BEAST`) running the long-lived shared
processes — the ZMQ hubs, the stream loggers, Analysis Manager, Device Status —
and **arbitrarily many client PCs**, each running whatever devices are plugged
into it plus its own applets and GUI. Clients are not a fixed pair or a fixed
list: a new lab PC is just another `HOSTS` entry with devices pointed at it.

What runs where is decided **per config entry**, not per machine. A device's
`"host"` is what determines which PC its server runs on — independent of where
the GUI that shows it was launched. The server GUI can start and stop only what
`check_host` says is local; everything else it merely displays.

## `CONFIG` is the single source of truth

`pytweezer/configuration/config.py`:

- `HOSTS` — hostname → IP. `SERVER_HOST` resolves from the `SIMULATING`/`LOCAL`
  flags at the top of the file (both `False` today → the real `PH-BEAST`; set
  either `True` to bind everything to localhost for dev/sim).
- `CONFIG["Servers"]` — hubs (`Imagehub`/`Commandhub`/`Datahub`/`Propertyhub`/
  `Messagehub`), the stream loggers, `Analysis Manager`, `Device Status`.
- `CONFIG["Devices"]` — one entry per physical device, each with its own `host`.
- `CONFIG["Loggers"]` — InfluxDB loggers.
- `CONFIG["GUI"]` — standalone GUI tools (StreamMonitor, Applet Launcher, …).

Every category names its `"script"` explicitly **except Devices**: they all run
`device_server.py`, differentiated by `"class"`, so the launcher path lives once
in `DEVICE_SERVER_SCRIPT` (`pytweezer/configuration/paths.py`) and spawn sites
(`DevicesPanel`, `process_cleanup`) reference that constant rather than
`params["script"]`.

**Append new entries, never insert them.** Ports come from `get_next_port()` in
declaration order, so inserting one renumbers every entry below it.

`get_config()` — defined next to `CONFIG` itself — is the accessor everything
uses. Tests monkeypatch it **per importing module** to inject a fake config, so
call it at use time rather than binding `CONFIG` at import time.

Filesystem layout lives separately in `pytweezer/configuration/paths.py`
(`tweezerpath`, `configpath`, `icon_path`, `propertyfilename`,
`DEVICE_SERVER_SCRIPT`, `load_properties()`); `tweezerpath`/`icon_path` are also
re-exported from `pytweezer.servers`.

**Two different things called "configuration".** The **root** `configuration/`
directory holds JSON data files (`properties/properties.json`, the Properties
startup state) — not the Python `CONFIG` dict.

## Launch protocol: `python <script> <name>`

`bin/managed_panel.py`'s `ManagedRow` starts every managed process the same way:
`subprocess.Popen(['python3', script, name])`, where `name` is the CONFIG key.

Every server/device/logger script's `main()` therefore takes that `name` as a
positional/optional CLI arg and, when given, looks itself up under
`CONFIG[category][name]` instead of reading `--host`/`--port` flags. **Preserve
this dual mode** when writing or editing a `main()`: config-driven when launched
from a GUI tile, flag-driven for manual CLI use.

## Messaging fabric

`pytweezer/servers/xsub_xpub.py` runs the XSUB/XPUB hub processes that fan out
pub/sub traffic, so publishers and subscribers only ever address a hub, never
each other. Because the hub fans a stream out to every subscriber, any number of
applets and analyses on any number of PCs can watch the same stream at once —
and unlike a device, whose PC is pinned by its config `host`, an applet runs
wherever an operator launched it.

- `pytweezer/servers/clients.py` — `GenericClient` and its subclasses
  `DataClient`, `ImageClient`, `CommandClient`: the publish/subscribe helpers
  used throughout drivers, servers, analyses and applets.
- `pytweezer/servers/properties.py` — `Properties(name)`, a ZMQ pub/sub link to
  the Propertyhub used as shared key-value state (applet catalogues, stream
  subscriptions, per-viewer settings). It is **shared, always**: every `set()` is
  broadcast to all clients and persisted centrally, with no local-only write, so
  anything per-operator belongs in local `QSettings` instead. It also spawns
  **non-daemon** threads that loop forever; see the `pytweezer-gui-internals`
  skill for the shutdown consequences.
- `pytweezer/servers/model_sync.py` exists but its `CONFIG["Servers"]` entry is
  commented out — present-but-inactive, not dead code to delete.

## Logging

`pytweezer/logging_utils.py`'s `get_logger(name)` is the standard factory
everywhere (not bare `logging.getLogger`) — it adds structured JSONL output
under `logs/` (or `$PYTWEEZER_LOG_DIR`) alongside console output.

Do not confuse the pub/sub stream loggers (`datalogger`/`imagelogger`/
`propertylogger`, which archive ZMQ streams) with **InfluxDB** metric logging,
which is entirely opt-in and separate. Streams are never auto-forwarded to
Influx.

## Where to go next

| Task | Go to |
| --- | --- |
| Device server/client internals, composites, coordinators | `pytweezer-device-framework` skill |
| GUI shell, panels, tabs, teardown | `pytweezer-gui-internals` skill |
| Add a driver for new hardware | `add-device-driver` skill |
| Add a live viewer window | `add-applet` skill |
| Add a streaming analysis | `add-analysis-script` skill |
| Record a value in InfluxDB | `add-logger` skill |
| Run/screenshot the GUI | `run-pytweezer` skill |

Each of those skills is self-contained — the conventions and constraints that
are not readable off the code live in them, not in a separate document.
