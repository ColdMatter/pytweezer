# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Control software for an atom-tweezer experiment (Rb and CaF systems). PyQt5 GUIs,
ZMQ messaging, and sipyco RPC servers coordinate device drivers (cameras,
MotMaster experiment sequencers) across multiple lab PCs.

## Commands

This project uses Poetry. There is no configured linter, formatter, or automated
test suite (no pytest config; `tests/` holds notebooks and ad hoc scripts, not a
runnable suite). Verify changes by actually running them through Poetry.

```bash
poetry install                      # (re)generate console scripts after editing pyproject.toml
poetry run python <script>.py       # run any script inside the venv
poetry env info                     # show the venv path (env name looks like pytweezer-<hash>-py3.13)
```

Entry points (`[tool.poetry.scripts]` in `pyproject.toml`):

```bash
poetry run pytweezer-server         # full-control GUI, run on the server PC
poetry run pytweezer-client         # view-only GUI, run on client PCs
poetry run pytweezer-device <name>  # start one device's RPC server standalone
```

`start_servers.bat` / `start_client.bat` just wrap the first two.

**Headless/offscreen testing:** PyQt5 windows can't construct without a display.
Set `QT_QPA_PLATFORM=offscreen` before running any script that imports Qt widgets
in a non-interactive shell.

**`os._exit(0)` swallows piped stdout.** `bin/gui.py`'s `_run()` hard-exits after
the Qt loop ends (see "Non-daemon threads" below). If you're capturing output
from a script that goes through that path, `sys.stdout.flush()` before exit or
the output will be lost when piped.

Pyright type checking is explicitly off (`pyrightconfig.json`); don't expect or
enforce type-check cleanliness.

## Architecture

### Two-PC role split, one config file

The system spans a **server PC** (runs the long-lived hub/logger/manager
processes) and one or more **client PCs** (launch applets + local device
drivers). `pytweezer/configuration/config.py` is the single source of truth:

- `HOSTS`: hostname → IP map. `SERVER_HOST` resolves based on `SIMULATING`/`LOCAL`
  flags at the top of the file (both `True` today → everything binds to
  localhost for dev/sim work).
- `CONFIG["Servers"]`: hubs (`Imagehub`/`Commandhub`/`Datahub`/`Propertyhub`/
  `Messagehub`), loggers, `Analysis Manager`, `Device Status`.
- `CONFIG["Devices"]`: one entry per physical device (MotMaster sequencers,
  cameras), each with its own `host` — **this is what determines which PC a
  device's server actually runs on**, independent of where the GUI is launched.
- `CONFIG["GUI"]`: standalone GUI tool entries (StreamMonitor, Applet Launcher, etc).

Don't confuse this with the **root** `configuration/` directory — that holds
`properties.json` (Properties startup state) and other JSON data files, not the
Python `CONFIG` dict. `ConfigReader.getConfiguration()` (in
`pytweezer/servers/configreader.py`) always returns the dict from
`pytweezer/configuration/config.py`.

### Process launching: ProcessTile → `python <script> <name>`

`bin/process_tile_base.py`'s `ProcessTile` is the base unit for starting/stopping
a process from the GUI: it runs `subprocess.Popen(['python3', script, name])`,
where `name` is the CONFIG key. Every server/device script's `main()` accepts
that `name` as a positional/optional CLI arg and, if given, looks up its own
config under `CONFIG[category][name]` instead of reading `--host`/`--port`/etc.
flags directly. When adding or modifying a server script's `main()`, preserve
this dual mode (config-driven when launched by a tile, flag-driven for manual
CLI use).

`bin/process_manager.py` builds two flavors of this: `ServerManager` (no host
filtering — server PC controls everything) and `DeviceManager` (filters devices
to `host == this machine`, case-insensitively, via `check_host`).

### Device control framework (server + client)

Rather than each driver module hand-rolling its own RPC server/client
boilerplate, there's a generic pair:

- `pytweezer/servers/device_server.py` — `run_device_server(name)` reads
  `CONFIG["Devices"][name]["driver"]`, looks it up in `DRIVER_REGISTRY` (maps a
  driver key like `"motmaster"`/`"imagemx2"`/`"blackfly"` to a factory that lazily
  imports the real backend and builds a `DeviceServerSpec`), then runs
  `sipyco.pc_rpc.simple_server_loop`. Every device's `CONFIG` entry points
  `"script"` at this same file — `"driver"` is what actually differentiates
  behavior. Backend imports are lazy per-factory so an unavailable hardware lib
  (e.g. `rotpy` for the Blackfly) never breaks importing the launcher itself.
  `resolve_device(name)` does whitespace-/case-insensitive config-key matching
  so CLI callers don't need to quote/space-match names exactly.
- `pytweezer/servers/device_client.py` — `get_device(name)` looks up the same
  config entry and returns a transparent `sipyco.pc_rpc.Client` using
  `AutoTarget` (safe because every device server exposes exactly one RPC
  target). Prefer this over hand-built `Client(...)` calls in new experiment code.

Full writeup: `docs/device_framework.md`.

### GUI shell: `TabbedGUI` vs the legacy `BWidget`/`Properties` stack

`pytweezer/GUI/pytweezerQt.py` defines the legacy base classes `BWidget`/
`BFrame`/`BMainWindow`, which each construct a `Properties(name)` connection
(`pytweezer/servers/properties.py`) — a ZMQ pub/sub link to the Propertyhub.
`Properties` spawns **non-daemon** `event_monitor` threads that loop forever;
a process using it will hang on interpreter shutdown unless killed harder.

`bin/gui.py`'s `TabbedGUI` (the shell behind `pytweezer-server`/`pytweezer-client`)
is deliberately a **plain `QMainWindow`**, not a `BMainWindow`, to avoid that
hang and avoid depending on the hub being reachable just to open a window. Its
`_run()` wrapper calls `logging.shutdown(); os._exit(0)` right after the Qt loop
ends specifically to bypass lingering non-daemon threads from embedded panels
(e.g. `AppletLauncher`) — this is intentional, not a workaround to "fix" by
switching to a normal return. Geometry persistence (`QSettings`) needs an
explicit `.sync()` before that hard exit, or the write never reaches disk.

Full writeup, including the panel teardown chain and the two status-panel
mechanisms (`ServerStatusPanel` client-probe vs `DeviceStatusPanel` server-published
feed): `docs/gui_architecture.md`.

### Messaging fabric

`pytweezer/servers/xsub_xpub.py` implements XSUB/XPUB hub processes
(Imagehub/Commandhub/Datahub/Propertyhub/Messagehub) that fan out pub/sub
traffic between publishers and subscribers without them needing to know each
other's endpoints directly. `pytweezer/servers/clients.py` (`DataClient`,
`ImageClient`, `CommandClient`) are the publish-side helpers used throughout the
drivers/servers. `pytweezer/servers/model_sync.py` exists (client/server pair for
syncing schedule/prep models) but its `CONFIG["Servers"]` entry is currently
commented out — treat it as present-but-inactive, not dead code to remove.

### Logging

`pytweezer/logging_utils.py`'s `get_logger(name)` is the standard logger factory
used everywhere (not bare `logging.getLogger`) — it adds structured JSONL file
output under `logs/` (or `$PYTWEEZER_LOG_DIR`) in addition to console output.
