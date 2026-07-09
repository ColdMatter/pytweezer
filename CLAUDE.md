# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Control software for an atom-tweezer experiment (Rb and CaF systems). PyQt5 GUIs,
ZMQ messaging, and sipyco RPC servers coordinate device drivers (cameras,
MotMaster experiment sequencers) across multiple lab PCs.

## Commands

This project uses Poetry. There is no configured linter or formatter. There is a
real pytest suite under `tests/` (`conftest.py` sets `QT_QPA_PLATFORM=offscreen`
before any PyQt5 import, so Qt widget tests run headless) — run it after
touching code it covers, and add cases there instead of writing one-off
verification scripts when the change fits its scope.

```bash
poetry install                      # (re)generate console scripts after editing pyproject.toml
poetry run pytest tests/ -q         # run the test suite
poetry run python <script>.py       # run any script inside the venv
poetry env info                     # show the venv path (env name looks like pytweezer-<hash>-py3.13)
```

For behavior the suite doesn't cover (hardware drivers, multi-process
server/hub interaction, anything needing a real device), fall back to running
it manually through Poetry — but check `tests/` first; it may already exercise
what you're about to hand-verify.

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

**Don't run multi-line Python through `python -c`.** In this environment a `-c`
string with embedded or leading newlines frequently returns empty output — this
is a real quoting/flushing failure, not a cosmetic "shell quirk," and rerunning
the same `-c` command with slightly different quoting will not fix it. Do not
diagnose it as an environment glitch and move on. Instead, for anything beyond a
single trivial expression, write a `.py` file to the scratchpad dir (see the
Scratchpad section) and run it with `poetry run python <file>.py`. A script file
runs identically every time, shows full tracebacks, and can be re-read/edited.
Keep `-c` only for one true one-liner with no newlines. When a script's output
matters, end it with `sys.stdout.flush()` (or `print(..., flush=True)`).

Pyright type checking is explicitly off (`pyrightconfig.json`); don't expect or
enforce type-check cleanliness.

## Writing docs and comments

Docstrings and comments are for the next user or developer reading the code cold.
Write what helps *them*: what a module/function does, how to use it, the
non-obvious constraints, and the gotchas that would otherwise cost someone an
afternoon. Do **not** write documentation that narrates the conversation or task
that produced the code — no "why no X", no justifying a choice against an
alternative that was only ever discussed in chat, no "see Y for the one place we
weighed Z", no changelog-style notes about what was changed or considered. If a
design decision genuinely needs recording, state the constraint as a present-tense
fact ("sums use `float64` to avoid integer overflow on raw camera counts"), not as
a defense of the decision. When in doubt, ask whether the sentence would still
make sense to someone who has never seen this task — if not, cut it.

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

### Applet framework (viewers/plots)

Applets are small **local** processes that subscribe to data/image streams and
display them live (image viewer, live plot). They mirror the device framework's
"one base class + one launcher" shape on the display side:

- `pytweezer/GUI/applet.py` — `Applet(QWidget)` base class every applet
  subclasses. It owns the `Properties(name)` connection, window title (= `name`),
  geometry persistence, a `poll()` timer, and the shared subscription/configure
  dialogs; subclasses override `init_gui`/`poll`/`update_subscriptions` and set
  `stream_category` (`"Image"`/`"Data"`). `run_applet(cls, default_name)` is the
  standard `main`, parsing the `name` argv the launcher passes.
- `pytweezer/GUI/viewers/` — the applet scripts (`image_monitor.py`
  `ImageDisplay`, `live_plot.py` `LivePlot`). `viewers/archive/` holds
  retired/experimental viewers — leave them alone.
- `pytweezer/GUI/applet_launcher.py` — `AppletLauncher` panel (the **Applets**
  tab). Launches each applet as `python <script> <name>`; the applet list is
  persisted in Properties under the `"Applets"` key, seeded from
  `DEFAULT_APPLETS`. `name` is the label, Properties namespace, and title at once.

Full writeup: `docs/applets.md`.

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

### InfluxDB metric logging (separate from the above)

Durable time-series values go into a self-hosted **InfluxDB 2.x**, via a framework
that deliberately mirrors the device framework. Don't confuse this with the pub/sub
"loggers" above (`datalogger`/`imagelogger`/`propertylogger`), which monitor ZMQ
streams and are **not** forwarded to InfluxDB — Influx logging is entirely opt-in.

- `pytweezer/servers/influx_client.py` — `InfluxWriter` (the single write path;
  writes never raise) and the notebook convenience `log(measurement, **fields)`.
  Connection config is the `INFLUXDB` block in `configuration/config.py`
  (env-var overridable).
- `pytweezer/loggers/` — the generic `Logger` base (`base.py`) and concrete
  subclasses that each **own their own device** (e.g. `ni_adc_logger.py`, an NI
  ADC via `nidaqmx`), analogous to `pytweezer/drivers/`. Loggers do not poll
  existing device RPC servers; to log off an existing driver (e.g. a camera), give
  that driver its own `InfluxWriter` instead.
- `pytweezer/servers/logger_server.py` — the launcher with `LOGGER_REGISTRY`
  (`"logger"` key → subclass factory) and dual-mode `main()`, exactly analogous
  to `device_server.py`. `CONFIG["Loggers"]` is its config category, surfaced as
  the **Loggers** tab (`LoggerManager`, server PC only).

Full writeup, including the self-hosted InfluxDB one-command setup: `docs/influx_logging.md`.
