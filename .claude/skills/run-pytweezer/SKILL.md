---
name: run-pytweezer
description: Launch, run, and screenshot the pytweezer PyQt5 control GUIs (pytweezer-server / pytweezer-client). Use when asked to run, start, launch, open, or screenshot the pytweezer app / GUI, or to confirm a GUI change renders in the real window rather than only in tests.
---

# Run pytweezer

pytweezer's user-facing app is two PyQt5 windows — `pytweezer-server` (full
process control, server PC) and `pytweezer-client` (view-only, client PCs) —
built from `bin/gui.py`. They spawn a window and block forever, so the way to
drive one non-interactively and *see* it is the screenshot driver at
[.claude/skills/run-pytweezer/driver.py](driver.py): it builds a GUI in-process,
spins the Qt event loop briefly so it fully paints (and, on a networked lab PC,
fetches live server/device status), then grabs a PNG of the window and of each
tab.

All paths below are relative to the repo root. Everything runs through Poetry
(env already set up: Python 3.13, `pytweezer-<hash>-py3.13`).

## Prerequisites

- Poetry with the project env installed. If imports fail, run `poetry install`.
- On Windows with an active desktop session (a lab PC), no display flags are
  needed — the driver uses the native Qt backend and the window flashes briefly.
- On a truly headless box (SSH, no session), prefix the run with
  `QT_QPA_PLATFORM=offscreen` (see Gotchas — text won't render there).

## Run (agent path) — screenshot driver

Build the **client** GUI (safe: view-only, spawns no subprocesses) and dump
screenshots:

```bash
poetry run python .claude/skills/run-pytweezer/driver.py client
```

Output (7 PNGs) lands in `.claude/skills/run-pytweezer/shots/`:

- `client_full.png` — whole window, active tab raised
- `client_tab_<Label>.png` — one per tab: Server_Status, Devices, Applets,
  Analysis, Properties, Streams

Pass a second arg to redirect the output dir:

```bash
poetry run python .claude/skills/run-pytweezer/driver.py client /tmp/shots
```

On a lab PC the shots show **live** state: e.g. the Server Status tab lists
Imagehub / Commandhub / Datahub / … as "Running" on `10.59.3.1`, and Devices
shows the real MotMaster / camera status feed. Off-network, rows show
"Unknown"/"Stopped" but the window still builds and paints.

### The `server` GUI — do not run off the server PC

`driver.py server` builds the **server** GUI, whose control panels
`subprocess.Popen` every active hub *on construction*. On a non-server machine
those children try to bind the configured server address (`10.59.3.1`) and fail,
and can collide with the live experiment's real hubs. Only run `server` mode on
the actual server PC (`PH-BEAST`), or in simulation mode (set `SIMULATING = True`
in `pytweezer/configuration/config.py`, which rebinds everything to localhost).
For a screenshot on any other machine, use `client`.

## Run (human path)

The real entry points open a window and block:

```bash
poetry run pytweezer-client     # or start_client.bat
poetry run pytweezer-server     # server PC only; or start_servers.bat
```

Useless in a non-interactive shell (nothing to look at, blocks forever). Ctrl-C
to quit. Use the driver instead when you need a screenshot.

## Test

```bash
poetry run pytest tests/ -q --ignore=tests/test_gui.py
```

→ `81 passed`. **`tests/test_gui.py` is currently broken** (pre-existing, not
caused by this skill): it imports `from bin.process_manager import ...`, but that
module was split into `bin/managed_panel.py` and `bin/process_manager.py` no
longer exists, so a plain `poetry run pytest tests/ -q` aborts at collection with
`ModuleNotFoundError: No module named 'bin.process_manager'`. Ignore that file
(as above) to run the rest.

## Gotchas

- **Offscreen renders no text.** Under `QT_QPA_PLATFORM=offscreen` the layout,
  colors, and status dots draw, but the Windows offscreen font database renders
  no normal glyphs — every label comes out blank. Only the native backend
  (default, needs a session) produces readable screenshots. So: prefer the
  default; reach for `offscreen` only when there is genuinely no display, and
  expect textless shots.
- **`os._exit(0)` swallows piped stdout.** `bin/gui.py` and the driver hard-exit
  after the Qt loop (to skip lingering non-daemon ZMQ threads). The driver
  `flush()`es first, but redirect to a file rather than piping through
  `head`/`tee` if you need the printed screenshot paths.
- **`grab()` too early = blank text.** A few `processEvents()` calls don't paint
  text; the driver runs the real event loop for ~1.2s (`_paint`) before grabbing.
  That delay is also what lets the reachability probe / device-status feed
  populate, so shorter waits give emptier-looking (but not wrong) panels.
- **Host not in config is expected.** A dev machine whose hostname isn't in
  `config.HOSTS` logs `Host <name> not found in config. Defaulting to localhost`
  and shows no device Start/Stop toggles (it owns no devices) — harmless.
- **Unreachable hubs don't abort the window.** Panels that connect to the
  Propertyhub / Analysis Manager are wrapped in `_safe_panel`; when those are
  down the tab becomes an "unavailable — see logs" placeholder and the rest of
  the window builds fine.

## Troubleshooting

- `ModuleNotFoundError: No module named 'bin.process_manager'` — you ran the full
  `pytest tests/`. Add `--ignore=tests/test_gui.py` (see Test).
- Blank/textless screenshots — you're on the offscreen backend with no session.
  Run on a machine with a desktop session, or accept textless layout shots.
- Empty printed output from the driver — it was piped through `head`; redirect to
  a file instead (the `os._exit` gotcha above).
