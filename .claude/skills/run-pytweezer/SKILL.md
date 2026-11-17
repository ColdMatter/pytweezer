---
name: run-pytweezer
description: Launch, run, and screenshot the pytweezer PyQt6 control GUIs (pytweezer-server / pytweezer-client). Use when asked to run, start, launch, open, or screenshot the pytweezer app / GUI, or to confirm a GUI change renders in the real window rather than only in tests.
---

# Run pytweezer

pytweezer's user-facing app is two PyQt6 windows — `pytweezer-server` (full
process control, server PC) and `pytweezer-client` (client PCs: the same tabs,
with Servers and Loggers view-only) — built from `bin/gui.py`. They spawn a
window and block forever, so the way to drive one non-interactively and *see*
it is the screenshot driver at
[.claude/skills/run-pytweezer/driver.py](driver.py): it builds a GUI in-process,
spins the Qt event loop briefly so it fully paints (and, on a networked lab PC,
fetches live server/device status), then grabs a PNG of the window and of each
tab.

All paths below are relative to the repo root. Everything runs through uv
(env is `.venv` in the repo root).

## Prerequisites

- uv with the project env installed. If imports fail, run `uv sync`.
- On Windows with an active desktop session (a lab PC), no display flags are
  needed — the driver uses the native Qt backend and the window flashes briefly.
- On a truly headless box (SSH, no session, or Linux without a display), prefix
  the run with `QT_QPA_PLATFORM=offscreen` (see Gotchas — on Windows text won't
  render there).
- Off the lab network, the Results tab lists nothing (the Experiment Manager
  is unreachable); set `PYTWEEZER_DATA_DIR` to a directory of measurement files
  to screenshot it with content.

## Run (agent path) — screenshot driver

Build the **client** GUI and dump screenshots:

```bash
uv run python .claude/skills/run-pytweezer/driver.py client
```

The client spawns no servers or loggers, but its Devices tab starts every
`"run_at_startup"` device whose `host` is this PC, as the real client does. On a dev
machine (host not in config) that is nothing; on a lab client PC it starts that
PC's devices, so don't run it there while that PC's GUI is up.

Output (13 PNGs) lands in `.claude/skills/run-pytweezer/shots/`:

- `client_full.png` — whole window, active tab raised
- `client_tab_<Label>.png` — one per tab: Servers, Devices, Loggers, Streams,
  Applets, Analysis, Experiments, Results, Saved_parameters, Optimisers,
  NI_Cards (only while CONFIG has an `NICards` device), Properties

Pass a second arg to redirect the output dir:

```bash
uv run python .claude/skills/run-pytweezer/driver.py client /tmp/shots
```

On a lab PC the shots show **live** state: e.g. the Servers tab lists
Imagehub / Commandhub / Datahub / … as "Running" on `10.59.3.1`, and Devices
shows the real MotMaster / camera status feed. Off-network, rows show
"Unknown"/"Stopped" but the window still builds and paints.

### The `server` GUI — simulates anywhere but the server PC

`driver.py server` builds the **server** GUI, whose control panels
`subprocess.Popen` every active hub, device server and the Experiment Manager
*on construction*. Like `pytweezer-server` itself, it marks the session
(`PYTWEEZER_ROLE=server`) before importing anything, so on any machine other
than `SERVER_PC` (`PH-BEAST`) the whole session runs in simulation on localhost,
whatever `SIMULATING` says. On `PH-BEAST` it starts the real servers, so don't
run it there while the experiment is live. The driver closes the window before
exiting so the children it started are stopped.

When driving a server GUI yourself, never pump events with `app.quit()`: in
Qt 6 it closes the window, and closing the server GUI stops every server.

## Run (human path)

The real entry points open a window and block:

```bash
uv run pytweezer-client     # or start_client.bat
uv run pytweezer-server     # or start_servers.bat; simulates off PH-BEAST
```

On the lab's Windows PCs, Smart App Control blocks uv's console-script `.exe`
launchers, so the `.bat` files run `uv run python -m bin.launch` (server) and
`uv run python -m bin.launch client` instead; use those forms there.

Only one GUI runs per PC: a launch while one is already up (server or client,
any user) shows a message, brings the running window to the front and exits.
`driver.py` builds its window directly and is not refused, so it can run
beside a live GUI.

Useless in a non-interactive shell (nothing to look at, blocks forever). Ctrl-C
to quit. Use the driver instead when you need a screenshot.

### Restarting a GUI on another lab PC over SSH

Use `pytweezer-machines` (the PCs are the `MACHINES` table in
`pytweezer/configuration/config.py`; no names means all of them):

```bash
uv run pytweezer-machines status                    # reach, branch, commit, behind/ahead, dirty, whether the GUI runs
uv run pytweezer-machines stop PH-BEAST             # close the GUI as if its window were closed, then kill stale processes
uv run pytweezer-machines update PH-BEAST --branch <branch>   # fetch, fast-forward, uv sync
uv run pytweezer-machines start PH-BEAST            # start on the desktop
```

`update` skips a clone that is on another branch unless `--branch` is given.
Exit code is 0 only if every machine succeeded. A new PC needs
`uv run pytweezer-machines setup MACHINE` once, from a terminal: it installs
this PC's SSH key (asking for the password) and registers the launch tasks.

Never start the GUI straight from an SSH command: it lands in session 0 with
no window (and so do the MOTMasters it launches, with their dialogs), and dies
when SSH disconnects. `start` instead runs the on-demand scheduled task
(`pytweezer-server` or `pytweezer-client`) that `register_launch_tasks.ps1`
registers, which runs the `.bat` launcher on the logged-on desktop. Never
create scheduled tasks or run `setup` yourself: ask the user to run
`uv run pytweezer-machines setup <name>`, which prompts for the PC's password
and writes sshd's key files there.

## Test

```bash
uv run pytest tests/test_gui.py tests/test_experiment_gui.py tests/test_stream_monitor.py -q
```

The GUI tests run offscreen through the `qapp` fixture in `tests/conftest.py`.

## Gotchas

- **Offscreen renders no text on Windows.** Under `QT_QPA_PLATFORM=offscreen`
  the layout, colors, and status dots draw, but the Windows offscreen font
  database renders no normal glyphs — every label comes out blank. There, only
  the native backend (default, needs a session) produces readable screenshots,
  so prefer it. On Linux, offscreen renders text normally.
- **`os._exit(0)` swallows piped stdout.** `bin/gui.py` and the driver hard-exit
  after the Qt loop (to skip lingering non-daemon ZMQ threads). The driver
  `flush()`es first, but redirect to a file rather than piping through
  `head`/`tee` if you need the printed screenshot paths.
- **`grab()` too early = blank text.** A few `processEvents()` calls don't paint
  text; the driver runs the real event loop for ~1.2s (`_paint`) before grabbing.
  That delay is also what lets the reachability probe / device-status feed
  populate, so shorter waits give emptier-looking (but not wrong) panels.
- **Host not in config is expected.** A dev machine whose hostname isn't in
  `config.MACHINES` logs `Host <name> not found in config. Defaulting to localhost`
  and shows no device Start/Stop toggles (it owns no devices) — harmless.
- **Unreachable hubs don't abort the window.** Panels that connect to the
  Properties server / Analysis Manager are wrapped in `_safe_panel`; when those are
  down the tab becomes an "unavailable — see logs" placeholder and the rest of
  the window builds fine.

## Troubleshooting

- Blank/textless screenshots — you're on the offscreen backend on Windows.
  Run on a machine with a desktop session, or accept textless layout shots.
- Empty printed output from the driver — it was piped through `head`; redirect to
  a file instead (the `os._exit` gotcha above).
