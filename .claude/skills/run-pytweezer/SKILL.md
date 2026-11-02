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
- Off the lab network, also set `PYTWEEZER_DATA_DIR` to an empty scratch
  directory (see Gotchas — the Results tab otherwise stalls the run for minutes).

## Run (agent path) — screenshot driver

Build the **client** GUI and dump screenshots:

```bash
uv run python .claude/skills/run-pytweezer/driver.py client
```

The client spawns no servers or loggers, but its Devices tab starts every
`"active"` device whose `host` is this PC, as the real client does. On a dev
machine (host not in config) that is nothing; on a lab client PC it starts that
PC's devices, so don't run it there while that PC's GUI is up.

Output (13 PNGs) lands in `.claude/skills/run-pytweezer/shots/`:

- `client_full.png` — whole window, active tab raised
- `client_tab_<Label>.png` — one per tab: Servers, Devices, Loggers, Streams,
  Applets, Analysis, Experiments, Results, Saved_parameters, Optimisers,
  NI_Cards, Properties

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

Useless in a non-interactive shell (nothing to look at, blocks forever). Ctrl-C
to quit. Use the driver instead when you need a screenshot.

### Restarting a GUI on another lab PC over SSH

Never start the GUI straight from an SSH command: it lands in session 0 with
no window (and so do the MOTMasters it launches, with their dialogs), and dies
when SSH disconnects. Each PC instead has on-demand scheduled tasks,
registered once by a person with `register_launch_tasks.ps1`, that run the
`.bat` launchers on the logged-on desktop:

```bash
ssh 10.59.3.1 "taskkill /pid <gui pid> /t /f"                    # stop (whole tree)
ssh 10.59.3.1 "cd /d C:\Users\tweez\ColdMatter\pytweezer && git checkout <branch> && git pull --ff-only"
ssh 10.59.3.1 "schtasks /run /tn pytweezer-server"               # start on the desktop
```

`pytweezer-client` is the client-PC equivalent. Do not create scheduled tasks
yourself; if `schtasks /run` says the task does not exist, ask the user to run
the registration script on that PC. SSH hosts: `10.59.3.1` (PH-BEAST, user
`tweez`), `rb-mm-pc` (10.59.3.2); see `~/.ssh/config`.

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
- **Off-network, the Results tab stalls the run.** Off the Experiment Manager's
  PC, Results reads measurement files from it over RPC on the GUI thread, and
  each refresh waits up to 30 s for an unreachable host, so the driver runs
  for many minutes. `PYTWEEZER_DATA_DIR=<empty dir>`
  makes it read local files instead.
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
  Properties server / Analysis Manager are wrapped in `_safe_panel`; when those are
  down the tab becomes an "unavailable — see logs" placeholder and the rest of
  the window builds fine.

## Troubleshooting

- Driver hangs after `client_full.png` — the Results tab is waiting on an
  unreachable Experiment Manager; set `PYTWEEZER_DATA_DIR` (see Gotchas).
- Blank/textless screenshots — you're on the offscreen backend on Windows.
  Run on a machine with a desktop session, or accept textless layout shots.
- Empty printed output from the driver — it was piped through `head`; redirect to
  a file instead (the `os._exit` gotcha above).
