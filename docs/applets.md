# pytweezer applet framework

How viewer/plot **applets** are defined, launched, and managed. Complements
[`gui_architecture.md`](gui_architecture.md), which covers the GUI shell that
embeds the **Applets** tab, and [`device_framework.md`](device_framework.md),
whose "one base + one launcher" shape this mirrors on the display side.

## What an applet is

An **applet** is a small, standalone process that runs **locally on a lab PC**,
subscribes to one or more data/image streams on the messaging fabric, and
displays them live (an image viewer, a live plot, …). Applets are pure
*consumers*: they never own hardware and never publish — they `subscribe()` to
streams that device servers and analysis processes publish through the
Image/Data hubs (see `pytweezer/servers/clients.py`).

Because a stream is fanned out to every subscriber by the XSUB/XPUB hubs, any
number of applets on any number of PCs can watch the same stream at once. Where
an applet runs is decided by **which machine's Applet Launcher started it** —
unlike devices (whose PC is pinned by `CONFIG["Devices"][name]["host"]`), applets
are deliberately local and per-operator.

## Component map

```
pytweezer/GUI/applet.py                 (base class + entry-point helper)
├── Applet(QWidget)      props + title + icon + geometry + poll timer + dialogs
│   ├── stream_category  "Image" | "Data"  (which stream catalog to browse)
│   ├── poll_interval    ms between poll() calls (0 = no timer)
│   ├── default_size     (w, h) fallback window size before geometry is saved
│   ├── init_gui()             hook: build widgets
│   ├── poll()                 hook: pull new stream data, refresh display
│   ├── update_subscriptions() hook: (re)apply stream subscriptions
│   ├── stream_color(stream, index)   palette colour, per-stream overridable
│   ├── open_subscription_editor(streamkey=None)  shared "subscriptions" dialog
│   └── open_config_editor()                      shared "configure" dialog
└── run_applet(applet_cls, default_name)   theme + parse <name> argv + Qt loop

pytweezer/GUI/viewers/                   (the applet scripts themselves)
├── image_monitor.py       ImageDisplay(Applet)      image  (stream_category="Image")
├── image_plot_monitor.py  ImagePlotDisplay(Applet)  image + linked projections
├── live_plot.py           LivePlot(Applet)          data   (stream_category="Data")
└── scalar_history.py      ScalarHistory(Applet)     data, one header field vs shot
      (viewers/archive/ holds retired/experimental viewers — ignore them)

pytweezer/GUI/applet_launcher.py         (the manager panel)
└── AppletLauncher(BWidget)   add/start/stop/restart applet subprocesses;
                              applet list persisted in Properties under "Applets"
```

## The base class: `Applet`

Every applet subclasses `Applet` (a plain `QWidget`). The base handles the
machinery that used to be copy-pasted into each viewer:

- a `Properties(name)` connection, exposed as **both** `self.props` and
  `self._props` (the latter is what `PropertyAttribute` descriptors read);
- the **window title**, set to `name` — so the title always matches the label
  in the Applet Launcher — and the shared viewer window icon;
- **geometry persistence** via `QSettings("pytweezer", name)` (restored in
  `__init__`, saved in `closeEvent`), falling back to `default_size`;
- a repeating **poll timer** that calls `self.poll()` every `poll_interval` ms;
- the shared **"subscriptions"** and **"configure"** context-menu dialogs;
- the **theme**: `run_applet` calls `theme.apply_theme`, which sets the dark
  stylesheet *and* pyqtgraph's background/foreground. An applet is its own
  process with its own `QApplication`, so it inherits nothing from the main GUI
  — without this it would open in the platform's default light style. Applets
  should therefore never set a stylesheet or a literal colour of their own;
  `stream_color(stream, index)` hands out the shared curve palette, honouring a
  per-stream `<stream>/color` property when the user has set one.

A subclass describes only *what* it shows, by overriding a few hooks:

| Hook | When it runs | What to do |
|---|---|---|
| `init_gui(self)` | once, after props + title are ready | build widgets, create stream clients, wire menu actions |
| `poll(self)` | every `poll_interval` ms | pull new stream data (`has_new_data()` / `recv()`) and update the display |
| `update_subscriptions(self)` | after either dialog closes | unsubscribe + re-subscribe the stream clients from the current property list |

Class attributes tune behavior:

- `stream_category` — `"Image"` or `"Data"`. Chooses which catalog the
  subscription editor lists, and the default Properties key the subscription
  list lives under (`<category>.lower() + "streams"`, i.e. `imagestreams` /
  `datastreams`).
- `poll_interval` — ms between `poll()` calls; set `0` for an applet driven
  purely by Qt signals.

### Minimal example

```python
from pytweezer.servers import DataClient
from pytweezer.GUI.applet import Applet, run_applet

class MyPlot(Applet):
    stream_category = "Data"
    poll_interval = 10

    def init_gui(self):
        self.stream = DataClient(self.name)
        ...  # build widgets; wire menu -> self.open_subscription_editor / self.open_config_editor
        self.update_subscriptions()

    def update_subscriptions(self):
        self.stream.unsubscribe()
        for ch in self.props.get("datastreams", []):
            self.stream.subscribe(ch)

    def poll(self):
        if self.stream.has_new_data():
            msg, info, array = self.stream.recv()
            ...  # draw

def main(name):                 # config-/launcher-driven entry
    run_applet(MyPlot, default_name=name)

if __name__ == "__main__":      # standalone CLI entry
    run_applet(MyPlot, default_name="MyPlot")
```

`run_applet` is the single entry point: it parses the `name` positional arg
(supplied by the launcher — see below), constructs the applet, and runs the Qt
event loop. Keep both a `main(name)` (called by tooling that imports the module)
and the `__main__` guard.

## Launching & management: the Applet Launcher

`pytweezer/GUI/applet_launcher.py`'s `AppletLauncher` is the panel surfaced as
the **Applets** tab in both the server and client GUIs. It is a subprocess
manager, closely analogous to `ProcessTile`/`DeviceManager` on the device side:

- **Storage — shared list, local running-state.** The list of applets lives in
  **Properties** under the `"Applets"` key:
  `{ name: {"script": ..., "description": ...}, ... }`, so every client sees the
  same catalogue. `_ensure_defaults()` seeds it from `DEFAULT_APPLETS` the first
  time. Which applets are *running* is **per-machine** and deliberately kept out
  of Properties: it lives in local `QSettings("pytweezer", <launcher name>)`
  under `"active_applets"` (a list of names), the same local store used for
  window geometry. Properties has no local-only write — every `set()` is
  broadcast to all clients and persisted centrally by the propertylogger — so
  storing running-state there would make one PC's applets start on every other
  PC. A legacy `"active"` key in the shared entry is ignored and stripped.
- **Launch model.** Starting an applet runs
  `subprocess.Popen([sys.executable, script_path, name], cwd=tweezerpath)` —
  i.e. `python <script> <name>`. **`name` is the applet's label, its Properties
  namespace, and its window title, all at once.** Two applets with different
  names therefore get independent property subtrees, subscriptions, and saved
  geometry, even if they run the same script.
- **Script resolution.** Relative `script` paths resolve against `tweezerpath`;
  a missing script is reported and the row marked `missing`.
- **Controls.** The name column's checkbox = active/start-stop; `add` (with an
  optional template from `DEFAULT_APPLETS`), `del`, and `restart` buttons; a
  1 s timer reconciles each row's status against `process.poll()`.
- **Auto-start & teardown.** On open, `_start_active_applets()` starts exactly
  the applets this machine had running last session (names still present in the
  shared list; stale ones are pruned). On close, `closeEvent` terminates the
  child processes via `_terminate_process`, which **leaves the recorded set
  intact** — the distinction matters: stopping an applet from its Stop button
  (`_stop_applet`) or closing its own window means "don't start it next time",
  whereas quitting the GUI must preserve what was running so it comes back.
  This is why the GUI shell's teardown must call the panel's `close()` — see
  `gui_architecture.md`.

### Adding an applet

- **New instance of an existing viewer** (e.g. a second image monitor): just add
  it in the launcher UI, or add an entry to `DEFAULT_APPLETS` so it's offered as
  a template. No code.
- **New applet type:** add a script under `pytweezer/GUI/viewers/` whose class
  subclasses `Applet`, implement `init_gui`/`poll`/`update_subscriptions`, end
  with `run_applet(...)`, then point a launcher entry at it. Optionally add it to
  `DEFAULT_APPLETS` as a template.

The `add-applet` skill (`.claude/skills/add-applet/`) walks through the second
case, including the transport gotchas that decide whether an applet survives a
real stream. Its `scripts/preview_applet.py` builds any applet against fake
Properties and fake streams and saves a screenshot, so a viewer can be developed
and looked at without hubs, hardware, or a window appearing on screen.

## Not to be confused with…

- **The `GUI` config category** (`CONFIG["GUI"]`, e.g. the Applet Launcher
  itself, StreamMonitor) — those are standalone GUI *tools* started by
  `ProcessTile`, not applets. The launcher is the tool; the applets are what it
  launches.
- **Loggers** (`pytweezer/loggers/`) — those also consume streams/hardware but
  write to InfluxDB rather than displaying; see `influx_logging.md`.
