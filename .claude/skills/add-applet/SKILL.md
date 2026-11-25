---
name: add-applet
description: Create a new pytweezer applet — a live viewer window under pytweezer/GUI/viewers/ that subscribes to image or data streams and displays them. Use this whenever the user wants a new applet, viewer, monitor, live display, plot window, readout, or scope for a stream ("show me the atom number live", "a window that plots the trap frequency", "display the camera with the fit overlaid", "a histogram of pixel values"), or wants to change how an existing viewer looks or behaves. Applies even when they only describe the display without saying "applet" or "viewer".
---

# Adding an applet to pytweezer

An applet is one local process that subscribes to streams and draws them.
`pytweezer/GUI/applet.py` owns the machinery (Properties connection, title,
geometry, theme, poll timer, subscription/configure dialogs), so an applet is
**one `Applet` subclass** in a new file under `pytweezer/GUI/viewers/`, ending
in `run_applet(<Class>, default_name="<Name>")`.

There is no CONFIG entry. The Applet Launcher runs `python <script> <name>`;
`name` is the window title, the Properties namespace and the launcher label, so
two entries running the same script under different names are independent
instances. The catalogue lives in Properties under `"Applets"`
(`{name: {"script", "description"}}`, relative scripts resolve against
`tweezerpath`); which applets are running is per-machine, in `QSettings`.

## First check it is an applet

Applets **display**; they never compute a derived quantity and never publish. A
new number nothing streams yet (centre of mass, fit, atom count, running
average) belongs in an analysis (`add-analysis-script` skill); the applet plots
what it publishes. An applet calling `client.send()` is doing an analysis's job.

The one exception: an applet may **drive the one device it is about**, as the
Camera applet does. Make every device call through a single worker-thread link
(`CameraLink` in `viewers/camera_applet.py`) so the GUI never waits on RPC, and
go read-only while an Experiment Manager task holds the device (follow
`ExperimentFeed`, as the Camera applet does).

## The shape of the file

```python
class AtomCount(Applet):
    stream_category = "Data"  # "Data" or "Image": which catalogue the subscription editor lists
    poll_interval = 50
    default_size = (520, 300)

    _threshold = PropertyAttribute("threshold", 0.5)

    def init_gui(self):  # once: build widgets, make stream clients
        ...
        self.update_subscriptions()

    def update_subscriptions(self):  # after either dialog closes
        ...

    def poll(self):  # every poll_interval ms
        ...


def main(name):
    run_applet(AtomCount, default_name=name)


if __name__ == "__main__":
    run_applet(AtomCount, default_name="Atom Count")
```

Keep both entry points: `main(name)` for tooling that imports the module, the
`__main__` guard for the launcher.

`poll_interval` is a repaint budget, not a data rate: messages queue between
polls. 10 ms suits viewers that redraw one cheap item (image viewers, Live
Plot); use 50–100 ms for anything that rebuilds a plot.

## Reading a stream safely

Each of these has bitten an existing viewer.

**`recv()` returns a variable-length tuple**: `(channel, header)`,
`(channel, header, array)`, or `None` on timeout. Analyses publish scalars
header-only, so an unconditional three-way unpack raises `ValueError` on a
scalar stream. Check the length you need:

```python
message = self.stream.recv()
if message is None or len(message) < 3:  # drawing an array
    continue
channel, head, array = message

message = self.stream.recv()
if message is None or len(message) < 2:  # reading a header field
    continue
channel, head = message[0], message[1]
```

**Drain with `while`, and don't wait.** One `recv()` per poll falls ever further
behind. Use a zero timeout: the default 1 ms lasts a whole timer tick (~16 ms)
on Windows and stalls the event loop every poll.

```python
while self.stream.has_new_data(timeout_ms=0):
    ...
```

**`update_subscriptions` unsubscribes first, then rebuilds the view.** It runs
every time a dialog closes; otherwise subscriptions accumulate and removed
streams leave ghost curves. Unsubscribe, clear, re-subscribe, recreate.

**Skip `"--None--"`**, the placeholder of an unconfigured stream list; don't
create a curve or row for it.

**`poll` must not raise.** It runs from a QTimer and an exception can take the
window down. Guard the shape you need and skip anything else.

Give every stream client a distinct name (`DataClient(self.name + "_left")`):
the name is the publish channel.

Header keys are all optional; use `head.get(...)`. A camera frame
(`Camera.broadcast_image` in `drivers/camera_base.py`) carries `roi_x0`/`roi_y0`
(sensor pixel of the frame's corner) and `binning`, plus whatever the reader
adds: `timestamp`, `index` (place in the shot, or count since acquisition start
when live), `camera_timestamp` on live frames where the driver has one, and
`rid` when the experiment passes one. `_offset`/`_imgresolution` (origin in
binned pixels, metres per pixel) appear only from cameras with `object_pixel_m`
configured and from `experiments/slm_tweezer_array.py`. If you scale by
`_imgresolution`, range and label the axes in metres too, or the image shrinks
into a corner.

## Experiment results, saved parameters and trap sites

- **Experiment results.** `ExperimentFeed` (`GUI/experiments/feed.py`) mirrors
  the manager's state on its own Qt timer. `point_received` carries each point
  measured (`{"rid", "index", "values", "scalars", ...}`); points from before
  the window opened are in `held_points()`, taken once on the first
  `queue_changed`. Close the feed in `closeEvent`. To let the user pick result
  names, use `open_subscription_editor(streamkey=..., choices=[...])`.
  `viewers/live_plot.py` does all of this.
- **Saved parameters.** `get_saved_parameters()` (`experiment/client.py`) is an
  RPC: call it at start-up and when the feed snapshot's
  `saved_parameters_version` changes, never per poll, and catch
  `ManagerUnavailable`/`ManagerError` so a missing manager leaves the window up.
- **Trap sites** (`trap_sites`) are full-sensor `(y, x)` pixels. Place them with
  `analysis/trap_sites.to_roi(sites, head.get("roi_y0", 0), head.get("roi_x0", 0))`
  or they sit off the atoms whenever the camera reads an ROI. The Image Monitor
  draws row-major (row 0 at top, x along columns), matching the Optimisers tab
  and notebooks; subclasses inherit that.

## Where drawing code lives

`pytweezer/GUI/views/` holds base views that show what they are handed and know
nothing of streams, Properties or devices (`HistogramPlot`); the image and line
plot drawing still sit in `viewers/image_monitor.py` and `viewers/live_plot.py`.
A specialisation (image with trap sites, histogram with a fit) stays in its
applet until a second panel needs it, then moves to `GUI/views/`
(`TweezerLoadingPlots`). Applets (`GUI/viewers/`) and default-analysis views
(`GUI/default_analyses/`) never import from each other; share through
`GUI/views/`.

## Properties are the user's knobs

Anything the user might retune without editing code (threshold, history length,
colormap, axis range) is a `PropertyAttribute` with a default:
`_history = PropertyAttribute("history", 200)`. Reading it is cheap and also
writes the default back, which is what makes it appear in the configure dialog.

## Making it look right

`run_applet` calls `apply_theme` (stylesheet and pyqtgraph defaults). Never set
a stylesheet, a literal hex colour or `pg.setConfigOptions` in an applet; take
colours from `pytweezer/GUI/theme.py`. See `pytweezer-gui-design` for the wider
rules. For an applet:

- **One applet, one job.** Extra instances are free.
- **Let the display fill the window**: `setContentsMargins(0, 0, 0, 0)` on the
  outer layout.
- **Put `subscriptions` and `configure` in the view-box context menu**
  (`vb.menu.addAction("subscriptions")`).
- **Colour curves with `self.stream_color(stream, index)`**: it cycles the
  shared palette and honours a per-stream colour the user set.
- **Label axes with units**; `plot.addLegend()` once there are several curves.
- **Monospace numeric readouts** (`QFont("Consolas")` or
  `setStyleHint(QFont.Monospace)`), or live digits jitter.
- **`default_size`** fits the content on first run; geometry is remembered after.

A `HistogramLUTItem` owns its image's lookup table: `image_item.setLookupTable`
is silently discarded and the image renders grey. Use
`lut.gradient.setColorMap(cmap)`.

## Fit overlays and linked plots

The Image Monitor draws a cloud fit over the image with no new code: list the
fit's data stream (from `analyses/live/cloud_fit.py`, header fields
`analysis/cloud_fit.FIELDS` in the image's plot coordinates) in
`overlaystreams`; `n_contours` sets the ellipse count. Reuse
`overlay_paths(head, n)` from `viewers/image_monitor.py` in any viewer that
shows a fit (`[]` for a failed fit). Image and fit share a frame through the
header's `_offset`/`_imgresolution` (`analysis/cloud_fit.frame_geometry`).

To extend the Image Monitor, subclass `ImageDisplay` and override its hooks, as
`viewers/image_plot_monitor.py` does; don't copy it.

## Preview it before you call it done

`scripts/preview_applet.py` builds the applet against fake Properties and
streams, feeds synthetic frames or traces, and saves a PNG off-screen (nothing
appears on the user's display; no hubs or hardware).

```bash
uv run python .claude/skills/add-applet/scripts/preview_applet.py \
    pytweezer/GUI/viewers/atom_count.py \
    --prop 'datastreams=["Atom_number","Background"]'
```

`--prop KEY=JSON` (repeatable) seeds properties. **Read the PNG**: anything cut
off, data illegible, mostly chrome, two curves the same colour?

The printed summary: empty `subscribed` means `update_subscriptions` never ran
or read the wrong key; `consumed 0` means `poll` isn't reading; messages
`left unread` means draining with `if`; a `published` line means the applet
sends, which it must not.

Also run anything that consumes data streams with `--scalars` (header-only
messages; surfaces the unpack bug as a traceback) and `--physical` (headers with
`_offset`/`_imgresolution`). For other sources, `--points` emits an experiment
point per tick on every `ExperimentFeed`, `--saved KEY=JSON` seeds
`get_saved_parameters` (e.g. `--saved 'trap_sites=[[40,60],[40,80]]'`), and
`--roi X0,Y0` moves the frames' ROI origin, so a site overlay can be seen to
follow it.

The synthetic messages are generic frames and `[coords, values]` traces, so a
viewer of one analysis's own output (Tweezer Loading) draws nothing; test its
drawing on that analysis's header, as `tests/test_tweezer_loading_plots.py`
does. Then run the test files you touched, e.g. `uv run pytest tests/test_<viewer>.py -q`.

## Making it available

Nothing to register. The Applets tab's **template** list rereads
`pytweezer/GUI/viewers/` whenever it is opened and offers every script there
that calls `run_applet`, named by its `default_name="..."`, so a new viewer
shows up in a running GUI. The user adds it with the template list and the
**add** button. Tell them so.

Do **not** add a new viewer to `DEFAULT_APPLETS` in
`pytweezer/GUI/applet_launcher.py`: that list is the starting catalogue a
fresh PC gets, and only goes in front of the others in the template list.
Add one only when the user asks for it to be a default applet.

## Module docstring

For a labmate who has never seen the applet: what it displays, what it expects
on its streams, and a `Properties:` list. Model it on
`viewers/scalar_history.py`.

## Reference viewers (`pytweezer/GUI/viewers/`)

- `scalar_history.py`: the smallest complete applet; correct drain and unpack.
- `live_plot.py`: a grid of plots, each with its own subscriptions under its key
  in `plots`, including experiment results from `ExperimentFeed`.
  `stacked_plotter.py` re-lays it out by overriding `arrange`.
- `tweezer_loading.py`: a thin applet around a shared `GUI/views/` widget.
- `tweezer_monitor.py`: Image Monitor subclass drawing saved trap sites.
- `slm_monitor.py`: four image streams in a grid, in metres when the header
  says so.
- `image_monitor.py`: image, histogram LUT, ROIs (`CameraROI`), `DataSidebar`,
  a mask stream on its own `streamkey`, fit overlays.
- `image_plot_monitor.py`: Image Monitor plus x/y sums (`projectionstreams`)
  in plots linked to its axes.
- `camera_applet.py`: Image Monitor with camera settings inserted via
  `ImageDisplay.panel_layout` / `image_layout`; the device-driving pattern
  above.

`viewers/archive/` holds retired viewers: not models, though the image viewers
still import `zmq_ROI` from there.
