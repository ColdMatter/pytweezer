---
name: add-applet
description: Create a new pytweezer applet — a live viewer window under pytweezer/GUI/viewers/ that subscribes to image or data streams and displays them. Use this whenever the user wants a new applet, viewer, monitor, live display, plot window, readout, or scope for a stream ("show me the atom number live", "a window that plots the trap frequency", "display the camera with the fit overlaid", "a histogram of pixel values"), or wants to change how an existing viewer looks or behaves. Applies even when they only describe the display without saying "applet" or "viewer".
---

# Adding an applet to pytweezer

An applet is one small local process that subscribes to streams and draws them.
`pytweezer/GUI/applet.py` owns the process machinery — Properties connection,
window title and icon, geometry, theme, poll timer, the subscription/configure
dialogs — so writing an applet means writing **one `Applet` subclass** in a new
file under `pytweezer/GUI/viewers/`, then pointing a launcher entry at it.

There is no config category to edit. The Applet Launcher runs
`python <script> <name>`, where `name` is the window title, the Properties
namespace, and the launcher label at once. Two entries running the same script
under different names are fully independent instances.

Background, including the launcher and the storage format:
[`docs/applets.md`](../../../docs/applets.md).

## First check it is an applet

Applets **display**; they never compute a derived quantity and never publish.
If the request needs a new number that nothing currently streams — a centre of
mass, a fit, an atom count, a running average — that number belongs in an
analysis process (use the `add-analysis-script` skill), and the applet only
plots what the analysis publishes. Splitting it this way is what lets several
people watch the same derived quantity from different PCs.

An applet that finds itself calling `client.send()` has taken on an analysis's
job. The preview script warns when it sees this.

## The shape of the file

```python
class AtomCount(Applet):
    stream_category = "Data"     # "Data" or "Image" — which catalog the
    poll_interval = 50           #   subscription editor lists
    default_size = (520, 300)

    _threshold = PropertyAttribute("threshold", 0.5)

    def init_gui(self):          # once: build widgets, make stream clients
        ...
        self.update_subscriptions()

    def update_subscriptions(self):   # after either dialog closes
        ...

    def poll(self):              # every poll_interval ms
        ...

def main(name):
    run_applet(AtomCount, default_name=name)

if __name__ == "__main__":
    run_applet(AtomCount, default_name="Atom Count")
```

Keep both entry points: `main(name)` for tooling that imports the module, and
the `__main__` guard for the launcher and the command line.

`poll_interval` is a repaint budget, not a data rate — nothing is lost between
polls because messages queue. The existing viewers use 10 ms because they redraw
one cheap item; 50–100 ms is kinder for anything that rebuilds a plot, and the
display still looks live.

## Reading a stream without losing or crashing on messages

Four things about the transport decide whether an applet survives contact with a
real stream. All four have bitten the existing viewers.

**`recv()` returns a variable-length tuple.** `(channel, header)` for a
header-only message, `(channel, header, array)` when an array follows, and
`None` on timeout. Analyses publish scalars as header-only messages, so
unpacking three values unconditionally raises `ValueError` the moment someone
points your applet at a scalar stream. Length-check against what you actually
need — an applet that draws an array requires three, one that reads a header
field is happy with two:

```python
message = self.stream.recv()
if message is None or len(message) < 3:     # drawing an array
    continue
channel, head, array = message

message = self.stream.recv()
if message is None or len(message) < 2:     # reading a header field
    continue
channel, head = message[0], message[1]      # tolerate a trailing array
```

**Drain the queue with `while`, not `if`.** One `recv()` per poll falls behind
a stream faster than it arrives and never catches up — the display drifts
further into the past the longer it runs.

```python
while self.stream.has_new_data():
    ...
```

**`update_subscriptions` must unsubscribe first, then rebuild the view.** It
runs every time a dialog closes, so leaving old subscriptions attached silently
accumulates them, and leaving old curves in the plot leaves ghosts of streams the
user just removed. Unsubscribe, clear, re-subscribe, recreate.

**Skip `"--None--"`.** That is the placeholder an unconfigured stream list
carries. Subscribing to it is harmless, but creating a curve or a row for it puts
a fake entry in front of the user.

Header keys are all optional — use `head.get(...)`. `_imgindex` (position within
a shot; `0` marks a new sequence, the natural reset for anything stateful),
`timestamp`, and the pair `_offset`/`_imgresolution` (pixel origin and metres per
pixel) are set by some publishers and not others. If you scale an image by
`_imgresolution`, keep the axis labels honest — an image drawn in metres inside
a view ranged in pixels shrinks into the corner and looks like a broken applet.

Give every stream client a distinct name (`DataClient(self.name + "_left")`).
The name is the publish channel, and two clients sharing one are indistinguishable
on the hub.

Finally, **`poll` must not raise.** It runs from a QTimer; an exception there
escapes into the Qt loop and, depending on the message, can take the window with
it. Guard the shape you require and skip anything else.

## Properties are the user's knobs

Anything the user might retune without editing code — a threshold, a history
length, a colormap, an axis range — is a `PropertyAttribute` with a default:

```python
_history = PropertyAttribute("history", 200)
```

Reading `self._history` is cheap (a local dict the Properties thread keeps in
sync) and the user can change it live from the **configure** dialog. Reading a
property also writes its default back, which is what makes the knob appear in
that dialog in the first place — so a property nobody has touched still shows up
with its default, ready to edit.

## Making it look right

The theme is handled: `run_applet` calls `apply_theme`, which applies the shared
dark stylesheet and the pyqtgraph background/foreground defaults. An applet is
its own process and inherits nothing from the main GUI, so this is the only thing
standing between your window and the platform's default light grey. Never set a
stylesheet or a literal hex colour in an applet — take colours from
`pytweezer/GUI/theme.py` so every window in the lab keeps matching.

What is left to you is the composition:

- **One applet, one job.** A window that shows one thing well beats a window
  with three panels and a toolbar. Extra instances are free — the user adds
  another launcher entry.
- **Let the display fill the window.** `layout.setContentsMargins(0, 0, 0, 0)`
  on the outer layout. A plot framed in 11 px of background is 11 px of data the
  user did not get.
- **Put controls in the view-box context menu, not in buttons.** This is the
  established pattern (`vb.menu.addAction("subscriptions")`), and it is why the
  viewers stay all-data. A button bar costs vertical space on every frame to
  serve a click that happens once a week. Always wire up `subscriptions` and
  `configure`; add your own actions beside them.
- **Colour curves with `self.stream_color(stream, index)`.** It cycles the
  shared palette so several streams are distinguishable immediately, while still
  honouring a per-stream colour the user has set. Hardcoding a pen colour means
  every curve on the plot is the same one.
- **Label axes with units**, and use `plot.addLegend()` once there is more than
  one curve.
- **For numeric readouts, use a monospace font** (`QFont("Consolas")`, or
  `setStyleHint(QFont.Monospace)`). Proportional digits change width as the value
  changes, so a live number visibly jitters and is hard to read at a glance.
- **`default_size`** should fit the content without the user resizing on first
  run; geometry is remembered after that.

Two pyqtgraph traps worth knowing:

- A `HistogramLUTItem` **takes ownership of its image's lookup table**. Calling
  `image_item.setLookupTable(...)` alongside one is silently discarded and the
  image renders greyscale. Set the colormap on the histogram instead:
  `lut.gradient.setColorMap(cmap)`.
- `pg.setConfigOptions` is global and already set by `apply_theme`. Re-setting
  `background`/`foreground` in an applet only puts it out of step with the rest.

## Preview it before you call it done

`scripts/preview_applet.py` builds the applet against fake Properties and fake
streams, feeds it synthetic frames or traces on a timer, and saves a PNG. No
hubs, no hardware, and nothing appears on the user's screen — it paints
off-screen via `WA_DontShowOnScreen` rather than mapping a window.

```bash
poetry run python .claude/skills/add-applet/scripts/preview_applet.py \
    pytweezer/GUI/viewers/atom_count.py \
    --prop 'datastreams=["Atom_number","Background"]'
```

`--prop KEY=JSON` seeds properties (repeatable) — this is how you drive the
applet into the configuration you want to see. Then **read the PNG** and judge
it: is anything cut off, is the data legible, is the window mostly data or
mostly chrome, are two curves the same colour?

The printed summary is the functional half. `subscribed` empty means
`update_subscriptions` never ran or read the wrong key. `consumed 0` means the
window built but `poll` is not reading. Messages `left unread` means you are
draining with `if` instead of `while`. A `published` line means the applet is
sending, which an applet should not do.

Two runs worth doing on anything that consumes data streams:

```bash
# header-only messages, as an analysis publishes scalars
... preview_applet.py <script> --prop '...' --scalars
# headers carrying _offset/_imgresolution, so coordinates are in metres
... preview_applet.py <script> --prop '...' --physical
```

The first is how the `recv()` unpack bug surfaces — a traceback rather than a
picture. Iterate on the code and re-run until the screenshot is something you
would be happy to put on a lab monitor.

Then run the suite, which covers the framework the applet sits on:

```bash
poetry run pytest tests/ -q
```

## Register it

Add an entry to `DEFAULT_APPLETS` in `pytweezer/GUI/applet_launcher.py` so it is
offered as a template in the **Applets** tab:

```python
{
    "name": "Atom Count",
    "script": "pytweezer/GUI/viewers/atom_count.py",
    "description": "Rolling history of the atom number",
},
```

The list seeds the Properties `"Applets"` key only on a *first* run, so on a
machine that already has applets configured the user adds the row through the
tab's **add** button. Say so rather than implying the entry alone makes it
appear.

## Document it in the module docstring

The docstring is what a labmate reads before running your applet. State what it
displays, what it expects on its streams, and what the configure dialog lets
them set:

```python
"""Rolling history of a scalar published in a data stream's header.

Subscribes to the streams named in ``datastreams`` and plots the header field
named by ``field`` against arrival order, keeping the last ``history`` points.
Point it at an analysis that publishes scalars (``image_stats.py``,
``centre_of_mass.py``).

Properties:
    *   datastreams: ([str]) input data streams.
    *   field: (str) header key to plot.
    *   history: (int) number of points kept.
"""
```

Write it for someone who has never seen this task — what the applet is for and
the constraints that would otherwise cost them an afternoon, not a narration of
what was built or why.

## Reference viewers

- `viewers/live_plot.py` — the smallest complete applet: one curve per data
  stream, correct drain and unpack handling.
- `viewers/image_monitor.py` — image stream, colormap + histogram LUT, ROIs, and
  a hideable sidebar; shows a second stream category (masks) on its own
  `streamkey`.
- `viewers/image_plot_monitor.py` — image with axis-linked projection plots;
  the example to copy for multiple linked plots in one window.

`viewers/archive/` holds retired viewers — do not use them as models, though
`zmq_ROI` is still imported from there by the image viewers.
