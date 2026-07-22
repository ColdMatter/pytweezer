---
name: add-analysis-script
description: Write a new streaming analysis process in pytweezer/analysis/ that subscribes to a live image or data stream, transforms each message, and republishes the result. Use this whenever the user wants to create, add, or write an analysis script, an analysis filter, or an image/data processing step — computing a centre of mass, fitting a peak, thresholding, counting atoms, averaging or subtracting frames, extracting a profile, deriving any quantity from a live camera or data stream — or asks how to get a new derived quantity onto a live plot or into the Analysis Manager. Applies even when the user only describes the maths ("find the centre of mass of images", "fit a Lorentzian to the trace") without saying "analysis" or "script".
---

# Adding a streaming analysis to pytweezer

An analysis is one small long-lived process: it subscribes to one or more pub/sub
streams, receives `(message, head, data)` in a loop, transforms each one, and
republishes. `pytweezer/analysis/analysis_base.py` owns that loop, so writing an
analysis means writing **one class with one `process` method** in a new file
under `pytweezer/analysis/`.

There is no registry to edit and no config entry to add. The Analysis Manager
scans the directory at runtime, so a correctly written file shows up in the GUI
by existing. That makes the two things worth getting right (1) the `process`
transform and (2) the handful of conventions that decide whether the GUI can see
your script at all.

## How the framework runs your class

`AnalysisProcess.__init__` opens a `Properties` connection named after the
process, creates the pub/sub client, and subscribes it to the streams named in
the `imagestreams`/`datastreams` property. Then `run()` loops forever:

```
msg = client.recv()  →  head, data  →  process(head, data)  →  send_result(...)
```

`process` returning `(head, data)` republishes that message on the process's own
stream; returning `None` drops it. That's the whole contract. Because
construction opens sockets and the loop never exits, everything testable lives
inside `process` — keep it that way and verification stays cheap.

## Pick the base class by what you consume

`ImageAnalysis` subscribes to an image stream via `ImageClient` and republishes
on channel `_`. `DataAnalysis` subscribes to a data stream via `DataClient` and
republishes on the default channel.

Choose by the **input**, not the output. An analysis that eats frames and emits
numbers is still an `ImageAnalysis` — it just publishes through a second client
it makes itself:

```python
class ImageStats(ImageAnalysis):
    def __init__(self, name):
        super().__init__(name)
        # Stats are scalars, so publish them on the Data hub.
        self.dataq = DataClient(name.split("/")[-1])
```

then `self.dataq.send(head, data, channel=...)` inside `process` and `return
None`, since the base's republish path would put them on the wrong hub.
`projections.py` (image in, two data profiles out) and `image_stats.py` are the
two examples to copy from.

## Writing `process`

```python
def process(self, head, data):
    if data.ndim != 2:      # guard the shape you require
        return None
    ...
    out_head = dict(head)   # copy before mutating
    out_head["centre_x"] = float(cx)
    return out_head, result
```

A few conventions that come from the transport rather than the maths:

- **Copy `head` before you change it.** The incoming dict belongs to the
  message you were handed; mutating it in place surprises anything that reads it
  after you.
- **Send scalars in the header with `data=None`.** `DataClient` publishes a
  header-only message fine, and that's how the viewers' data sidebar and
  `live_plot.py` pick up derived numbers. Values must be JSON-safe — wrap numpy
  scalars in `float()`/`int()`, or they will not survive the hop.
- **Emit 1-D traces as a 2xN `[coords, values]` array**, so `live_plot.py` and
  `gaussian_fit.py` can read `A[0]` as the axis and `A[1]` as the values.
- **`np.ascontiguousarray` any crop or slice before sending.** A view carries the
  parent frame's strides, and ZMQ would ship the whole buffer behind it.
- **Accumulate in `float64`** (`data.sum(axis=0, dtype=np.float64)`) — raw camera
  frames are integer and overflow quietly.
- **Return `None` rather than raising** on a message you can't handle. An
  exception kills the process, and the GUI shows it as simply stopped.

Useful header keys, when present: `_imgindex` (position within a shot, `0` marks
a new sequence — the natural reset signal for anything stateful), `timestamp`,
`_offset` (pixel origin of a crop within its parent frame) and `_imgresolution`
(metres per pixel). Treat all four as optional with `head.get(...)`; not every
publisher sets them. If your result is a position in the frame, decide whether to
report pixels or physical units and say which in the docstring — `roi_slice.py`
shows the conversion both ways.

## Properties are the user's knobs

Anything the user might want to change without editing code — a threshold, a fit
method, a number of frames to average — is a `PropertyAttribute` with a sensible
default:

```python
class CentreOfMass(ImageAnalysis):
    _threshold = PropertyAttribute("threshold", 0.0)
```

Reading `self._threshold` fetches from a local dict the Properties thread keeps
in sync, so per-frame reads are cheap and the user can retune it live from the
GUI while the analysis runs. It is a deep copy on each access, though, so don't
read a large list-valued property inside a per-pixel loop.

## Document it in the module docstring

The docstring is the only thing a labmate sees before running your analysis, and
the properties section is how they know what the GUI's property editor lets them
set. Follow the existing shape exactly:

```python
"""One-line summary of what this computes.

Input:
    One image stream.

Output:
    A data message (no array) carrying ``centre_x``, ``centre_y`` ... published
    on ``<name>``.

Properties:
    *   imagestreams: ([str]) input image streams.
    *   threshold: (float) counts below this are treated as background.
"""
```

Then any non-obvious constraint as prose below it — what the coordinates mean,
what happens on an empty frame, why a dtype was chosen. Write it for someone
who has never seen this task.

## The category trap

`AnalysisManager._classify_script` decides whether your script appears under
**Image** or **Data** in the GUI by *text-searching the whole file*:

- contains `imagestreams` or `ImageAnalysis`, and not the data equivalents → Image
- contains `datastreams` or `DataAnalysis`, and not the image equivalents → Data
- **contains both, or neither → the script is silently invisible in the GUI**

So an image analysis must never contain the literal string `datastreams` or
`DataAnalysis` anywhere — including in a comment or docstring. This is easy to
trip by writing "publishes two datastreams" in prose; `projections.py` says
"two *data* streams" with a space precisely to stay classifiable. Importing
`DataClient` is safe (it matches neither pattern).

After writing the file, check it the way the GUI will:

```bash
grep -c "imagestreams\|ImageAnalysis" pytweezer/analysis/<yourfile>.py
grep -c "datastreams\|DataAnalysis" pytweezer/analysis/<yourfile>.py
```

Exactly one of those must be non-zero.

## End the file with the standard main

```python
if __name__ == "__main__":
    run_analysis(CentreOfMass)
```

`run_analysis` parses the single `name` argument the Analysis Manager passes when
it launches `python <script> Analysis/<Category>/<name>`.

## Verify it with a test

Add a case to `tests/test_analysis.py`. If that file doesn't exist yet, copy the
bundled starter first — it carries the helpers plus worked cases for the existing
analyses:

```bash
cp .claude/skills/add-analysis-script/assets/test_analysis.py tests/test_analysis.py
```

`build(cls, props)` allocates the analysis **without running `__init__`**, so no
hubs, no sockets, and no non-daemon threads: it installs a dict-backed fake for
`_props` (which every `PropertyAttribute` reads through) and recording fakes for
`client` and `dataq`. You then push a synthetic frame or trace through `process`
and assert on what came back or what got published:

```python
def test_centre_of_mass_finds_a_single_bright_spot():
    from pytweezer.analysis.centre_of_mass import CentreOfMass

    com = build(CentreOfMass, {"threshold": 0.0})
    frame = np.zeros((8, 8), dtype=np.uint16)
    frame[5, 3] = 100

    head, data = com.process({}, frame)

    assert head["centre_x"] == pytest.approx(3.0)
    assert head["centre_y"] == pytest.approx(5.0)
```

Test the cases that actually bite: a known-answer input where you can compute the
right number by hand, the degenerate input (empty frame, all-background, wrong
ndim) that must return `None` rather than raise, and any property that changes
behaviour. Run:

```bash
poetry run pytest tests/ -q
```

## Turning it on

Nothing else to edit. In the GUI's **Analysis** tab, *Add* creates a filter:
give it a name, pick Image or Data, pick your script from the dropdown, and
select the input stream. The *Properties* button opens the editor at
`/Analysis/<Category>/<name>/` where the defaults you declared appear, ready to
tune. Starting the filter launches your script as its own process.

Tell the user this is where they finish the job; the skill's work ends at a
tested file. A passing test proves the transform, not that the stream names line
up or that the numbers mean what the physics needs — say so plainly rather than
implying it's verified end to end.
