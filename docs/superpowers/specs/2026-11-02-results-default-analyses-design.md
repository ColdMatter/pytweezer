# Default analyses in the Results tab

## Intent

When a measurement finishes, the Results tab should show the analysis that
fits that experiment without anyone opening a notebook. The first example is
the tweezer loading histogram that already runs live (the
`tweezer_loading.py` analysis feeding `TweezerLoadingViewer`). It should also
work on a single h5 file after the fact.

Each experiment class declares the analyses that make sense for its files. The
Results tab offers them in a dropdown for the selected file. Today's
"Plot … against … one curve per …" quick plot becomes one entry in that
dropdown, and every measurement gets it.

Builds on #44, which lands the tweezer optimisers and the live loading
analysis on `main`. The PR for this work says `Depends on #44`.

## Goals

- An `Experiment` class declares its default analyses. Subclasses inherit
  the declaration and can extend or replace it.
- The Results tab has a dropdown above the plot area: **Plot against**
  (universal, always first) followed by the file's declared analyses.
  Choosing one replaces the plot area with that analysis's view.
- A **tweezer histogram** analysis. Given a file, it shows the same three
  panels as `TweezerLoadingViewer` (per-site histograms, all-sites histogram
  with its two-Gaussian fit and threshold, loading-probability map) and the
  same readout line.
- The live and per-file histograms share their numerics and their drawing
  code, so they cannot disagree.
- `TweezerImaging`, `MolassesRecapture` and their optimisers record the
  per-site photon counts of every shot, so their files can be histogrammed.
  `LoadTweezers` files already hold the frames.

## Non-goals

- Analyses do not write back into the h5 file. Files stay as the experiment
  wrote them.
- No live image-based analysis of a running measurement. The manager's feed
  carries only 0-d scalars. While a measurement runs, only **Plot against**
  updates live; the other entries say they are available once it finishes.
- No user-defined analyses from the GUI, and no analysis that spans several
  files.
- No change to the live applet's behaviour or properties.

## Anti-goals

- The Results tab must not import experiment modules. Experiment modules
  pull in drivers, cupy and so on, and the GUI runs on client PCs. The tab
  already relies only on the file and the archive, and it keeps doing so.
- Experiment modules must not import Qt or pyqtgraph. Workers and the
  catalogue subprocess import them headless.
- The tab must not freeze while a large file loads and is analysed.
- No change to the results of existing analysis. New methods are fine, but
  anything that computes an existing quantity (sites, photons, threshold,
  fidelity, loading, survival) must give results equivalent to the
  `rearrangement-optimisation-RnD` branch's `analysis.py`, which is the
  authority on how analysis is done. A suspected error in existing analysis is
  not fixed: it is reproduced and marked with a `# TODO: check ...` comment
  saying what looks wrong.

## Design

### Declaring analyses

An analysis is referred to by a registered name string, not an object, so
experiment modules stay free of GUI code:

```python
class LoadTweezers(MotMasterExperiment):
    analyses = ("tweezer_histogram",)
```

`Experiment.analyses` defaults to `()`. **Plot against** is implicit and is
never listed. Subclasses inherit the tuple as normal class attributes do.

`runner.open_writer` (shared by queued tasks and `run_local`) writes the class's `analyses` into the file's root attrs (for
example `attrs["analyses"]`, a JSON list) when the file is created, so
the file says what to offer. The tab reads this attribute from the metadata it
already loads and never needs the catalogue or the experiment class. Files
written before this change, and files of classes that declare nothing, get
only **Plot against**. *(My call: this keeps the tab independent of the
manager's catalogue and of later edits to the class. The cost is that an
analysis added to a class later does not appear on its older files.)*

### Analysis registry and the two halves of an analysis

Each analysis has:

1. **A compute half**: GUI-free, numpy only, under `pytweezer/analysis/`.
   It takes the measurement (through the archive, so it works remotely) plus
   any parameters, and returns a plain result. For the histogram that is the
   same `(head, photons)` pair the live analysis publishes. A notebook
   can call it directly.
2. **A view half**: a QWidget under `pytweezer/GUI/`. It draws that
   result and offers the analysis's controls.

The registry maps a name to a display label, the compute function, the view
class, and the result names it needs from the file. The registry lives on the
GUI side and the compute functions it points to live in `pytweezer/analysis/`.
An unknown name in a file's `analyses` attr shows up in the dropdown as
disabled, with a tooltip, rather than raising.

### The tweezer histogram

**Shared numerics.** Split `LoadingStatistics.summary` in
`pytweezer/analysis/tweezer_loading.py` so that the step from photons to
statistics (threshold fit, fidelity, per-site loading, building the header)
is a function of `photons` and site positions alone. The live analysis
and the per-file analysis both call it. Finding the sites in the mean
filtered image and summing the boxes into photons stays in shared functions
too.

**Shared drawing.** Move the plots, curves, map, colour bar and readout out of
`TweezerLoadingViewer` into a reusable widget, for example
`TweezerLoadingPlots`, with a `show(head, photons)` method. The applet becomes
a thin wrapper that subscribes and calls it. The Results tab embeds the same
widget.

**Where the photons come from**, by what the file contains:

- **Frames recorded** (`LoadTweezers`: `first_image`, `second_image` per
  point). Filter the first frames, find the sites in their mean, and sum the
  boxes, just as the live analysis does. The number of sites, feature size,
  window size and detection step are not in these files. They appear as
  small spin boxes in the view, defaulting to the live analysis's defaults
  (64, 10, 5, 100), and the analysis reruns when one changes.
- **Photons recorded** (`TweezerArrayMeasurement` subclasses, after this
  change). Use the recorded per-shot photons with the file's own
  `trap_sites` constant. Nothing needs re-finding.

**Threshold.** The view matches `get_array_loading_statistics(...,
show_histogram=True)` and the live viewer: it fits the two Gaussians to the
shots shown and draws that fit and its threshold. If the fit fails, it uses
the file's calibration `threshold`, or the live analysis's fallback when the
file has none. The measurements' own scoring is not changed. Loading is judged
against a per-batch refit, and survival against the calibration threshold.
This is how the `rearrangement-optimisation-RnD` branch's
`OptimiseMolassesShimParameters` and the imaging-optimiser notebook do it, and
that branch is the authority on analysis. The `score_batch` comment, which
says occupancy is always judged against the calibration threshold, is wrong
and gets corrected. Whether loading should really use a per-batch refit
while survival uses the fixed threshold is left as it is, with a `# TODO:
check` comment.

**Which shots.** A selector in the view chooses **all points** (the default)
or one scan value, pooling its repetitions. One scan value is the meaningful
choice in an optimiser or a scan, where points differ in their settings. It
is a scan value, not a point, because a `LoadTweezers` point is a single
shot. For frames-recorded files, the sites are found once in the mean of
all the file's frames and reused for every selection. *(My call.)*

**Which image.** A first/second image choice, like the live analysis's
`image_index`. The histogram of the second image is the one survival is
judged on. *(My call.)*

### Recording photons in `TweezerArrayMeasurement`

`score_batch` already computes the filtered first and second frames. It also
returns, or records, the per-site photon counts of the batch: a
`(shots, sites)` float array for the first images and another for the
second. These are stored per point as results `first_photons` and
`second_photons`. The sum and conversion reuse the same helpers as the
histogram. `measure()`'s scalar results stay as they are. Since
`BayesianOptimisation`/`StagedScan` take `measure()`'s return value as
`dict[str, float]`, the arrays must be recorded directly inside the
measurement, not returned through `measure()`. These classes declare
`analyses = ("tweezer_histogram",)`, as does `LoadTweezers`.

### Results tab changes (`pytweezer/GUI/experiments/results.py`)

- A combo box ("Analysis") heads the plot region. Its entries are
  **Plot against**, then the file's declared analyses in order.
- The plot region's body becomes a stacked widget. **Plot against** is the
  current controls plus plot, unchanged in behaviour (including live
  plotting). Each other analysis gets its view widget.
- The choice is remembered per experiment class for the session: selecting
  another `LoadTweezers` file keeps the histogram showing. Falls back to
  **Plot against** when the new file doesn't offer the remembered choice.
- Image-based analyses load their result arrays through their own archive
  client and compute off the GUI thread. `RemoteArchive` holds one lock for
  the whole RPC call, so sharing the tab's client would block its polling
  behind a large load. *(My call.)* The view shows "Analysing…" until done. A
  newer selection discards a stale result.
- For a running measurement, the non-default entries are disabled, with a
  tooltip saying they become available when it finishes. The tab already
  reloads the file on finish.
- Errors in an analysis (missing results, fit failure, too few sites found)
  show in the view's message line, as `plot_message` does now. They never
  raise into the tab.

## Testing

Under `tests/`, following the `write-tests` skill:

- The shared statistics function gives the same header for the same photons
  whether reached from the live analysis or the per-file path.
- Equivalence: on the same synthetic frames, the new code's photons,
  threshold, fidelity and per-site loading match
  `analysis.get_array_loading_statistics`.
- Per-file histogram compute on a small synthetic file of each kind (frames
  recorded, photons recorded), using `run_local` with a simulated or fake
  camera where possible.
- The `analyses` attr is written into the file and inherited by subclasses.
- Results tab: the dropdown lists **Plot against** plus a file's analyses,
  disables them while running, and switches views (extend
  `tests/test_results_browser.py`).

## Left to the builder

- Exact names: the attr key, the registry module, the widget class, the
  result names `first_photons`/`second_photons`.
- How the off-thread compute is done (QThreadPool or a plain thread plus a
  signal), as long as the GUI never blocks on it.
- Layout and wording of the view's parameter row and point selector, within
  the `pytweezer-gui-design` skill's rules.
- Whether the registry entry also declares which result names to fetch, or
  the compute function fetches them itself.
