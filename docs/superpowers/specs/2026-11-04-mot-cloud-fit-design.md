# MOT cloud fits: 2D Gaussian, projections and image overlays

## Why

Cloud sizes are still measured by hand in the CaF lab-book notebooks
(`03Nov26.ipynb`, `get_size_dropouts`/`get_temperature_2dGaussDropouts`): each
scan point's frames are background-subtracted, a rotated 2D Gaussian with an
offset is fitted to the mean image, eight white contours of the fit are drawn
over it, every frame is refitted for σ error bars, and σ is plotted against the
scanned parameter. pytweezer should do this itself, live while the experiment
runs and afterwards from the measurement file, so a molasses-shim or MOT scan
gives its sizes without a notebook.

## Goals

1. **One fit, shared everywhere.** A Qt-free 2D Gaussian fit used by the live
   analysis, the experiments and the Results tab alike.
2. **Live.** A live analysis, run by the Analysis Manager on the server PC (the
   beast), fits each frame of any image stream and publishes the results on a
   data stream; the fit contour is an overlay on image monitors; the results go
   on the existing live/scalar-history plots and the image monitor's data panel.
3. **Experiments.** `CaFFluorescence` and `LoadTweezers` (via an ORCA image
   per shot) fit their MOT frames and record the results per point; the
   Results tab offers a view of each point's image with its contours, and
   "Plot against" plots the sizes against any scanned parameter.
4. **Projections.** A live analysis of horizontal and vertical sums, each with
   a 1D Gaussian fit, and an image-monitor subclass that draws the sums and
   their fits above and to the left of the image.
5. **Image statistics.** Max, min, mean and standard deviation of each image,
   shown in the image monitor's data panel.
6. **Data panel.** The image monitor's data sidebar moves below the image and
   is shown by default.
7. **Temperature.** A default Results analysis that fits σ² against t² for a
   time-of-flight scan, as `get_temperature_2dGaussDropouts` does, giving the
   temperature along x and y.

## Non-goals

- Several clouds per image, non-Gaussian clouds, atom-number calibration.
- numba, Cython or CUDA for the fit (see Performance).

## Anti-goals

- A fit that stalls the analysis loop or the experiment on a bad frame: every
  fit returns (with `success=False`) rather than raising or hanging.
- Width parameters that swap meaning between frames. The notebook fits σx, σy
  along the Gaussian's own rotated axes, so with θ free its "axial" and
  "radial" can swap from point to point.
- A second copy of the image monitor. The existing `image_plot_monitor.py`
  duplicates `image_monitor.py`; the projections monitor must be a subclass.

## The fit (`pytweezer/analysis/cloud_fit.py`, new)

Model: `offset + amplitude * exp(-½ (r−r₀)ᵀ Σ⁻¹ (r−r₀))`, parameterised by
amplitude, centre (x₀, y₀), σ_major, σ_minor, angle θ (of the major axis from
+x, in (−π/2, π/2]) and offset.

Reported per fit (all lengths in metres when the pixel scale is known, else in
pixels, with a `units` field saying which):

| field | meaning |
|---|---|
| `x0`, `y0` | centre |
| `sigma_x`, `sigma_y` | rms width along the image x and y axes, √Σₓₓ and √Σᵧᵧ: unambiguous whatever θ is, and what a time-of-flight analysis along each axis needs |
| `sigma_major`, `sigma_minor`, `angle` | the ellipse itself |
| `amplitude`, `offset` | peak above offset; offset |
| `integral` | 2π·amplitude·σ_major·σ_minor, in counts·pixel (the fitted cloud's summed counts) |
| `success` | False when there is no cloud or the fit failed |

Method (from timing the notebook's fit on this laptop: 3.5 ms at 64², 28 ms at
200², 330 ms at 512², scaling with pixel count):

1. Moments estimate on the whole image (robust baseline, clipped weights), as
   `gaussian_moments` in `gaussian_fit.py` does in 1D, gives the seed and the
   window.
2. Least squares (`scipy.optimize.least_squares`, Levenberg–Marquardt) with an
   **analytic Jacobian**, on a window of about ±4σ around the seed, binned so
   σ spans a few pixels. This keeps any frame, a full ORCA frame included, to a
   few ms.
3. Capped evaluations; on failure return the moments estimate with
   `success=False`.

The same module holds the 1D fit used by the projections (same windowing and
analytic-Jacobian approach; `gaussian_fit.py` keeps working and may be reduced
to a wrapper of it), and `contour_ellipses(fit, n)` giving the n contour
ellipses (levels evenly spaced in fraction of the amplitude, as matplotlib's 8
auto levels roughly were) for the overlay and the Results view.

## Performance

Pixel count, not interpreter speed, sets the cost, so windowing and an analytic
Jacobian are the optimisation. numba would gain at most another 2–3×; a GPU only
pays for batches of hundreds of fits, and one fit per frame at shot rates
(≲ a few Hz) is far from that. Analyses run as separate processes on the
beast's many cores, so many running at once is not a GIL problem. Keep the fit a
plain function of arrays so a batched GPU fit could replace it later if
per-frame fits of large stacks ever matter. *(my call)*

## Pixel scale

No camera's CONFIG entry says how big a pixel is at the atoms. Add an optional
`object_pixel_m` key (metres per unbinned sensor pixel at the atoms: sensor
pitch / magnification) to each camera entry. `CameraBase.broadcast_image` then
adds `_imgresolution = [object_pixel_m·binning]*2` and `_offset = [roi_x0,
roi_y0] / binning` (the convention `image_monitor.py`, `roi_slice.py` and
`datamgr.py` already use: offset in binned pixels, multiplied by the
resolution). Without the key, headers are as now and everything is in pixels.

Values to fill in are open (see Open values). The builder must check every
consumer of camera streams still behaves once these keys appear (image
monitors switch from pixel to metre axes; `CameraROI.copy_as_camera_roi`,
`roi_slice.py`, the tweezer-loading analysis placing sites by
`roi_x0`/`roi_y0`).

## Background

The experiments already broadcast every shot's raw frames on the camera's
stream (`CameraExperiment.take_shot`, header `rid`). `take_shot` gains a
`background: bool = False` keyword that adds `"background": True` to the
broadcast headers; `CaFFluorescence.prepare` uses it for its background shot.

A live analysis `shot_background.py` (replacing the old file-based
`subtract_bg.py`, which takes `abs()` of the difference; the builder should
check nothing configured still uses it) subscribes to image streams, keeps the
mean of the latest background frames per input stream and rid, and republishes
every non-background frame minus the background of its rid on its own image
stream. Frames with no matching background pass through unchanged (the fit's
offset absorbs a uniform background). Stats, projections and the 2D fit then
subscribe to this stream, or straight to a camera stream.

## Live analyses (`pytweezer/analysis/`, run by the Analysis Manager)

All three subscribe to image streams and publish on the Data hub; header
fields `rid`, `index`, `_imgresolution`, `_offset` are carried through.

- **`cloud_fit.py`** (the module above, with `run_analysis` in `__main__`, as
  `gaussian_fit.py` does): publishes one data message per frame on `<name>`,
  the fit table as header fields (no array).
- **`projections.py`** (existing, extended): publishes `<name>_x` and
  `<name>_y` as before (`[coords, sums]`, now in metres when the scale is
  known) plus a third row, the 1D fit curve, and the 1D fit's `center`,
  `sigma`, `amplitude`, `offset`, `success` in the header. The builder checks
  `live_plot.py` and other readers of `[coords, values]` still work with a
  third row.
- **`image_stats.py`** (existing): already publishes max, min, mean, std
  (plus stderr, sum, n). No change beyond carrying `rid`/`index`.

## Image monitor (`pytweezer/GUI/viewers/image_monitor.py`)

- **Orientation.** The monitor draws frames row-major (row 0 at the top, x
  along the columns), as the Tweezer Monitor, the Optimisers tab and the
  notebooks do; it now draws them transposed (pyqtgraph's col-major default).
  Everything here (σx/σy, overlays, the projection plots, `roi_slice.py`)
  takes x as the column index. Every lab Image Monitor turns; ROIs saved on
  them are redrawn by hand once, not migrated. `camera_roi` already handles
  both orders; the builder checks `roi_slice.py` and anything else reading a
  monitor's ROIs against the new order, and the Tweezer Monitor's docstring
  note about the transposed Image Monitor goes. *(user's decision)*

- **Overlays.** A new `overlaystreams` property (Data streams), set from a
  view-box context-menu entry "Overlays", like "Image masks". Each subscribed
  stream's latest message that carries the 2D fit fields is drawn as
  `n_contours` (property, default 8) white ellipses in the image's coordinates;
  a message with `success=False` clears that stream's ellipses. Ellipses, not a
  contour image, because the fit's contours are exactly ellipses and drawing
  them from the parameters costs nothing.
- **Data panel.** The sidebar moves below the image, shown by default (the
  property default becomes True). It shows the latest message of **each**
  subscribed stream (now it shows only whichever arrived last), so stats and fit
  results sit side by side. Its layout follows `pytweezer-gui-design`.

## Projections monitor (replaces `image_plot_monitor.py`)

`ImageProjectionDisplay(ImageDisplay)`: everything the image monitor does
(masks, overlays, ROIs, data panel, index selector), plus a plot above the image
sharing its x-axis and a plot left of it sharing its y-axis. One property,
`projectionstreams`, names projection analyses; `<name>_x` draws in the top
plot and `<name>_y` in the left, each as the sum plus its fit curve.
`image_plot_monitor.py` is deleted; the builder moves any Properties/CONFIG
references to it.

## Experiments

A helper (on `CameraExperiment` or a function in `cloud_fit.py`; builder's
choice) fits one camera's background-subtracted frames for a point:

- fits the mean image; records `<camera>_<field>` for `x0`, `y0`, `sigma_x`,
  `sigma_y`, `sigma_major`, `sigma_minor`, `angle`, `amplitude`, `integral`;
- fits each frame, seeded from the mean fit, and records `<camera>_<field>_error`
  for the sizes and centre as the standard error over the successful frame fits
  (NaN with fewer than two). The notebook used the standard deviation; the SEM
  matches `CaFFluorescence.counts_error`. *(my call)*
- records the mean background-subtracted image `<camera>_mean_image`
  (float32), when the experiment's `save_mean_image` argument (default True)
  is set, for the Results view.

**`CaFFluorescence`** (and so `OptimiseMOTLasers`): marks its background shot,
and records the ThorCam fit per point beside `counts`.

**`LoadTweezers`**: `RbTweezerBasic2026_2` fires `tMOTCamTrig` (Rb pattern
board) with the first HamCam image. Add an `ORCA` `CameraShots`, one frame per
shot, and record its fit per point. The ORCA is plugged into the CaF MOTMaster
PC (ph-bonesaw) but images both species; here it images the Rb MOT, so
`cloud_species` is Rb. It is used only while the `ORCA` CONFIG entry is active;
how an optional camera is declared and reserved is the builder's to design
within the reservation rules. No background shot: the offset term takes the
background.

**The ORCA's CONFIG entry** is renamed from `"CaF ORCA"` to `"ORCA"` (stream
`orca`), and the stale "keep exactly one of these two entries active" comment
goes. Every reference moves with it: `experiments/camera_settings.py`, the
`camera_experiment.py` docstring example, and the `add-device-driver` skill's
example and `check_device.py` usage, which still name the deleted `"Rb ORCA"`.
Dated specs and plans under `docs/superpowers/` stay as written. *(user's
decision)*

Both declare the new Results analysis in `analyses`.

## Results tab

A new `AnalysisView`, registered as `"cloud_fit"` ("MOT Gaussian fit") in
`pytweezer/GUI/experiments/analyses.py`, with its computation Qt-free in
`pytweezer/analysis/`. For a finished file it shows each point's
`<camera>_mean_image` with its contour ellipses (point chosen by slider or
list, labelled with the scanned values), and a plot of `sigma_x`, `sigma_y`
with error bars against the scan axis. Points sharing a scan value are pooled
(mean ± SEM over them, through `pytweezer/experiment/summary.py` as "Plot
against" does); a value run once shows its per-frame error, NaN for a single
frame. The user picks the scanned argument to plot against, so multi-axis and
staged scans work. *(my call)* It reads only what the file recorded; a
file with no mean images shows the plot only. "Plot against" already plots any
recorded field against any scanned one.

A second view, `"cloud_temperature"` ("MOT temperature (time of flight)"),
declared by the same experiments. The user picks which scanned argument is the
free-fall time (defaulting to one whose name contains `FreeExp`, `FreeFall` or
`Drop`, as the notebooks' `FreeExpTime`/`tFreeFallDuration`/`tDropDuration`),
converted to seconds (MOTMaster time parameters are in clock ticks; the
builder finds the tick length the MOTMaster wrapper already knows). Per axis it
fits σ² = σ₀² + (k_B T/m) t² weighted by the recorded errors, and shows σx²,
σy² against t² with the fits, T_x, T_y ± error in µK and σ₀. The mass comes
from a constant the experiment records, `cloud_mass_amu` (a class attribute
`cloud_species`: CaF for `CaFFluorescence`; for `LoadTweezers`' ORCA image,
Rb). The centre against
t² is also plotted, with a fitted g, as the notebook did, as a check of the
time axis. With fewer than three distinct times it says so instead of fitting.

## Skills to update

`add-analysis-script` (the fit module, background step), `add-applet`
(overlays, projections monitor, data panel), `add-experiment` (background
flag, recorded fit fields, the new analysis). Minor edits go in with the code;
new sections are proposed to the user first.

## Open values

- `object_pixel_m` for each camera (ORCA, CaF ThorCam, Rb ThorCam, Rb
  HamCam). Until given, those cameras report in pixels.

## Left to the builder

Window size and binning rule; contour levels' exact spacing; the data panel's
layout; how the optional ORCA is declared; whether `gaussian_fit.py` becomes a
wrapper; plot styling within `pytweezer-gui-design`; tests per `write-tests`.
