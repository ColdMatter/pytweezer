# Trap-site finder, array saved parameters and tweezer viewer

## Intent

Today every tweezer-array measurement (`TweezerArrayMeasurement` in
`pytweezer/experiments/tweezer_array.py`) finds the trap sites itself in
`prepare()`: it averages calibration shots and lowers a brightness threshold
from the brightest pixel in steps of 100 until exactly N spots appear
(`analysis.detect_trap_sites`). When no step gives exactly N, it falls through
to every noise peak: rid 110 (`OptimiseTweezerImaging`, 2026-11-02) failed
with "found 4978 trap sites in the calibration image, not 64". Each optimiser
run also spends its first shots re-finding sites that have not moved.

Instead, finding the sites becomes its own optimiser, run by hand whenever the
user chooses. It uses the array geometry the rig knows (positions, not just a
count) and fits it onto the camera image, so every site is placed even when
some are dim or empty. The user accepts the result in the Optimisers tab like
any other proposal; it is stored as a saved parameter, an array with
unambiguous site indexing, and every other tweezer script uses it. A new
**tweezer viewer** shows the live camera image with the accepted sites drawn
as squares, so the user can check at a glance that they are in a sensible
place.

## Goals

- A trap-site finder optimiser that fits the rig's geometry (or a default
  8×8, 5 µm square) to an averaged calibration image, free in position,
  rotation and scale, and proposes the site positions.
- Saved parameters can hold number arrays.
- Accepted sites are the single source of trap sites for the tweezer
  experiments and the live loading analysis.
- A tweezer viewer: the image viewer plus a square at each accepted site.

## Non-goals

- Running the finder automatically (on SLM change, on a timer, before each
  optimiser). The user runs it when they feel like it.
- Tracking drift between finder runs.
- Editing array saved parameters by hand in the GUI.
- Changing how the rig makes or reports arrays.

## Anti-goals

- Sites silently changing: a new fit must never be used until accepted.
- Ambiguous indexing: site *i* must always mean the same trap, the rig's site
  *i*, whether or not it was bright in the calibration image.
- Changing the camera ROI silently shifting every site.

## Decided by the user

- The finder asks the rig for the array geometry
  (`Rearrangement.get_array()`, `pytweezer/coordinators/rearrangement.py`) and
  falls back on a default square grid, 8×8 with 5 µm spacing, when the rig
  does not know which array is on the SLM.
- It uses the given geometry: the grid shape is fitted to the image, with
  position, rotation and scale free (scale because the camera's
  `hamcam_px_per_um = 2.07` in `analysis.py` may be slightly off).
- It records the site locations.
- Positions are saved as an array with unambiguous site indexing.
- Saved parameters are extended to hold arrays (rather than storing a pointer
  to a run file).
- The proposal waits for the user to accept it in the Optimisers tab.
- Other scripts use the accepted sites.
- The tweezer viewer is based on the image viewer
  (`pytweezer/GUI/viewers/image_monitor.py`) and draws the sites as squares.
  Its purpose is to verify the sites are in a sensible location.

## Design

### Array saved parameters

`pytweezer/experiment/saved_parameters.py`: `Value` gains a number array,
stored in the JSON book and the JSONL change log as nested lists of finite
floats. Rules, extending `_checked`:

- A key keeps its type: an array key only takes arrays, a scalar key only
  scalars.
- An array key keeps its number of dimensions and every axis length but the
  first, so `trap_sites` is always N×2 while N may change with the rig's
  array. *(my call)*
- Every element must be finite.

Arrays come back to callers as `numpy.ndarray` from `get_saved_parameters()`
and the client (`pytweezer/experiment/client.py`). The RPC wire format is the
builder's choice, as long as the manager never receives numpy types it can't
serialise.

`pytweezer/GUI/saved_parameters.py`: an array shows as a read-only summary
("64 × 2 array"); its detail sheet may show the values but does not edit
them. The history shows array changes the same way. *(my call)*

### Finder optimiser

A new class `FindTrapSites` in `pytweezer/experiments/tweezer_array.py` (or a
sibling module), built on the existing shot machinery (`LoadTweezers.shot`,
`TweezerArrayMeasurement._batch`), plus a new optimiser kind in
`pytweezer/experiment/optimisers.py` for optimisers whose proposals come from a
fit rather than from a search over settings. The existing kinds (`StagedScan`,
`BayesianOptimisation`) assume an objective and per-point settings; the new
kind proposes keys it declares directly, and has no objective. Its name and
exact shape are left to the builder, but it must appear in the Optimisers tab
with its proposals acceptable through the existing accept flow
(`proposed.<key>` constants).

Run:

1. Take `calibration_shots` shots with the tweezers off during the background
   drop (as `_calibrate` does now: `BackgroundDrop=1`,
   `tTweezerSetVVA0/1=0`), and form the mean background-subtracted image
   (`an.tweezer_show_bg_subtracted`).
2. Get the geometry: `rig.get_array()`'s `x_um`/`y_um` if it returns an array,
   else the default 8×8, 5 µm square centred on the origin. Record which
   (`sites_from`: `"rig"` or `"default"`). The default's size and spacing are
   form arguments defaulting to 8, 8 and 5 µm. *(my call)*
3. Map µm to sensor pixels with a fixed mirror then a similarity transform:
   the camera sees the hologram's x mirrored and y not, so
   `x_img = -s·x' + x0`, `y_img = s·y' + y0`, where `(x', y')` is the
   geometry rotated by θ. Start from `s = hamcam_px_per_um` and θ = −2.5°,
   the convention `analysis.index_grid_positions` uses to pair camera spots
   with hologram sites in the rearrangement notebooks. *(my call, from that
   code; the mirror is fixed, never fitted, since a square grid fits equally
   well mirrored and only the indexing would be wrong. The viewer's site
   labels are how the user checks it.)* Fit `s`, θ, `x0` and `y0` so the
   template sites sit on the bright spots of the
   (top-hat-filtered) mean image. The fitting method is the builder's: e.g.
   a coarse offset search by cross-correlating the image with the template
   rendered as spots, then a local optimisation of all four parameters on the
   summed intensity in a window at each site. The fit must place every site,
   bright or not.
4. Sum each site's photons per calibration shot and fit the occupancy
   threshold as `_calibrate` does now (`an.get_array_loading_statistics`).
5. Record the mean image, the geometry used, the fitted scale, rotation and
   offset, the site positions, the threshold and the calibration fidelity.
6. Propose:
   - `trap_sites`: N×2 float array of `(y, x)` positions in **full-sensor
     pixels**, row *i* = site *i* of the geometry (the rig's order; for the
     default grid, row-major from the top-left of the image).
   - `trap_threshold`: the fitted occupancy threshold, photons. *(my call:
     sites and threshold were calibrated together before; keeping them
     together keeps the measurements equivalent.)*
   - `trap_window_size` and `trap_feature_size`: the summing-window side and
     top-hat size (px, ints) the threshold was fitted with. They move from
     `TweezerArrayMeasurement`'s arguments to the finder's; measurements use
     the saved values, so a threshold is never applied to sums it was not
     fitted on. *(my call)*
   - `trap_geometry`: N×2 array of the `(x_um, y_um)` the sites were fitted
     against (the rig's, or the default grid's). *(my call)*

   The Optimisers tab's accept creates a proposed key that does not exist
   yet (group "Tweezer array"; units "px", "photons", "px", "px", "µm"),
   rather than refusing it: accepting is already an explicit human act, and
   an array key has no sensible hand-made first value. *(my call)*

Fail clearly (and propose nothing) when the fit cannot converge or when the
fitted scale is far from `hamcam_px_per_um` (the bound is the builder's
choice, e.g. ±20 %).

Full-sensor pixels: a site at `(y, x)` in an image read with ROI origin
`(roi_y0, roi_x0)` is at `(y - roi_y0, x - roi_x0)` in that image. The finder
adds its own ROI origin to what it fits. *(my call)*

### Using the accepted sites

`TweezerArrayMeasurement.prepare()` stops calibrating. It reads `trap_sites`
and `trap_threshold` from saved parameters and:

- fails at once, with a message saying to run and accept the finder, if
  either is missing;
- fails at once if the rig reports an array whose `(x_um, y_um)` differ from
  `trap_geometry` (count, or any position by more than 0.1 µm), so sites
  fitted to another array are never used; when the rig does not know its
  array, the saved sites are used unchecked, as the user is running without
  the rig's mask; *(my call)*
- converts the sites to its ROI's pixels, rounds them to whole pixels (sites
  are summed by slicing) and fails if any window falls outside the ROI;
- records the sites it used (ROI pixels, as `trap_sites` is recorded now, so
  `pytweezer/analysis/tweezer_histogram.py` keeps working) and the threshold.

The `calibration_shots`, `sites`, `initial_threshold`, `window_size` and
`feature_size` arguments move to the finder. Every subclass (`TweezerImaging`, `OptimiseTweezerImaging`,
`MolassesRecapture`, `OptimiseMolassesShims`) inherits the change.

The live loading analysis (`pytweezer/analysis/tweezer_loading.py`) also uses
the accepted `trap_sites` instead of finding sites itself, converting them to
each frame's ROI pixels, with the same `trap_geometry` check against the rig
(it reports the mismatch in its output header instead of failing). *(my
call: "other scripts use these trap sites"; two independent site finders
would disagree.)* It keeps fitting its own threshold per window, as now.

### ROI origin in camera frames

Frame headers on the image streams don't currently say which ROI they were
read from. Cameras add `roi_x0` and `roi_y0` (the last `set_roi`) to every
broadcast frame's header (`pytweezer/drivers/camera_base.py`:
`acquire_n_frames` and the live broadcast). Consumers treat a missing origin
as `(0, 0)`.

### Optimisers tab

`pytweezer/GUI/optimisers.py` shows the finder's run as a plot of its mean
image with a square at each fitted site, in place of the objective plots the
search kinds show, so the fit can be judged before accepting. Its
`trap_sites` proposal shows as an array summary, and the run's fitted
threshold as an ordinary scalar proposal.

### Tweezer viewer

A new applet, `pytweezer/GUI/viewers/tweezer_monitor.py` (name is the
builder's), derived from `image_monitor.py`:

- The live camera image, with the image viewer's colour map, levels and ROIs.
- A square at each accepted `trap_sites` position, converted to the frame's
  ROI pixels from its header.
- It reads `trap_sites` from saved parameters when it starts and again when
  the saved parameters change (the manager publishes a
  `saved_parameters_version`, as the Optimisers tab uses).
- Squares are `trap_window_size` pixels across, the box the experiments sum.
  One property, `show_site_labels` (number each square with its index,
  default off). *(my call)*
- Accepted sites only; with none saved, it shows the image and a line saying
  so.

## Left to the builder

- The optimiser kind's name and how it plugs into `optimiser_schema` and the
  catalogue.
- The fitting algorithm and its convergence criteria, within the decisions
  above.
- Array wire format over RPC and in the manager's JSON.
- Moving or deleting `detect_trap_sites` once nothing calls it.
- Tests, per the `write-tests` skill: at least the array saved-parameter
  rules, the fit recovering a known transform from a synthetic image with
  some sites dark, and the ROI conversion.
