---
name: tweezer-array
description: The Rb tweezer-array experiments and optimisers — SlmTweezerArray (an array on the SLM through the Rb Rearrangement Rig), LoadTweezers and its subclasses (loading, imaging survival, HamCam frames), the trap_* saved parameters and TrapSites scoring, FindTrapSites, and the TweezerOptimisation family (OptimiseTweezerImaging, OptimiseMolassesShims/Light, OptimiseMOTLight and their MinTime variants). Use whenever the user works on tweezer loading, trap sites, site occupancy, loading/survival/fidelity scores or thresholds, the SLM array or rearrangement rig from an experiment, tweezer histograms, the live tweezer loading analyses, or any optimiser that scores tweezer shots — even if they only say "optimise the molasses for loading" or "the sites are in the wrong place". For general experiment mechanics use add-experiment; for the optimiser mixins and the Optimisers tab use add-optimiser.
---

# Rb tweezer array

The experiments and optimisers that put an array on the SLM, load it from the
Rb MOT and score it on the HamCam. General mechanics (arguments, devices,
`record`, MOTMaster, saved parameters) are in add-experiment; the
`StagedScan`/`BayesianOptimisation`/`Fit` mixins and accepting proposals are in
add-optimiser.

## The workflow

1. **`SlmTweezerArray`** (`experiments/slm_tweezer_array.py`) puts a
   rectangular array on the SLM through `Rb Rearrangement Rig`
   (`rig.make_array(...)`, one mask per point) and broadcasts the simulated
   intensity, mask and optimisation convergence for the SLM Monitor applet. Its optics are
   `slm_*` saved parameters (SI): blaze, Fresnel lens and pupil as saved-backed
   arguments; generator, Zernike rows `[index, coefficient]` and depth envelope
   from `array_hologram.SAVED_KEYS`. The rig remembers the array:
   `rig.get_array()` gives `sites`, `x_um`, `y_um` (`None` if the rig didn't
   write the mask), and `rig.get_array_hologram()` the whole `make_array`
   result, which the SLM Monitor shows when it opens.
   Its arguments and mask step are the `SlmArray` mixin (`read_slm_settings`,
   `write_slm_array`, and `update_slm_array`, which rewrites the mask only when
   an SLM argument changes). `LoadTweezersSlmArray` mixes it into
   `LoadTweezers`; `MergeCaFTweezer` (`experiments/merge_caf_tweezer.py`) into
   one that runs `MERGE.CaFTweezer` linked on the CaF MOTMaster and
   records one HamCam and one CaFMOT Cam frame per shot;
   `MergeCaFTweezerHamCamBackground` runs MERGE.CaFTweezerHamCamBackground and
   also records a HamCam background frame. The HamCam set-up is
   the function `set_up_hamcam` in `load_tweezers.py`.
   `update_slm_array` records each point's `slm_array` (an index), and
   `record_slm_arrays()` (in `finish`) the constant `slm_arrays`: per array
   written, trap positions, weights, blaze and the simulated trap plane
   (`array_summary`).
   MergeCaFTweezer also records `hamcam_roi_origin` and has saved-backed
   `site_*` arguments (`caf_hamcam_centre_x/_y`, `caf_hamcam_px_per_m`,
   `caf_hamcam_angle`): the CaF imaging's own map from focal-plane µm to HamCam
   sensor pixels (the `geometry_to_pixels` model), separate from Rb's `trap_*`.
   Its Results-tab analysis `caf_tweezer_sites`
   (`analyses/default/caf_tweezer_sites.py`) places the array's sites with them
   and scores each shot's frame at the sites against empty positions
   (`analysis/site_photometry.py`).
2. **`FindTrapSites`** (`optimisers/find_trap_sites.py`, a `Fit` over
   `LoadTweezers`) fits the array on the camera and proposes the trap-site
   saved parameters. Rerun it after changing the array.
3. **`LoadTweezers`** and the optimisers measure with those sites.

## Trap sites

`TRAP_SITE_KEYS` (`experiments/load_tweezers.py`): `trap_sites` (full-sensor
pixels, `(y, x)`), `trap_threshold` (photons), `trap_window_size`,
`trap_feature_size` (top-hat size), `trap_geometry` (the array in µm it was
fitted to).

`LoadTweezers.prepare()` calls `read_trap_sites()`, which calls
`get_trap_site_positions(saved)` → `(TrapSites | None, reason)`. It moves the
sites into the ROI by the origin `set_roi` actually returned
(`self.roi_origin`), refuses sites whose summing window leaves the ROI, and
warns (but carries on) when `trap_geometry` isn't the rig's current array
(`same_geometry`; an unreachable rig is not checked). An empty reason means
"this experiment wants none": `FindTrapSites` overrides it to
`return None, ""`.

**Score frames only through `TrapSites`** (`analysis/array_loading.py`):
`counts`, `photons`, `occupied`, `fit_loading` (refits the threshold on a
batch → `BatchLoading` with `loading`, `fidelity`, `threshold`, `photons`) and
`survival`. Never inline the maths in an experiment.

## `LoadTweezers` and subclasses

`LoadTweezers` (`MotMasterExperiment`, `hamcam = Device("Rb HamCam")`,
`rig = Device("Rb Rearrangement Rig")`) takes one shot of two HamCam frames per
point.

- The sequence runs at saved parameters bound by `saved=` on its `MotMaster`
  (`LoadTweezers.rb.saved`; script parameter → saved key, see add-experiment);
  `use_motmaster_parameter_overrides=False` sends none of them, but the trap
  sites are still read.
- HamCam set-up (`set_up_hamcam`): binning 1 in LoadTweezers (sites are in
  sensor pixels; MergeCaFTweezer takes `hamcam_binning`, a Choice of 1, 2, 4), ROI `hamcam_roi_x0/_y0/_width/_height` (unbinned) and `hamcam_gain`
  (EM-gain sensitivity) are `scannable=False`.
- Per point: `signal_counts`; with sites, `first_occupied`/`second_occupied`
  (bool per site), `loading`, imaging `survival` (NaN if nothing loaded), and
  with `save_tweezer_counts` raw `first_counts`/`second_counts`. Frames go to
  `hamcam_images` (`(2, H, W)`) only with `save_raw_images`. Constants:
  `trap_sites` (ROI pixels), `trap_window_size`, `trap_feature_size`,
  `threshold`.
- Without sites it runs only if `save_raw_images` is on.
- `LoadTweezersMOTImage` adds the ORCA's frame of the MOT from the same shot,
  recorded as an `orca_*` cloud fit (add-experiment, "Cameras and cloud fits").
  `LoadTweezers` itself never touches the ORCA.
- `LoadTweezersNImages` runs the multiple-images script and records nothing;
  its frames are only broadcast live.
- `TrapFrequencyScan` (`experiments/trap_frequency.py`) modulates the tweezers
  between the two images at `modulation_frequency` (via the VCO voltage
  `tTweezerModulationVCF`, recorded as `modulation_voltage`); scan it and plot
  `survival` for parametric-heating dips. Its first analysis, `trap_frequency`
  (`analyses/default/trap_frequency.py`), pools survival per frequency from
  `first_occupied`/`second_occupied` (else `survival`), finds the dips, and
  takes the highest-frequency one at least `min_relative_depth` (default 0.5)
  as prominent as the deepest as `2 ν_r`, fitting a Gaussian to it.

`analyses = ("tweezer_histogram", "tweezer_photon_histogram")` for the Results
tab (`analyses/default/tweezer_histogram.py`); they read `first_counts`/
`second_counts` with `trap_sites` and `threshold`, else per-site photons, else
`hamcam_images`. They expect one shot per point.

## Batch-of-shots classes

`TweezerOptimisation.take_batch(**overrides)` and
`FindTrapSites.take_calibration_shots()` loop the shot themselves
(`start_shot`, `run_motmaster_sequence(**overrides)`, `wait_frames`). Their
`first_occupied`/`second_occupied`/`*_counts` are `(shots, sites)` per point,
and `save_raw_images` (off by default here) records `hamcam_images` as
`(shots, 2, H, W)`. They declare no analyses.

## The tweezer optimisers

Bayesian ones subclass `(TweezerOptimisation, LoadTweezers)`
(`optimisers/tweezer_optimisation.py`). `TweezerOptimisation` adds `shots`,
refuses to run without trap sites, and gives a generic `measure()`: each
setting goes to its dimension's `save_as` script parameter (`None` keeps the
saved value), plus the drop from `drop` — a class attribute naming the
argument holding the drop duration, or `None` for `BackgroundDrop=0`.
`score_batch()` returns `loading` and `fidelity` (threshold refitted on the
batch) and `survival` (against the accepted threshold).

- Objectives are penalty forms in `objective_of(scores, settings)`:
  `shortfall_penalty` (10) × each shortfall below `loading_threshold` (0.55) /
  `survival_threshold` (0.8). At 10 a threshold is soft (a shortfall of 0.1
  costs the whole time range); a near-hard one needs ~100.
- MOT and molasses optimisers come in pairs per module: the bare one searches
  at the saved time; its `…MinTime` subclass adds the time as a whole-number
  dimension (`load_time`, or `MOLASSES_TIME`) and scores
  `shortness(time, bounds)` less the penalties, proposing the shortest time
  that meets the thresholds.
- `OptimiseTweezerImaging`: spans around saved `tImgCoolVCO`/`tImgCoolVVA` (0–5 V), no
  drop, scored `fidelity_weight` (0.5) × fidelity + the rest × survival.
  Overrides `measure()` because one setting sets both imaging stages
  (`tImgCoolVCO1/2`).
- `OptimiseMolassesShims`: spans around `tMolassesShim{X,Y,Z}ccValue` (limits
  ±10 V), `drop = "drop_duration"`, survival less the loading penalty.
- `OptimiseMolassesLight`: both molasses stages' light,
  `tMol{Cool,Rep}{VCO,VVA}{1,2}` (eight dimensions, 0–5 V), scored like the shims.
- `OptimiseMOTLight`: MOT-loading `tMOTLoad{Cool,Rep}{VCO,VVA}` (0–5 V) for
  the best `loading`, no drop. Its MinTime variant's `MOT_LOAD_TIME` name and
  limits are placeholders (FIXME in the module).

`FindTrapSites` takes `calibration_shots` with no drop, fits the rig's array
geometry (or a `default_rows` × `default_columns` grid) to the top-hat
filtered mean of their first images (`analysis/trap_sites.py`), fits the
threshold with a `TrapSites`, and proposes `TRAP_SITE_KEYS`. The Optimisers
tab's fit view draws its `calibration_image` and `roi_sites`.

## Live analyses

Shots tag their HamCam frames with `start_shot({"rid": self.rid})`.
`analyses/live/tweezer_loading.py` (loading statistics over the current task,
restarting when the rid changes) and `tweezer_occupancy_counts.py` read the
saved `trap_*` parameters and place the sites by each frame's
`roi_x0`/`roi_y0` header.

Tests: `tests/test_tweezer_array.py`, `test_trap_sites.py`,
`test_array_loading.py`, `test_slm_tweezer_array.py`,
`test_tweezer_histogram.py`, `test_tweezer_loading_plots.py`.
