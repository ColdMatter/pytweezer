# Rearrange tweezers: a queued rearrangement experiment

## Why

Rearrangement runs today only from the `rearrangement-optimisation-RnD`
notebooks (`tests/rearrangement-data-acquisition.ipynb`,
`tests/rearrangement-testing-psf_better_timing.ipynb`): they build the
coordinator in-process with their own phase-mask generator, start
`arm_rearrangement` in a thread, sleep 0.1 s, fire MOTMaster by hand, and pickle
the images. None of it goes through the queue, so nothing is reserved, recorded
in a measurement file or repeatable from a recipe.

## Goal

A queued experiment, `RearrangeTweezers`, that characterises rearrangement: one
shot per point (scan `repetitions` for many), each recording the occupancy image,
the two follow-up images, which atom went where, how full the target array is,
and how long the move took.

## Non-goals

- Handing a filled array to later experiments. `RearrangeTweezers` is built so a
  later design can extend it for that.
- The PSF matched filter (`SiteScorer`) for occupancy. A later design; it changes
  the calibration workflow.
- The preload-and-hardware-trigger upload (`arm_rearrangement_preload_trigger`):
  it needs a pulser object `main` has no device for.
- Results-tab analyses for rearrangement (`analyses = ()` for now).
- Changing `initialise`/`arm_rearrangement` and their variants: the RnD
  notebooks keep using them with an in-process generator.

## Session order

`SlmTweezerArray` (puts the initial array on the SLM through the rig) →
`FindTrapSites` (fits its sites on the HamCam, proposes `trap_*`) →
`RearrangeTweezers`.

## One shot

The rig shows the initial mask and blocks reading the HamCam; MOTMaster's
`RbTweezerBasic2026_2_Assemble` loads the tweezers and triggers three HamCam
frames:

1. the occupancy image, before the move;
2. a hold while the rig rearranges;
3. an image straight after the move, scored for target fill;
4. a second image of the target array, separating losses in imaging from losses
   in the move.

Three frames per shot, fixed by the script.

## The rig (`pytweezer/coordinators/rearrangement.py`)

### `make_array` remembers what a move needs

- `array_hologram.array_hologram` also returns the optimised trap phases as
  `phases` (numpy), beside `weights`, `x_um`, `y_um`.
- The rig keeps the array's optimised terms `(w, φ, x, y)` and how its mask was
  built: the generator, the envelope (or `None`), blaze, Fresnel focal length,
  Zernike flag, pupil radius and dump grating.
- `get_array()` is unchanged; it returns `None` once anything else writes the
  SLM, and both new methods refuse then.

### `prepare_rearrangement(...) -> dict`

Keyword arguments:

- target geometry: `kind` (`"square"` or `"kagome"`), `rows`, `columns` (square),
  `sites` (kagome, via `generate_kagome_lattice`), `spacing_um`, `angle_deg`;
- target optimisation: `iterations`, `damping`, `phase_randomness`;
- occupancy scoring: `occupancy_sites` (`{index: (y, x)}` in ROI pixels, in the
  initial array's order), `threshold_photons`, `window_size`, `feature_size`;
- `follow_up_frames` (2).

It builds the target terms with the initial array's generator, centred on the
blaze spot like the initial array and weighted by the same envelope, optimises
them with `superposition_optimization`, and stores everything a shot needs.
Returns `x_um`, `y_um`, `weights`, `uniformity`, `min_max_ratio`, `mse` (numpy
and floats only).

Raises if there is no GPU (the existing clear error), if `get_array()` is `None`,
or for an unknown `kind`.

### `rearrange(*, d0_um, profile, header) -> dict`

One shot:

1. Show the initial mask; `start_acquisition()` on the HamCam (set up by the
   experiment for a 3-frame snap).
2. Read frame 0 (`acquire_n_frames(1)`); occupancy is the top-hat box sum at the
   occupancy sites, in photons (`array_loading.photons_at_sites`, as
   `LoadTweezers.site_photons`), above the threshold.
3. With no atoms, skip planning (the planner raises on an empty occupancy;
   `n_frames` = 0). Otherwise `plan_rearrangement(..., background=...)` and play
   `iter_planned_sequence(plan, pupil=...)` through the existing paced, pipelined
   upload, so the first move frame is built on exactly the layers of the
   displayed mask.
4. Read frames 1 and 2 (`acquire_n_frames(2, start_frame=1)`). In a `finally`:
   stop the acquisition and show the initial mask again.
5. Broadcast the three frames on the camera's stream with `header` (the task's
   `rid`).

Returns: `frames` `(3, H, W)`; `occupied` (bool per initial site);
`destination` (int per initial site: the target index its atom was sent to, −1
if empty or discarded); `n_frames`; `occupancy_ms`, `calculation_and_upload_ms`,
`total_rearrangement_ms`.

`destination` comes from the plan's `moving_idx`/`final_idx` through a pure
numpy helper; occupancy scoring is a pure helper too, so both are tested without
a GPU.

### Phase-mask generator (`pytweezer/helper_scripts/phasemask.py`)

`OptimisationBasedPhasemaskGeneratorGPU` only:

- `plan_rearrangement(..., background=None)`: the phase the frames are built
  on; `None` keeps today's `fresnel + blaze + zernike`.
- `iter_planned_sequence(plan, *, to_host=False, pupil=None)`: yields the
  plan's frames; `pupil=(radius_um, dump_dx_um, dump_dy_um)` replaces the phase
  outside the pupil with the dump grating, as `array_hologram.apply_pupil` does.

Existing callers are unaffected. (Already done on this branch: the rearrangement
functions copy their initial terms before the in-place update kernel walks them.)

## The experiment (`pytweezer/experiments/rearrange_tweezers.py`)

`RearrangeTweezers(LoadTweezers)`:

- `rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2_Assemble",
  saved=LoadTweezers.rb.saved)`; inherits `hamcam`, `rig`, the camera arguments,
  `save_raw_images`, `save_tweezer_counts`, and the trap-site loading and
  geometry check.
- `analyses = ()`.

Arguments, SI with display units as `SlmTweezerArray`:

| Group | Argument | Default |
| --- | --- | --- |
| Target array | `target_kind` (Choice: square, kagome) | kagome |
| | `target_rows`, `target_columns` | 6, 6 |
| | `target_sites` | 70 |
| | `target_spacing` (µm) | 5 µm |
| | `target_angle` (deg) | 2.5° |
| | `target_iterations`, `target_damping`, `target_phase_randomness` | 5000, 0.5, 1.0 |
| Rearrangement | `d0` (µm per frame) | 0.5 µm |
| | `profile` (Choice: linear, minimum_jerk) | linear |
| | `arm_delay` (s) | 0.1 s |

Scanning: `d0`, `profile`, `arm_delay`, `hamcam_gain` and script parameters scan
(passed or re-sent per shot). Target-array arguments and the ROI are refused in
`planned_points`.

### `prepare`

1. `super().prepare()`; refuse if `trap_sites` is `None` (with the inherited
   reason). Set the HamCam up for `setup_acquisition(3, mode="snap")`.
2. Refuse if `rig.get_array()` is `None`.
3. Least-squares affine fit from the array's `x_um`/`y_um` to `trap_sites` (ROI
   pixels, same order). Refuse if the RMS residual exceeds `trap_window_size / 2`.
4. `rig.prepare_rearrangement(...)`; map the target µm positions to pixels with
   the fit; refuse if any target's summing window leaves the ROI.
5. Record constants: `target_sites` (px, `(targets, 2)`), `target_x_um`,
   `target_y_um`, `target_weights`, `target_uniformity`, `target_min_max_ratio`,
   `target_mse`, `affine_residual_px`; the inherited `trap_sites`, `threshold`,
   `trap_window_size`, `trap_feature_size` stay.

### `run_point`

Re-send `hamcam_gain` if it changed, then

```python
shot, _ = run_parallel(
    lambda: self.rig.rearrange(d0_um=..., profile=..., header={"rid": self.rid}),
    after(self.arm_delay, self.run_sequences),
)
```

Records per point (fixed shapes, so shots stack):

| Name | Shape / unit |
| --- | --- |
| `rearrangement_images` (with `save_raw_images`) | `(3, H, W)` |
| `occupied` | bool `(sites,)` |
| `loading` | fraction |
| `destination` | int `(sites,)`, −1 for none |
| `target_filled_1`, `target_filled_2` | bool `(targets,)`, frames 1 and 2 |
| `fill_fraction_1`, `fill_fraction_2` | fraction |
| `initial_counts` (with `save_tweezer_counts`) | `(sites,)` counts, frame 0 |
| `target_counts` (with `save_tweezer_counts`) | `(2, targets)` counts, frames 1 and 2 |
| `n_frames` | int |
| `occupancy_ms`, `calculation_and_upload_ms`, `total_rearrangement_ms` | ms |

Target fill uses the same scoring as occupancy: top-hat box sum at the target
pixels, in photons, above the `FindTrapSites` threshold.

## Failures

- **MOTMaster fails, or fires before the camera is armed:** the rig blocks
  reading frame 0 until the HamCam timeout (20 s in CONFIG), stops the
  acquisition and restores the initial mask in `finally`, and raises. The task
  fails; a frame-0 timeout reads "no occupancy frame: MOTMaster may have fired
  before the camera was armed (arm_delay)".
- **More atoms than targets, or fewer:** the existing pairing decides; surplus
  atoms' traps are switched off and recorded with `destination` −1, unfilled
  targets show in `target_filled_*`.
- **No atoms loaded:** no planning, `n_frames` 0, frames 1 and 2 still read and
  recorded.
- **Interrupt or abort:** a shot in flight cannot be cut short (`rearrange` is
  one synchronous RPC call); the task stops after it.
- **No GPU** (simulation, a non-GPU host): `prepare_rearrangement` raises the
  coordinator's error and the task fails in `prepare`.

## Tests

Following `write-tests`:

- `tests/test_rearrange_tweezers.py`, with fakes at the device edges (rig,
  HamCam, MOTMaster) as in `tests/test_tweezer_array.py`:
  - a shot runs the rig and MOTMaster, MOTMaster after `arm_delay`, and records
    every field with a fixed shape;
  - target sites land at the right pixels under a known affine map;
  - refusals: no array on the SLM, sites that don't fit the array, a target
    window outside the ROI, a scan of the target geometry or ROI;
  - a failed `Go` fails the task.
- `tests/test_rearrangement.py`: `make_array` keeps the array's terms and layers;
  the `destination` and occupancy helpers; `prepare_rearrangement` and
  `rearrange` with the GPU parts faked as the existing tests fake them; a
  GPU-only check that the first move frame equals the `make_array` mask.

## Skills and docs

Approved with this spec:

- `add-experiment`: step 5 in "Tweezer array measurements".
- `pytweezer-device-framework`: the rig's new methods.
- `docs/notes/rearrangement_coordinator.md`: the queued path.

## Breaking changes

None. `array_hologram`'s result gains `phases`; existing coordinator methods are
unchanged.

## Left to the builder

Internal structure of the rig's stored state; helper names; error wording; how
the affine fit is solved.
