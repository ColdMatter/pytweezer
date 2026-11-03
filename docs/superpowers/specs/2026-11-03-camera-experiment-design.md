# CameraExperiment: cameras and saved parameters as declarations

## Intent

Many experiments are "set up cameras A, B, C; run MOTMaster script X; get
N_A, N_B, N_C frames from the cameras". Today each one copies the same
camera setup, arm/acquire/stop code and recording from the last
(`LoadTweezers` is the model), and `LoadTweezers.prepare` also carries a
block that reads saved parameters and merges them into every sequence run.
The aim is that such an experiment is written as declarations, with code only
for what is specific to it (analysis, driver extras).

## Goals

1. **`CameraExperiment(MotMasterExperiment)`**, a concrete base class
   (not mixins) that runs the cameras around the MOTMaster sequences.
2. **Cameras are declared as class attributes**, like `MotMaster(...)`:

   ```python
   class PureRecipe(CameraExperiment):
       rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
       ham = CameraShots("Rb HamCam", frames=2, exposure=20e-3, roi=(0, 512, 0, 512))
       thor = CameraShots("Rb ThorCam", frames=1)
       caf = CameraShots("CaF ThorCam", frames=3)
   ```

   `CameraShots` is a `Device` subclass (so it is declared, reserved and in the
   schema like any device); the class attribute name is the camera's key.
   Declarations stay class attributes because the manager imports the module,
   without running it, to list arguments and reserve devices.
3. **Camera settings are arguments** in the Experiments form, editable and
   scannable (`ListAxis(argument="ham.exposure", ...)`), with the declaration's
   values as defaults. Named with dots, as MOTMaster script parameters are:
   `<camera>.frames`, `<camera>.exposure`, `<camera>.roi_x0`, `.roi_width`,
   `.roi_y0`, `.roi_height`. They are always shown in the form, not behind the
   search box.
4. **Generic loops over `self.cameras`** (a dict `{attribute name: device client}`):
   - `prepare`: apply each camera's ROI, exposure, trigger source and
     `setup_acquisition(frames, mode="snap")`, then call the hook
     `configure_camera(name, camera)` for driver extras (EM gain, sensitivity,
     `set_external_exposure_mode`).
   - `take_shot(**overrides)`: start acquisition on every camera, run the
     sequences, read each camera's `frames` frames, stop every camera. Returns
     `{name: (frames, H, W) array}`. Cameras are armed before the master fires.
   - default `run_point`: `take_shot()`, then record each camera's frames as
     one stack, `<name>_images`. Users index into the stack; there are no
     per-frame labels.
   - `record_frames(frames)` does that recording for a `run_point` that wants it
     alongside its own analysis.
   - Frames are broadcast with the `rid` header, as `LoadTweezers.shot` does today,
     so applets keep working.
5. **Saved-parameter binding on the `MotMaster` declaration**, generic for any
   script:

   ```python
   rb = MotMaster(
       "Rb MotMaster",
       script="RbTweezerBasic2026_2",
       saved=(*SAVED_UNDER_OWN_NAME, {"tImgCoolVCO1": "tImgCoolVCO", "tImgCoolVCO2": "tImgCoolVCO"}),
   )
   ```

   - A bare name in `saved` means the saved key has the same name as the script
     parameter; a dict entry maps a script parameter to a differently named
     saved key (several parameters may share one key).
   - `MotMasterExperiment.prepare` resolves the bindings once from the manager,
     with `LoadTweezers.prepare`'s present errors (manager unreachable raises
     `RuntimeError`; a missing saved key raises `KeyError` naming all missing
     keys), and records the values used as `saved_sequence` (`/constants`).
   - `run_sequences` applies them automatically. Precedence, highest first: an
     override passed to `run_sequences`; a script parameter set in the form
     (declared or by dotted name, including a scanned one); the saved value; the
     script's own default. `saved_sequence` holds only the saved values that
     were not beaten by the form, as today.
   - No experiment copies the reading/merging block again.
6. **`LoadTweezers` is rewritten** as a `CameraExperiment`:
   `ham = CameraShots("Rb HamCam", frames=2, ...)`, `rb = MotMaster(..., saved=...)`.
   It keeps the `Rearrangement` rig device, trap-site loading and checks,
   `signal_counts`, `loading`, `first_/second_occupied`, `first_/second_counts`,
   `take_shots` and the `analyses`, now reading the two frames from
   the `take_shot()` stack (`frames["ham"][0]` and `[1]`). Its saved files do
   not change: it overrides `run_point` and records `first_image`,
   `second_image` (and `first_images`, `second_images` per batch in
   `take_shots`) exactly as now, still gated by `save_raw_images`, and does not
   also write `ham_images`. `save_raw_images` and `save_tweezer_counts` stay
   `LoadTweezers` arguments. `tweezer_histogram.FRAMES` and the existing
   measurement files therefore keep working. `TweezerArrayMeasurement` and the
   optimisers that subclass it keep working. The module-level
   `SAVED_SEQUENCE_PARAMETERS` is replaced by the `saved=` declaration (the
   rewrite chooses how to keep the single list of names importable).

## Non-goals and anti-goals

- No mixins (a concrete base class first; split later if a real need to
  combine capabilities appears).
- Not the `globalParameters.json` overlap in `saved_parameters.py` (the FIXME);
  that is a separate decision.
- Not editing notebooks. List any that import `SAVED_SEQUENCE_PARAMETERS` or
  `LoadTweezers` internals that break, for the user.
- Not changing camera drivers or `camera_base.py`; the generic loops use the
  `Camera` interface every driver shares (`set_roi`, `set_exposure_time`,
  `set_trigger_source`, `setup_acquisition`, `start_acquisition`,
  `acquire_n_frames`, `stop_acquisition`). Driver extras go in
  `configure_camera`.
- Anti-goal: a design where an experiment author must still write per-camera
  loops or the saved-parameter block to get the standard behaviour.

## Constraints

- No hardware at import or class definition (the manager imports modules to
  list arguments).
- Every camera used must be declared (reserved, `UndeclaredDeviceError`
  otherwise); `CameraShots` being a `Device` gives this.
- Tests follow the design: update existing tests broken by the rewrite
  (`tests/test_tweezer_array.py` uses `SAVED_SEQUENCE_PARAMETERS`; others may
  touch `roi_x0` and friends) instead of keeping the old design. Do not add new
  test files or cases unless the user asks.
- Docstrings Google style, UK English, no change history, comments only for
  the surprising (project CLAUDE.md). Use `get_logger`.
- The `add-experiment` skill describes `MotMasterExperiment` and
  `LoadTweezers`: minor edits (names, signatures) go in with the code;
  adding a camera section is a *significant* skill change, so propose the text
  to the user and wait.

## Decided by the user

- Base class name `CameraExperiment`; concrete base class, not mixins.
- Camera settings are form arguments (editable, scannable), not fixed in the class.
- Frames recorded as one stack per camera, `<name>_images`, indexed by position.
- Cameras kept in a dict, with loops doing setup, arm, run and read.
- Saved-parameter binding is generic, not copied into `prepare`.
- `LoadTweezers` is rewritten on the new base.

## My calls (not yet confirmed)

- The declaration is called `CameraShots`; the hook `configure_camera`; the
  module `pytweezer/experiment/camera_experiment.py`.
- Saved binding is a `saved=` keyword on `MotMaster`, accepting bare names and
  a dict of renames.
- Precedence of saved values below form values, above script defaults (the
  current behaviour of `LoadTweezers`).
- Frames are broadcast with the `rid` header by default.
- The generic `<name>_images` records belong to `CameraExperiment`'s default
  `run_point` only. `LoadTweezers` keeps its on-disk names (see goal 6), so no
  analysis, optimiser or saved measurement file changes.
- `roi=(x0, width, y0, height)`, the order `set_roi` takes.
- A subclass can lock a camera setting: `CameraShots(..., frames=2,
  locked=("frames",))` leaves it out of the form and fixes it at the declared
  value. `LoadTweezers` locks `ham.frames`, as it needs exactly two frames. Other
  settings of a locked camera stay editable.
- `LoadTweezers`' `roi_x0`, `roi_y0`, `roi_width` and `roi_height` become
  `ham.roi_x0` and so on. `sensitivity` stays a `LoadTweezers` argument,
  applied in `configure_camera`. Saved recipes that name the old arguments stop
  validating (`unknown_arguments` in `recipes.py` reports them); write a
  migration only if it is trivial.

## Left to the builder

- How dotted camera arguments are generated and exposed through `arguments()`,
  `to_schema`, the form and `extra_argument` machinery (they must be always
  shown, unlike undeclared MOTMaster parameters).
- Which settings are optional: HamCam is exposed by the external trigger pulse
  (`set_external_exposure_mode`), so `exposure=None` (do not call
  `set_exposure_time`) and `roi=None` must be allowed. Check each driver's
  `set_trigger_source` and choose the default trigger source accordingly.
- Whether `frames` is scannable per point (changing it needs re-arming the
  camera) or fixed at `prepare`.
- Exact shape and storage of `saved_sequence`, error text, and how the
  default `run_point` names records.
