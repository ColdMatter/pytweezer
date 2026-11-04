# CameraExperiment: cameras and saved parameters as declarations

## Intent

Many experiments are "set up cameras A, B, C; run MOTMaster script X; get
N_A, N_B, N_C frames from the cameras". Today each one copies the same
camera setup, arm/acquire/stop code and recording from the last
(`LoadTweezers` takes a fixed number of frames per shot; `CaFFluorescence`
takes whatever a free-running camera caught), and `LoadTweezers.prepare` also
carries a block that reads saved parameters and merges them into every
sequence run. The aim is that such an experiment is written as declarations,
with code only for what is specific to it (analysis, driver extras).

## Goals

1. **`CameraExperiment(MotMasterExperiment)`**, a concrete base class
   (not mixins) that runs the cameras around the MOTMaster sequences.
2. **Cameras are declared as class attributes**, like `MotMaster(...)`:

   ```python
   class PureRecipe(CameraExperiment):
       rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
       hamcam = CameraShots("Rb HamCam", frames=2, exposure=20e-3, roi=(0, 512, 0, 512))
       thorcam = CameraShots("Rb ThorCam", frames=1, stream=True)
       orca = CameraShots("CaF ORCA", frames=3, binning=2, gain=1)
   ```

   `CameraShots` is a `Device` subclass (so it is declared, reserved and in the
   schema like any device); the class attribute name is the camera's key.
   Declarations stay class attributes because the manager imports the module,
   without running it, to list arguments and reserve devices.
3. **Camera settings are arguments** in the Experiments form, editable and
   scannable (`ListAxis(argument="hamcam.exposure", ...)`), with the
   declaration's values as defaults. Named with dots, as MOTMaster script
   parameters are: `<camera>.frames`, `<camera>.exposure`, `<camera>.binning`,
   `<camera>.gain`, `<camera>.roi_x0`, `.roi_width`, `.roi_y0`, `.roi_height`.
   They are always shown in the form, not behind the search box.
   - `binning` and `gain` are optional: `None` (the default) means the
     experiment never sets them and the camera keeps what it has. They use
     `Camera.set_binning` and `Camera.set_gain`. `gain` is in the camera's own
     unit: dB on the ThorCams, the contrast-gain index on the ORCAs, and on the
     HamCam its sensitivity, which takes effect only once EM gain is enabled.
   - The ROI is in unbinned sensor pixels, as pylablib's DCAM and Thorlabs
     backends take it; a camera binning `b` returns frames of
     `(roi_height // b, roi_width // b)` and may round the ROI to a multiple of
     `b`. The origin it used (`self.roi_origins[name]`) is in sensor pixels too.
4. **Two acquisition modes per camera:**
   - **Snap** (the default), as `LoadTweezers` does: `setup_acquisition(frames,
     mode="snap")` at `prepare`, then per shot `start_acquisition`, run the
     sequences, `acquire_n_frames(frames)`, `stop_acquisition`.
   - **Stream** (`CameraShots(..., stream=True)`), as `CaFFluorescence` does:
     `camera.arm()` at `prepare` (and again, a no-op while it runs, before every
     shot, so a shared camera another task stopped comes back). Each shot notes
     `frames_acquired()` as a mark, runs the sequences, waits with
     `pytweezer.drivers.shot_frames.settle_frame_counts` until every streaming
     camera's count has been still for `frame_settle_s` (at most
     `frame_timeout_s`), then reads `read_frames_from(mark)`. The camera keeps
     running after the shot and the task. Stream mode is needed for the
     ThorCams: their driver ignores `mode`, so a snap loop would re-read stale
     ring-buffer frames from the second shot on.
5. **Generic loops over `self.cameras`** (a dict `{attribute name: device client}`):
   - `prepare`, per camera, in this order: binning, ROI (`set_roi` keeps the
     binning; `set_binning` can reset the ROI), exposure, trigger source, the
     hook `configure_camera(name, camera)` for driver extras (on the HamCam:
     `set_external_exposure_mode`, enabling EM gain), gain, then
     `setup_acquisition(frames, mode="snap")` or, streaming, `arm()`. A
     streaming camera is stopped first if any of these is to be set. Settings
     left `None` are skipped.
   - `take_shot(**overrides)`: start every snap camera and mark every streaming
     one, run the sequences, read every camera's frames, and stop the snap
     cameras even on failure. Returns `{name: (frames, H, W) array}`. Cameras
     are armed before the master fires.
   - default `run_point`: `take_shot()`, then record each camera's frames as
     one stack, `<name>_images`. Users index into the stack; there are no
     per-frame labels.
   - `record_frames(frames)` does that recording for a `run_point` that wants it
     alongside its own analysis.
   - Snap frames are broadcast with the `rid` header, as `LoadTweezers.shot`
     does today, so applets keep working. Streaming cameras are not
     re-broadcast: a live one (the CaF ThorCam) already broadcasts every frame.
6. **Saved-parameter binding on the `MotMaster` declaration**, generic for any
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
     keys), and records the values used as `saved_sequence` (`/constants`),
     keyed `"<attribute>.<parameter>"` (e.g. `"rb.tMOTLoadRepVCO"`).
   - `run_sequences` applies them automatically. Precedence, highest first: an
     override passed to `run_sequences`; a script parameter set in the form
     (declared or by dotted name, including a scanned one); the saved value; the
     script's own default. `saved_sequence` holds only the saved values that
     were not beaten by the form, as today.
   - No experiment copies the reading/merging block again.
7. **`LoadTweezers` is rewritten** as a `CameraExperiment`:
   `hamcam = CameraShots("Rb HamCam", frames=2, binning=1, gain=600, roi=(110, 256, 220, 256), locked=("frames", "binning"))`,
   `rb = MotMaster(..., saved=...)`. It keeps the `Rearrangement` rig device,
   trap-site loading and checks, `signal_counts`, `loading`,
   `first_/second_occupied`, `first_/second_counts`, `take_shots` and the
   `analyses`, now reading the two frames from the `take_shot()` stack
   (`frames["hamcam"][0]` and `[1]`). It records the camera frames as
   `hamcam_images`, `(2, H, W)` per point (`(shots, 2, H, W)` per point in
   `take_shots`), replacing `first_image`, `second_image`, `first_images` and
   `second_images`; still gated by `save_raw_images`. `save_raw_images` and
   `save_tweezer_counts` stay `LoadTweezers` arguments. `LoadTweezers.sensitivity`
   is removed in favour of `hamcam.gain`. Binning is locked at 1 because the
   trap sites are saved in sensor pixels and placed in the frame by the ROI
   origin. The default analysis (`tweezer_histogram.FRAMES` and the code reading
   it), the optimisers, their tooltips and the tests move to index
   `hamcam_images`. Measurement files written before the change, which hold
   `first_image` and `second_image`, are not read by the new analysis.
   `TweezerArrayMeasurement` and the optimisers that subclass it keep working.
   The module-level `SAVED_SEQUENCE_PARAMETERS` is replaced by the `saved=`
   declaration (`LoadTweezers.rb.saved` is the importable mapping).
8. **`CaFFluorescence` is ported** onto `CameraExperiment`: its `shot()` becomes
   `take_shot()` on `thorcam = CameraShots("CaF ThorCam", frames=3, stream=True,
   trigger=None, locked=("frames",))`, one frame per MOTMaster iteration. Its
   background shot, laser setting and restore logic stay. `OptimiseMOTLasers`,
   `Rig.auto_mot` and the Optimisers tab keep working unchanged.

## Non-goals and anti-goals

- No mixins (a concrete base class first; split later if a real need to
  combine capabilities appears).
- Not the `globalParameters.json` overlap in `saved_parameters.py` (the FIXME);
  that is a separate decision.
- Not editing notebooks. List any that import `SAVED_SEQUENCE_PARAMETERS` or
  `LoadTweezers` internals that break, for the user.
- Not changing camera drivers or `camera_base.py`; the generic loops use the
  `Camera` interface every driver shares (`set_roi`, `set_exposure_time`,
  `set_trigger_source`, `set_binning`, `set_gain`, `setup_acquisition`,
  `start_acquisition`, `acquire_n_frames`, `stop_acquisition`, `arm`,
  `frames_acquired`, `read_frames_from`). Driver extras go in
  `configure_camera`.
- Not `ContinuousAcquisition` (`experiments/camera_continuous.py`), which leaves
  one chosen camera free-running and is unrelated. Its undotted `binning`,
  `gain` and `roi_*` arguments do not collide with the dotted camera arguments.
- Anti-goal: a design where an experiment author must still write per-camera
  loops or the saved-parameter block to get the standard behaviour.

## Constraints

- No hardware at import or class definition (the manager imports modules to
  list arguments).
- Every camera used must be declared (reserved, `UndeclaredDeviceError`
  otherwise); `CameraShots` being a `Device` gives this.
- Tests follow the design: update existing tests broken by the rewrite
  (`tests/test_tweezer_array.py`, `tests/test_tweezer_histogram.py`,
  `tests/test_caf_fluorescence.py`) instead of keeping the old design. Do not
  add new test files or cases unless the user asks.
- Docstrings Google style, UK English, no change history, comments only for
  the surprising (project CLAUDE.md). Use `get_logger`.
- The `add-experiment` skill describes `MotMasterExperiment` and
  `LoadTweezers`: minor edits (names, signatures) go in with the code;
  adding a camera section is a *significant* skill change, so propose the text
  to the user and wait.

## Decided by the user

- Base class name `CameraExperiment`; concrete base class, not mixins.
- Camera attribute names `hamcam`, `thorcam`, `orca`; records `<name>_images`
  (`hamcam_images`); `LoadTweezers.camera` becomes `hamcam`.
- Camera settings are form arguments (editable, scannable), not fixed in the
  class. `frames`, `roi_width`, `roi_height` and `binning` cannot be scanned
  (they change the recorded shape); `exposure`, `gain`, `roi_x0` and `roi_y0`
  can, and are re-applied before a shot when they change. `LoadTweezers`
  refuses `hamcam.roi_x0`/`roi_y0` scans, because its trap sites are placed in
  the ROI fixed at `prepare`.
- Snap and stream modes, as in Goal 4; snap is the default.
- `binning` and `gain` are optional `CameraShots` settings, applied in the
  order in Goal 5. `LoadTweezers.sensitivity` is replaced by `hamcam.gain`.
- Frames recorded as one stack per camera, `<name>_images`, indexed by
  position, including by `LoadTweezers`, its analyses and the optimisers.
- Cameras kept in a dict, with loops doing setup, arm, run and read.
- Saved-parameter binding is generic, not copied into `prepare`; `saved=` on
  `MotMaster`, bare names and a dict of renames; precedence as in Goal 6;
  `saved_sequence` keys are dotted.
- `LoadTweezers` is rewritten on the new base; `CaFFluorescence` is ported
  after it.
- Frames are broadcast with the `rid` header by default.
- Old measurement files are not supported by the updated analyses (a notebook
  is a record; no compatibility shim). Saved recipes naming the old arguments
  (`roi_x0`, `sensitivity`, ...) stop validating (`unknown_arguments` in
  `recipes.py` reports them); a migration only if it is trivial.
- `roi=(x0, width, y0, height)`, the order `set_roi` takes.
- A subclass can lock a camera setting: `CameraShots(..., frames=2,
  locked=("frames",))` leaves it out of the form and fixes it at the declared
  value. Other settings of a locked camera stay editable.
- No new tests; existing ones are updated to the design.

## Left to the builder (settled in the plan)

- How dotted camera arguments are generated and exposed through `arguments()`,
  `to_schema`, the form and `extra_argument` machinery.
- Which settings are optional: HamCam is exposed by the external trigger pulse
  (`set_external_exposure_mode`), so `exposure=None` and `roi=None` must be
  allowed, as must `trigger=None` (the CaF ThorCam's bulb trigger is set from
  its CONFIG entry and must not be overwritten).
- Where the stream settle timing lives, and whether a streaming camera that
  catches a different number of frames than `frames` is an error.
- Exact error text, and how the default `run_point` names records.
