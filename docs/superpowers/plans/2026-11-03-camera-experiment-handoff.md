# CameraExperiment: handoff to the implementation session

Written 2026-11-03 at the end of the brainstorming session. Read this first, then
the design (`docs/superpowers/specs/2026-11-03-camera-experiment-design.md`) and the
plan (`docs/superpowers/plans/2026-11-03-camera-experiment.md`).

## State

- Branch `camera-experiment`, pushed, based on main at c46518d.
- Main now contains PR #50 (CaF laser-tuning optimiser) and the camera
  binning/gain/continuous commit (f06e317). Both were merged on 2026-11-03.
- **The design and plan on this branch are out of date.** They were written before
  #50 and before the decisions below. The user approved the decisions below; the
  first job is to bring the design and the plan up to date with them (the user
  wants this done in this new session), then execute the plan.
- Nothing has been built. No product code has changed on this branch.

## Decisions the user approved (apply to the design and plan)

1. **Camera attribute names are `hamcam`, `thorcam`, `orca`**, replacing `ham`,
   `thor` and `caf` everywhere (examples, argument names, records). Arguments
   `hamcam.roi_x0`, `hamcam.exposure`, `hamcam.gain`; records `hamcam_images`;
   `LoadTweezers.camera` becomes `hamcam`. In the design's example the third
   camera is `orca = CameraShots("CaF ORCA", ...)` (the real CONFIG names are
   "Rb HamCam", "Rb ThorCam", "CaF ThorCam", "CaF ORCA", "Rb ORCA"; check
   `pytweezer/configuration/config.py`).
2. **Streaming mode.** `CameraShots(..., stream=True)` arms the camera once per
   task (`camera.arm()`). Each shot notes `frames_acquired()` as a mark, runs the
   sequences, waits with `pytweezer.drivers.shot_frames.settle_frame_counts`, then
   reads `read_frames_from(mark)`, as `CaFFluorescence.shot` does. `frames` is the
   expected count (decide in the plan whether a mismatch raises). Snap mode
   (`setup_acquisition(n, "snap")`, start/stop per shot, `acquire_n_frames`) stays
   the default, as `LoadTweezers` does. Reason: the Thorlabs driver ignores
   `mode`, so a snap loop would re-read stale ring-buffer frames on a ThorCam from
   the second shot on. Decide where the settle timing arguments live
   (`CaFFluorescence` has `frame_settle_s`, `frame_timeout_s`, `frame_poll_s`).
3. **Port `CaFFluorescence`** (pytweezer/experiments/caf_fluorescence.py) onto
   `CameraExperiment` as a task after the `LoadTweezers` rewrite. Its `shot()`
   becomes `take_shot()` on a streaming `thorcam = CameraShots("CaF ThorCam", ...)`;
   keep its background shot, laser setting and restore logic. Update
   `tests/test_caf_fluorescence.py` (and anything else from #50 that breaks); tests
   follow the design. Check that `OptimiseMOTLasers`, `Rig.auto_mot` and the
   Optimisers tab still work, and whether anything they submit names `camera` or
   settle arguments that move.
4. **`binning` and `gain` are optional `CameraShots` settings** (None = not set),
   arguments `<camera>.binning` and `<camera>.gain`, using
   `Camera.set_binning` / `Camera.set_gain` from f06e317. The HamCam's `set_gain`
   is its sensitivity and takes effect only once EM gain is enabled.
   `LoadTweezers.sensitivity` is removed in favour of `hamcam.gain`; saved recipes
   naming `sensitivity` (like the old `roi_*` names) stop validating, with no
   migration unless trivial. **Apply order in `prepare`:** binning, then ROI
   (`set_roi` keeps binning; `set_binning` can reset the ROI), exposure, trigger,
   then `configure_camera` (which enables EM gain on the HamCam), then gain, then
   `setup_acquisition`. This replaces the design's order. `binning` cannot be
   scanned (it changes the frame shape, like `roi_width`); `gain` can. Check what
   units the ROI is in once binned, and say so.
5. **Answers to the plan's five questions** (all "yes, as the plan proposes"):
   - Swap the hook order (see 4: the plan's reason was that changing DCAM settings
     after the buffer is allocated is risky).
   - `<camera>.frames`, `.roi_width`, `.roi_height` (and `.binning`) cannot be
     scanned; `exposure`, `gain`, `roi_x0`, `roi_y0` can, re-applied before a shot
     when they change. `LoadTweezers` refuses `hamcam.roi_x0`/`roi_y0` scans because
     its trap sites are placed in the ROI fixed at prepare.
   - No new tests, only update existing ones that break (the user's standing rule;
     see below).
   - `saved_sequence` keys become dotted, e.g. `"rb.tMOTLoadRepVCO"`.
   - `LoadTweezers.camera` is renamed `hamcam`.
6. The plan's first task is no longer "check conflicts with an open PR". #50 and
   the camera branch are merged, so drop any stacking or ordering text and list
   what they changed that the plan must account for.
7. `ContinuousAcquisition` (pytweezer/experiments/camera_continuous.py) is
   unrelated to `CameraExperiment`. Leave it alone, but check it does not collide
   with the new camera argument names or the skill text.
8. The `add-device-driver`, `add-experiment`, `add-optimiser` and `notebook-rig`
   skills were all edited by #50 and the camera commit. Fix the plan's skill-edit
   tasks accordingly. Minor edits go in with the code; the new "Camera
   experiments" section of `add-experiment` is a *significant* skill change, so
   draft it in the plan and wait for the user's approval before applying it.

## Earlier decisions (already in the design)

- Class name `CameraExperiment(MotMasterExperiment)`; a concrete base class, not
  mixins.
- Camera settings are form arguments (editable, scannable), with the declaration's
  values as defaults, named with dots like MOTMaster parameters.
- The base class collects `CameraShots` declarations into `self.cameras`
  (dict) and loops over it: set up, arm, run sequences, read.
- Frames are recorded as one stack per camera, `<name>_images`, indexed by
  position; no frame labels.
- `saved=` binding on the `MotMaster` declaration; precedence: override passed to
  `run_sequences` > form value > saved value > script default.
- `LoadTweezers` is rewritten on the base class, records `hamcam_images`, and its
  analysis (`tweezer_histogram`), the optimisers, their tooltips and the tests
  move to index the stack. Old measurement files are not supported by the updated
  analysis; no compatibility shim. `save_raw_images` and `save_tweezer_counts`
  stay `LoadTweezers` arguments.
- `roi=(x0, width, y0, height)`, the order `set_roi` takes.
- `locked=("frames",)` on a `CameraShots` leaves a setting out of the form;
  `LoadTweezers` locks `hamcam.frames`.
- Frames are broadcast with the `rid` header by default.
- Out of scope: the `globalParameters.json` overlap (the FIXME in
  `saved_parameters.py`); editing notebooks (list any that break); camera driver
  changes.

## Standing rules to respect

- **No new tests** unless the user asks (their global rule). Update or delete
  existing tests that the design breaks. Run the existing suite with
  `uv run pytest tests/ -q`.
- Notebooks are records: never edit them. List any that break.
- Work on the feature branch, commit and push in coherent tested steps; run
  `uv run ruff check --fix` and `ruff format` on changed Python files first. Open a
  PR for the branch at the end.
- GUI checks stay offscreen (`QT_QPA_PLATFORM=offscreen`), never on the live
  display.
- Never merge PRs without the user's explicit say-so (the permission system
  blocked a merge once in the brainstorming session until the user allowed it).
- Docstrings Google style, UK English, minimal comments, `get_logger`.

## Open

- Who writes and executes the plan: the user asked whether an Opus 5.5 agent
  should do the implementation (they already had one write the first plan). They
  have not said yes or no to implementation by an agent; ask.
- The first plan was written by an Opus agent that read the pre-#50 code. Re-check
  every path, signature and claim in it against current main.
