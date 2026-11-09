# Plain camera experiments: remove CameraExperiment, one MOTMaster per experiment

## Why

`CameraExperiment` was meant to make MOTMaster-plus-camera experiments easy to
write. With only a handful of experiments using it, the generalisation costs
more than it saves: `LoadTweezers` has `run_point`, `take_shot`, `take_shots`,
`record_shot` and a `planned_points` guard; `LoadTweezersMOTImage` has to
override `configure_camera` just to keep the HamCam's EM-gain calls off the
ORCA; and `LoadTweezers` carries code (`take_shots`, the batch helpers) that
only its subclasses use. Likewise `MotMasterExperiment` supports several
MOTMasters with master/follower threading that no real experiment uses:
two machines are driven by MOTMaster's linked mode, or explicitly by the
experiment.

Go back to basics: an experiment that uses a camera sets that camera up and
reads it in plain code you can read top to bottom.

## Goals

- `LoadTweezers` and `LoadTweezersMOTImage` read straight through: the devices,
  the arguments, a `prepare` that sets the cameras up, a shot, and what is
  recorded. No hooks to override to special-case a camera.
- **A class holds only what it uses itself.** A subclass may inherit from
  `LoadTweezers` when it reuses enough to justify it, but anything only the
  subclass needs lives in the subclass's file, never in `LoadTweezers`.
- `MotMasterExperiment` supports exactly one MOTMaster, and its script
  parameters are named without a prefix.

## Non-goals

- A replacement camera abstraction. There is deliberately no shared camera
  helper; if one is wanted later it comes back as its own design once there
  are enough experiments to justify it.
- Changing what `LoadTweezers` records or how the Results-tab analyses read it.
- Changing linked mode, `saved=` bindings, or the Experiment Manager.

## Anti-goals

- Re-growing generic hooks (`configure_camera`, per-camera recorders,
  `record_shot`-style extension points) under new names.
- Moving subclass-only code back into `LoadTweezers` "for reuse".

## What is removed

- `pytweezer/experiment/camera_experiment.py` entirely (`CameraExperiment`,
  `CameraShots`, `ShotSettings`, `record_cloud_fit`, etc.).
- `pytweezer/experiments/caf_mot.py` (`CaFMOT`, `LASERS`, `frame_counts`) and
  `pytweezer/experiments/caf_fluorescence.py` — not in use.
- `pytweezer/optimisers/optimise_mot_lasers.py` (`OptimiseMOTLasers`), built on
  `CaFFluorescence`.
- The notebook rig: `pytweezer/notebook/rig.py` (`Rig`, `Shot`,
  `plot_auto_mot_results`) and its export from `pytweezer/notebook/__init__.py`.
- The tests that cover only removed code (`tests/test_camera_experiment.py`,
  `tests/test_caf_fluorescence.py`, `tests/test_optimise_mot_lasers.py`,
  `tests/test_notebook_rig.py`).
- Any `CONFIG["GUI"]`/pipeline/analysis registration of the removed
  experiments and optimiser.

**Kept (builder's call, made here):** `PycafCamera`
(`pytweezer/notebook/pycaf_camera.py`), `pytweezer/notebook/zips.py` and
`pytweezer/drivers/shot_frames.py`. `PycafCamera` is separate from `Rig`, was
added in the latest commit, and is in use. It does borrow
`_motmaster_file_config` from `rig.py`, and its test borrows `MONTHS` and
`FakeCamera` from `tests/test_notebook_rig.py`: move those into
`pycaf_camera.py` and `tests/test_notebook_pycaf_camera.py` before deleting
`rig.py` and its test.

## MotMasterExperiment: one MOTMaster

- A class may declare at most one `MotMaster`; declaring two is a `TypeError`
  at class creation that points at linked mode. (A base class with none is
  fine, as now.)
- Remove `master=`, `follower_timeout`, `follower_arm_delay`, the follower
  threads and fresh follower clients, `master_name()`, and `device=` on
  `MotMasterNumber`/`MotMasterInteger` (and `motmaster_device` in their
  schema). `run_sequences` just sends the parameters to the one MOTMaster and
  blocks on `start_motmaster_experiment`.
- Linked mode stays as it is, including releasing the link in
  `close_devices`.
- **Script parameters lose the attribute prefix** everywhere: the form's
  search-box rows and scans use `tPulse`, not `rb.tPulse`;
  `run_sequences(**overrides)` takes plain parameter names
  (`run_sequences(BackgroundDrop=1)`); `saved_sequence` is keyed by plain
  parameter name. A script parameter is an extra argument when it is not a
  declared argument name; a declared argument always wins (builder's call: no
  error for the shadowed script parameter, but say so in the docstring).
- The Experiments tab (`pytweezer/GUI/experiments/arg_editor.py`,
  `motmaster_params.py`, `panel.py`) shows one MOTMaster box and reads and
  writes unprefixed names.
- Existing h5 files keep their prefixed `saved_sequence` keys; nothing needs
  to read them back, so no migration.
- **Stored requests with old names are refused, not translated** (user's
  decision). Recipes, the Experiments tab's last submission, "Resubmit" from
  an old h5 file and tasks still in the persisted queue may carry `rb.tX` or
  `hamcam.gain`. Dotted names are no longer valid arguments: they fail with the
  existing "no longer has argument(s) … load the recipe, check it and save it
  again" message (`recipes.unknown_arguments`), and the form must not turn them
  into MOTMaster rows named `hamcam.gain`. Loading an old last submission into
  the form may simply leave those values out. No translation layer, no rewrite
  of `recipes.json`.
- **Misspelt names fail at task start, not at submit** (user's decision).
  Without the prefix, any undeclared name is taken as a script parameter, and
  only MOTMaster knows the script's parameters, so submit accepts it and
  `prepare` refuses it before any shot runs. Submitting does not contact
  MOTMaster. To keep this safe: extras must be valid Python identifiers; a name
  that clashes with an existing attribute of the experiment (e.g. `rig`,
  `threshold`, `rid`) is refused, since extras are set on the instance; and the
  "did you mean" hint also suggests declared argument names.

## LoadTweezers

Inherits `MotMasterExperiment` directly. Declares:

- `rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2", saved=...)`,
  the same bindings as now.
- `hamcam = Device("Rb HamCam")`, `rig = Device("Rb Rearrangement Rig")`.
- Arguments, group "Camera", same defaults as today: `hamcam_gain` (600),
  `hamcam_roi_x0`, `hamcam_roi_width`, `hamcam_roi_y0`, `hamcam_roi_height`
  (110, 256, 220, 256, unbinned sensor pixels), `save_raw_images`,
  `save_tweezer_counts`.
- `analyses = ("tweezer_histogram", "tweezer_photon_histogram")`.

`prepare`, in plain calls: remember whether the HamCam was live and stop live;
binning 1; ROI (keep the origin the camera returns); trigger source `"ext"`;
external exposure mode; EM gain on; direct EM gain on; gain; set up 2 frames in
snap mode. Then load and record the saved trap sites as now.

A shot: start acquisition, `run_sequences(**overrides)`, read the two frames
(broadcast on the camera's stream with the task's `rid`), stop acquisition in
a `finally`. `run_point` takes one shot and records exactly what it records
today: `hamcam_images` (with `save_raw_images`), `signal_counts`,
`first_occupied`, `second_occupied`, `loading`, `first_counts`/`second_counts`
(with `save_tweezer_counts`), plus the prepare-time `trap_sites`,
`trap_window_size`, `trap_feature_size`, `threshold`.

`finish` puts the HamCam back in live mode if it was live.

Scanning: gain may be scanned and is re-sent before a shot whose gain differs
from the one last sent. The ROI cannot be scanned (the trap sites are placed in
the ROI set at `prepare`, and its size sets the frame shape); `LoadTweezers`
refuses an ROI axis in `planned_points`, since it is `LoadTweezers`' own
constraint.

`LoadTweezers` keeps `saved_trap_sites`, `site_photons`, `site_counts` and
`top_hat` because its own `run_point` uses them. `take_shots` leaves it.

## LoadTweezersMOTImage

Inherits `LoadTweezers` and adds only the ORCA:

- `orca = Device("ORCA")`, `save_mean_image` (Bool, True, group "Camera"),
  `analyses = (*LoadTweezers.analyses, "cloud_fit", "cloud_temperature")`.
- `prepare` also stops the ORCA's live mode (remembering it), sets trigger
  `"ext"` and one frame in snap mode, and records `cloud_mass_amu` for Rb.
  `finish` restores its live mode.
- Each shot also reads the ORCA's one frame from the same MOTMaster run. One
  way that needs no hook (builder may choose another plain one): start the
  ORCA, call `LoadTweezers`' shot, then read the ORCA's buffered frame, stopping
  the ORCA in a `finally`.
- Records the cloud fit as `orca_*` exactly as `record_cloud_fit` did (same
  field names, units, NaNs, `object_pixel_m` handling, `orca_mean_image` with
  `save_mean_image`), so the `cloud_fit`/`cloud_temperature` analyses keep
  working. That code now lives in `LoadTweezersMOTImage`, its only user.

## Subclasses of LoadTweezers

- **`TweezerArrayMeasurement`** keeps inheriting `LoadTweezers`. Its batch of
  shots (`n` shots at given script parameters, recording `hamcam_images` with
  `save_raw_images`) lives in `tweezer_array_measurement.py`.
  `OptimiseTweezerImaging` and `OptimiseMolassesShims` pass plain script
  parameter names.
- **`FindTrapSites`** keeps inheriting `LoadTweezers`. Its own batch of shots
  (tweezers off during the background drop), and not loading saved trap sites,
  live in `find_trap_sites.py`. It reads the ROI origin from whatever
  `LoadTweezers` keeps it in.
- A few lines of batch loop duplicated between the two is accepted over a
  shared helper in `LoadTweezers`.

## Skills, docs and docstrings

Approving this document approves these skill changes (they are significant
under CLAUDE.md's rules):

- `add-experiment`: replace the `CameraExperiment`/`CameraShots` guidance with
  "set the camera up in `prepare` and read it in your shot", using
  `LoadTweezers` as the example; one MOTMaster per experiment; unprefixed
  script parameters.
- `add-optimiser`: drop `OptimiseMOTLasers`/`CaFMOT` references.
- `notebook-rig`: remove everything about `Rig`; keep it only for `PycafCamera`
  and the zip helpers, with its `description` updated to match (or delete the
  skill if nothing worth a skill is left). Update its row in CLAUDE.md's skill
  table accordingly.
- `add-device-driver`: drop `CameraShots` references.

Update docstrings that point at removed code (`analysis/cloud_results.py`,
`experiment/camera_settings.py`, `GUI/viewers/image_monitor.py`,
`analysis/tweezer_histogram.py`). Old specs and plans under `docs/superpowers/`
are history and stay as they are.

## Tests

Delete the tests for removed code. Update `tests/test_motmaster_experiment.py`
(one MOTMaster, unprefixed names), `tests/test_tweezer_array.py`,
`tests/test_experiment_gui.py`, `tests/test_experiment_reservations.py` and any
other test that breaks, following the `write-tests` skill. The suite and ruff
pass before each push.

## Breaking changes to tell the lab

- Form/scan names: `hamcam.gain` → `hamcam_gain`, `hamcam.roi_x0` →
  `hamcam_roi_x0`, etc.; `rb.tSomething` → `tSomething`.
- Notebooks importing `Rig`, `Shot`, `plot_auto_mot_results`, `CaFMOT`,
  `CaFFluorescence`, `OptimiseMOTLasers` or `CameraExperiment` stop working.
- `saved_sequence` keys in new files are unprefixed.

## Delivery

Feature branch `plain-camera-experiments`, with a PR. Commit and push after
each coherent, tested step (e.g. MotMasterExperiment simplified; LoadTweezers
rewritten; removals).

## Left to the builder

Method names and internal structure within each class; how the shot is split
between methods; error wording; the order of the steps.
