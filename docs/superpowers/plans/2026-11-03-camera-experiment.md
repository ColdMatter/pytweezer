# CameraExperiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiments that run "set up cameras, run MOTMaster, read N frames per camera" are written as declarations (`CameraShots`, `MotMaster(saved=...)`); `LoadTweezers` and `CaFFluorescence` are rebuilt on them.

**Architecture:** `MotMaster` gains a `saved=` binding that `MotMasterExperiment` resolves in `prepare()` and merges in `run_sequences()`. A new `CameraExperiment(MotMasterExperiment)` in `pytweezer/experiment/camera_experiment.py` turns each `CameraShots` class attribute into dotted form arguments (real class attributes, so the catalogue, form, scans and recipes see them with no other change) and runs generic loops over `self.cameras`, in snap mode (fixed frames per shot) or stream mode (free-running, frames read from a mark). `LoadTweezers` records `hamcam_images`; the tweezer histogram reads that stack; `CaFFluorescence` streams from `thorcam`.

**Tech Stack:** Python 3, numpy, h5py (via `MeasurementWriter`), sipyco RPC clients, PyQt6 (form only, unchanged), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-camera-experiment-design.md`. Decisions behind this revision: `docs/superpowers/plans/2026-11-03-camera-experiment-handoff.md`.

## What main changed since the first plan

PR #50 (CaF laser-tuning optimiser) and f06e317 (camera binning, gain, continuous acquisition) are merged. What this plan accounts for:

- `Camera.set_binning(binning)` and `Camera.set_gain(gain)` (`drivers/camera_base.py`): the base raises `NotImplementedError`; ThorCam (`set_gain` in dB, `set_binning` through pylablib's `set_roi(hbin=, vbin=)`, which resets the ROI to full frame), ORCA (contrast-gain index), ImagEM X2 (`set_gain` is `set_sensitivity`) and `SimulatedCamera` implement them. `set_roi` keeps the camera's binning on every driver.
- `drivers/shot_frames.py:settle_frame_counts(cameras, settle_s, timeout_s, poll_s)`, moved out of `Rig`.
- `experiments/caf_fluorescence.py:CaFFluorescence(MotMasterExperiment)`: `camera = Device("CaF ThorCam")`, `shot()` arms, marks, runs, settles and reads; arguments `frame_settle_s`, `frame_timeout_s` (group "Camera"), class attribute `frame_poll_s`; raises `RuntimeError` naming the device when a shot catches no frames.
- `optimisers/optimise_mot_lasers.py:OptimiseMOTLasers(StagedScan, CaFFluorescence)`; `Rig.auto_mot` submits it with laser arguments and the user's `**arguments`; `experiment/optimisers.py` holds the accept logic the Optimisers tab and `Rig.auto_mot` share. None of them names `camera`; tests pass `frame_settle_s=0.0`, which keeps its name here.
- `experiments/camera_continuous.py:ContinuousAcquisition`: unrelated; its undotted `binning`, `gain`, `roi_*` arguments do not collide.
- The CaF ThorCam's CONFIG entry sets `trigger_mode: "bulb"`, binning 16, 12 dB and `live: True`, so `CaFFluorescence` sets none of these.
- Skills edited since: `add-device-driver` (binning/gain, `set_roi` keeps binning), `add-optimiser` ("CaF laser tuning"), `notebook-rig` (frame collection, CaF ThorCam), `add-experiment`.
- The form lives in `pytweezer/GUI/experiments/arg_editor.py` (`load_request`, `_add_motmaster_boxes`): it builds a row for every entry in `schema["arguments"]` and treats a dotted name as a MOTMaster parameter only when the part before the dot is a MOTMaster attribute, so dotted camera arguments need no GUI change.

## Decisions taken (from the handoff; approved by the user)

1. Camera attributes are `hamcam`, `thorcam`, `orca`; records `<name>_images`.
2. `CameraShots(..., stream=True)` streams; snap is the default.
3. `CaFFluorescence` is ported after `LoadTweezers`.
4. `binning` and `gain` are optional settings. Apply order in `prepare`: binning, ROI, exposure, trigger, `configure_camera`, gain, then `setup_acquisition` (snap) or `arm()` (stream). `LoadTweezers.sensitivity` is removed in favour of `hamcam.gain`.
5. `frames`, `roi_width`, `roi_height`, `binning` cannot be scanned; `exposure`, `gain`, `roi_x0`, `roi_y0` can. `LoadTweezers` refuses `hamcam.roi_x0`/`roi_y0` scans. No new tests. `saved_sequence` keys are dotted. `LoadTweezers.camera` becomes `hamcam`.

## Decisions on the spec's "Left to the builder"

- **Dotted arguments:** `CameraExperiment.__init_subclass__` `setattr`s the generated `Integer`/`Number` objects onto the class under their dotted names, with `.name` set by hand because `setattr` skips `__set_name__`. `collect()`, and therefore `arguments()`, `schema()`, `coerce_arguments`, `Scan.points`, `unknown_arguments` and the form, sees them as declared arguments. `getattr(self, "hamcam.exposure")` works because an instance's `__dict__` overrides a non-data descriptor. Group: `f"Camera: {name}"`. Only `CameraShots` in `vars(cls)` generate arguments; subclasses inherit them.
- **Locked settings** generate no argument. If an inherited argument has that name, the subclass sets the class attribute to `None`, which `collect()` treats as removing it. Submitting a locked or missing setting by name is refused as an unknown argument.
- **Optional settings:** `exposure`, `binning`, `gain`, `roi` and `trigger` may be `None`, meaning "never set it"; none of them then generates an argument. `roi_origins[name]` is `None` for a camera with no `roi`.
- **Trigger source:** the keyword `CameraShots(trigger="ext")`, not a form argument; `"ext"` is valid on every driver (pylablib DCAM and TLCam). `CaFFluorescence` passes `trigger=None` to keep the CONFIG bulb trigger. The HamCam's `set_external_exposure_mode()` refines `"ext"` in `configure_camera`.
- **Settle timing:** a class that declares any streaming camera gets the arguments `frame_settle_s = Number(0.3, min=0, unit="s", group="Camera")` and `frame_timeout_s = Number(5.0, min=0, unit="s", group="Camera")`, generated in `__init_subclass__` with `CaFFluorescence`'s tooltips, and `CameraExperiment.frame_poll_s: ClassVar[float] = 0.05`. One settle wait covers every streaming camera. The names are unchanged, so everything that passes `frame_settle_s` keeps working.
- **Stream frame count:** `frames` is the expected count. A streaming camera that catches none raises `RuntimeError` naming the device and script (today's `CaFFluorescence` text). Any other count that differs from `frames` logs a warning and returns what arrived, as `Rig.run` warns. Reason: a free-running camera (and every simulated one) can catch a stray frame, which must not abort an optimiser run; an experiment that records a varying stack will then fail at `record`, which already refuses a shape change.
- **Re-applying:** before a shot, a camera whose `shot_settings` differ from those last applied is stopped and has its whole `apply_camera_settings` run again (all settings, `configure_camera`, then `setup_acquisition` or `arm`), so the order in decision 4 always holds.
- **Errors:** keep today's texts: `RuntimeError("the saved parameters could not be read: …")` and `KeyError("no saved parameter 'a', 'b'")`. A bound parameter missing from the script raises the existing `_check_parameter` `ValueError` in `prepare()`.
- **Records:** the default `run_point` records `<name>_images` (`(frames, H, W)` per point). `record_frames(frames)` records every entry of a `{name: array}` dict as `<name>_images`, so `take_shots` can pass `{"hamcam": (shots, 2, H, W)}`.
- **Old recipes** for `LoadTweezers`/`TweezerArrayMeasurement` naming `roi_x0`, `roi_y0`, `roi_width`, `roi_height` or `sensitivity` are refused (`unknown_arguments`). No migration. Loading one into the form ignores the old names (the new defaults equal the old ones: `110/220/256/256`, gain 600), and saving it again fixes it.
- **`LoadTweezers` binning** is locked at 1: trap sites are saved in sensor pixels and placed in the frame by subtracting the ROI origin, which holds only unbinned. Leaving it `None` would inherit whatever binning `ContinuousAcquisition` last set.

## Global Constraints

- Branch: `camera-experiment` (already pushed, based on main at c46518d), with a PR. Commit and push after each task; stage only your own files.
- Before each commit, run `uv run ruff check --fix <files>` and `uv run ruff format <files>`, then `git add` the files whole. Never use `--no-verify`.
- Each task ends with `uv run pytest tests/ -q` green.
- Add no new test files or test cases. Update or delete only the existing tests the change breaks. Tests follow the design; never add hooks to production code for a test.
- Google-style docstrings that describe current behaviour only, UK English, comments only where the code surprises, `get_logger` for logging.
- Don't touch hardware at import or class definition.
- Don't change camera drivers or `camera_base.py`.
- Run any Qt check with `QT_QPA_PLATFORM=offscreen`, never on the live display.
- Notebooks are never edited.
- Never merge the PR without the user's say-so.
- Names fixed by the spec: `CameraExperiment`, `CameraShots`, `configure_camera`, `take_shot`, `record_frames`, `self.cameras`, `self.roi_origins`, `<camera>.frames`, `.exposure`, `.binning`, `.gain`, `.roi_x0`, `.roi_width`, `.roi_y0`, `.roi_height`, `<camera>_images`, `roi=(x0, width, y0, height)`, `locked=(...)`, `stream=True`, `MotMaster(..., saved=...)`, `saved_sequence`, `hamcam`, `thorcam`.

## Review Focus

Under the no-new-tests rule, the scratch scripts in Tasks 2, 3 and 5 check each of these by hand.

1. **Scanning `thorcam.exposure`, `.gain` or `.roi_x0`.** Each point's shot uses that point's value: the camera is re-applied before it is armed, and only when a setting changed.
2. **Scanning a shape-changing setting** (`frames`, `roi_width`, `roi_height`, `binning`), or `hamcam.roi_x0`/`roi_y0` in `LoadTweezers`. Refused at `planned_points` with a `ValueError` that names the argument and says why; the task fails before any shot.
3. **A sequence or frame read that raises mid-shot.** Every snap camera is still stopped (`finally`), streaming cameras are left running, and the original error reaches the task.
4. **A saved-bound script parameter set in the form or scanned.** The form or scan value is sent and the saved one is not. `saved_sequence` leaves out a parameter set in the form.
5. **A locked setting submitted by name** (`run_local(LoadTweezers, **{"hamcam.frames": 3})`) or a pre-change name (`roi_x0`, `sensitivity`). Refused, naming the argument.
6. **Apply order.** On a camera with every setting given, the calls are `set_binning`, `set_roi`, `set_exposure_time`, `set_trigger_source`, `configure_camera`, `set_gain`, then `setup_acquisition` (snap) or `arm` (stream, after `stop_acquisition`).

---

### Task 1: Saved-parameter binding on `MotMaster`

**Files:**
- Modify: `pytweezer/experiment/motmaster.py` (`MotMaster.__init__`, `to_schema`, `MotMasterExperiment.prepare`, `run_sequences`, module docstring)
- Modify: `pytweezer/experiments/load_tweezers.py`: delete `SAVED_SEQUENCE_PARAMETERS` and the reading/merging block in `prepare`, and declare `rb = MotMaster(..., saved=…)`. `shot()` stops adding saved values.
- Modify: `pytweezer/experiment/saved_parameters.py:31`: the FIXME comment's example becomes "a `MotMaster`'s `saved=` bindings".
- Modify: `.claude/skills/add-experiment/SKILL.md` (minor: see step 6)
- Test (update only): `tests/test_tweezer_array.py`

**Interfaces:**
- Produces:
  - `MotMaster(device_name, *, script, master=False, linked=False, iterations=1, save=False, follower_timeout=60.0, saved: Iterable[str | dict[str, str]] = (), timeout=None)`. `MotMaster.saved: dict[str, str]` is `{script parameter: saved key}`, normalised from bare names (same key) and dicts (renames). Raises `TypeError` at class definition for an entry that is neither a `str` nor a `dict` of `str` to `str`, or a parameter bound to two different keys. `to_schema()["motmaster"]` gains `"saved"`.
  - `MotMasterExperiment.saved_parameters: dict[str, Any]` is every saved parameter read in `prepare()`, or `{}` when no `MotMaster` binds any. Read in exactly one call to `get_saved_parameters()`, imported at module level in `motmaster.py` (tests patch `pytweezer.experiment.motmaster.get_saved_parameters`).
  - `MotMasterExperiment.saved_sequence: dict[str, Any]`, `{"<attribute>.<parameter>": value}` for bound parameters the form did not set at `prepare()`. Recorded as the constant `saved_sequence` only when some binding exists.
  - `run_sequences(**overrides)` precedence, highest first: `overrides`, then `motmaster_parameters()` (form, dotted, scanned), then the saved values, then the script default.

- [ ] **Step 1: Check out the branch.** `git switch camera-experiment && git pull`.

- [ ] **Step 2: Update the existing tests to the new design.** In `tests/test_tweezer_array.py`:
  - imports: drop `SAVED_SEQUENCE_PARAMETERS` and `from pytweezer.experiments import load_tweezers`; add `from pytweezer.experiment import motmaster`.
  - `SAVED` builds from `LoadTweezers.rb.saved.values()`.
  - the `rig` fixture patches `get_saved_parameters` on `(motmaster, optimisers)`.
  - `test_every_shot_runs_at_the_saved_parameters_unless_the_form_sets_one`: `expected = {parameter: SAVED[key] for parameter, key in LoadTweezers.rb.saved.items()} | {"tMolCoolVCO1": 7.5}`.
  - `test_load_tweezers_needs_the_saved_parameters` patches `motmaster.get_saved_parameters`.

- [ ] **Step 3: Run the tests and check they fail.** `uv run pytest tests/test_tweezer_array.py -q`. Expected: errors on `LoadTweezers.rb.saved` (`AttributeError`).

- [ ] **Step 4: Implement it.** Do the binding in `MotMasterExperiment.prepare()`, after the existing setup and `_check_parameters`:
  - if no `MotMaster` has `saved`, set `saved_parameters = {}` and `saved_sequence = {}`, then return without contacting the manager;
  - otherwise read once, mapping `(ManagerUnavailable, ManagerError)` to the `RuntimeError` above;
  - raise `KeyError` naming every missing key, sorted;
  - `_check_parameter` each bound parameter;
  - keep per attribute `{parameter: saved value}` for `run_sequences`, and record `saved_sequence`.

  In `run_sequences`, merge each attribute's saved values *under* `motmaster_parameters()`, then apply `overrides`.

  In `load_tweezers.py`, keep the names in the private tuple `_SAVED_UNDER_OWN_NAME` and declare:
  ```python
  rb = MotMaster(
      "Rb MotMaster",
      script="RbTweezerBasic2026_2",
      saved=(
          *_SAVED_UNDER_OWN_NAME,
          {"tImgCoolVCO1": "tImgCoolVCO", "tImgCoolVCO2": "tImgCoolVCO",
           "tImgCoolVVA1": "tImgCoolVVA", "tImgCoolVVA2": "tImgCoolVVA"},
      ),
  )
  ```
  `LoadTweezers.prepare` passes `self.saved_parameters` to `saved_trap_sites`. Remove the `client` imports it no longer uses.

- [ ] **Step 5: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_array.py tests/test_motmaster_experiment.py tests/test_tweezer_histogram.py tests/test_caf_fluorescence.py tests/test_optimise_mot_lasers.py -q`, then `uv run pytest tests/ -q`. Expected: all pass. (`CaFFluorescence` binds nothing, so it never reads saved parameters; `test_a_laser_with_no_saved_set_point_names_the_key` patches `client.get_saved_parameters` for the optimiser and is unaffected.)

- [ ] **Step 6: Make the minor skill edits** in `add-experiment/SKILL.md`:
  - In "MOTMaster sequences", replace "(`LoadTweezers.shot()` puts the saved sequence parameters under these)" with one sentence on `saved=`: bare name or `{parameter: key}`; read once in `prepare()` (unreachable manager → `RuntimeError`, missing keys → `KeyError`); precedence `run_sequences` override > form/scan > saved > script default; constant `saved_sequence` `{"rb.<parameter>": value}`.
  - In "Tweezer array measurements" item 3, replace `SAVED_SEQUENCE_PARAMETERS` with `LoadTweezers.rb.saved` (`saved=` on its `MotMaster`).

- [ ] **Step 7: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiment/motmaster.py pytweezer/experiments/load_tweezers.py pytweezer/experiment/saved_parameters.py tests/test_tweezer_array.py
  uv run ruff format pytweezer/experiment/motmaster.py pytweezer/experiments/load_tweezers.py pytweezer/experiment/saved_parameters.py tests/test_tweezer_array.py
  git add pytweezer/experiment/motmaster.py pytweezer/experiments/load_tweezers.py pytweezer/experiment/saved_parameters.py tests/test_tweezer_array.py .claude/skills/add-experiment/SKILL.md
  git commit -m "MotMaster saved= binds script parameters to saved parameters"
  git push -u origin camera-experiment
  ```

### Task 2: `CameraShots` and `CameraExperiment`

**Files:**
- Create: `pytweezer/experiment/camera_experiment.py`
- Modify: `.claude/skills/add-device-driver/SKILL.md` (minor: one clause that `CameraExperiment` calls `set_binning`/`set_gain` only when a `CameraShots` sets them, so a driver without them still works there)
- No test changes (nothing existing uses it yet).

**Interfaces:**
- Consumes: `MotMasterExperiment.prepare()` / `run_sequences(**overrides)` from Task 1; `Device`, `Integer`, `Number`, `collect` from `arguments.py`; `settle_frame_counts` from `drivers/shot_frames.py`.
- Produces:
  - `SETTINGS = ("frames", "exposure", "binning", "gain", "roi")`.
  - `CameraShots(Device)`, built as `CameraShots(device_name, *, frames: int = 1, exposure: float | None = None, binning: int | None = None, gain: float | None = None, roi: tuple[int, int, int, int] | None = None, trigger: str | None = "ext", stream: bool = False, locked: Iterable[str] = (), timeout: float | None = None)`. At class definition it raises `ValueError` for `frames < 1`, `exposure <= 0`, `binning < 1`, `gain < 0`, an `roi` that is not four whole numbers, or a `locked` entry outside `SETTINGS`.
    - `CameraShots.arguments(name) -> dict[str, Argument]`: `{f"{name}.frames": Integer(frames, min=1), f"{name}.exposure": Number(exposure, unit="ms", scale=1e-3, min=0), f"{name}.binning": Integer(binning, min=1, tooltip="pixels per side"), f"{name}.gain": Number(gain, min=0, tooltip=<the camera's unit, as ContinuousAcquisition.gain says>), f"{name}.roi_x0": Integer(x0, min=0, unit="px"), ".roi_width": Integer(min=1, unit="px"), ".roi_y0": Integer(min=0, unit="px"), ".roi_height": Integer(min=1, unit="px")}` (ROI tooltips: "unbinned sensor pixels"), leaving out every `None` or locked setting, all with `group=f"Camera: {name}"`.
    - `CameraShots.to_schema()` adds `"camera": {"frames", "exposure", "binning", "gain", "roi", "trigger", "stream", "locked"}` to `Device.to_schema()`.
  - `ShotSettings`, a frozen dataclass of `(frames: int, exposure: float | None, binning: int | None, gain: float | None, roi: tuple[int, int, int, int] | None)`.
  - `CameraExperiment(MotMasterExperiment)`:
    - `frame_poll_s: ClassVar[float] = 0.05`.
    - `__init_subclass__`: for each `CameraShots` in `vars(cls)`, `setattr` its `arguments(name)` onto `cls` and hide inherited arguments of locked settings (see Decisions); if any camera of the class (`camera_shots()`) streams and `frame_settle_s` is not already an argument, add `frame_settle_s` and `frame_timeout_s`.
    - `camera_shots() -> dict[str, CameraShots]` (classmethod, `collect(cls, CameraShots)`).
    - `cameras -> dict[str, Any]` (property; `{name: getattr(self, name)}`).
    - `shot_settings(name) -> ShotSettings`: the declared value for locked or `None` settings, else the argument's current value.
    - `apply_camera_settings(name) -> None`: for a streaming camera with anything to set, `stop_acquisition()` first; then, skipping `None`, `set_binning`, `set_roi(*roi)` (keeping the return in `self.roi_origins[name]` as a tuple), `set_exposure_time`, `set_trigger_source(trigger)`, `configure_camera(name, camera)`, `set_gain`, then `setup_acquisition(frames, mode="snap")` (snap) or `arm()` (stream). It then stores the settings in `self._applied[name]`.
    - `configure_camera(name: str, camera: Any) -> None`: a hook that does nothing by default.
    - `prepare()`: `super().prepare()`, then `apply_camera_settings` for every camera.
    - `planned_points(scan)`: raises `ValueError` if a scan axis is `<camera>.frames`, `.binning`, `.roi_width` or `.roi_height` ("…changes the shape of `<camera>_images`; set it for the whole task"), else `super().planned_points(scan)`.
    - `take_shot(**overrides) -> dict[str, np.ndarray]`: re-apply any camera whose `shot_settings` differ from `_applied`; `start_acquisition` on every snap camera; `arm()` then `marks[name] = frames_acquired()` on every streaming camera; `run_sequences(**overrides)`; snap cameras `acquire_n_frames(frames, broadcast=True, header={"rid": self.rid})` as `np.asarray`; then one `settle_frame_counts(streaming, self.frame_settle_s, self.frame_timeout_s, self.frame_poll_s)` and `read_frames_from(marks[name])` per streaming camera, with the count rule in Decisions; `stop_acquisition` on every snap camera in a `finally`.
    - `record_frames(frames: dict[str, np.ndarray]) -> None`: `record(f"{name}_images", stack)` per entry.
    - `run_point()`: `self.record_frames(self.take_shot())`.

- [ ] **Step 1: Write `camera_experiment.py`** to the interfaces above. Its module docstring is the spec's `PureRecipe` example plus one paragraph each on dotted arguments and locking, snap versus stream, and which settings can be scanned (and that the ROI is in unbinned sensor pixels). No comments beyond the surprising: the `setattr` of dotted names, the `None` that hides a locked inherited argument, and why binning precedes the ROI.

- [ ] **Step 2: Check it by hand with a simulated run** (no hardware, no manager). Write `<scratchpad>/camera_smoke.py`, outside the repo:
  ```python
  import pytest
  from pytweezer.experiment import ListAxis, Scan, run_local
  from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
  from pytweezer.experiment.motmaster import MotMaster

  class TwoCameras(CameraExperiment):
      rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
      thorcam = CameraShots("Rb ThorCam", frames=1, exposure=10e-3, gain=0.0, roi=(0, 64, 0, 32))
      orca = CameraShots("Rb ORCA", frames=2, binning=2, locked=("frames",))

  assert {"thorcam.frames", "thorcam.exposure", "thorcam.gain", "thorcam.roi_x0",
          "thorcam.roi_width", "thorcam.roi_y0", "thorcam.roi_height", "orca.binning"} <= set(TwoCameras.arguments())
  assert "orca.frames" not in TwoCameras.arguments()
  assert "frame_settle_s" not in TwoCameras.arguments()
  m = run_local(TwoCameras, Scan(axes=[ListAxis(argument="thorcam.exposure", values=[5e-3, 20e-3])]), simulate=True)
  assert m.status == "completed", m.attrs.get("error")
  assert m.results["thorcam_images"].shape == (2, 1, 32, 64)
  assert m.results["orca_images"].shape[:2] == (2, 2)
  bad = run_local(TwoCameras, Scan(axes=[ListAxis(argument="thorcam.frames", values=[1, 2])]), simulate=True)
  # planned_points raises before a file exists:
  ```
  Finish the last check to match how `run_local` reports a `planned_points` `ValueError` (read `runner.run_local`): either `pytest.raises(ValueError, match="thorcam.frames")` or `bad.status == "failed"` with `thorcam.frames` in the error; do the same for `orca.binning`. Also check:
  - Review Focus 5: `run_local(TwoCameras, simulate=True, **{"orca.frames": 3})` is refused with "orca.frames" named.
  - Review Focus 1: subclass with a `configure_camera` that appends `(name, self.shot_settings(name).exposure)` to a list, run the exposure scan, and expect `[("thorcam", 10e-3), ("orca", None), ("thorcam", 5e-3), ("thorcam", 20e-3)]`: one apply per camera at `prepare` and one per change; the ORCA is not re-applied. Repeat for a `thorcam.gain` scan.
  - Review Focus 6: wrap one simulated camera client in a recorder of method names and check the order of the calls `apply_camera_settings` makes, snap and stream.
  - Review Focus 3: subclass `TwoCameras` with a `run_sequences` that raises `RuntimeError`, run it with `simulate=True`, and from its `finish()` assert every `self.cameras[name]._acquisition_in_progress()` is `False` (the simulated backend is the in-process object) and that the run failed with the `RuntimeError`.
  - Stream mode: `class Streaming(CameraExperiment)` with `thorcam = CameraShots("Rb ThorCam", frames=1, stream=True)`; `frame_settle_s` and `frame_timeout_s` are arguments; `run_local(Streaming, simulate=True, frame_settle_s=0.05, frame_timeout_s=0.5)` completes with at least one frame per point, and the camera is still acquiring afterwards.

  Run with `uv run python <scratchpad>/camera_smoke.py`. Expected: no assertion errors.

- [ ] **Step 3: Check the form, offscreen.** `QT_QPA_PLATFORM=offscreen uv run python <scratchpad>/camera_form.py`, where the script builds an `ArgumentEditor()` with `fetcher.threaded = False` and `set_parameter_source(lambda device, script: {"tDelay1": 5})` (as `tests/test_experiment_gui.py::sequenced_editor` does), calls `set_experiment(TwoCameras.schema())`, and asserts:
  - `"thorcam.exposure"` is in `editor.rows` and `"orca.frames"` is not;
  - `editor.request().args["thorcam.roi_width"] == 64`;
  - `load_request(editor.request())` round-trips the edited value.

  No GUI code change is expected. If one is needed, stop and report it.

- [ ] **Step 4: Run the suite.** `uv run pytest tests/ -q`. Expected: all pass.

- [ ] **Step 5: Make the minor skill edit** in `add-device-driver/SKILL.md` (see Files).

- [ ] **Step 6: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiment/camera_experiment.py && uv run ruff format pytweezer/experiment/camera_experiment.py
  git add pytweezer/experiment/camera_experiment.py .claude/skills/add-device-driver/SKILL.md
  git commit -m "CameraExperiment: cameras declared as CameraShots, settings as dotted arguments"
  git push
  ```

### Task 3: `LoadTweezers` on `CameraExperiment`; optimisers and their tests to `hamcam_images`

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py` (whole class body)
- Modify: `pytweezer/experiments/tweezer_array_measurement.py` (`save_raw_images` tooltip)
- Modify: `pytweezer/optimisers/find_trap_sites.py` (`roi_origin` → `roi_origins["hamcam"]`; tooltip)
- Modify: `.claude/skills/add-experiment/SKILL.md` (minor: see step 6)
- Test (update only): `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: Task 2's `CameraShots`, `CameraExperiment`, `take_shot`, `record_frames`, `shot_settings`, `roi_origins`, `configure_camera`, `planned_points`; Task 1's `saved_parameters`.
- Produces:
  - `LoadTweezers(CameraExperiment)` with `hamcam = CameraShots("Rb HamCam", frames=2, binning=1, gain=600, roi=(110, 256, 220, 256), locked=("frames", "binning"))`. `exposure` stays `None`, because the trigger pulse exposes the HamCam. `hamcam.gain` gets the tooltip "EM gain sensitivity".
  - `sensitivity` is removed. `save_raw_images` and `save_tweezer_counts` keep group `"Camera"`.
  - `configure_camera(name, camera)` calls `set_external_exposure_mode()`, `enable_em_gain(True)` and `enable_direct_em_gain(True)`; the base then calls `set_gain`, which sets the sensitivity on the HamCam.
  - `planned_points(scan)` raises `ValueError` for a `hamcam.roi_x0` or `hamcam.roi_y0` axis: "the trap sites are placed in the ROI set before the first point". Otherwise it defers to `super()`.
  - `take_shots(n_shots, **sequence) -> tuple[list[np.ndarray], list[np.ndarray]]` has an unchanged signature and return. With `save_raw_images` it records `hamcam_images`, `(shots, 2, H, W)` per point.
  - `run_point` records `hamcam_images` `(2, H, W)` (with `save_raw_images`) through `record_frames`. `signal_counts`, `first_/second_occupied`, `loading`, `first_/second_counts` are unchanged, computed from `frames["hamcam"][0]` and `[1]`.
  - `saved_trap_sites` reads the origin from `self.roi_origins["hamcam"]` and the ROI size from `self.shot_settings("hamcam").roi`.
  - `shot()` and `finish()` are deleted: `take_shot` stops the camera itself.

- [ ] **Step 1: Update the existing tests to the new design.** In `tests/test_tweezer_array.py`:
  - `FakeHamCam` gains `def set_trigger_source(self, source): ...`, `def set_binning(self, binning): ...` and `def set_gain(self, gain): ...`, and drops `set_sensitivity`.
  - `test_a_measurements_counts_are_the_raw_pixel_sums_of_its_frames`: frames are `measurement.results["hamcam_images"][0][:, i]` for `i, name in enumerate(("first", "second"))`.
  - `test_load_tweezers_records_each_sites_raw_counts_and_occupancy`: `window_sums(results["hamcam_images"][:, i])`.
  - `test_load_tweezers_without_raw_images_records_no_frames`: `assert "hamcam_images" not in measurement.results`, replacing the two `first_/second_image` asserts.
  - `test_load_tweezers_without_saved_sites_records_frames_only`: `assert "hamcam_images" in measurement.results`.
  - `test_a_measurement_records_occupancy_and_frames_only_when_asked`: `("hamcam_images" in measurement.results) == save_raw_images` and shape `(1, FAST["shots"], 2, 128, 128)`.
  - Any test reading `rig.camera` keeps working: that is the fixture's fake, not the experiment attribute.

- [ ] **Step 2: Run the tests and check they fail.** `uv run pytest tests/test_tweezer_array.py -q`. Expected: FAIL on `hamcam_images` missing (`KeyError`/assert).

- [ ] **Step 3: Rewrite `LoadTweezers`** to the interfaces above, and update its docstring for `hamcam_images`, `hamcam.gain` and the saved binding. `TweezerArrayMeasurement` and `FindTrapSites` change only in their tooltip ("record every shot's camera frames as hamcam_images") and in `FindTrapSites.fit`'s `self.roi_origins["hamcam"][::-1]`.

- [ ] **Step 4: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_array.py tests/test_tweezer_histogram.py -q`, then `uv run pytest tests/ -q`. Expected: all pass (`tests/test_tweezer_histogram.py` still records its own `first_image` until Task 4).

- [ ] **Step 5: Check Review Focus 2, 4 and 5 by hand** in `<scratchpad>/load_tweezers_smoke.py`, reusing the `rig` fakes by importing `FakeSequencer`, `FakeHamCam`, `FakeRig` and `SAVED` from `tests.test_tweezer_array` and patching as the fixture does:
  - a `ListAxis(argument="hamcam.roi_x0", …)` scan fails, naming `hamcam.roi_x0`;
  - `run_local(LoadTweezers, **{"hamcam.frames": 3})`, `**{"hamcam.binning": 2}`, `roi_x0=100` and `sensitivity=500` are refused, naming the argument;
  - a `hamcam.gain` scan of `[500, 700]` calls `set_gain(500)` then `set_gain(700)`, each after `enable_em_gain`;
  - `run_local(LoadTweezers, Scan(axes=[ListAxis(argument="rb.tMolCoolVCO1", values=[7.5, 8.5])]))` sends 7.5 then 8.5, never the saved value.

  Also run an offscreen `ArgumentEditor` on `LoadTweezers.schema()` and check that it shows `hamcam.gain` and `hamcam.roi_x0`…`hamcam.roi_height` in the group "Camera: hamcam", and no `hamcam.frames`, `hamcam.binning` or `sensitivity`.

- [ ] **Step 6: Make the minor skill edits** in `add-experiment/SKILL.md` "Tweezer array measurements":
  - item 3: ROI from `hamcam.roi_x0`/`hamcam.roi_y0`/`hamcam.roi_width`/`hamcam.roi_height` and EM-gain sensitivity from `hamcam.gain`, with sites moved by the origin `set_roi` returned (`roi_origins["hamcam"]`); binning locked at 1; frames recorded as `hamcam_images` `(2, H, W)` only with `save_raw_images`; ROI-origin scans refused.
  - item 4: `save_raw_images` records `hamcam_images`, `(shots, 2, H, W)` per point, replacing `first_images`/`second_images`.

  In `add-optimiser/SKILL.md`, nothing names the frames or `camera`; leave it.

- [ ] **Step 7: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py
  uv run ruff format pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py
  git add pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py .claude/skills/add-experiment/SKILL.md
  git commit -m "LoadTweezers runs on CameraExperiment and records hamcam_images"
  git push
  ```

### Task 4: Tweezer histogram reads `hamcam_images`

**Files:**
- Modify: `pytweezer/analysis/tweezer_histogram.py` (`FRAMES`, `RESULTS`, the frames branch of `tweezer_histogram`, module docstring)
- Modify: `.claude/skills/add-experiment/SKILL.md` ("Running" paragraph on what the histograms read)
- Test (update only): `tests/test_tweezer_histogram.py`

**Interfaces:**
- Produces: `FRAMES = "hamcam_images"`; `RESULTS["frames"] == (FRAMES,)`. The frames branch reads `stack = measurement.results[FRAMES]` (`(points, 2, H, W)`), finds sites in the mean of the filtered `stack[:, 0]` and scores `stack[:, settings.image]`. `needed_results` of a frames file is `["hamcam_images"]`. Files with only `first_image`/`second_image` have no source (`KeyError` "this file recorded no tweezer images, counts or photons"), as decided.

- [ ] **Step 1: Update the existing tests.** In `tests/test_tweezer_histogram.py`:
  - `ImagesArray.run_point` records `self.record("hamcam_images", np.stack([patterned_frame(self.point.index), tweezer_frame([], seed=1000 + self.point.index)]))`, with the docstring "Records the two frames of each point as one stack, as LoadTweezers does; the second is empty.";
  - `test_needed_results_names_only_what_the_source_uses` expects `["hamcam_images"]` for the frames file.

- [ ] **Step 2: Run the tests and check they fail.** `uv run pytest tests/test_tweezer_histogram.py -q`. Expected: the frames tests FAIL with `KeyError` "recorded no tweezer images".

- [ ] **Step 3: Implement the interface above** in `tweezer_histogram.py`.

- [ ] **Step 4: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_histogram.py tests/test_results_analyses.py -q`, then `uv run pytest tests/ -q`. Expected: all pass.

- [ ] **Step 5: Make the minor skill edit.** In the "Running" paragraph, change "else frames" to "else the `hamcam_images` stack".

- [ ] **Step 6: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py
  uv run ruff format pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py
  git add pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py .claude/skills/add-experiment/SKILL.md
  git commit -m "Tweezer histogram reads frames from hamcam_images"
  git push
  ```

### Task 5: `CaFFluorescence` on `CameraExperiment`

**Files:**
- Modify: `pytweezer/experiments/caf_fluorescence.py`
- Modify: `.claude/skills/add-optimiser/SKILL.md` (minor: "CaF laser tuning" says the measurement streams from `thorcam`, a `CameraExperiment`)
- Test (update only): `tests/test_caf_fluorescence.py`

**Interfaces:**
- Consumes: Task 2's streaming `CameraShots`, `take_shot`, generated `frame_settle_s`/`frame_timeout_s`, `frame_poll_s`.
- Produces:
  - `CaFFluorescence(CameraExperiment)` with `thorcam = CameraShots("CaF ThorCam", frames=3, stream=True, trigger=None, locked=("frames",))`: one frame per iteration of `caf`'s `iterations=3`; no exposure, binning, gain, ROI or trigger is set, so the CONFIG bulb trigger, binning 16 and 12 dB stand, and no `thorcam.*` argument appears in the form.
  - `camera`, `shot()`, the hand-declared `frame_settle_s`/`frame_timeout_s` and the `frame_poll_s` class attribute are removed (the base provides the last three under the same names and defaults).
  - `prepare` takes the background with `self.take_shot(**{"caf.MOTCoilsCurrentValue": self.bg_field})["thorcam"]`, as float; `measure` uses `self.take_shot()["thorcam"]`, as float. The no-frames `RuntimeError` now comes from `take_shot`, still naming "CaF ThorCam" and the script.
  - `OptimiseMOTLasers(StagedScan, CaFFluorescence)` is unchanged. `StagedScan` sets its own points, so `CameraExperiment.planned_points` is not reached for it; it has no camera axis to refuse anyway.

- [ ] **Step 1: Update the existing tests.** In `tests/test_caf_fluorescence.py`, `OneIteration` also declares `thorcam = CameraShots("CaF ThorCam", frames=1, stream=True, trigger=None, locked=("frames",))`, so its shots catch the frames it expects. Nothing else should need to change: the fakes are looked up by device name, and `FakeCamera` already has `arm`, `frames_acquired` and `read_frames_from`. If a test does break, update it to the design and say why in the commit.

- [ ] **Step 2: Port `CaFFluorescence`** to the interfaces above and update its docstring (frames come from `thorcam`, streaming).

- [ ] **Step 3: Run the tests.** `uv run pytest tests/test_caf_fluorescence.py tests/test_optimise_mot_lasers.py tests/test_notebook_rig.py tests/test_experiment_gui.py -q`, then `uv run pytest tests/ -q`. Expected: all pass, including `test_runs_against_the_simulated_motmaster_and_camera`.

- [ ] **Step 4: Check the consumers by hand.**
  - `OptimiseMOTLasers.schema()["arguments"]` still has `frame_settle_s` and `frame_timeout_s` and no `thorcam.*` or `camera`; `OptimiseMOTLasers.reserved_devices()` is the same set as on main (print it at Task 4's commit and again now).
  - `Rig.auto_mot` submits only laser arguments plus the caller's `**arguments`: nothing to change. Its docstring example (`mot_field`) still names a real argument.
  - Build the Optimisers tab offscreen on `OptimiseMOTLasers` as `tests/test_experiment_gui.py` does and check it lists the laser stages as before.

- [ ] **Step 5: Make the minor skill edit** in `add-optimiser/SKILL.md` (see Files). `notebook-rig/SKILL.md` describes the rig's own frame collection, which does not change; leave it.

- [ ] **Step 6: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiments/caf_fluorescence.py tests/test_caf_fluorescence.py
  uv run ruff format pytweezer/experiments/caf_fluorescence.py tests/test_caf_fluorescence.py
  git add pytweezer/experiments/caf_fluorescence.py tests/test_caf_fluorescence.py .claude/skills/add-optimiser/SKILL.md
  git commit -m "CaFFluorescence streams its frames from a CameraShots"
  git push
  ```

### Task 6: Whole-branch checks and PR

- [ ] **Step 1: Run lint, the suite and the docs build.** `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -q`, then `uv sync --group docs && cd docs && uv run make html`. Expected: clean, all pass, and the docs build without new warnings for `camera_experiment`.
- [ ] **Step 2: Check for stale names.** `grep -rn "SAVED_SEQUENCE_PARAMETERS\|first_image\|second_image\|\.shot(\|roi_origin\b\|sensitivity\|ham_images\|\"ham\"" pytweezer/ tests/ .claude/skills/`. Expected: no hits except drivers'/`camera_base`'s `roi_origin` and `set_sensitivity`, `ContinuousAcquisition`'s `roi_origin` record, and `tests/test_analysis.py`'s live-analysis test name.
- [ ] **Step 3: Open the PR** from `camera-experiment` to `main` (GitHub MCP `create_pull_request`), with a body that summarises the design, links the spec, lists the decisions above, and lists the consumers below. End it with the attribution line. Do not merge it.
- [ ] **Step 4: Ask the user about the camera section.** Show them the draft below and ask for approval. Apply it only after they agree, as its own commit.

## Consumers that may break

- **Notebooks:** re-grep the Oct–Nov 2026 tweezer and CaF notebooks for `LoadTweezers`, `SAVED_SEQUENCE_PARAMETERS`, `TweezerArrayMeasurement`, `FindTrapSites`, `tweezer_histogram`, `first_image`, `CaFFluorescence` (and its `.shot(` or `.camera`) and `sensitivity=`. The first plan found no hits for the tweezer names (the CaF hits were pycaf's own `*first_image(...)` helpers); `CaFFluorescence` is new since. List any hit for the user; never edit a notebook.
- **Saved recipes** for `LoadTweezers`/`TweezerArrayMeasurement`/the Rb optimisers that name `roi_x0`, `roi_y0`, `roi_width`, `roi_height` or `sensitivity` stop validating. Load each into the form and save it again. Recipes for `CaFFluorescence`/`OptimiseMOTLasers` are unaffected.
- **Measurement files** from before the change that hold `first_image`/`second_image` and no counts get no tweezer histogram. Files with counts are unaffected.
- **Docs and notes:** `docs/notes/motmaster_go_overhead_handoff.md` mentions `LoadTweezers` only by name. `add-applet` says `LoadTweezers` passes `rid`, which is still true.

## Draft camera section for `add-experiment` (significant change: for approval, not applied)

Proposed as a new `## Camera experiments` section after "MOTMaster sequences", replacing that section's hand-written `Tof` camera calls with a pointer here:

````markdown
## Camera experiments

"Set up cameras, run the sequences, read N frames from each" is declared, not
written (`pytweezer/experiment/camera_experiment.py`):

```python
from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
from pytweezer.experiment.motmaster import MotMaster


class PureRecipe(CameraExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
    hamcam = CameraShots("Rb HamCam", frames=2, roi=(0, 512, 0, 512), locked=("frames",))
    thorcam = CameraShots("Rb ThorCam", frames=1, exposure=20e-3, stream=True)
```

- Each `CameraShots` is a device (declared, reserved) whose settings become
  form arguments named `<camera>.frames`, `.exposure`, `.binning`, `.gain`,
  `.roi_x0`, `.roi_width`, `.roi_y0`, `.roi_height` (group "Camera: <camera>"),
  defaulting to the declaration. `roi=(x0, width, y0, height)` in unbinned
  sensor pixels, as `set_roi` takes it; binned frames are `width // binning`
  wide. `gain` is in the camera's unit (ThorCam dB, ORCA contrast-gain index,
  HamCam sensitivity). A setting left `None` is never set; a `locked` one is
  fixed at the declared value and not in the form. `trigger="ext"` by default;
  `None` keeps the camera's CONFIG trigger.
- `prepare()` applies, per camera: binning, ROI, exposure, trigger,
  `configure_camera(name, camera)` (override it for driver extras: EM gain,
  `set_external_exposure_mode`), gain, then `setup_acquisition(frames,
  mode="snap")`, or `arm()` for a `stream=True` camera. Origins the cameras
  used are in `self.roi_origins[name]`.
- Snap cameras (the default) are started before the sequences and read for
  exactly `frames` frames (broadcast with `{"rid": self.rid}`), then stopped,
  even on failure. Stream cameras run free (use them for ThorCams, whose
  driver ignores snap mode): each shot reads the frames caught since a mark
  once the counts have been still for `frame_settle_s` (at most
  `frame_timeout_s`); none is an error, a count other than `frames` a warning.
- `take_shot(**overrides)` does one shot (overrides as for `run_sequences`):
  `{name: (frames, H, W) array}`. `self.cameras` is `{name: client}`.
- The default `run_point` records each stack as `<camera>_images`;
  `record_frames({name: stack})` does the same for your own `run_point`.
- `exposure`, `gain`, `roi_x0` and `roi_y0` can be scanned (re-applied before
  the point's shot); `frames`, `binning`, `roi_width` and `roi_height` change
  the recorded shape, so scanning them is refused.
````
