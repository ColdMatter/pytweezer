# CameraExperiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiments that run "set up cameras, run MOTMaster, read N frames per camera" are written as declarations (`CameraShots`, `MotMaster(saved=...)`), and `LoadTweezers` is rebuilt on them.

**Architecture:** `MotMaster` gains a `saved=` binding that `MotMasterExperiment` resolves in `prepare()` and merges in `run_sequences()`. A new `CameraExperiment(MotMasterExperiment)` in `pytweezer/experiment/camera_experiment.py` turns each `CameraShots` class attribute into dotted form arguments (real class attributes, so the catalogue, form, scans and recipes see them with no other change) and runs generic arm/run/read loops over `self.cameras`. `LoadTweezers` becomes a `CameraExperiment`, recording `ham_images`; the tweezer histogram reads that stack.

**Tech Stack:** Python 3, numpy, h5py (via `MeasurementWriter`), sipyco RPC clients, PyQt6 (form only, unchanged), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-camera-experiment-design.md`

## Questions for the user (raised while reading the code)

Each has a default the plan follows; say if you want otherwise.

1. **Hook order.** The spec lists `setup_acquisition` before `configure_camera`. Today `LoadTweezers` sets the external exposure mode, EM gain and sensitivity *before* `setup_acquisition(2, "snap")`, and DCAM may refuse some property changes once a buffer is allocated. The plan calls `configure_camera` **before** `setup_acquisition`, so the HamCam sees exactly today's call order.
2. **Frames and ROI size can't be scanned.** A recorded result's shape is fixed by its first point, so scanning `<camera>.frames`, `.roi_width` or `.roi_height` would fail at the second point when `<camera>_images` changes shape. The plan has `CameraExperiment.planned_points` refuse these scans with an explanation. `exposure`, `roi_x0` and `roi_y0` stay scannable and are re-applied before the shot of any point that changes them. `LoadTweezers` also refuses `ham.roi_x0`/`ham.roi_y0` scans, because its trap sites are placed in the ROI fixed at `prepare()`.
3. **No new tests** (your standing rule). Only `LoadTweezers` and its subclasses use `CameraExperiment`, so their existing tests cover it with one camera. Several cameras, argument generation, locking and re-applying scanned settings are checked by a scratch script in the scratchpad (Task 2), which is not kept in the repo. Should these get permanent tests in `tests/test_motmaster_experiment.py` (or a new `tests/test_camera_experiment.py`)?
4. **`saved_sequence` changes shape.** It becomes `{"rb.tMOTLoadRepVCO": value, ...}` (dotted, so it covers several MotMasters), where it was `{"tMOTLoadRepVCO": value}`. Nothing in the repo or the recent notebooks reads it. As today, a script parameter that is only scanned is still listed, because scanned values reach the instance after `prepare()`.
5. **`LoadTweezers.camera` becomes `LoadTweezers.ham`**, the declaration in the spec. Nothing outside `load_tweezers.py` uses `.camera`.

No part of the design conflicts with the code. The GUI form needs no code change: it builds a row for every entry in `schema["arguments"]` and treats a dotted name as a MOTMaster parameter only when the part before the dot is a MOTMaster attribute (`arg_editor.py` `load_request`, `_add_motmaster_boxes`). `recipes.unknown_arguments` and `scan.points` read the same declared arguments.

## Global Constraints

- Branch: `camera-experiment` off `main`, with a PR. Commit and push after each task; stage only your own files.
- Before each commit, run `uv run ruff check --fix <files>` and `uv run ruff format <files>`, then `git add` the files whole. Never use `--no-verify`.
- Each task ends with `uv run pytest tests/ -q` green.
- Add no new test files or test cases. Update or delete only the existing tests the change breaks. Tests follow the design; never add hooks to production code for a test.
- Google-style docstrings that describe current behaviour only, UK English, comments only where the code surprises, `get_logger` for logging.
- Don't touch hardware at import or class definition (the manager imports modules to list their arguments).
- Don't change camera drivers or `camera_base.py`. The generic loops use only `set_roi`, `set_exposure_time`, `set_trigger_source`, `setup_acquisition`, `start_acquisition`, `acquire_n_frames`, `stop_acquisition`.
- Run any Qt check with `QT_QPA_PLATFORM=offscreen`, never on the live display.
- Notebooks are never edited.
- Names fixed by the spec: `CameraExperiment`, `CameraShots`, `configure_camera`, `take_shot`, `record_frames`, `self.cameras`, `<camera>.frames`, `<camera>.exposure`, `<camera>.roi_x0`, `.roi_width`, `.roi_y0`, `.roi_height`, `<camera>_images`, `roi=(x0, width, y0, height)`, `locked=("frames",)`, `MotMaster(..., saved=...)`, `saved_sequence`.

## Decisions on the spec's "Left to the builder"

- **Dotted arguments:** `CameraExperiment.__init_subclass__` `setattr`s the generated `Integer`/`Number` objects onto the class under their dotted names, with `.name` set by hand because `setattr` skips `__set_name__`. `collect()`, and therefore `arguments()`, `schema()`, `coerce_arguments`, `Scan.points`, `unknown_arguments` and the form, sees them as declared arguments. `getattr(self, "ham.exposure")` works because an instance's `__dict__` overrides a non-data descriptor. Group: `f"Camera: {name}"`.
- **Locked settings** generate no argument. If an inherited argument has that name, the subclass sets the class attribute to `None`, which `collect()` treats as removing it. Submitting a locked or missing setting by name is refused as an unknown argument.
- **Optional settings:** `exposure=None` means "don't call `set_exposure_time`" and `roi=None` means "don't call `set_roi`". Neither generates arguments. `roi_origins[name]` is `None` for such a camera.
- **Trigger source:** the keyword `CameraShots(trigger="ext")`, not a form argument. `"ext"` is a valid mode for every camera driver (pylablib DCAM: ImagEM/ORCA; Thorlabs TLCam: both ThorCams) and suits frames taken around a MOTMaster run. The HamCam's `set_external_exposure_mode()` then refines it in `configure_camera`.
- **Scanning:** see Question 2.
- **Errors:** keep today's texts: `RuntimeError("the saved parameters could not be read: …")` and `KeyError("no saved parameter 'a', 'b'")`. A bound parameter missing from the script raises the existing `_check_parameter` `ValueError` in `prepare()`.
- **Records:** the default `run_point` records `<name>_images` (`(frames, H, W)` per point). `record_frames(frames)` records every entry of a `{name: array}` dict as `<name>_images`, so `take_shots` can pass `{"ham": (shots, 2, H, W)}`.
- **Old LoadTweezers recipes** that name `roi_x0` etc. are refused (`unknown_arguments`). No migration. Loading one into the form ignores the old names (the new defaults equal the old ones, `110/220/256/256`), and saving it again fixes it.

## Review Focus

Under the no-new-tests rule, the scratch scripts in Tasks 2 and 3 check each of these by hand.

1. **Scanning `thor.exposure` or `thor.roi_x0`.** Each point's shot must use that point's value: the setting is re-applied before the cameras are armed, and only when it changed.
2. **Scanning a shape-changing setting**, or any `ham.roi_*` in `LoadTweezers`. Refused at `planned_points` with a `ValueError` that names the argument and says why; the task fails before any shot.
3. **A sequence or frame read that raises mid-shot.** Every camera is still stopped (`finally`), and the original error reaches the task.
4. **A saved-bound script parameter set in the form or scanned.** The form or scan value is sent and the saved one is not. `saved_sequence` leaves out a parameter set in the form.
5. **A locked setting submitted by name** (`run_local(LoadTweezers, **{"ham.frames": 3})`) or a pre-change argument name (`roi_x0`). Refused, naming the argument.

---

### Task 1: Saved-parameter binding on `MotMaster`

**Files:**
- Modify: `pytweezer/experiment/motmaster.py` (`MotMaster.__init__`, `MotMasterExperiment.prepare`, `run_sequences`, module docstring)
- Modify: `pytweezer/experiments/load_tweezers.py`: delete `SAVED_SEQUENCE_PARAMETERS` and the reading/merging block in `prepare`, and declare `rb = MotMaster(..., saved=…)`. `shot()` stops adding saved values.
- Modify: `pytweezer/experiment/saved_parameters.py:31`: the FIXME comment's example becomes "a `MotMaster`'s `saved=` bindings".
- Modify: `.claude/skills/add-experiment/SKILL.md` (minor: see step 6)
- Test (update only): `tests/test_tweezer_array.py`

**Interfaces:**
- Produces:
  - `MotMaster(device_name, *, script, ..., saved: Iterable[str | dict[str, str]] = (), timeout=None)`. `MotMaster.saved: dict[str, str]` is `{script parameter: saved key}`, normalised from bare names (same key) and dicts (renames). Raises `TypeError` at class definition for an entry that is neither a `str` nor a `dict` of `str` to `str`, or a parameter bound to two different keys.
  - `MotMasterExperiment.saved_parameters: dict[str, Any]` is every saved parameter read in `prepare()`, or `{}` when no `MotMaster` binds any. Read in exactly one call to `get_saved_parameters()`, imported at module level in `motmaster.py` so tests can patch `pytweezer.experiment.motmaster.get_saved_parameters`.
  - `MotMasterExperiment.saved_sequence: dict[str, Any]`, `{"<attribute>.<parameter>": value}` for bound parameters the form did not set at `prepare()`. Recorded as the constant `saved_sequence` only when some binding exists.
  - `run_sequences(**overrides)` precedence, highest first: `overrides`, then `motmaster_parameters()` (form, dotted, scanned), then the saved values, then the script default.

- [ ] **Step 1: Branch.** `git switch -c camera-experiment main`.

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

  In `load_tweezers.py`, keep the names in one private tuple and declare:
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
  `LoadTweezers.prepare` passes `self.saved_parameters` to `saved_trap_sites`. Remove the `client` imports it no longer uses. `LoadTweezers.rb.saved` is the importable list of names.

- [ ] **Step 5: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_array.py tests/test_motmaster_experiment.py tests/test_tweezer_histogram.py -q`, then `uv run pytest tests/ -q`. Expected: all pass.

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
- No test changes (nothing existing uses it yet).

**Interfaces:**
- Consumes: `MotMasterExperiment.prepare()` / `run_sequences(**overrides)` from Task 1; `Device`, `Integer`, `Number`, `collect` from `arguments.py`.
- Produces:
  - `SETTINGS = ("frames", "exposure", "roi")`.
  - `CameraShots(Device)`, built as `CameraShots(device_name, *, frames: int = 1, exposure: float | None = None, roi: tuple[int, int, int, int] | None = None, trigger: str = "ext", locked: Iterable[str] = (), timeout: float | None = None)`. At class definition it raises `ValueError` for `frames < 1`, `exposure <= 0`, an `roi` that is not four whole numbers, or a `locked` entry outside `SETTINGS`.
    - `CameraShots.arguments() -> dict[str, Argument]`: `{f"{name}.frames": Integer(frames, min=1), f"{name}.exposure": Number(exposure, unit="ms", scale=1e-3, min=0), f"{name}.roi_x0": Integer(x0, min=0, unit="px"), ".roi_width": Integer(min=1, unit="px"), ".roi_y0": Integer(min=0, unit="px"), ".roi_height": Integer(min=1, unit="px")}`, leaving out every `None` or locked setting, all with `group=f"Camera: {name}"`.
    - `CameraShots.to_schema()` adds `"camera": {"frames", "exposure", "roi", "trigger", "locked"}` to `Device.to_schema()`.
  - `ShotSettings`, a frozen dataclass of `(frames: int, exposure: float | None, roi: tuple[int, int, int, int] | None)`.
  - `CameraExperiment(MotMasterExperiment)`:
    - `__init_subclass__`: for each `CameraShots` in `vars(cls)`, `setattr` its `arguments()` onto `cls` and hide inherited arguments of locked settings (see Decisions).
    - `camera_shots() -> dict[str, CameraShots]` (classmethod, `collect(cls, CameraShots)`).
    - `cameras -> dict[str, Any]` (property; `{name: getattr(self, name)}`).
    - `shot_settings(name) -> ShotSettings`: the declared value for locked or `None` settings, else the argument's current value.
    - `apply_camera_settings(name) -> None`: in this order `set_roi(*roi)` (keeping the return in `self.roi_origins[name]` as a tuple), `set_exposure_time`, `set_trigger_source(trigger)`, `configure_camera(name, camera)`, `setup_acquisition(frames, mode="snap")`. It then stores the settings in `self._applied[name]`.
    - `configure_camera(name: str, camera: Any) -> None`: a hook that does nothing by default.
    - `prepare()`: `super().prepare()`, then `apply_camera_settings` for every camera.
    - `planned_points(scan)`: raises `ValueError` if a scan axis is `<camera>.frames`, `.roi_width` or `.roi_height` ("…changes the shape of `<camera>_images`; set it for the whole task"), else `super().planned_points(scan)`.
    - `take_shot(**overrides) -> dict[str, np.ndarray]`: re-apply any camera whose `shot_settings` differ from `_applied`; `start_acquisition` on every camera; `run_sequences(**overrides)`; `acquire_n_frames(frames, broadcast=True, header={"rid": self.rid})` per camera as `np.asarray`; `stop_acquisition` on every camera in a `finally`.
    - `record_frames(frames: dict[str, np.ndarray]) -> None`: `record(f"{name}_images", stack)` per entry.
    - `run_point()`: `self.record_frames(self.take_shot())`.

- [ ] **Step 1: Write `camera_experiment.py`** to the interfaces above. Its module docstring is the spec's `PureRecipe` example plus one paragraph on dotted arguments, locking and which settings can be scanned. No comments beyond the surprising: the `setattr` of dotted names and the `None` that hides a locked inherited argument.

- [ ] **Step 2: Check it by hand with a simulated run** (no hardware, no manager). Write `<scratchpad>/camera_smoke.py`, outside the repo:
  ```python
  import pytest
  from pytweezer.experiment import ListAxis, Scan, run_local
  from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
  from pytweezer.experiment.motmaster import MotMaster

  class TwoCameras(CameraExperiment):
      rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
      thor = CameraShots("Rb ThorCam", frames=1, exposure=10e-3, roi=(0, 64, 0, 32))
      orca = CameraShots("Rb ORCA", frames=2, locked=("frames",))

  assert {"thor.frames", "thor.exposure", "thor.roi_x0", "thor.roi_width",
          "thor.roi_y0", "thor.roi_height"} <= set(TwoCameras.arguments())
  assert not any(n.startswith("orca.") for n in TwoCameras.arguments())
  m = run_local(TwoCameras, Scan(axes=[ListAxis(argument="thor.exposure", values=[5e-3, 20e-3])]), simulate=True)
  assert m.status == "completed", m.attrs.get("error")
  assert m.results["thor_images"].shape == (2, 1, 32, 64)
  assert m.results["orca_images"].shape[:2] == (2, 2)
  bad = run_local(TwoCameras, Scan(axes=[ListAxis(argument="thor.frames", values=[1, 2])]), simulate=True)
  # planned_points raises before a file exists:
  ```
  Finish the last check to match how `run_local` reports a `planned_points` `ValueError` (read `runner.run_local`): either `pytest.raises(ValueError, match="thor.frames")` or `bad.status == "failed"` with `thor.frames` in the error. Also check that `run_local(TwoCameras, simulate=True, **{"orca.frames": 3})` is refused with "orca.frames" named (Review Focus 5). Then check Review Focus 1: subclass with a `configure_camera` that appends `(name, self.shot_settings(name).exposure)` to a list, run the exposure scan, and expect the list to read `[("thor", 10e-3), ("orca", None), ("thor", 5e-3), ("thor", 20e-3)]`. That is one apply per camera at `prepare` and one per change; ORCA is not re-applied. Then check Review Focus 3: subclass `TwoCameras` with a `run_sequences` that raises `RuntimeError`, run it with `simulate=True`, and from its `finish()` assert every `self.cameras[name]._acquisition_in_progress()` is `False` (the simulated backend is the in-process object) and that the run failed with the `RuntimeError`. Run with `uv run python <scratchpad>/camera_smoke.py`. Expected: no assertion errors.

- [ ] **Step 3: Check the form, offscreen.** `QT_QPA_PLATFORM=offscreen uv run python <scratchpad>/camera_form.py`, where the script builds an `ArgumentEditor()` with `fetcher.threaded = False` and `set_parameter_source(lambda device, script: {"tDelay1": 5})` (as `tests/test_experiment_gui.py::sequenced_editor` does), calls `set_experiment(TwoCameras.schema())`, and asserts:
  - `"thor.exposure"` is in `editor.rows` and `"orca.frames"` is not;
  - `editor.request().args["thor.roi_width"] == 64`;
  - `load_request(editor.request())` round-trips the edited value.

  No GUI code change is expected. If one is needed, stop and report it.

- [ ] **Step 4: Run the suite.** `uv run pytest tests/ -q`. Expected: all pass.

- [ ] **Step 5: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiment/camera_experiment.py && uv run ruff format pytweezer/experiment/camera_experiment.py
  git add pytweezer/experiment/camera_experiment.py
  git commit -m "CameraExperiment: cameras declared as CameraShots, settings as dotted arguments"
  git push
  ```

### Task 3: `LoadTweezers` on `CameraExperiment`; optimisers and their tests to `ham_images`

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py` (whole class body)
- Modify: `pytweezer/experiments/tweezer_array_measurement.py` (`save_raw_images` tooltip)
- Modify: `pytweezer/optimisers/find_trap_sites.py` (`roi_origin` → `roi_origins["ham"]`; tooltip)
- Modify: `.claude/skills/add-experiment/SKILL.md` (minor: see step 5)
- Test (update only): `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: Task 2's `CameraShots`, `CameraExperiment`, `take_shot`, `record_frames`, `shot_settings`, `roi_origins`, `configure_camera`, `planned_points`; Task 1's `saved_parameters`.
- Produces:
  - `LoadTweezers(CameraExperiment)` with `ham = CameraShots("Rb HamCam", frames=2, roi=(110, 256, 220, 256), locked=("frames",))`. `exposure` stays `None`, because the trigger pulse exposes the HamCam.
  - `LoadTweezers.sensitivity` stays an `Integer(600, min=0, group="Camera: ham", …)`, so it sits with the HamCam's settings. `save_raw_images` and `save_tweezer_counts` keep group `"Camera"`.
  - `configure_camera(name, camera)` calls `set_external_exposure_mode()`, `enable_em_gain(True)`, `enable_direct_em_gain(True)` and `set_sensitivity(self.sensitivity)`.
  - `planned_points(scan)` raises `ValueError` for any `ham.roi_*` axis: "the trap sites are placed in the ROI set before the first point". Otherwise it defers to `super()`.
  - `take_shots(n_shots, **sequence) -> tuple[list[np.ndarray], list[np.ndarray]]` has an unchanged signature and return. With `save_raw_images` it records `ham_images`, `(shots, 2, H, W)` per point.
  - `run_point` records `ham_images` `(2, H, W)` (with `save_raw_images`) through `record_frames`. `signal_counts`, `first_/second_occupied`, `loading`, `first_/second_counts` are unchanged, computed from `frames["ham"][0]` and `[1]`.
  - `saved_trap_sites` reads the origin from `self.roi_origins["ham"]` and the ROI size from `self.shot_settings("ham").roi`.
  - `shot()` and `finish()` are deleted: `take_shot` stops the camera itself.

- [ ] **Step 1: Update the existing tests to the new design.** In `tests/test_tweezer_array.py`:
  - `FakeHamCam` gains `def set_trigger_source(self, source): ...`.
  - `test_a_measurements_counts_are_the_raw_pixel_sums_of_its_frames`: frames are `measurement.results["ham_images"][0][:, i]` for `i, name in enumerate(("first", "second"))`.
  - `test_load_tweezers_records_each_sites_raw_counts_and_occupancy`: `window_sums(results["ham_images"][:, i])`.
  - `test_load_tweezers_without_raw_images_records_no_frames`: `assert "ham_images" not in measurement.results`, replacing the two `first_/second_image` asserts.
  - `test_load_tweezers_without_saved_sites_records_frames_only`: `assert "ham_images" in measurement.results`.
  - `test_a_measurement_records_occupancy_and_frames_only_when_asked`: `("ham_images" in measurement.results) == save_raw_images` and shape `(1, FAST["shots"], 2, 128, 128)`.

- [ ] **Step 2: Run the tests and check they fail.** `uv run pytest tests/test_tweezer_array.py -q`. Expected: FAIL on `ham_images` missing (`KeyError`/assert).

- [ ] **Step 3: Rewrite `LoadTweezers`** to the interfaces above, and update its docstring for `ham_images` and the saved binding. `TweezerArrayMeasurement` and `FindTrapSites` change only in their tooltip ("record every shot's camera frames as ham_images") and in `FindTrapSites.fit`'s `self.roi_origins["ham"][::-1]`.

- [ ] **Step 4: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_array.py tests/test_tweezer_histogram.py -q`, then `uv run pytest tests/ -q`. Expected: all pass (`tests/test_tweezer_histogram.py` still records its own `first_image` until Task 4).

- [ ] **Step 5: Check Review Focus 2, 4 and 5 by hand** in `<scratchpad>/load_tweezers_smoke.py`, reusing the `rig` fakes by importing `FakeSequencer`, `FakeHamCam`, `FakeRig` and `SAVED` from `tests.test_tweezer_array` and patching as the fixture does:
  - a `ListAxis(argument="ham.roi_x0", …)` scan fails, naming `ham.roi_x0`;
  - `run_local(LoadTweezers, **{"ham.frames": 3})` and `run_local(LoadTweezers, roi_x0=100)` are refused, naming the argument;
  - `run_local(LoadTweezers, Scan(axes=[ListAxis(argument="rb.tMolCoolVCO1", values=[7.5, 8.5])]))` sends 7.5 then 8.5, never the saved value.

  Also run an offscreen `ArgumentEditor` on `LoadTweezers.schema()` and check that it shows `ham.roi_x0`…`ham.roi_height` and `sensitivity` in the group "Camera: ham", and no `ham.frames`.

- [ ] **Step 6: Make the minor skill edits** in `add-experiment/SKILL.md` "Tweezer array measurements":
  - item 3: ROI from `ham.roi_x0`/`ham.roi_y0`/`ham.roi_width`/`ham.roi_height`, with sites moved by the origin `set_roi` returned (`roi_origins["ham"]`); frames recorded as `ham_images` `(2, H, W)` only with `save_raw_images`; ROI scans refused.
  - item 4: `save_raw_images` records `ham_images`, `(shots, 2, H, W)` per point, replacing `first_images`/`second_images`.

- [ ] **Step 7: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py
  uv run ruff format pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py
  git add pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array_measurement.py pytweezer/optimisers/find_trap_sites.py tests/test_tweezer_array.py .claude/skills/add-experiment/SKILL.md
  git commit -m "LoadTweezers runs on CameraExperiment and records ham_images"
  git push
  ```

### Task 4: Tweezer histogram reads `ham_images`

**Files:**
- Modify: `pytweezer/analysis/tweezer_histogram.py` (`FRAMES`, `RESULTS`, the frames branch of `tweezer_histogram`, module docstring)
- Modify: `.claude/skills/add-experiment/SKILL.md` ("Running" paragraph on what the histograms read)
- Test (update only): `tests/test_tweezer_histogram.py`

**Interfaces:**
- Produces: `FRAMES = "ham_images"`; `RESULTS["frames"] == (FRAMES,)`. The frames branch reads `stack = measurement.results[FRAMES]` (`(points, 2, H, W)`), finds sites in the mean of the filtered `stack[:, 0]` and scores `stack[:, settings.image]`. `needed_results` of a frames file is `["ham_images"]`. Files with only `first_image`/`second_image` have no source (`KeyError` "this file recorded no tweezer images, counts or photons"), as decided.

- [ ] **Step 1: Update the existing tests.** In `tests/test_tweezer_histogram.py`:
  - `ImagesArray.run_point` records `self.record("ham_images", np.stack([patterned_frame(self.point.index), tweezer_frame([], seed=1000 + self.point.index)]))`, with the docstring "Records the two frames of each point as one stack, as LoadTweezers does; the second is empty.";
  - `test_needed_results_names_only_what_the_source_uses` expects `["ham_images"]` for the frames file.

- [ ] **Step 2: Run the tests and check they fail.** `uv run pytest tests/test_tweezer_histogram.py -q`. Expected: the frames tests FAIL with `KeyError` "recorded no tweezer images".

- [ ] **Step 3: Implement the interface above** in `tweezer_histogram.py`.

- [ ] **Step 4: Run the tests and check they pass.** `uv run pytest tests/test_tweezer_histogram.py tests/test_results_analyses.py -q`, then `uv run pytest tests/ -q`. Expected: all pass.

- [ ] **Step 5: Make the minor skill edit.** In the "Running" paragraph, change "else frames" to "else the `ham_images` stack".

- [ ] **Step 6: Commit and push.**
  ```bash
  uv run ruff check --fix pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py
  uv run ruff format pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py
  git add pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py .claude/skills/add-experiment/SKILL.md
  git commit -m "Tweezer histogram reads frames from ham_images"
  git push
  ```

### Task 5: Whole-branch checks and PR

- [ ] **Step 1: Run lint, the suite and the docs build.** `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -q`, then `uv sync --group docs && cd docs && uv run make html`. Expected: clean, all pass, and the docs build without new warnings for `camera_experiment`.
- [ ] **Step 2: Check for stale names.** `grep -rn "SAVED_SEQUENCE_PARAMETERS\|first_image\|second_image\|\.shot(\|roi_origin\b" pytweezer/ tests/ .claude/skills/`. Expected: no hits except driver/`camera_base` `roi_origin` and `tests/test_analysis.py`'s live-analysis test name.
- [ ] **Step 3: Open the PR.** `gh pr create --base main --head camera-experiment` with a body that summarises the design, links the spec, lists the Questions above with the defaults taken, and states that no notebooks were found to break (see below). End it with the attribution line.
- [ ] **Step 4: Ask the user about the camera section.** Show them the draft below and ask for approval. Apply it only after they agree, as its own commit.

## Consumers that may break

- **Notebooks:** grepping the Oct–Nov 2026 tweezer notebooks (`caftweezers/ipython_notebooks/*.ipynb`, `2026/09 Sep`, `scripts`, `Experiment Optimisation Scripts`, `tests`) and the CaF notebooks (`cafmot/.../2026/10-Oct`, `03Nov26.ipynb`) found no use of `LoadTweezers`, `SAVED_SEQUENCE_PARAMETERS`, `TweezerArrayMeasurement`, `FindTrapSites`, `tweezer_histogram` or the `first_image` result (the CaF hits are pycaf's own `*first_image(...)` helpers). Older folders were not searched; `LoadTweezers` was added on 2026-10-28.
- **Saved recipes** for `LoadTweezers`/`TweezerArrayMeasurement` that name `roi_x0`, `roi_y0`, `roi_width` or `roi_height` stop validating. Load each into the form and save it again.
- **Measurement files** from before the change that hold `first_image`/`second_image` and no counts get no tweezer histogram. Files with counts are unaffected.
- **Docs and notes:** `docs/notes/motmaster_go_overhead_handoff.md` mentions `LoadTweezers` only by name, so it is unaffected. `add-applet` says `LoadTweezers` passes `rid`, which is still true.

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
    ham = CameraShots("Rb HamCam", frames=2, roi=(0, 512, 0, 512), locked=("frames",))
    thor = CameraShots("Rb ThorCam", frames=1, exposure=20e-3)
```

- Each `CameraShots` is a device (declared, reserved) whose settings become
  form arguments named `<camera>.frames`, `.exposure`, `.roi_x0`, `.roi_width`,
  `.roi_y0`, `.roi_height` (group "Camera: <camera>"), defaulting to the
  declaration. `roi=(x0, width, y0, height)`, as `set_roi` takes it. A setting
  left `None` (`exposure` for a camera exposed by its trigger pulse) is never
  set; a `locked` one is fixed at the declared value and not in the form.
  `trigger="ext"` by default.
- `prepare()` sets each camera's ROI, exposure and trigger, calls
  `configure_camera(name, camera)` (override it for driver extras: EM gain,
  `set_external_exposure_mode`), then `setup_acquisition(frames, mode="snap")`.
  Origins the cameras used are in `self.roi_origins[name]`.
- `take_shot(**overrides)` arms every camera, runs the sequences (overrides as
  for `run_sequences`), reads each camera's frames (broadcast with
  `{"rid": self.rid}`) and stops them, even on failure:
  `{name: (frames, H, W) array}`. `self.cameras` is `{name: client}`.
- The default `run_point` records each stack as `<camera>_images`;
  `record_frames({name: stack})` does the same for your own `run_point`.
- `exposure`, `roi_x0` and `roi_y0` can be scanned (re-applied before the
  point's shot); `frames`, `roi_width` and `roi_height` change the recorded
  shape, so scanning them is refused.
````
