# Plain Camera Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `CameraExperiment`, the unused CaF experiments and the notebook `Rig`; reduce `MotMasterExperiment` to one MOTMaster with unprefixed script parameters; rewrite `LoadTweezers` and `LoadTweezersMOTImage` as plain code.

**Architecture:** `MotMasterExperiment` drives exactly one `MotMaster`; any undeclared identifier argument is a script parameter, held in the experiment's `extra_values` dict, not as an attribute. `LoadTweezers` sets the HamCam up in `prepare` and reads it in its own `take_shot`; `LoadTweezersMOTImage` wraps that shot with the ORCA. Batch loops live in the subclasses that use them.

**Tech Stack:** Python 3, PyQt6 (Experiments tab), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-09-plain-camera-experiments-design.md`

## Global Constraints

- One `MotMaster` per class; two is a `TypeError` at class creation that names linked mode.
- Script parameters are unprefixed everywhere: form rows, scans, `run_sequences(**overrides)`, `saved_sequence` keys.
- Dotted names (`rb.tX`, `hamcam.gain`) are not valid arguments: refused by `recipes.unknown_arguments` and at task start, not translated. The form drops them when loading an old request.
- A misspelt script parameter passes submit and fails in the task before any `start_motmaster_experiment`; the "did you mean" hint includes declared argument names.
- Script-parameter names must be valid Python identifiers. A declared argument always wins over a script parameter of the same name (say so in the docstring).
- `LoadTweezers` records exactly what it records today (field names, units, shapes); `LoadTweezersMOTImage` records `orca_*` exactly as `record_cloud_fit` did.
- `LoadTweezers` argument defaults: `hamcam_gain` 600, `hamcam_roi_x0` 110, `hamcam_roi_width` 256, `hamcam_roi_y0` 220, `hamcam_roi_height` 256, group "Camera".
- No shared camera helper, no `configure_camera`-style hooks; subclass-only code stays in the subclass's file.
- Keep `PycafCamera`, `notebook/zips.py`, `drivers/shot_frames.py`.
- UK English, Google-style docstrings, `get_logger`, ruff-clean; suite passes before each push.
- Work on the session's designated branch `ccr-bb840753-gv9frj` (the harness requires it; the spec suggested `plain-camera-experiments`).

**Decision taken here (deviation from the spec's mechanism, same goal):** the spec refuses script parameters whose names clash with experiment attributes (`rig`, `threshold`, `rid`) "since extras are set on the instance". Attributes set later in `prepare` (`threshold`) can't be known when the task starts, so instead extras are not set on the instance at all: `Experiment` keeps them in `self.extra_values`. Nothing can clobber them and they can clobber nothing, so no clash check is needed.

## Review Focus

1. An old recipe, last submission or persisted task carrying `rb.tX`/`hamcam.gain`: the recipe is refused with "no longer has argument(s)"; the form loads the rest and leaves those out, never making a row named `hamcam.gain`. Tests in Task 2.
2. A misspelt script parameter, set in the form or only scanned, fails before any MOTMaster run, with a hint that includes declared argument names. Tests in Task 2.
3. A shot that raises mid-sequence still stops the HamCam's (and ORCA's) acquisition, and `finish` puts a camera that was live back into live mode. Tests in Task 3.
4. A script parameter that shares a name with an experiment attribute (`threshold` on `TweezerArrayMeasurement`) reaches MOTMaster unchanged and leaves the attribute alone. Test in Task 2.
5. A scanned `hamcam_gain` is re-sent before each shot whose gain differs; an ROI axis is refused before the first point. Tests in Task 3.

---

### Task 1: Remove the CaF experiments, OptimiseMOTLasers and the notebook Rig

**Files:**
- Delete: `pytweezer/experiments/caf_mot.py`, `pytweezer/experiments/caf_fluorescence.py`, `pytweezer/optimisers/optimise_mot_lasers.py`, `pytweezer/notebook/rig.py`, `tests/test_caf_fluorescence.py`, `tests/test_optimise_mot_lasers.py`, `tests/test_notebook_rig.py`
- Modify: `pytweezer/notebook/__init__.py` (export only `PycafCamera`), `pytweezer/notebook/pycaf_camera.py` (own `_motmaster_file_config`), `tests/test_notebook_pycaf_camera.py` (own `MONTHS`, `FakeCamera`), `pytweezer/experiment/optimisers.py` (docstrings mentioning `Rig.auto_mot`)

- [ ] **Step 1:** Move `_motmaster_file_config` from `rig.py` into `pycaf_camera.py` (module level, replacing the local import) and `MONTHS`/`FakeCamera` from `tests/test_notebook_rig.py` into `tests/test_notebook_pycaf_camera.py`.
- [ ] **Step 2:** Delete the files listed; drop `Rig`, `Shot`, `plot_auto_mot_results` from `pytweezer/notebook/__init__.py`; reword the two `Rig.auto_mot(accept=True)` mentions in `experiment/optimisers.py` to name only the Optimisers tab.
- [ ] **Step 3:** Verify: `grep -rn "caf_mot\|caf_fluorescence\|optimise_mot_lasers\|notebook.rig\|test_notebook_rig\|Rig.auto_mot" pytweezer tests bin` prints nothing; `uv run pytest tests/test_notebook_pycaf_camera.py tests/test_notebook_zips.py tests/test_optimisers.py -q` passes.
- [ ] **Step 4:** ruff, commit "Remove CaFMOT, CaFFluorescence, OptimiseMOTLasers and the notebook Rig", push.

### Task 2: MotMasterExperiment drives one MOTMaster; script parameters are unprefixed

**Files:**
- Modify: `pytweezer/experiment/experiment.py` (extras in `extra_values`), `pytweezer/experiment/runner.py:93-99`, `pytweezer/experiment/motmaster.py`, `pytweezer/experiment/recipes.py:129-148`, `pytweezer/experiments/motmaster_arguments_test.py`, `pytweezer/experiments/load_tweezers.py` (`take_shots` drops the `rb.` prefix only), `pytweezer/GUI/experiments/arg_editor.py`, `pytweezer/GUI/experiments/motmaster_params.py` (box title), `pytweezer/GUI/experiments/panel.py` if it reads per-attribute boxes
- Test: `tests/test_motmaster_experiment.py`, `tests/test_experiment_recipes.py`, `tests/test_experiment_gui.py`, `tests/test_tweezer_array.py` (`rb.tMolCoolVCO1` → `tMolCoolVCO1`)

**Interfaces:**
- Produces:
  - `Experiment.extra_values: dict[str, Any]` — undeclared arguments, by name, in the order first set. Replaces `_extra_names`; `set_argument` puts undeclared names here instead of `setattr`; `argument_values()` includes them.
  - `MotMaster(device_name, *, script, linked=False, iterations=1, save=False, saved=(), timeout=None)`; schema `"motmaster": {script, linked, iterations, save, saved}`.
  - `MotMasterParameter(*args, parameter=None, **kwargs)`; schema adds only `motmaster_parameter`.
  - `MotMasterExperiment.motmaster() -> tuple[str, MotMaster]` — the attribute name and declaration of the one MOTMaster (classmethod; `TypeError` if none).
  - `MotMasterExperiment.motmaster_parameters() -> dict[str, Any]` — `{script parameter: value}` set in the form (declared and extras).
  - `MotMasterExperiment.run_sequences(**overrides) -> None` — plain parameter names.
  - `saved_sequence: dict[str, Any]` keyed by plain parameter name.

- [ ] **Step 1: Update tests to the new rules.** In `tests/test_motmaster_experiment.py` remove `Pair`, `Trio`, `Configured`'s second MOTMaster and every follower/master test; turn `rb.X` into `X` throughout. Add/adjust:

```python
def test_two_motmasters_are_refused_at_class_definition():
    with pytest.raises(TypeError, match="linked"):
        class Two(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            b = MotMaster("CaF MotMaster", script="s")

@pytest.mark.parametrize("name", ["rb.tDelay1", "hamcam.gain", "rb.", "1x", "tDelay1"])
def test_dotted_non_identifier_and_declared_target_names_are_not_arguments(name):
    # "tDelay1" is set by the declared `tof`; a second route to it is refused.
    with pytest.raises(ValueError, match="no argument"):
        Single({name: 1})

def test_a_misspelt_parameter_fails_before_any_run_and_suggests_declared_names(rig):
    measurement = run_local(Single, tofx=1)   # close to declared `tof`
    assert measurement.status == "failed"
    assert "tof" in measurement.attrs["error"] and rig.goes == []

def test_a_script_parameter_named_like_an_attribute_is_sent_and_leaves_it_alone(rig):
    # Single defines `threshold = 0.5` as a plain class attribute for this test.
    run_local(Single, threshold=3)
    assert rig.goes[-1]["threshold"] == 3 and Single.threshold == 0.5
```

Keep the existing parametrised "bad script parameter fails before any sequence runs" cases, with `nShots`, `tDelayy` and scan-only `tDelayy` unprefixed. `test_declared_and_searched_parameters_are_sent_typed` asserts `measurement.arguments["bField"] == 3.0` and `measurement.points["nShots"]`. Saved-binding tests assert `saved_sequence` keys are plain. In `tests/test_experiment_recipes.py`, `test_unknown_arguments_lists_names_the_experiment_lacks` uses args `["atoms", "tPulse", "rb.tPulse", "cam.exposure", "rb", "1x"]` and expects `["1x", "cam.exposure", "gone", "rb.tPulse"]` (`rb` is a valid identifier and would be a script parameter), plus a schema with no MOTMaster where `tPulse` is unknown.

In `tests/test_experiment_gui.py`: `editor.motmaster_box` replaces `motmaster_boxes["rb"]`; `add_motmaster_row("tDelay1")`; rows named `tDelay1`, `coil_current`; add

```python
def test_load_request_leaves_out_old_dotted_names(qapp):
    editor = sequenced_editor(qapp, {"rb.tDelay1": 9, "hamcam.gain": 600, "tDelay1": 4})
    assert "rb.tDelay1" not in editor.rows and "hamcam.gain" not in editor.rows
    assert editor.rows["tDelay1"].value.value() == 4

def test_the_search_box_does_not_offer_a_declared_argument_name(qapp): ...
    # box.choose("pulse") (declared name) adds no row
```

(Use whatever helper the file already has for building a `Sequenced` editor from a request.)

- [ ] **Step 2:** Run `uv run pytest tests/test_motmaster_experiment.py tests/test_experiment_recipes.py tests/test_experiment_gui.py -q`; expect failures on the new/changed tests.
- [ ] **Step 3: Implement.**
  - `experiment.py`: `extra_values` as above; `runner._argument_schema` iterates `experiment.extra_values`.
  - `motmaster.py`: `__init_subclass__` raises `TypeError` for more than one `MotMaster` (message points at `MotMaster(..., linked=True)`), still checks two declared arguments targeting one parameter. `extra_argument(name)` returns a `_MotMasterOverride` (group "MOTMaster") when the class has a MOTMaster, `name.isidentifier()`, and no declared `MotMasterParameter` targets `name`; else `None`. `prepare` sets up the one sequencer (`set_linked`, script, run-until-stopped off, iterations, save toggle, `set_trigger_mode(False)`), checks parameters, binds saved values. `_check_parameter(parameter)` hints with `difflib.get_close_matches` over script parameters plus declared argument names. `run_sequences` merges saved < form < overrides, checks, types, then one blocking `start_motmaster_experiment`. Delete `master`, `follower_timeout`, `follower_arm_delay`, threads, fresh clients, `master_name`, `_parameter_device`, `device=`. `close_devices` releases the link when linked. Rewrite the module docstring for one MOTMaster with unprefixed names, including the declared-argument-wins rule.
  - `recipes.unknown_arguments`: a name is known if declared, or the schema has a MOTMaster device, `name.isidentifier()` and no declared argument has `motmaster_parameter == name`.
  - `motmaster_arguments_test.py`: `sent = self.motmaster_parameters()`.
  - `arg_editor.py`: one `self.motmaster_box` (or `None`); `_searched: set[str]` of row names; `add_motmaster_row(parameter, value=None)`; rows named by the bare parameter; `declared` passed to the box = declared `motmaster_parameter`s ∪ declared argument names; `load_request` adds a searched row only for an identifier name that is not a declared row (dotted names are skipped). `MotMasterBox` title `"MOTMaster ({script}[, linked])"`.
- [ ] **Step 4:** The three test files plus `tests/test_tweezer_array.py`, `tests/test_experiment_runner.py`, `tests/test_experiment_reservations.py` pass; then the full suite.
- [ ] **Step 5:** ruff, commit "MotMasterExperiment: one MOTMaster, unprefixed script parameters", push.

### Task 3: LoadTweezers and LoadTweezersMOTImage in plain code; remove CameraExperiment

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py`, `pytweezer/experiments/tweezer_array_measurement.py`, `pytweezer/optimisers/find_trap_sites.py`, `docs/conf.py` (drop `_skip_dotted_member`), docstrings in `pytweezer/analysis/cloud_results.py`, `pytweezer/experiment/camera_settings.py`, `pytweezer/GUI/viewers/image_monitor.py`, `pytweezer/analysis/tweezer_histogram.py`
- Delete: `pytweezer/experiment/camera_experiment.py`, `tests/test_camera_experiment.py`
- Test: `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: `run_sequences(**overrides)`, `saved_parameters` from Task 2.
- Produces:
  - `LoadTweezers.take_shot(**overrides) -> np.ndarray` — one shot's `(2, H, W)` HamCam frames, broadcast with `{"rid": self.rid}`; re-sends `hamcam_gain` first if it changed.
  - `LoadTweezers.roi_origin: tuple[int, int]` — `(x0, y0)` the HamCam returned from `set_roi`.
  - `LoadTweezers.record_shot`-free `run_point`; keeps `saved_trap_sites`, `top_hat`, `site_photons`, `site_counts`.
  - `TweezerArrayMeasurement.take_batch(n_shots, **sequence) -> tuple[list, list]` and `FindTrapSites`' own equivalent (same name, own copy); both record `hamcam_images` `(shots, 2, H, W)` with `save_raw_images`.

- [ ] **Step 1: Tests.** Make `FakeHamCam`/`FakeORCA` record calls (`self.calls.append((name, args))`) and `is_live` return a settable `live`. Rename scan/override names (`hamcam_gain`). Add:

```python
def test_load_tweezers_sets_the_hamcam_up_for_two_triggered_frames(rig):
    run_local(LoadTweezers)
    setup = [call for call in rig.camera.calls if call[0] != "acquire_n_frames"]
    assert setup[:9] == [("stop_live", ()), ("set_binning", (1,)),
        ("set_roi", (110, 256, 220, 256)), ("set_trigger_source", ("ext",)),
        ("set_external_exposure_mode", ()), ("enable_em_gain", (True,)),
        ("enable_direct_em_gain", (True,)), ("set_gain", (600,)),
        ("setup_acquisition", (2, "snap"))]

def test_a_scanned_gain_is_sent_before_each_shot_that_changes_it(rig):
    run_local(LoadTweezers, Scan(axes=[ListAxis(argument="hamcam_gain", values=[600, 600, 300])]))
    assert [a for n, a in rig.camera.calls if n == "set_gain"] == [(600,), (300,)]

@pytest.mark.parametrize("axis", ["hamcam_roi_x0", "hamcam_roi_width", "hamcam_roi_y0", "hamcam_roi_height"])
def test_the_roi_cannot_be_scanned(rig, axis):
    measurement = run_local(LoadTweezers, Scan(axes=[ListAxis(argument=axis, values=[100, 120])]))
    assert measurement.status == "failed" and rig.goes == []

@pytest.mark.parametrize("cls", [LoadTweezers, LoadTweezersMOTImage])
def test_a_failed_shot_stops_the_cameras_and_puts_live_ones_back_in_live(rig, cls):
    rig.camera.live = rig.orca.live = True
    rig.fail = True                       # FakeSequencer raises in start_motmaster_experiment
    measurement = run_local(cls)
    assert measurement.status == "failed"
    assert rig.camera.calls[-2:] == [("stop_acquisition", ()), ("start_live", ())]
    if cls is LoadTweezersMOTImage:
        assert ("start_live", ()) in rig.orca.calls and ("stop_acquisition", ()) in rig.orca.calls
```

(Adjust the exact `setup_acquisition` argument shape to how the driver is called, `(2, mode="snap")`.) Existing tests for recorded fields, `orca_*` fit, broadcast headers, `test_load_tweezers_never_touches_the_orca`, ROI-origin placement and FindTrapSites stay and must pass unchanged apart from names. Also assert `LoadTweezersMOTImage` records `cloud_mass_amu` for Rb.

- [ ] **Step 2:** Run `uv run pytest tests/test_tweezer_array.py -q`; new tests fail.
- [ ] **Step 3: Implement** per the spec's LoadTweezers / LoadTweezersMOTImage / Subclasses sections:
  - `LoadTweezers(MotMasterExperiment)`: `hamcam = Device("Rb HamCam")`, camera `Integer`/`Number` arguments (gain `Number(600, min=0)`, ROI `Integer`s with `unit="px"`, tooltip "Unbinned sensor pixels"), `planned_points` refusing the four ROI axes, `prepare` in the spec's order then trap sites, `take_shot` (start acquisition → `run_sequences` → `acquire_n_frames(2, broadcast=True, header={"rid": self.rid})`, stop acquisition in `finally`), `run_point` recording today's fields, `finish` restoring live mode (try/finally around `super().finish()`).
  - `LoadTweezersMOTImage(LoadTweezers)`: `orca = Device("ORCA")`, `save_mean_image`; `prepare` adds ORCA stop-live/trigger `"ext"`/`setup_acquisition(1, mode="snap")` and records `cloud_mass_amu`; `run_point` starts the ORCA, calls `take_shot`, reads its one frame, stops it in `finally`, records the shot as `LoadTweezers` does and the cloud fit as `orca_*` (moved verbatim from `record_cloud_fit`/`_cloud_geometry`, reading `get_device_config("ORCA")` and `self.orca.frame_geometry()`); `finish` restores ORCA live mode.
  - Split `LoadTweezers.run_point` so the recording of one shot is a plain method both classes call (e.g. `record_tweezers(frames)`); it is LoadTweezers' own, not a hook.
  - `TweezerArrayMeasurement.take_batch` and `FindTrapSites`' batch (with `BackgroundDrop=1`, tweezers off) each loop `take_shot`; `FindTrapSites` uses `self.roi_origin[::-1]`.
  - Delete `camera_experiment.py` and its test; fix the four docstrings and `docs/conf.py`.
- [ ] **Step 4:** `uv run pytest tests/test_tweezer_array.py tests/test_cloud_results.py tests/test_results_analyses.py -q`, then the full suite; `grep -rn "camera_experiment\|CameraExperiment\|CameraShots\|record_cloud_fit" pytweezer tests bin docs --include=*.py` prints nothing.
- [ ] **Step 5:** ruff, commit "LoadTweezers sets up and reads its cameras in plain code; remove CameraExperiment", push.

### Task 4: Skills, CLAUDE.md and docs

**Files:**
- Modify: `.claude/skills/add-experiment/SKILL.md`, `.claude/skills/add-optimiser/SKILL.md`, `.claude/skills/notebook-rig/SKILL.md`, `.claude/skills/add-device-driver/SKILL.md`, `CLAUDE.md` (skill table row)

- [ ] **Step 1:** Edit the skills as the spec's "Skills, docs and docstrings" section lists: `add-experiment` (one MOTMaster, unprefixed names, extras in `extra_values`, "set the camera up in `prepare`, read it in your shot" with `LoadTweezers` as the example; drop `CaFMOT`/`CameraExperiment`); `add-optimiser` (drop `OptimiseMOTLasers`/`CaFMOT`/`Rig.auto_mot`); `notebook-rig` cut to `PycafCamera` and the zip helpers with a matching `description`; `add-device-driver` (drop `CameraShots`/`CameraExperiment`/`Rig`).
- [ ] **Step 2:** Verify: `grep -rn "CameraExperiment\|CameraShots\|CaFMOT\|CaFFluorescence\|OptimiseMOTLasers\|Rig.auto_mot\|from pytweezer.notebook import Rig\|master=True\|follower" .claude CLAUDE.md` prints nothing; `cd docs && uv run --group docs make html` builds.
- [ ] **Step 3:** Commit "Skills: plain camera experiments, one MOTMaster", push.
