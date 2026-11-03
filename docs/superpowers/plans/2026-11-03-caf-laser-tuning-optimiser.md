# CaF laser-tuning optimiser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A staged-scan optimiser for the six CaF wavemeter-lock lasers, run by the Experiment Manager, whose accepted set points are saved parameters and are also applied to the lock, called from the notebook by `Rig.auto_mot`.

**Architecture:** Three small additions to `StagedScan` (per-stage on/off, span-around-a-centre stages whose centre defaults to a saved parameter, an `apply` hook the Optimisers tab calls after an accept) carry the framework side. A `CaFFluorescence` measurement (`MotMasterExperiment` on `CaF MotMaster` + `CaF ThorCam`) provides `measure(**set_points)`; `OptimiseMOTLasers(StagedScan, CaFFluorescence)` declares six stages. `Rig.auto_mot` submits it and reads the file back.

**Tech Stack:** Python 3.11, pytest, PyQt6 (offscreen for the tab), h5py, sipyco RPC, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-caf-laser-tuning-optimiser-design.md`

## Global Constraints

- UK spelling; Google-style docstrings describing current behaviour only; comments only where genuinely surprising (CLAUDE.md).
- Logging via `get_logger(name)` from `pytweezer/logging_utils.py`.
- No hardware is available: every test runs against fakes or the simulated MOTMaster/camera, in the style of `tests/test_motmaster_experiment.py` and `tests/test_notebook_rig.py`. Do not add parameters or hooks to production code just so a test can reach it.
- Nothing at module level in `pytweezer/optimisers/` or `pytweezer/experiments/` may touch hardware.
- Set points are THz floats; scan spans and the `save_as` values are in THz.
- Stage defaults, the only values taken from the 27 Oct notebook: spans (THz) v0 `10e-6`, BX `30e-6`, v1 `40e-6`, v2 `150e-6`, v3 `150e-6`, TCool `40e-6`; points 30, 30, 20, 30, 30, 30; enabled by default v0, v1, TCool. Stage order v0, BX, v1, v2, v3, TCool. Script `"0. DDSFourSidebands5W_noDDS"`, 3 iterations.
- A run leaves every laser at the set point it had when the run started, including after a failure or termination.
- Branch `caf-laser-tuning-optimiser` with a PR (a feature). Stage only your own files: the working tree already holds unrelated edits (`pytweezer/drivers/camera_base.py`, `imagemX2.py`, `orca.py`, `thorcam.py`, `tweezermonitorcam.py`, `configuration/properties/properties.json`, `experiments/camera_continuous.py`, `tests/test_camera_continuous.py`); do not touch or commit them. Before each commit: `uv run ruff check --fix <files>` and `uv run ruff format <files>`, `git add` the files whole, never `--no-verify`.
- Run the suite with `uv run pytest tests/ -q`; `QT_QPA_PLATFORM=offscreen` for anything importing Qt.
- Do not edit `.claude/skills/add-optimiser` without the user's go-ahead (Task 8).

## Review Focus

Inputs and conditions the spec implies but does not spell out; each is pinned by a test in the task that owns the code.

1. A run with every stage switched off: refused at `prepare()` with a message naming the `_run` arguments (Task 1).
2. A laser's saved set point does not exist when its stage is enabled and no centre is given: `KeyError` naming the key before any laser moves (Tasks 2, 6).
3. A scan that fails or is terminated mid-stage: lasers back at their starting set points (Task 5).
4. A shot that returns no camera frames: a clear `RuntimeError`, never a NaN objective that silently drops the point (Task 5).
5. Accepting only some of the ticked proposals: `apply` receives exactly the ticked keys; an `apply` that raises leaves the saved values written and reports the failure beside the run (Task 3).

---

### Task 0: Branch

- [ ] **Step 1:** `git switch -c caf-laser-tuning-optimiser` (the unrelated working-tree edits come along and stay unstaged).
- [ ] **Step 2:** `git add docs/superpowers/specs/2026-11-03-caf-laser-tuning-optimiser-design.md docs/superpowers/plans/2026-11-03-caf-laser-tuning-optimiser.md` and commit (`Design and plan for the CaF laser-tuning optimiser`).

### Task 1: Per-stage on/off in `StagedScan`

**Files:**
- Modify: `pytweezer/experiment/optimisers.py` (`Stage`, `StagedScan`)
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Produces: `Stage(setting, *, start, stop, points, save_as, run: bool = True)`. `StagedScan.generated_arguments()` adds `Bool(stage.run, group=SEARCH)` named `<setting>_run`. `StagedScan.enabled_stages() -> list[tuple[int, Stage]]` gives `(index in self.stages, stage)` for each enabled stage. The recorded `stage` of a point stays its index in the full `stages` list; `stages_total` and `stages_done` count enabled stages.

- [ ] **Step 1: Write failing tests** in `tests/test_optimisers.py`, using `DemoResonance`-based subclasses defined in the test module:
  - `test_a_stage_switched_off_is_not_scanned`: `run_local(DemoStagedScan, power_run=False)` records only `stage == 0` points, has `proposed.demo_resonance` and no `proposed.demo_power`, `stages_total == 1`.
  - `test_a_later_stage_runs_at_the_best_of_the_enabled_earlier_ones`: with `detuning_run=False`, the power stage's `measure` is called with no `detuning` keyword (spy via a subclass overriding `measure`).
  - `test_a_run_with_no_stage_enabled_is_refused`: `run_local(DemoStagedScan, detuning_run=False, power_run=False)` returns a measurement with `status == "failed"` whose `attrs["error"]` contains `detuning_run`.
  - `test_stage_run_defaults_follow_the_declaration`: a class with `Stage(..., run=False)` has `<setting>_run` default `False`.
- [ ] **Step 2: Run** `uv run pytest tests/test_optimisers.py -q -k "stage"`. Expected: the four new tests FAIL.
- [ ] **Step 3: Implement** `run` on `Stage`, the generated Bool, `enabled_stages()`, and make `_schedule`, `prepare` (refuse none enabled with a `ValueError` naming the `_run` arguments), `run_point` (earlier best values keyed by setting, from enabled stages only), `finish` and `number_of_points` use enabled stages only. `_best_values` becomes a dict keyed by stage index.
- [ ] **Step 4: Run** `uv run pytest tests/test_optimisers.py tests/test_experiment_gui.py -q`. Expected: all PASS (the existing staged-scan and tab tests are unchanged).
- [ ] **Step 5: Commit** (`Staged scans: switch each stage on or off`).

### Task 2: Span stages with a saved-backed centre

**Files:**
- Modify: `pytweezer/experiment/optimisers.py` (`Stage`, `StagedScan.check_declarations`/`generated_arguments`/`_schedule`)
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Consumes: Task 1's `Stage.run`.
- Produces: `Stage(setting, *, points, save_as, span: float | None = None, start: float | None = None, stop: float | None = None, run: bool = True)`; a stage gives exactly one of `span` or both `start` and `stop`. A span stage generates `<setting>_span` (`Number(span, min=0, group=SEARCH)`), `<setting>_points`, and `<setting>_centre` (`Number(saved=stage.save_as, group=SEARCH)`, which the framework fills from the saved parameter when unset, and which a user may set). Its sweep is `linspace(centre - span, centre + span, points)`. Declaration errors raise `TypeError` when the class is defined, as `Dimension`'s do.

- [ ] **Step 1: Write failing tests**:
  - `test_a_span_stage_scans_around_its_saved_value`: a `StagedScan` over `DemoResonance` with `Stage("detuning", span=2.0, points=5, save_as="demo_resonance")`, saved `demo_resonance` patched to 1.0 (use the fixture the other saved-parameter tests in this file use); the recorded settings are `[-1, 0, 1, 2, 3]`.
  - `test_a_span_stage_scans_around_a_chosen_centre`: `detuning_centre=10.0` gives settings `8 … 12`, and the saved value is not read.
  - `test_the_centre_used_is_in_the_file`: `measurement.arguments["detuning_centre"]` is the resolved centre; if the runner does not store resolved saved-backed arguments, record it as constant `detuning_centre_used` and assert that instead.
  - `test_a_span_stage_without_its_saved_parameter_names_the_key`: `KeyError` matching the `save_as` before any point runs.
  - `test_bad_span_declarations_are_refused`: parametrised: span with start/stop, neither span nor start/stop, non-positive span, only one of start/stop; each `TypeError` at class definition.
- [ ] **Step 2: Run** `uv run pytest tests/test_optimisers.py -q -k "span or centre"`. Expected: FAIL.
- [ ] **Step 3: Implement.** Confirm in `runner.py` that `resolve_saved_arguments` runs before `prepare()`; the schedule is built in `prepare`, so the centre must already be set. Compute values from the centre in `_schedule` for span stages.
- [ ] **Step 4: Run** `uv run pytest tests/test_optimisers.py tests/test_saved_parameters.py tests/test_experiment_arguments.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** (`Staged scans: stages around a saved or chosen centre`).

### Task 3: `apply` on accept

**Files:**
- Modify: `pytweezer/experiment/optimisers.py` (`Optimiser`), `pytweezer/GUI/optimisers.py` (`accept`)
- Test: `tests/test_optimisers.py`, `tests/test_experiment_gui.py`

**Interfaces:**
- Produces: `Optimiser.apply(cls, proposals: dict[str, Any]) -> None`, a classmethod doing nothing by default, called with `{saved key: value}` of the ticked values. `optimiser_schema()` gains `"applies": bool`, true when a subclass overrides `apply`, so the tab imports a class only when it must. After `set_saved_parameters` succeeds, `accept()` imports the optimiser's class from `row.key` (module, class name), calls `apply` off the GUI thread, then shows either the existing "Accepted: …" status with "and applied" appended, or an error-state status "Accepted, but applying failed: <error>". A failing `apply` does not undo the save. Read the `pytweezer-gui-internals` skill for how long calls are kept off the GUI thread.

- [ ] **Step 1: Write failing tests**:
  - in `tests/test_optimisers.py`: `test_apply_does_nothing_by_default_and_the_schema_says_so` (`DemoStagedScan.optimiser_schema()["applies"] is False`; a subclass overriding `apply` gives `True`).
  - in `tests/test_experiment_gui.py`, beside the existing accept tests: `test_accepting_calls_apply_with_exactly_the_ticked_values`, `test_a_failing_apply_leaves_the_values_saved_and_says_so`. Use a small optimiser class defined in a module the catalogue can import (or the module the existing accept tests use) whose `apply` records into a module-level list or raises.
- [ ] **Step 2: Run** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_optimisers.py tests/test_experiment_gui.py -q -k "apply"`. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** the same two files in full. Expected: PASS.
- [ ] **Step 5: Commit** (`Optimisers: apply accepted values to hardware`).

### Task 4: Shared frame settling

**Files:**
- Create: `pytweezer/drivers/shot_frames.py`
- Modify: `pytweezer/notebook/rig.py` (`Rig._settle`)
- Test: `tests/test_notebook_rig.py`

**Interfaces:**
- Produces: `settle_frame_counts(cameras: Mapping[str, Any], settle_s: float, timeout_s: float, poll_s: float) -> None`, blocking until every camera's `frames_acquired()` has stayed unchanged for `settle_s` or `timeout_s` has passed. `Rig._settle` calls it with the rig's `frame_settle_s`, `frame_timeout_s`, `frame_poll_s`; behaviour is unchanged.

- [ ] **Step 1: Write failing test** `test_settle_frame_counts_waits_for_a_late_frame` in `tests/test_notebook_rig.py`, using `FakeCamera(1, lag_polls=3)`: after one `trigger(1)` and `settle_frame_counts({"CaF": camera}, 0.05, 2.0, 0.01)`, `camera.frames_acquired() == 1`.
- [ ] **Step 2: Run** `uv run pytest tests/test_notebook_rig.py -q -k settle_frame_counts`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** by moving `Rig._settle`'s loop into the new function.
- [ ] **Step 4: Run** `uv run pytest tests/test_notebook_rig.py -q`. Expected: all PASS, including the late-frame tests that already exist.
- [ ] **Step 5: Commit** (`Move camera frame settling out of Rig`).

### Task 5: `CaFFluorescence` measurement

**Files:**
- Create: `pytweezer/experiments/caf_fluorescence.py`
- Test: `tests/test_caf_fluorescence.py`

**Interfaces:**
- Consumes: `settle_frame_counts` (Task 4); `MotMasterExperiment`, `MotMaster`, `MotMasterNumber` from `pytweezer/experiment/motmaster.py`.
- Produces: `LASERS = ("v0", "BX", "v1", "v2", "v3", "TCool")`; class `CaFFluorescence(MotMasterExperiment)` with `caf = MotMaster("CaF MotMaster", script="0. DDSFourSidebands5W_noDDS", iterations=3)`, `camera = Device("CaF ThorCam")`, `mot_field = MotMasterNumber(1.0, parameter="MOTCoilsCurrentValue")`, `bg_field = Number(0.0)`, `settle_s = Number(0.0, min=0)`, `frame_settle_s`/`frame_timeout_s` Numbers (defaults 0.3 and 5.0, as `Rig`), and
  `measure(self, *, v0: float | None = None, BX: float | None = None, v1: float | None = None, v2: float | None = None, v3: float | None = None, TCool: float | None = None) -> dict[str, float]` returning `{"counts": mean over frames of the background-subtracted frame sum, "counts_error": standard error, NaN for one frame}`. `None` leaves a laser alone; a value ramps it with `ramp_laser_set_point_wml`, then waits `settle_s` once per call that moved a laser. `run_point` records `measure()`'s results at the current set points, so the experiment runs on its own.
- `prepare` stores `self.start_set_points = laser_set_points_wml()`, records it as constant `start_set_points`, and takes one background shot (`caf.MOTCoilsCurrentValue` overridden to `bg_field`) whose frames' mean is the background. `finish` ramps every laser back to `start_set_points`, in a `finally`, whatever happened. Frames per shot: `camera.arm()`, mark `frames_acquired()`, `run_sequences(...)`, `settle_frame_counts(...)`, `read_frames_from(mark)`.

- [ ] **Step 1: Write failing tests** in `tests/test_caf_fluorescence.py` with a fake sequencer and fake camera in the style of `tests/test_motmaster_experiment.py` / `tests/test_notebook_rig.py` (a camera whose frame value depends on a laser's current set point):
  - `test_counts_are_background_subtracted_frame_sums`: counts equal `(frame - background).sum()` averaged over the iterations; `counts_error` is the standard error.
  - `test_counts_error_is_nan_for_one_frame`.
  - `test_none_leaves_a_laser_alone_and_a_value_ramps_it`: the sequencer log shows a `ramp` for `v0` only when `v0=` is given.
  - `test_lasers_return_to_where_they_started`: after a normal run, after a `measure` that raises mid-run, and after a terminated run, every laser's final set point equals the one at `prepare`.
  - `test_a_shot_with_no_frames_is_an_error`: a camera that returns no frames makes `measure` raise `RuntimeError` naming the camera.
  - `test_the_background_shot_has_the_field_off`: the first script start has `MOTCoilsCurrentValue == 0.0`, measured shots `1.0`.
- [ ] **Step 2: Run** `uv run pytest tests/test_caf_fluorescence.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `CaFFluorescence` as in Interfaces. Keep the laser-moving code in its own method (`set_lasers(**set_points)`) so a later recapture measurement can reuse it.
- [ ] **Step 4: Run** `uv run pytest tests/test_caf_fluorescence.py tests/test_motmaster_experiment.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** (`Add the CaF fluorescence measurement`).

### Task 6: `OptimiseMOTLasers`

**Files:**
- Create: `pytweezer/optimisers/optimise_mot_lasers.py`
- Test: `tests/test_optimisers.py` (or `tests/test_optimise_mot_lasers.py` if it grows past a screen)

**Interfaces:**
- Consumes: Tasks 1–3 (`Stage` with `run` and `span`, `Optimiser.apply`), Task 5 (`CaFFluorescence`, `LASERS`).
- Produces: `LASER_KEYS = {laser: f"caf_laser_{laser}" for laser in LASERS}`; `class OptimiseMOTLasers(StagedScan, CaFFluorescence)` with `objective = "counts"`, `goal = "max"`, six span stages in `LASERS` order, `save_as` from `LASER_KEYS`, with the spans, points and default `run` flags of Global Constraints; `proposal_units = {key: "THz"}` and `proposal_group = "CaF lasers"`; `apply(cls, proposals)` ramps `CaF MotMaster` (via `get_device`) to each accepted value, only for the keys passed. `create_saved_set_points(set_points: dict[str, float] | None = None) -> list[str]` creates each missing `LASER_KEYS` saved parameter (`create=True`, group, unit `"THz"`, reason `"initial laser set points"`) from `set_points`, or from the lock's current ones read through `get_device("CaF MotMaster")` when `None`; it never overwrites an existing key and returns the keys it created.

- [ ] **Step 1: Write failing tests**:
  - `test_the_stages_and_defaults_match_the_notebook`: stage order, spans, points, and that only v0, v1 and TCool default to running.
  - `test_each_enabled_laser_is_scanned_around_its_saved_set_point_and_best_values_chain`: with fakes from Task 5 where fluorescence peaks at a known set point per laser, `run_local` records proposals at those peaks and a later laser's `measure` sees the earlier laser's best.
  - `test_a_laser_with_no_saved_set_point_names_the_key`: `KeyError` matching `caf_laser_v0` before any laser moves.
  - `test_apply_ramps_only_the_accepted_lasers`: `OptimiseMOTLasers.apply({"caf_laser_v0": x})` makes one `ramp` call, for `v0`.
  - `test_create_saved_set_points_creates_only_the_missing_ones`.
  - `test_the_optimiser_is_in_the_catalogue_and_imports_without_hardware`: appears in `optimisers` listing; `optimiser_schema()["applies"]` is true.
- [ ] **Step 2: Run** `uv run pytest tests/test_optimisers.py -q -k "mot_lasers or MOTLasers or saved_set_points"`. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces.
- [ ] **Step 4: Run** `uv run pytest tests/ -q`. Expected: PASS apart from failures that already exist on `main` (run the suite on `main` first if any appear, and report them rather than fixing them here).
- [ ] **Step 5: Commit** (`Add the CaF laser-tuning optimiser`).

### Task 7: `Rig.auto_mot` runs the optimiser

**Files:**
- Modify: `pytweezer/notebook/rig.py` (`Rig.auto_mot`, `plot_auto_mot_results` unchanged), `.claude/skills/notebook-rig/SKILL.md` (minor edit)
- Test: `tests/test_notebook_rig.py`

**Interfaces:**
- Consumes: `OptimiseMOTLasers`, `LASERS`, `LASER_KEYS` (Task 6); `submit`, `wait` from `pytweezer.experiment.client`; `load_measurement`, `data_root` from `pytweezer.experiment.storage`.
- Produces: `Rig.auto_mot(self, stages: Sequence[str], *, spans: dict[str, float] | None = None, points: dict[str, int] | None = None, centres: dict[str, float] | None = None, display: bool = False, **arguments) -> dict[str, dict[str, Any]]`. It submits `OptimiseMOTLasers` with `<laser>_run` true for exactly `stages` and false for the other lasers, plus `<laser>_span`/`_points`/`_centre` from the dicts and any other `arguments`; waits; raises if the run did not complete; and returns for each laser in `stages`, `scan_range`, `numbers`, `errors`, `set_point` (the proposal) and `start_set_point` (the lock's value when the run started, from the file's `start_set_points` constant), the shape `plot_auto_mot_results` takes. It never accepts. An unknown laser name raises `KeyError` naming `LASERS`. The previous `script`, `scan_ranges`, `camera`, `field_parameter`… parameters are removed; the `Rig` no longer scans lasers itself for this, but `scan_laser` and `laser_set_points` stay.

- [ ] **Step 1: Write failing tests** replacing the five `test_auto_mot_*` tests: `test_auto_mot_submits_the_chosen_stages` (patch `submit` and `wait` in `pytweezer.notebook.rig`'s namespace and assert the submitted `<laser>_run` flags and arguments), `test_auto_mot_returns_each_stage_in_the_shape_the_plot_takes` (a measurement from `run_local(OptimiseMOTLasers, simulate…)` or a hand-made `Measurement`, patched into `load_measurement`), `test_auto_mot_refuses_an_unknown_laser`, `test_auto_mot_raises_for_a_run_that_did_not_complete`, `test_auto_mot_does_not_accept`.
- [ ] **Step 2: Run** `uv run pytest tests/test_notebook_rig.py -q -k auto_mot`. Expected: FAIL.
- [ ] **Step 3: Implement** `Rig.auto_mot` as in Interfaces; delete the old body. Update the `notebook-rig` skill's `auto_mot` description and signature (a minor edit under CLAUDE.md).
- [ ] **Step 4: Run** `uv run pytest tests/test_notebook_rig.py -q`, then `uv run ruff check .` and `uv run ruff format --check .`. Expected: PASS and clean.
- [ ] **Step 5: Commit** (`Rig.auto_mot: run the laser optimiser`).

### Task 8: Skill update, docs, PR

**Files:**
- Modify: `.claude/skills/add-optimiser/SKILL.md` (only after the user agrees)

- [ ] **Step 1: Describe to the user** the `add-optimiser` changes this work needs: the `run` flag and `<setting>_run` argument, span stages and the saved-backed `<setting>_centre`, the `apply` hook and the schema's `applies`, and the new `OptimiseMOTLasers` under a "CaF laser tuning" heading. State why each is needed and wait for their go-ahead; these are significant changes under CLAUDE.md.
- [ ] **Step 2: After the go-ahead**, make the edits; check `tests/test_optimisers.py` is still the listed test for the section.
- [ ] **Step 3: Build the docs** (`cd docs && uv run make html`) only if docstrings changed API pages; expected: no new warnings.
- [ ] **Step 4: Push the branch and open a PR** to `main` with the PR-body attribution line from the session reminder, listing for the user what could not be checked without hardware: the ramp and settle timing on the real wavemeter lock, the ThorCam's frame count per shot against `iterations`, `apply` against the live `CaF MotMaster` from the GUI PC, and the one-off `create_saved_set_points` call that must be run once before the first run.
