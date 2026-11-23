# Tweezer-Array Scores Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each tweezer optimiser states its goal as a plain penalty-form
objective, and `TweezerOptimisation` does the repetitive parts (script
mapping, drop configuration). FindTrapSites stops doing the drop.

**Architecture:**
- The framework's `objective_of` gains a `settings` argument, and
  `BayesianOptimisation` keeps its search bounds.
- Two helpers, `shortfall` and `shortness`, live in
  `experiment/optimisers.py`.
- `TweezerOptimisation` gets a generic `measure` and a declared `drop`.
- The four tweezer optimisers rewrite their objectives in penalty form.

**Tech Stack:** Python, scikit-optimize, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-23-tweezer-array-scores-design.md` (approved).

**Branch:** `tweezer-array-scores`, in the worktree `/home/user/pytweezer/.worktrees/tweezer-array-scores`, stacked on PR #71.

## Global Constraints

- **Penalty-form objectives, exactly as the spec writes them.** Defaults:
  - `loading_threshold` 0.55;
  - `survival_threshold` 0.8;
  - `shortfall_penalty` 10;
  - `fidelity_weight` 0.5.
- **Placeholder parameter.** `MOT_LOAD_TIME = "tMOTLoadTime"`, with a
  `FIXME: placeholder, replace with the script's parameter name` comment.
- **Every evaluation still takes two images.** Loading and fidelity come
  from the first image only.
- **The threshold policy and its TODO stay unchanged.** Loading is judged
  against a threshold refitted on each batch, survival against the accepted
  one.
- **Recorded results don't change**, except that the objective's value now
  differs.
- **Tests follow the design, and there are no test hooks in production code
  (CLAUDE.md).** The user does not want exhaustive tests: test the objective
  arithmetic and the shot configuration, not every wiring line.
- **Follow CLAUDE.md's commit procedure:** run ruff, `git add` whole files,
  never pass `--no-verify`. Commits end with the session's `Co-Authored-By`
  and `Claude-Session` lines. Use Google docstrings and UK spelling.

## Review Focus

1. **Every setting that clears its thresholds must outscore every one that
   doesn't by enough.** At the default penalty of 10, a 10 % shortfall costs
   as much as the whole time range. Task 3 pins this for MOT: the shortest
   time at a loading of 0.50 scores below the longest time at 0.55.
2. **A time dimension's normalisation uses this run's search bounds**, not
   fixed numbers. Task 1 pins this.
3. **The placeholder script parameter.** `run_motmaster_sequence` checks
   parameters (`_check_parameters`), so a name the script lacks is refused.
   On hardware that is the intended FIXME. In tests, the fake script's
   parameter list may include it; add nothing to production code for it.

---

### Task 1: The framework passes settings to `objective_of`; `shortfall` and `shortness`

**Files:**
- Modify: `pytweezer/experiment/optimisers.py`
  - `Optimiser.objective_of` (~line 197);
  - the `StagedScan` call (~line 393);
  - `BayesianOptimisation.prepare` and `run_point` (~lines 560-590).
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Produces:
  - **`Optimiser.objective_of(self, results: dict[str, float], settings:
    dict[str, float]) -> float`.** The default still returns
    `results[self.objective]`.
    - In `StagedScan`, `settings` is the `settings` dict passed to
      `measure`.
    - In `BayesianOptimisation`, it is `{dimension.setting: value}`.
  - **`BayesianOptimisation.bounds: dict[str, tuple[float, float]]`**, set in
    `prepare` from `search_bounds()`.
  - **`shortfall(value: float, threshold: float) -> float`**:
    `max(0.0, threshold - value)`.
  - **`shortness(value: float, bounds: tuple[float, float]) -> float`**:
    `1 - (value - low) / (high - low)`.
  - Both helpers are module-level functions in `experiment/optimisers.py`.
- Only the tweezer optimisers override `objective_of` (grep to confirm). Task
  3 rewrites them, so this task only keeps them working: add a `settings`
  parameter to the three overrides and ignore it for now.

- [ ] **Step 1: Write the failing test** in `tests/test_optimisers.py`:
  `test_a_bayesian_objective_sees_the_settings_it_scores`. A tiny
  `BayesianOptimisation` subclass (follow the file's existing demo pattern)
  whose `objective_of` returns `shortness(settings["x"], self.bounds["x"])`
  with x searched over (0, 10). Run it locally (`run_local`). Assert that each
  point's recorded objective equals `1 - x/10` for that point's recorded `x`.
- [ ] **Step 2: Check it fails.** Run
  `uv run pytest tests/test_optimisers.py -q` and expect a failure on the
  missing argument or attribute.
- [ ] **Step 3: Implement** the Interfaces above. Update the `add-optimiser`
  skill line about `objective_of(results)` (~line 68) to
  `objective_of(results, settings)`, and name the two helpers there. This is a
  minor edit.
- [ ] **Step 4: Check it passes.** Run
  `uv run pytest tests/test_optimisers.py tests/test_tweezer_array.py -q` and
  expect a pass.
- [ ] **Step 5: Commit**: "Optimisers see the settings they score; shortfall and shortness helpers".

### Task 2: `TweezerOptimisation` maps settings itself and declares its drop

**Files:**
- Modify: `pytweezer/optimisers/tweezer_optimisation.py`
- Modify: the `measure` of `optimise_mot_light.py`,
  `optimise_molasses_light.py` and `optimise_molasses_shims.py` (delete it)
  and `optimise_tweezer_imaging.py` (keep it, using the drop).
- Test: `tests/test_tweezer_array.py`. Update the existing configuration tests
  (e.g. `test_each_point_keeps_the_tweezers_on_and_drops_as_its_measurement_needs`,
  `test_each_molasses_evaluation_runs_at_its_shims`,
  `test_each_light_evaluation_runs_at_its_settings`) only as the new shape
  needs.

**Interfaces:**
- Produces, on `TweezerOptimisation`:
  - `drop: ClassVar[str | None] = None` names the argument holding the drop
    duration, or `None` for no drop.
  - `drop_overrides(self) -> dict` returns `{"BackgroundDrop": 0}` when `drop`
    is `None`, else `{"BackgroundDrop": 1, "tDropDuration": getattr(self,
    self.drop)}`.
  - `measure(self, **settings) -> dict[str, float]` maps each setting that
    isn't `None` to its `Dimension.save_as` and returns
    `self.score_batch(**self.drop_overrides(), **mapped)`.
  - `score_batch` is otherwise unchanged.
- The molasses optimisers set `drop = "drop_duration"`, keeping their
  `drop_duration` argument. MOT and imaging use no drop.
- Imaging's `measure` keeps its VCO1/VCO2 and VVA1/VVA2 mapping and calls
  `self.score_batch(**self.drop_overrides(), **imaging)`.

- [ ] **Step 1: Check the baseline.** Run
  `uv run pytest tests/test_tweezer_array.py -q` and expect a pass.
- [ ] **Step 2: Implement it**, deleting the three repetitive `measure`
  methods.
- [ ] **Step 3: Run the tests.** Repeat Step 1's command. The MOTMaster runs
  each optimiser sends must be the same as before; adjust a test only where
  it reached removed code.
- [ ] **Step 4: Commit**: "TweezerOptimisation maps settings to script parameters and declares its drop".

### Task 3: Penalty-form objectives, and the new time dimensions

**Files:**
- Modify: `optimise_mot_light.py`, `optimise_molasses_light.py`,
  `optimise_molasses_shims.py` and `optimise_tweezer_imaging.py`, all in
  `pytweezer/optimisers/`.
- Test: `tests/test_tweezer_array.py`. Replace
  `test_the_score_is_survival_only_above_the_fidelity_floor` and
  `test_the_molasses_score_counts_recapture_only_once_the_array_loads`.

**Interfaces:**
- **MOT.**
  - Add the module constant `MOT_LOAD_TIME` (placeholder, with a FIXME).
  - Add `Dimension("load_time", save_as=MOT_LOAD_TIME, low=10, high=500)`,
    with no span, so nothing is looked up under the placeholder at
    `prepare`. Its limits are placeholders; say so in the FIXME.
  - Add the arguments `loading_threshold = Number(0.55, min=0, max=1,
    group=SEARCH)` and `shortfall_penalty = Number(10, min=0, group=SEARCH)`.
  - `objective_of`:
    `shortness(settings["load_time"], self.bounds["load_time"]) -
    self.shortfall_penalty * shortfall(results["loading"],
    self.loading_threshold)`.
  - Set `objective = "score"`. The docstring states the goal in one sentence.
- **Molasses (light and shims).**
  - Add `Dimension("molasses_time", save_as="tMolassesDuration", low=1,
    high=20)`.
  - Add `loading_threshold` (0.55), `survival_threshold` (0.8) and
    `shortfall_penalty` (10). Remove `loading_floor`.
  - `objective_of`: `shortness(molasses_time) − penalty × shortfall(loading)
    − penalty × shortfall(survival)`, as in the spec.
- **Imaging.**
  - Add `fidelity_weight = Number(0.5, min=0, max=1, group=SEARCH)` and
    remove `fidelity_floor`.
  - `objective_of`: `w * results["fidelity"] + (1 - w) * results["survival"]`.
- The `objective_of` overrides use the design's names for their arguments
  (`scores`, `settings`), with `results` accepted positionally from the
  framework.

- [ ] **Step 1: Write the failing tests.** Call each optimiser's
  `objective_of` directly on an instance built the way the file's fixtures
  build one, with `bounds` set. One test per optimiser:
  - `test_mot_scores_shorter_loads_higher_once_loading_clears_the_threshold`:
    at loading 0.6, load time 10 scores higher than 500. The shortest time at
    loading 0.50 scores below the longest at 0.55.
  - `test_molasses_penalises_each_shortfall`: with loading and survival above
    their thresholds, the score equals `shortness`. Each 0.1 shortfall costs 1.
  - `test_imaging_weights_fidelity_against_survival`: with `fidelity_weight`
    0.25, fidelity 0.9 and survival 0.5, the score is 0.6.
- [ ] **Step 2: Check they fail.** Run
  `uv run pytest tests/test_tweezer_array.py -q` and expect failures on the
  new tests.
- [ ] **Step 3: Implement it**, and delete the two replaced tests.
- [ ] **Step 4: Check they pass.** Run `uv run pytest tests/ -q` and expect
  everything to pass.
- [ ] **Step 5: Commit**: "Tweezer optimisers state their goals as penalty-form objectives".

### Task 4: FindTrapSites takes no drop; skills; verify; PR

**Files:**
- Modify: `pytweezer/optimisers/find_trap_sites.py`,
  `pytweezer/analysis/array_loading.py` (remove `background_subtracted_mean`
  if nothing else uses it) and `.claude/skills/add-optimiser/SKILL.md`.
- Test: `tests/test_tweezer_array.py`. Rewrite
  `test_find_trap_sites_takes_its_shots_with_the_tweezers_off_in_the_background_drop`.
  Remove `background_subtracted_mean`'s test, if one exists, along with the
  function.

**Interfaces:**
- `take_calibration_shots` runs `self.run_motmaster_sequence(BackgroundDrop=0)`,
  without the tweezer-off VVAs.
- `fit` builds its calibration image as `np.mean(firsts, axis=0)`; the second
  images are unused.
- Everything else `fit` records and proposes is unchanged.

- [ ] **Step 1: Rewrite the test** as
  `test_find_trap_sites_takes_its_shots_without_a_drop`. Every run has
  `BackgroundDrop == 0` and no `tTweezerSetVVA0`/`tTweezerSetVVA1` override.
  The existing FindTrapSites geometry tests must still pass, so check the
  fake camera's first frames still carry the atoms.
- [ ] **Step 2: Check it fails.** Run
  `uv run pytest tests/test_tweezer_array.py -q` and expect a failure.
- [ ] **Step 3: Implement it.** Then update the `add-optimiser` skill to the
  approved rules:
  - the tweezer optimisers' penalty-form objectives;
  - `drop`;
  - the generic `measure`;
  - FindTrapSites without the drop.

  Replace its `fidelity_floor`/`loading_floor` descriptions (~line 148).
- [ ] **Step 4: Verify everything.**
  - `uv run pytest tests/ -q`: all pass.
  - `uv run ruff check .` and `uv run ruff format --check .`: clean.
- [ ] **Step 5: Commit and push.** Commit: "FindTrapSites calibrates without a
  drop; add-optimiser follows the new objectives". Then
  `git push -u origin tweezer-array-scores`.
- [ ] **Step 6: Report** the placeholders the user must replace before a
  real run: `MOT_LOAD_TIME`, and the `load_time` and `molasses_time` limits.
