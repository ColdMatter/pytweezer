# Optimisers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A general optimiser framework for the task system — experiments that set their own points, `StagedScan` and `BayesianOptimisation` building blocks over a `measure()` experiment, run-ID history, and an Optimisers dashboard that accepts proposals into saved parameters — built and tested against a simulated demo.

**Architecture:** An optimiser is an ordinary `Experiment` that mixes a building block from `pytweezer/experiment/optimisers.py` into a measurement experiment. A new `planned_points` hook lets it run its own number of points through the unchanged runner; `finish()` records its proposals as constants. The dashboard finds optimisers through the catalogue schema, reads runs through the results archive, and accepts through `set_saved_parameters(..., rid=...)`.

**Tech Stack:** Python 3, scikit-optimize (already a dependency), numpy, PyQt6 + pyqtgraph, pytest.

**Spec:** `docs/superpowers/specs/2026-10-31-optimisers-design.md` (read it first).

## Global Constraints

- The framework knows nothing about the lab: no MOTMaster, tweezer, MOT or device names in `optimisers.py` or the tab. Lab optimisers are out of scope.
- One place runs tasks: the Experiment Manager on the server PC. Client code reaches it at the address CONFIG gives.
- Tests follow the design (CLAUDE.md): change existing tests and helpers to fit; add nothing to production code just so a test can reach it.
- UK English; Google-style docstrings describing current behaviour only; comment only what is surprising; `get_logger(name)`.
- `uv run ruff check` / `uv run ruff format` on touched files only; `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q` passes after every task.
- Generated argument names: `<setting>_start`, `<setting>_stop` (`Number`), `<setting>_points` (`Integer`, `min=1`) per stage; `<setting>_low`, `<setting>_high` (`Number`) per dimension. Mixin arguments: `repeats` (`Integer`, default 1, `min=1`) for `StagedScan`; `n_calls` (default 30, `min=1`), `n_initial` (default 5, `min=1`), `seed` (default 0) (`Integer`) and `xi_start` (0.1), `xi_end` (0.001) (`Number`) for `BayesianOptimisation`.
- Recorded constants: `proposed.<key>` per proposal; `stages_done`, `stages_total` (staged); `evaluations_done`, `evaluations_total` (Bayesian).
- Schema entry: `"optimiser": {"kind": "staged" | "bayesian", "proposes": [keys]}`.
- Accept reason: `accepted from run <rid>`. Partial labels: `stage <done> of <total> finished`, `<done> of <total> evaluations`.
- Tab title `Optimisers`, directly after `Saved parameters` in `bin/gui.py`.
- Demo keys: `demo_resonance`, `demo_power`; demo objective `counts`, goal `"max"`.
- Commit after each task, plain-sentence messages, ending with the session's attribution lines.

## Review Focus

1. A shot whose objective is `nan` (a failed fit, zero atoms) must not poison the run: the default `best` ignores non-finite objectives, and a Bayesian run tells the optimiser the worst finite objective seen so far (skipping the tell if there is none). Tests in Tasks 2 and 3.
2. A run terminated before any stage or evaluation finishes records no proposals, and the dashboard shows no Accept rows for it. Tests in Tasks 2, 3 and 5.
3. A stage whose start equals its stop, or with one point, proposes that value. Test in Task 2.
4. A proposal the store refuses (key missing, wrong type) leaves every saved value unchanged and the refusal visible in the sheet. Test in Task 5.
5. Changing a stage's `<setting>_points` argument changes the number of points the run reports (the hook reads argument values, not class defaults). Test in Task 2.

---

### Task 1: Experiments set their own points

**Files:**
- Modify: `pytweezer/experiment/experiment.py`, `pytweezer/experiment/runner.py:62`, `pytweezer/experiment/client.py` (`_resolve`), `docs/superpowers/specs/2026-10-31-optimisers-design.md` (section 2)
- Test: `tests/test_optimisers.py` (new)

**Interfaces:**
- Produces: `Experiment.planned_points(self, scan: Scan) -> list[Point]`, an instance method (the count may depend on argument values), defaulting to `scan.points(type(self))`. `open_writer` calls `experiment.planned_points(scan)` after constructing the experiment. `_resolve` replaces `scan.axis_values(experiment)` with `experiment(args or {}).planned_points(scan)`.

- [ ] **Step 1: Write the failing tests**

```python
class Counts(Experiment):
    level = Integer(3, min=1)

    def planned_points(self, scan):
        if scan.axes or scan.repetitions != 1:
            raise ValueError("Counts sets its own points")
        return [Point(i, 0, (), {}) for i in range(self.level)]

    def run_point(self):
        self.record("i", self.point.index)


def test_an_ordinary_experiment_runs_its_scan(): ...
    # RabiDemo with a 3-value ListAxis over pulse_time → n_done == 3

def test_an_experiment_can_set_its_own_number_of_points():
    measurement = run_local(Counts, level=5)
    assert measurement.n_done == 5 and list(measurement.results["i"]) == [0, 1, 2, 3, 4]

def test_a_scan_an_experiment_refuses_is_refused_at_submission():
    with pytest.raises(ValueError, match="sets its own points"):
        _resolve(Counts, Scan(repetitions=2))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_optimisers.py -q`
Expected: FAIL (`n_done == 3`, not 5; no error from `_resolve`).

- [ ] **Step 3: Implement the hook, the two call sites, and correct spec section 2** (it says a class method `points(scan)`; make it the instance method `planned_points(scan)` and say why in one sentence).

- [ ] **Step 4: Run the full suite** — all pass.

- [ ] **Step 5: Commit** — `git commit -m "Let an experiment set its own points"`

---

### Task 2: `StagedScan` and the demo measurement

**Files:**
- Create: `pytweezer/experiment/optimisers.py`
- Modify: `pytweezer/experiment/experiment.py` (`schema()` merges `cls.optimiser_schema()` when present), `pytweezer/experiments/demo.py`
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Consumes: Task 1's `planned_points`.
- Produces, in `optimisers.py`:
  - `Stage(setting: str, *, start: float, stop: float, points: int, save_as: str)` (a frozen dataclass)
  - `class Optimiser` (base of both mixins): class attributes `objective: str`, `goal: str = "max"`; `objective_of(self, results: dict[str, float]) -> float` (default `results[self.objective]`); `best(self, settings: list[float], objectives: list[float]) -> float`; `optimiser_schema(cls) -> dict`; `__init_subclass__` that adds the generated arguments as class attributes.
  - `class StagedScan(Optimiser)`: class attribute `stages: list[Stage]`; argument `repeats`; overrides `planned_points`, `prepare`, `run_point`, `finish`.
  - In `demo.py`: `DemoResonance(Experiment)` with `measure(self, *, detuning: float = 0.0, power: float = 1.0) -> dict[str, float]` returning `{"counts": ...}` (Lorentzian centred at 1.3, width 0.8, peak 1000 × power, Poisson noise from `np.random.default_rng(self.noise_seed)`; argument `noise_seed = Integer(0)`), whose `run_point` records `measure()`'s results; `DemoStagedScan(StagedScan, DemoResonance)` with `objective = "counts"`, stages over `detuning` (−5 to 5, 21 points, `save_as="demo_resonance"`) and `power` (0.1 to 1.0, 10 points, `save_as="demo_power"`).
- Default `best`: group objectives by setting, average each group ignoring non-finite values, return the setting with the best mean for `goal`; if no setting has a finite mean, raise `ValueError` naming the stage.
- Each point records `stage` (int), `setting` (float), `objective` (float) and every result of `measure()`.

- [ ] **Step 1: Write the failing tests**

```python
class SpyStagedScan(DemoStagedScan):
    calls = []

    def measure(self, **settings):
        SpyStagedScan.calls.append(settings)
        return super().measure(**settings)


def test_stage_two_runs_at_stage_ones_best_value():
    SpyStagedScan.calls.clear()
    m = run_local(SpyStagedScan, detuning_points=11, power_points=4)
    best_detuning = m.constants["proposed.demo_resonance"]
    assert best_detuning == pytest.approx(1.0, abs=1.0)
    assert [c["detuning"] for c in SpyStagedScan.calls[11:]] == [best_detuning] * 4

def test_the_default_best_averages_repeats_and_ignores_nan(): ...
    # Optimiser.best([1, 1, 2, 2], [10, nan, 8, 8]) == 1 for goal "max"

def test_an_overridden_best_replaces_the_default(): ...

def test_a_single_point_stage_proposes_its_value(): ...
    # detuning_start == detuning_stop == 0.5, detuning_points=1 → proposed 0.5

def test_the_points_argument_sets_the_number_of_points():
    assert run_local(DemoStagedScan, detuning_points=3, power_points=2, repeats=2).n_done == 10

def test_a_run_terminated_in_stage_two_proposes_only_stage_one(): ...
    # open_writer + run_points with a control returning Action.TERMINATE once
    # progress.done > detuning_points → proposed.demo_resonance present,
    # proposed.demo_power absent, stages_done == 1, stages_total == 2

def test_a_run_terminated_in_stage_one_proposes_nothing(): ...

def test_a_recipe_stores_the_generated_arguments(make_manager): ...
    # save_recipe(DemoStagedScan, "quick", detuning_points=5, client=in_process_client(...))
    # → the saved recipe's args include detuning_points == 5


def test_the_schema_lists_generated_arguments_and_proposals():
    schema = DemoStagedScan.schema()
    assert schema["arguments"]["detuning_points"]["default"] == 21
    assert schema["optimiser"] == {"kind": "staged", "proposes": ["demo_resonance", "demo_power"]}
```

- [ ] **Step 2: Run them to see them fail** — `ImportError` on `optimisers`.

- [ ] **Step 3: Implement `optimisers.py` (`Stage`, `Optimiser`, `StagedScan`) and the demo classes**

`__init_subclass__` adds an argument only if the subclass does not already declare one by that name. `measure()` is called with keyword settings only.

- [ ] **Step 4: Run the full suite** — all pass.

- [ ] **Step 5: Commit** — `git commit -m "Add StagedScan and a simulated demo resonance"`

---

### Task 3: `BayesianOptimisation`

**Files:**
- Modify: `pytweezer/experiment/optimisers.py`, `pytweezer/experiments/demo.py`
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Consumes: Task 2's `Optimiser`.
- Produces: `Dimension(setting: str, *, low: float, high: float, save_as: str)`; `class BayesianOptimisation(Optimiser)` with class attribute `dimensions: list[Dimension]` and the mixin arguments in Global Constraints; `DemoBayesian(BayesianOptimisation, DemoResonance)` with `objective = "counts"` and dimensions `detuning` (−5, 5, `demo_resonance`) and `power` (0.1, 1.0, `demo_power`).
- `prepare()` builds `skopt.Optimizer(dimensions=[(low, high), ...], base_estimator="GP", acq_func="EI", n_initial_points=n_initial, random_state=seed)`. Before each ask, set `acq_func_kwargs` `xi` to `xi_start * (xi_end / xi_start) ** (i / max(n_calls - 1, 1))` (geometric annealing, as `optimisation.py` does). Tell `-objective` for `"max"`. A non-finite objective is told as the worst finite objective so far; with none, the tell is skipped.
- Each point records every setting by its name, `objective`, `xi`, and every result.
- `best` for Bayesian: the settings of the best finite measured objective.

- [ ] **Step 1: Write the failing tests**

```python
def test_bayesian_finds_the_simulated_resonance():
    m = run_local(DemoBayesian, n_calls=25, seed=1)
    assert m.constants["proposed.demo_resonance"] == pytest.approx(1.3, abs=0.5)
    assert m.constants["evaluations_done"] == 25

def test_xi_is_annealed_from_start_to_end(): ...
    # m.results["xi"][0] == 0.1, [-1] == pytest.approx(0.001)

def test_a_nan_objective_does_not_stop_the_run(): ...
    # subclass whose measure() returns nan on the 3rd call → run completes, proposals finite

def test_a_bayesian_run_terminated_before_any_evaluation_proposes_nothing(): ...
```

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite** — all pass.
- [ ] **Step 5: Commit** — `git commit -m "Add BayesianOptimisation"`

---

### Task 4: Run-ID history and accepting end to end

**Files:**
- Modify: `pytweezer/experiment/saved_parameters.py` (`SavedParameterLog.read` gains `rid`), `pytweezer/servers/experiment_manager.py` (`_cmd_saved_parameter_history` passes `rid`), `pytweezer/experiment/client.py` (method and module function gain `rid: int | None = None`)
- Test: `tests/test_saved_parameters.py`, `tests/test_experiment_e2e.py`

**Interfaces:**
- Produces: `saved_parameter_history(key=None, *, rid=None, limit=None)` on the log, the command, the client method and the module function.

- [ ] **Step 1: Write the failing tests**

```python
def test_history_filters_by_run(tmp_path): ...
    # two changes with rid 7, one with rid 8 → read(rid=7) returns the two, newest first

def test_an_optimiser_run_can_be_accepted(harness):
    # write labexp/optim.py: "from pytweezer.experiments.demo import DemoStagedScan"
    # (inside the test, after the fixture has started — keeps the catalogue test's
    # single-module assertion intact); create demo_resonance and demo_power with
    # set_saved_parameters(..., create=True); submit "labexp.optim:DemoStagedScan"
    # with detuning_points=5, power_points=3; wait for completed; read proposals
    # from the measurement; set_saved_parameters(proposals,
    # "accepted from run <rid>", rid=rid) → values saved, and
    # saved_parameter_history(rid=rid) has two entries.
```

Use `until(..., timeout=30)` for catalogue and run waits.

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement the `rid` filter through all four layers.**
- [ ] **Step 4: Run the full suite** — all pass.
- [ ] **Step 5: Commit** — `git commit -m "Filter saved parameter history by run, and accept an optimiser run end to end"`

---

### Task 5: The Optimisers tab

**Files:**
- Create: `pytweezer/GUI/optimisers.py`
- Modify: `bin/gui.py` (tab after `Saved parameters`, built with the shared feed and a results archive as `make_results` builds one), `pytweezer/GUI/experiments/panel.py` only if opening an experiment by name needs a public method
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: catalogue entries' `"optimiser"` schema (Task 2); `Task.h5_path`, `client.history()`, the snapshot's `running`/`queue`/`history`; `archive.load(path)` with `.constants` and `.results`; `saved_parameters()`, `set_saved_parameters(..., rid=)`, `saved_parameter_history(rid=)` (Task 4); recipes via `client.recipes()`.
- Produces: `class OptimisersPanel(QWidget)`: `__init__(self, client=None, feed=None, archive=None, experiments_panel=None, parent=None)`; `select(module: str, class_name: str)`; `accept()`; `discard()` (hides the sheet for that run on this PC); `run_selected()`; `edit_arguments()` (calls `experiments_panel.load_request` with the optimiser's last request or defaults and raises that tab); attributes `rows` (one per optimiser) and `accept_rows` (key → checkbox) for tests.
- Row state, from the optimiser's latest task: `queued`/`running`; `failed`; `terminated` → `partial`; `completed` → `accepted` if `saved_parameter_history(rid=)` is non-empty, else `not accepted`.
- Plots: staged — one plot per stage of `objective` against `setting`, chosen value marked; Bayesian — `objective` per evaluation and running best. Use pyqtgraph as the Results tab does.

Layout and wording: spec section 6 and mockup A; follow the `pytweezer-gui-design` skill.

- [ ] **Step 1: Write the failing tests** (fake client, fake feed, fake archive returning a `Measurement`-like object with `.constants`/`.results`)

```python
def test_rows_show_each_run_state(qapp): ...
    # optimisers whose latest tasks are running, failed, terminated (partial label
    # "stage 1 of 2 finished"), completed+accepted, completed+not accepted

def test_run_is_disabled_while_queued_or_running(qapp): ...

def test_accept_writes_only_ticked_rows_with_the_run(qapp):
    panel = make_optimisers_panel()
    panel.select(DEMO, "DemoStagedScan")
    panel.accept_rows["demo_power"].setChecked(False)
    panel.accept()
    assert panel.client.calls[-1] == ("set_saved_parameters", {"demo_resonance": 1.2},
                                      "accepted from run 41", 41)

def test_a_failed_run_offers_no_accept(qapp): ...
def test_a_run_with_no_proposals_offers_no_accept(qapp): ...

def test_a_refused_accept_changes_nothing_and_says_why(qapp): ...
    # fake client raises ManagerError("no saved parameter 'demo_power'") → sheet
    # status state "error" with that text; no other write attempted
```

- [ ] **Step 2: Run them to see them fail** — `ModuleNotFoundError`.
- [ ] **Step 3: Implement the panel and register the tab.**
- [ ] **Step 4: Run the tests, then screenshot** with the design skill's `scripts/shoot.py` (fake data: a staged run with two stages, a Bayesian run, one row per state) and fix what its checklist finds.
- [ ] **Step 5: Commit** — `git commit -m "Add the Optimisers tab"`
