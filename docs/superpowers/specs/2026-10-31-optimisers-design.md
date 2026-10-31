# Optimisers — design

Date: 2026-10-31. Status: design, awaiting review.
Second of two specs; it builds on saved parameters
(`2026-10-30-saved-parameters-design.md`). The decisions behind it are in
`2026-10-30-optimisers-brainstorm.md` (numbered there; cited here as D1–D19).

## Intent

Calibrations such as finding a resonance or tuning shim currents are today
notebook routines that drive the hardware directly and write their results
straight into `parameters.json`. This spec gives the task system a general way
to run them: an optimiser is an ordinary queued experiment that scans or
optimises settings of a measurement, records the values it proposes for saved
parameters, and leaves a person to accept them on a dashboard.

The framework knows nothing about the lab. It is built and tested against a
simulated demo; the real optimisers (MOT position, resonances, MOT number,
MOT and molasses shims) are later work, one short experiment each.

## Goals

1. An optimiser is a short class: a building block (`StagedScan` or
   `BayesianOptimisation`) combined with a measurement experiment.
2. It runs as one task through the existing queue, with progress, pause,
   terminate and its measurement file working as for any experiment (D2–D4).
3. It proposes values for named saved parameters; nothing is saved until a
   person accepts them (D1, D6).
4. A dashboard shows every optimiser, its latest run and whether that run was
   accepted, and accepts the rows a person ticks (D8, D17, D18).
5. Its settings are ordinary arguments, so presets are recipes (D16).

## Non-goals

- **The real optimisers.** Only the demo is built here.
- **Optimiser structure designed in the GUI.** Stages and dimensions are code
  (D16).
- **Resuming an interrupted optimisation.** Its state lives in the worker (D3).
- **Accept creating keys.** A proposed key must already exist.

## 1. The measurement

A measurement experiment provides

```python
def measure(self, **settings) -> dict[str, float]: ...
```

one shot with the named settings overridden, returning its results. Its own
`run_point` calls `measure()` and records the results, so it is usable by hand.
What a setting means is the measurement experiment's business; the framework
only passes names and values through.

## 2. Experiments set their own points

`Experiment` gains

```python
@classmethod
def points(cls, scan: Scan) -> list[Point]:
    return scan.points(cls)
```

The runner (`open_writer`) and `submit()` (`_resolve`) call
`experiment_cls.points(scan)` in place of `scan.points(experiment_cls)`, so
existing experiments are unchanged. An optimiser returns its own number of
points with no argument values; the runner's progress, pause, terminate and
file handling work as before. An optimiser refuses a submitted scan with axes
or more than one repetition, at submission, with a message saying it sets its
own points.

## 3. The building blocks

Both are mixins in a new `pytweezer/experiment/optimisers.py`, combined with a
measurement experiment: `class DemoStagedScan(StagedScan, DemoResonance)`.

**Objective.** Each optimiser names one result of `measure()` as its
`objective` and a `goal` of `"max"` or `"min"`. `objective(results) -> float`
returns that result by default; an optimiser overrides it to combine results.

**Best value.** `best(settings, results) -> value` chooses a proposal. By
default it takes the setting whose objective, averaged over that setting's
repeat shots, is best for the goal (D19). An optimiser overrides it, for
example to fit a peak. If `best` raises, the run fails with that error.

### StagedScan

```python
stages = [
    Stage("detuning", start=-5.0, stop=5.0, points=21, save_as="demo_resonance"),
    Stage("power", start=0.1, stop=1.0, points=10, save_as="demo_power"),
]
```

- Each stage adds arguments `<setting>_start`, `<setting>_stop` (`Number`) and
  `<setting>_points` (`Integer`) with the stage's values as defaults (decision
  20 below). The mixin adds `repeats` (`Integer`, default 1).
- `points()` returns the sum over stages of points × repeats.
- The run goes stage by stage. Each shot calls `measure()` with the current
  stage's setting and every earlier stage's best value (D15). When a stage's
  points are done, `best()` gives its proposal.
- Each point records `stage` (index), `setting` (value), the objective and
  every result of `measure()`.

### BayesianOptimisation

```python
dimensions = [
    Dimension("detuning", low=-5.0, high=5.0, save_as="demo_resonance"),
    Dimension("power", low=0.1, high=1.0, save_as="demo_power"),
]
```

- Each dimension adds `<setting>_low` and `<setting>_high` (`Number`). The
  mixin adds `n_calls`, `n_initial` (`Integer`), `xi_start`, `xi_end`
  (`Number`) and `seed` (`Integer`).
- `points()` returns `n_calls` points. `prepare()` builds a
  `skopt.Optimizer` (Gaussian process, expected improvement); each point asks
  it for settings, calls `measure()`, and tells it the objective (negated for
  `"max"`). `xi` is annealed from `xi_start` to `xi_end` across the calls.
- Each point records every setting, the objective and every result.
- `best()` defaults to the best measured point.

### What a run records

In `finish()`, which the runner calls however the run ends:

- one constant per proposal, `proposed.<key>`, holding the proposed value;
- `stages_done` and `stages_total` (staged), or `evaluations_done` and
  `evaluations_total` (Bayesian), so a terminated run can be labelled partial.

A staged run proposes only for the stages it finished. A Bayesian run
proposes if it made at least one evaluation.

## 4. Finding optimisers

`Experiment.schema()` gains, for an optimiser, an `"optimiser"` entry:
`{"kind": "staged" | "bayesian", "proposes": [saved keys]}`. The catalogue
carries it like the rest of the schema, so the dashboard lists optimisers
without any registration.

## 5. Accepting

Accepting writes the ticked proposals with
`set_saved_parameters(values, "accepted from run <rid>", rid=rid)`. It is
offered for completed and terminated runs; a terminated run is labelled
partial ("stage 1 of 2 finished", "34 of 60 evaluations"). Failed runs show
their plots but cannot be accepted (D18). A key that does not exist, or a
value of the wrong type, is refused by the store; the sheet shows the refusal
and nothing is written (the store is all-or-nothing).

`saved_parameter_history` and its client function regain an optional `rid`
filter, so the dashboard can tell whether a run was accepted.

## 6. The Optimisers tab

A new tab, `Optimisers`, after `Saved parameters`, in
`pytweezer/GUI/optimisers.py`. Mockup A in
https://claude.ai/artifact/2E9MRytEvLUcr7AVahEcPJ.

- **Well, top: one row per optimiser** in the catalogue:
  - name and the keys it proposes;
  - its latest run's state as colour and text: queued, running, completed
    (not accepted), accepted, terminated (partial), failed;
  - when it was last accepted, from the saved-parameter history;
  - a preset picker listing its recipes, and **Run**, which submits the
    chosen recipe or the defaults; disabled while that optimiser is queued
    or running;
  - **Edit arguments…**, which opens the optimiser in the Experiments tab.
- **Below, for the selected row:**
  - the latest run's plots, read from its measurement file as the Results tab
    reads one: for a staged scan one plot per stage (objective against
    setting, the chosen value marked); for a Bayesian run objective per
    evaluation and best so far;
  - beside them, the **Accept sheet**: saved key, current value, proposed
    value, a tick per row; **Accept** as the one primary button, and
    **Discard**, which hides the sheet for that run on this PC.

The tab follows the manager's snapshot through `ExperimentFeed`, refetching
history and saved values when their versions change, and polls nothing while
hidden. A run whose file cannot be read says so on its row; the rest of the
tab keeps working.

## 7. The demo

In `pytweezer/experiments/demo.py`, beside `RabiDemo`:

- `DemoResonance`: a simulated measurement whose
  `measure(detuning=…, power=…)` returns `counts`, a Lorentzian in detuning
  scaled by power, with Poisson noise from a seeded generator. It declares no
  devices.
- `DemoStagedScan(StagedScan, DemoResonance)` and
  `DemoBayesian(BayesianOptimisation, DemoResonance)`, both maximising
  `counts` and proposing `demo_resonance` and `demo_power`.

Accepting a demo result needs those keys to exist; tests create them.

## 8. Testing

Following the `write-tests` skill.

`tests/test_optimisers.py` (new), running the demo with `run_local`:

1. An ordinary experiment's points are unchanged; an optimiser runs its own
   number of points, and a scan with axes is refused at submission.
2. `StagedScan`: stage 2 runs at stage 1's best value; the default `best`
   averages repeat shots; an override replaces it; a run terminated in stage
   2 records only stage 1's proposal and `stages_done` 1.
3. `BayesianOptimisation`: with a fixed seed it finds the simulated optimum
   within a tolerance; `xi` falls from `xi_start` to `xi_end`; the proposals
   are the best measured point.
4. The generated arguments appear in `schema()` with the stages' defaults,
   and a recipe stores them; the schema's `"optimiser"` entry lists the
   proposed keys.

`tests/test_experiment_manager.py`:

5. `saved_parameter_history` filters by `rid`.

`tests/test_experiment_gui.py` (offscreen, fake client):

6. Rows show each run state, including partial; Run is disabled while queued
   or running; Accept writes only the ticked rows with the run ID and reason;
   a failed run has no Accept; a refusal is shown.

`tests/test_experiment_e2e.py`:

7. `DemoStagedScan` submitted to a real manager completes; accepting its
   proposals saves them, and the log entries carry its run ID.

## Decision 20

**A stage or dimension creates its own arguments** from the values given in
it: `<setting>_start`, `_stop`, `_points` for a stage; `<setting>_low`,
`_high` for a dimension. They appear in the Experiments form and recipes like
any other argument.
