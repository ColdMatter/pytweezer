# Optimisers tab: brainstorming notes (in progress)

Handover notes from a brainstorming session that stopped part-way. Nothing has
been built. This is not the design document; that comes once the open questions
below are settled.

## Where this came from

`pytweezer/helper_scripts/optimisation.py` holds notebook-style routines that
drive the Rb MOTMaster and cameras directly, not as pytweezer experiments. All of
them write their results to `pytweezer/experiment/parameters.json` through
`ExpParameterManager`:

- `DetectMOTPosition` (Rb ThorCam): takes background shots (MOT coils off,
  `tMOTccValue` 0) and image shots, then fits a Gaussian to find the MOT centre.
- `DetectResonances`: sweeps the imaging cooling VCO `tImgCoolVCO1` (3 to 5 V,
  20 points) and saves `cool_vco_resonance`. It then sweeps the repump VCO
  `tImgRepVCO1` (0 to 5 V) at that value and saves `rep_vco_resonance`. Each
  stage maximises the background-subtracted counts in a window around the MOT
  centre.
- `OptimiseMOTNumber`: the same pattern over four stages in turn:
  `tMOTLoadCoolVCO`, `tMOTLoadCoolVVA`, `tMOTLoadRepVCO`, `tMOTLoadRepVVA`.
- `OptimiseMOTShimParameters` and `OptimiseMolassesShimParameters` (Rb HamCam):
  Bayesian optimisation (skopt, EI, `xi` annealed from 0.1 to 0.001) of the three
  MOT or molasses shim currents.
  - The MOT shims maximise tweezer loading probability.
  - The molasses shims maximise survival, falling back to loading probability
    when loading is below 50 %.

`tests/test_optimisation.py` tests these routines against fakes.

## Original request

- Create pytweezer experiments covering these routines.
- Keep `optimisation.py`.
- Experiments get no individual tests. Instead, write generic tests that run over
  every experiment module:
  - Does it import cleanly the way the manager imports it
    (`pytweezer.experiment.introspect.describe`)?
  - Does every declared device exist in CONFIG (`reserved_devices()` /
    `resolve_address`)?
- Idea, not yet requested: run each experiment through
  `run_local(..., simulate=True)`, which would also catch a device used without
  being declared.
- Undecided: whether to keep `tests/test_tweezer_baseline.py`, an existing
  test for a single experiment.

## Revised direction

An **Optimisers tab** that runs **preset scans and Bayesian optimisations using
experiments**.

## Decided

1. **Results are accepted by hand.** When a run ends, the tab shows the best
   values and the data behind them. Nothing reaches `parameters.json` until the
   user accepts.
2. **Runs live on the server and go through the experiment queue.** The
   optimiser submits ordinary tasks, waits for them and reads their results.
   Closing the GUI does not stop a run.
3. **A Bayesian optimisation is a single task**, not one task per evaluation.
   It is an `Experiment`: `prepare()` builds the skopt `Optimizer`, each point
   is one evaluation (ask, run the shots, compute the objective, tell, record),
   and `finish()` records the best values as constants. Consequences:
   - The number of evaluations is known up front, so the existing runner,
     progress, pause and terminate (at point boundaries) and the measurement
     file work unchanged. The run's history is that file.
   - The task holds its devices for the whole optimisation, so no other task
     can change the sequencer's state between evaluations.
   - The optimiser's state lives in the worker's memory; a crashed or
     interrupted run does not resume (a later run could be seeded from the
     old file's history).
   - Its settings (bounds, number of calls, initial points, `xi` schedule) are
     experiment arguments, so saved presets are ordinary recipes.
4. **A staged scan is a single task too.** The experiment runs stage 1's
   points, takes the best value, then runs stage 2 at it; the total number of
   points is known up front. There is **no optimiser service**: the
   Optimisers tab launches ordinary tasks, shows their progress and offers
   Accept. All stages of one staged scan belong to one experiment.
5. **Optimisers inherit their measurement.** Each single-purpose experiment
   puts its shot in `measure(**motmaster_overrides) -> dict[str, float]`; its
   own `run_point` records that. Two generic mixins, `StagedScan` and
   `BayesianOptimisation`, own the point loop and record the best values in
   `finish()`. An optimiser is a short class such as
   `class OptimiseMOTShims(BayesianOptimisation, TweezerLoading)` naming its
   objective and its stages or dimensions. Rejected: one generic optimiser
   taking the target class as an argument (devices would depend on argument
   values), and shared plain functions (declarations copied into every
   optimiser). Needs one framework hook: the experiment, not the submitted
   `Scan`, says how many points it runs.
   - `measure()` follows the usual camera pattern (see `LoadTweezers.run_point`):
     start the acquisition, run the MOTMaster sequence (which triggers the
     camera; `Go()` blocks for the whole sequence while the camera buffers),
     then read the frames.
6. **Accept writes to saved parameters, which are their own namespace.** A
   `parameters.json` key is not a MOTMaster parameter, even where the names
   match: different experiments may keep different values for the same
   MOTMaster parameter under different keys. A stage or dimension saves under
   its MOTMaster parameter's name by default and takes `save_as=` for a
   distinct one, e.g. `Stage("rb.tImgCoolVCO1", values,
   save_as="cool_vco_resonance")`; results that are not MOTMaster settings
   (the MOT centre) always name their key. Accept only updates keys that
   already exist (`ExpParameterManager.set_parameter` refuses others) and
   shows each key's current and new value, with rows the user can untick.
7. **MOT counts reads the saved MOT centre** (`x_centre`, `y_centre`) in
   `prepare()`, as `TweezerBaseline` reads its settings; arguments can
   override it by hand. Finding the centre is a separate MOT Position run whose
   result is accepted first. Runs stay short, at the cost of a manual step and
   a wrong window if the MOT has moved since.
8. **The tab is a calibration dashboard** (mockup A in
   https://claude.ai/artifact/2E9MRytEvLUcr7AVahEcPJ):
   - Top well: one row per optimiser with what it saves, its last run (state
     as colour and text, including "completed, not accepted"), when it was
     last accepted, a preset (recipe) picker and a Run button (disabled while
     that optimiser is queued or running).
   - Below, for the selected row: its latest run's plots (one per stage, the
     chosen value marked; for a Bayesian run, objective per evaluation and best
     so far) and, beside them, the Accept sheet: saved parameter, current, new,
     a tick per row, Accept as the one primary button, and Discard.
   - Arguments are edited in the Experiments tab (a link from the dashboard);
     the run itself is also in Results. Rejected: a self-contained tab
     duplicating the argument editor, and Accept living only in Results (no
     view of which calibrations are stale).
9. **Saved parameters move out of git and become a standard facility.**
   `TweezerBaseline`'s behaviour (saved values feed MOTMaster parameters,
   overridable per task) becomes the norm for new experiments, so it is built
   properly now rather than copied:
   - The Experiment Manager owns the saved values, in its data directory next
     to `recipes.json`; the tracked `parameters.json` only seeds them. This
     stops Accept dirtying the server PC's checkout (and breaking `git pull`)
     and stops other PCs reading stale copies.
   - Accept is a manager command; each acceptance is logged (time, who, run,
     key, old, new) and the snapshot carries the latest per optimiser, so
     every PC's dashboard updates live.
   - Everything that reads `parameters.json` today moves to the manager:
     `TweezerBaseline`, `analysis.py` (reads `cool_vco_resonance` at import),
     notebooks.

## Proposed, not yet confirmed

- The routines split into single-purpose experiments, each also usable by hand:
  - MOT position: the background-subtracted image and its fitted centre.
  - MOT counts: windowed counts around a given centre.
  - Tweezer loading probability.
  - Molasses survival probability.
- A scan preset sweeps one argument, picks the setting that maximises a result,
  and can chain stages. DetectResonances, for example, is a cooling-VCO stage
  followed by a repump-VCO stage.
- Within one run, later stages use the earlier stages' optima directly, without
  waiting for acceptance.
- Presets are code (optimisation experiments) plus recipes (their settings);
  nothing is designed in the tab.
- Any PC can watch a run or accept its result.

## Open questions

- Split into two specs (saved parameters first, then optimisers)?
- How an experiment declares that an argument comes from a saved value, and
  whether it is resolved at submit or when the task starts.

## Still to cover

- Then the written design
  (`docs/superpowers/specs/2026-10-30-optimisers-design.md`), the builder check,
  and `superpowers:writing-plans`.

## Relevant code and skills

- The experiment framework, in `pytweezer/experiment/`:
  - `experiment.py`
  - `motmaster.py`
  - `scan.py`
  - `client.py` (`submit`/`wait`)
  - `introspect.py`
  - `catalogue.py`
- `pytweezer/experiments/tweezer_baseline.py`: an example that reads saved
  parameters in `prepare()`.
- Skills: `add-experiment`, `pytweezer-gui-internals`, `pytweezer-gui-design`,
  `write-tests`.
