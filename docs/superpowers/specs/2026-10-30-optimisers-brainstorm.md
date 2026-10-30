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

- How an optimisation reuses a single-purpose experiment's measurement
  (inheritance or a generic wrapper naming the experiment class).

## Still to cover

- The tab's layout (offer the visual companion for mockups).
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
