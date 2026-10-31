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
5. **Optimisers inherit their measurement.** A measurement experiment puts
   one shot in `measure(**settings) -> dict[str, float]`; its own `run_point`
   records that. Two generic mixins, `StagedScan` and `BayesianOptimisation`,
   own the point loop and record the best values in `finish()`. An optimiser
   is a short class combining a mixin with a measurement experiment, naming
   its objective and its stages or dimensions. Rejected: one generic optimiser
   taking the target class as an argument (devices would depend on argument
   values), and shared plain functions (declarations copied into every
   optimiser). Needs one framework hook: the experiment, not the submitted
   `Scan`, says how many points it runs.
6. **Accept writes to saved parameters, which are their own namespace.** A
   saved key is not an argument or a MOTMaster parameter, even where the
   names match. Each stage or dimension names the saved key it proposes a
   value for. Accept only updates keys that already exist and shows each
   key's current and new value, with rows the user can untick.
7. *(Moved to later work.)* Where a particular optimiser's inputs come from
   (the MOT centre, tweezer site positions) is decided when that optimiser is
   written, not in the framework.
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
   - Accept writes through `set_saved_parameters(..., rid=...)`, so each
     acceptance is logged with its run; spec 2 decides whether it needs a
     command of its own.
   - Everything that reads `parameters.json` today moves to the manager
     (later work, not spec 1):
     `TweezerBaseline`, `analysis.py` (reads `cool_vco_resonance` at import),
     notebooks.
10. **Two specs, built in order.** (1) Saved parameters: the manager-owned
    store and change log, get/set/history commands, client functions, the
    tab and an import from `parameters.json`. (2) Optimisers, the framework
    only: `measure()`, the `StagedScan` and `BayesianOptimisation` mixins, the
    `planned_scan()` hook, Accept and the dashboard, built and tested against
    a demo optimiser over `pytweezer/experiments/demo.py`. The real
    optimisers (MOT position, resonances, MOT number, MOT and molasses shims)
    are later work, one short experiment each. Each spec gets its own plan
    and PR.
11. **Saved parameters are a plain store with no special behaviour.** They
    are not tied to arguments or MOTMaster parameters. Code anywhere
    (experiments, notebooks, analysis) reads them with `get_saved_parameter(key)` /
    `get_saved_parameters()` and writes with `set_saved_parameters(values, reason)` from
    `pytweezer.experiment.client`, both going to the manager. A user who wants
    a saved value as an argument's default writes `Number(get_saved_parameter("x_centre"))`,
    read when the module is imported; the Experiments form submits exactly
    what it shows, and its Defaults button has the manager describe the
    experiment again so a changed saved value is picked up. One who wants it
    in a MOTMaster parameter reads it in `prepare()`. What is recorded in the
    measurement file is up to the experiment. Ordinary experiments are expected only to read; optimisers'
    values are written through Accept. Rejected along the way: a `saved=` /
    `Saved(...)` link on arguments, recording which arguments came from saved
    keys, and a `saved_parameters=` argument to `run_local`.
12. **Saved keys are flat and unique across the lab**; today's keys stay as
    they are. The old categories (`motmaster_parameters`, `misc_parameters`,
    `emccd_parameters`, `image_analysis`) become display groups. CaF keys get
    their own names when they arrive (e.g. `caf.tImgCoolVCO` by convention).
13. **The framework knows nothing about tweezers or MOTs.** The task system,
    saved parameters and optimisers are general; lab-specific behaviour lives
    only in the experiments written on top of them.
14. **One place runs tasks.** The Experiment Manager on the server PC runs
    every task in a worker it starts; code inside a task reaches the manager
    at the address CONFIG gives, like any other client.

## Spec 2 decisions (2026-10-31)

15. **Within one run, later stages use earlier stages' best values directly**,
    without waiting for acceptance.
16. **Presets are code plus recipes.** An optimiser's stages and dimensions are
    its class; its ranges, point counts and evaluation counts are arguments,
    so saved presets are recipes. Nothing is designed in the tab.
17. **Any PC can watch a run and accept its result**; accepting is a
    saved-parameter write through the manager.
18. **Completed and terminated runs can be accepted; failed runs cannot.** A
    terminated run's Accept sheet says it is partial ("stage 1 of 2 finished",
    "34 of 60 evaluations"); in a staged scan only finished stages propose
    values.
19. **A stage proposes the setting with the best mean result by default**
    (over that setting's repeat shots), for a named result and a goal of
    higher or lower; an optimiser overrides `best(settings, results) ->
    value` for anything else, such as a peak fit. The framework records
    whatever `best` returns and the plot marks it. A Bayesian optimisation
    uses the same hook, proposing the best measured point by default.

## Status

- Spec 1 revised 2026-10-31: `2026-10-30-saved-parameters-design.md`, awaiting
  review. Its plan is rewritten once the spec is approved.
- Then spec 2 (`2026-10-30-optimisers-design.md`), the framework only, and
  `superpowers:writing-plans` for it.

## Later work, not in either spec

The routines split into single-purpose experiments, each also usable by hand,
built on the spec 2 framework: MOT position, MOT counts, tweezer loading
probability, molasses survival probability; and the optimisers over them
(resonances, MOT number, MOT and molasses shims). Where each gets its inputs
(MOT centre, site positions) is decided then.

Moving the existing readers of `parameters.json` (`TweezerBaseline`,
`analysis.py`, `optimisation.py`, notebooks) to saved parameters, running the
import on the server PC, and deleting `parameters.json` and
`ExpParameterManager`.

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
