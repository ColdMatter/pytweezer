---
name: add-optimiser
description: Write, run and accept pytweezer optimisers — StagedScan, BayesianOptimisation and Fit classes in pytweezer/optimisers/ (mixins in pytweezer/experiment/optimisers.py) that run a measurement experiment and propose new saved-parameter values, accepted by hand in the Optimisers tab. Use whenever the user wants to write, add or change an optimiser or calibration search, find the best value of something ("find the best imaging detuning", "optimise the molasses shims", "fit the trap sites"), work on FindTrapSites, OptimiseTweezerImaging or OptimiseMolassesShims, accept or discard an optimiser run's proposals, or work on the Optimisers tab (pytweezer/GUI/optimisers.py) — even if they only describe the search without saying "optimiser". For the measurement experiment it runs, and for saved parameters themselves, use add-experiment.
---

# Optimisers

An optimiser is a class in a module under `pytweezer/optimisers/`, built on a
mixin from `pytweezer/experiment/optimisers.py`; it runs a *measurement
experiment* to propose saved-parameter values; nothing is saved until someone
accepts them in the **Optimisers** tab. It is an `Experiment` subclass, so
everything in the add-experiment skill (arguments, devices, recording, the
queue, files, saved parameters) applies; this skill covers what an optimiser
adds. Device-free examples: `DemoStagedScan`/`DemoBayesian` in
`optimisers/demo_staged_scan.py`/`demo_bayesian.py` (over
`experiments/demo_resonance.py`); real ones `optimisers/find_trap_sites.py`,
`optimise_tweezer_imaging.py`, `optimise_molasses_shims.py`.

## Search optimisers

The measurement provides `measure(**settings) -> dict[str, float]`, every
setting a keyword with a default (an early `StagedScan` stage passes only the
settings scanned so far; the tweezer ones take `None` as "keep the saved
value"); put the mixin first:

```python
class FindResonance(StagedScan, Resonance):  # or BayesianOptimisation
    objective = "counts"  # a key of measure()'s results
    goal = "max"  # or "min"
    stages: ClassVar[list[Stage]] = [
        Stage("detuning", start=-5.0, stop=5.0, points=21, save_as="resonance"),
    ]
    # BayesianOptimisation: dimensions = [Dimension("detuning", low=-5, high=5, save_as=...)]
```

- Each `Stage` generates `<setting>_run/_start/_stop/_points` arguments (each
  `Dimension` `<setting>_low/_high`); declare one of those names yourself to
  override it. `Dimension(..., span=0.5, save_as=...)` instead searches its
  `save_as` value at the start of the run ± `<setting>_span`; every Bayesian run
  records its `search_bounds`. `Dimension(..., whole=True)` asks only whole
  numbers (for an Int32 script parameter; a range holding fewer than two raises
  `ValueError` at `prepare()`). `Dimension(..., limits=(low, high))` clips
  either kind of range to what the hardware takes (an analog output's ±10 V);
  a range wholly outside them raises `ValueError` at `prepare()`. Duplicate
  settings or `save_as`, a bad `goal`, `low >= high`, neither bounds nor span,
  a non-positive span, or a span with bounds raise `TypeError` when the class
  is defined (a `Stage` likewise: `start`/`stop` or `span`, not both).
- `Stage(..., span=s, save_as=...)` is the alternative to `start`/`stop`: it
  generates `<setting>_span` and `<setting>_centre` instead of
  `_start/_stop`, and sweeps `centre - span` to `centre + span`. The centre is
  saved-backed: left empty it is the `save_as` value when the run starts, so
  every span stage's `save_as` must exist even when the stage is switched off,
  since the runner resolves it for all stages. A non-positive `_span` raises
  `ValueError` when the run starts. A subclass places the centre differently by
  overriding `centre_argument(stage)` (the argument's name and declaration) and
  `centre_of(stage)` (its value, read after `prepare()`, so it may use what
  `prepare()` measured).
- `Stage(..., run=False)` makes `<setting>_run` default to false; a stage whose
  `_run` is false is not scanned, passes nothing to later stages and proposes nothing
  (`enabled_stages()` lists `(index in stages, stage)` for those that run).
  `stage` records still use the index in `stages`, and `stages_total` counts
  enabled stages. A run with no stage enabled is refused (`ValueError` in
  `prepare()`).
- A `StagedScan` sweeps each enabled stage `repeats` times, then passes the best
  values of earlier stages to every later stage's `measure()`. A
  `BayesianOptimisation` (skopt GP, expected improvement) makes `n_calls`
  evaluations, the first `n_initial` random, with `xi` falling geometrically
  from `xi_start` to `xi_end`; a non-finite objective is told as the worst so
  far.
- Override `objective_of(results, settings)` for an objective computed from
  several results or the settings measured at, and `best(settings, objectives)`
  to choose differently; in a `StagedScan`, `self.stage` is the `Stage` being
  chosen for. `shortfall(value, threshold)` and `shortness(value, bounds)` in
  `experiment/optimisers.py` help build penalties; a Bayesian run's
  `self.bounds` holds its search bounds.
- The mixins record their own results — `stage`, `setting`, `objective`
  (staged); each setting name, `objective`, `xi` (Bayesian) — so `measure()`
  must not return those names. Progress constants: `stages_done/_total`,
  `evaluations_done/_total`.
- **Every `save_as` must already be a saved parameter**: accepting a search
  only sets existing keys, and a span dimension raises `KeyError` at
  `prepare()` without one. Create it first (`set_saved_parameters(...,
  create=True)`).
- `Optimiser.apply(cls, proposals)`, a classmethod that does nothing by default,
  puts accepted values to use (e.g. moves hardware); `optimiser_schema()["applies"]`
  says whether a class overrides it. It receives `{saved key: value}` of the
  accepted values, after they are saved.

## Fit

`class FindX(Fit, Measurement)` runs one point that calls `fit() -> {key:
value}` and proposes those values. The keys are those of `proposal_units`
(`{key: unit}`); `proposal_group` is the group they are created in. Accepting a
fit creates keys not yet saved, and its values are accepted together (no
per-key ticks). Values may be numpy arrays (array saved parameters). The tab's
fit view draws the results `calibration_image` and `roi_sites`
(`FindTrapSites`'); another fit needs its own view in `GUI/optimisers.py`
(`_plotted_names`, `_FitPlots`).

## Running and accepting

- An optimiser sets its own points (`planned_points`): submitting it with scan
  axes or repetitions is refused. Its proposals are constants
  `proposed.<key>`.
- In the GUI, optimisers are set up and run only in the **Optimisers** tab
  (`pytweezer/GUI/optimisers.py`), which lists the manager's optimisers
  catalogue (`pytweezer.optimisers`, fetched with the `optimisers` command);
  the Experiments tab lists only the experiments catalogue
  (`pytweezer.experiments`),
  and Results' Resubmit opens an optimiser run there. From code, `submit`
  without a scan works as for any experiment. The tab's form groups the
  mixin's own arguments (group `SEARCH`) first, starts from the last run's
  arguments, and saves presets as recipes. A running optimiser is plotted from
  the feed's published points; a finished one from its file. **Terminate**
  under the plots stops the running or paused run after its current point. `analyses` is
  `()`: optimiser runs are not analysed in the Results tab.
- Only a `completed` or `terminated` run can be accepted (a terminated staged
  scan proposes the stages it finished). **Accept** saves the ticked values
  with reason `accepted from run <rid>` and that `rid`, all or none; the log
  then lists every key named, unchanged ones included, which is how the tab
  knows a run was accepted (`saved_parameter_history(rid=...)`). **Discard**
  hides a run's proposals on that PC only.
- After saving, the tab calls `apply` with the saved values off the GUI thread
  if the optimiser `applies`. It skips it, saying "not applied: run N is using
  X", when a running or paused task reserves any device the optimiser reserves
  (`live_task_using`). A failing `apply` is reported but never undoes the save.
- The tab's accept logic lives in helpers in `experiment/optimisers.py`:
  `proposals`, `accept_proposals`, `changed_summary`, `live_task_using`; keep
  accept behaviour there, not in the tab, so any other caller shares it.

## The Rb tweezer optimisers

The measurements they run (`SlmTweezerArray`, `LoadTweezers`) are described
under "Tweezer array measurements" in add-experiment. The Bayesian ones are
`(TweezerOptimisation, LoadTweezers)`: `TweezerOptimisation`
(`optimisers/tweezer_optimisation.py`) adds `shots`, `score_batch()` and a
generic `measure()`, and refuses to run without accepted trap sites. `measure()`
passes each setting to its dimension's `save_as` script parameter (`None` keeps
the saved value) and takes the drop from `drop`: a class attribute naming the
argument that holds the drop duration (`tDropDuration`), or `None` (the
default) for `BackgroundDrop=0`. Only `OptimiseTweezerImaging` overrides
`measure()`, because one setting sets two script parameters. Every evaluation
takes two images; loading and fidelity come from the first. It scores each batch through
`TrapSites` (`analysis/array_loading.py`): `LoadTweezers.trap_sites` holds the
accepted sites with their window, feature size and threshold, and `score_batch`
only calls its `counts`, `fit_loading` and `survival`, never inline maths.

- `FindTrapSites` (`Fit`) takes `calibration_shots` with no drop, fits the
  rig's array geometry (or a `default_rows` × `default_columns` grid) to the
  top-hat filtered (`top_hat_image`) mean of their first images
  (`analysis/trap_sites.py`), fits the occupancy threshold with a `TrapSites`, and proposes
  `TRAP_SITE_KEYS`: `trap_sites` (sensor pixels, `(y, x)`), `trap_threshold`,
  `trap_window_size`, `trap_feature_size`, `trap_geometry`. Tweezer
  measurements refuse to run when the saved `trap_geometry` is not the rig's
  current array, so rerun it after changing the array.
- Their objectives are penalty forms, computed by `objective_of(scores,
  settings)` from `loading`, `survival` and `fidelity`, all measured by
  `score_batch()`. A search for the shortest time subtracts
  `shortfall_penalty` (10) times each shortfall below a threshold
  (`loading_threshold` 0.55, `survival_threshold` 0.8) from
  `shortness(time, bounds)`, so any miss ranks below any run that passes.
  The time is a whole-number dimension (`load_time`, `molasses_time`).
- `OptimiseTweezerImaging` is Bayesian with span dimensions around the saved
  `tImgCoolVCO`/`tImgCoolVVA`, no drop, scored `fidelity_weight` (0.5) ×
  fidelity + (1 − weight) × survival. It overrides `measure()` to set both
  imaging stages' VCO or VVA (`tImgCoolVCO1/2`).
- `OptimiseMolassesShims` has span dimensions around
  `tMolassesShim{X,Y,Z}ccValue` (limited to the card's ±10 V) and the whole
  `molasses_time` (`tMolassesDuration`, 1–20); `drop = "drop_duration"`.
- `OptimiseMOTLight` searches the MOT-loading cooling and repump VCOs and VVAs
  (`tMOTLoad{Cool,Rep}{VCO,VVA}`, limited to 0–10 V) and the whole `load_time`,
  for the shortest load that reaches `loading_threshold`, with no drop. Its
  `MOT_LOAD_TIME` script parameter name and `load_time` limits are
  placeholders (FIXME in the module).
- `OptimiseMolassesLight` searches both molasses stages' (`tMol{Cool,Rep}{VCO,VVA}{1,2}`,
  eight dimensions) and the molasses time, scored like `OptimiseMolassesShims`.

Tests: `tests/test_optimisers.py`,
`tests/test_tweezer_array.py`, `tests/test_trap_sites.py`,
`tests/test_saved_parameters.py`; the tab in `tests/test_experiment_gui.py`.
