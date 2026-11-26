# Tweezer-array scores and optimiser objectives

## Why

A tweezer array can be scored in several ways: loading probability, imaging
fidelity, imaging survival, or a combination of them. Each optimiser wants a
different combination and a different experiment configuration:

| Score | Images it uses | Drop between images |
| --- | --- | --- |
| loading probability | first | no |
| imaging fidelity | first | no |
| survival | first and second, compared | yes for molasses; none for imaging |

Today `TweezerOptimisation.score_batch` scores every evaluation the same way.
Each optimiser then works around that:

- every `measure()` re-lists the script parameters its `Dimension`s already
  name in `save_as`;
- every `measure()` passes `BackgroundDrop`/`tDropDuration` as raw overrides;
- each `objective_of` is a hand-made ladder that encodes no clear goal.

This change makes each optimiser state its goal as a plain cost function, and
moves the repetitive parts into the framework class.

## Goals

1. **`TweezerOptimisation.measure` is written once.**
   - It maps each search setting to its script parameter through the
     `Dimension`'s `save_as`.
   - A `None` setting keeps its saved value, as today.
   - An optimiser overrides `measure` only when one setting drives several
     script parameters, as the imaging optimiser's VCO1/VCO2 does.
2. **The drop is declared, not passed as raw overrides.**
   - `drop: str | None = None` is a class attribute.
   - `None` means no drop between the images: `BackgroundDrop=0`.
   - A string names the argument that holds the drop duration:
     `BackgroundDrop=1, tDropDuration=<that argument>`.
3. **Each optimiser writes its objective as plain maths.** It writes
   `objective_of(self, scores, settings)`, where:
   - `scores` is `{"loading", "fidelity", "survival"}` for this evaluation;
   - `settings` holds every search setting as actually run, with saved values
     filled in for `None`.
4. **Two helpers serve as the vocabulary for penalty-form objectives,** with
   no other new concepts:
   - `shortfall(value, threshold) = max(0, threshold − value)`;
   - `shortness(value, dimension) = 1 − (value − low)/(high − low)`, using the
     dimension's limits; it is 1 at the shortest allowed time and 0 at the
     longest.
5. **The objectives:**
   - **MOT** (`OptimiseMOTLight`): shortest load time with loading at or above
     `loading_threshold`:

     ```python
     shortness(settings["load_time"], self.dimension("load_time"))
     - self.shortfall_penalty * shortfall(scores["loading"], self.loading_threshold)
     ```

     The load time becomes a search dimension. No drop.
   - **Molasses** (`OptimiseMolassesLight`; `OptimiseMolassesShims` too, see
     Calls below): shortest molasses time with loading and drop survival each
     at or above a threshold:

     ```python
     shortness(settings["molasses_time"], self.dimension("molasses_time"))
     - self.shortfall_penalty * shortfall(scores["loading"], self.loading_threshold)
     - self.shortfall_penalty * shortfall(scores["survival"], self.survival_threshold)
     ```

     The molasses time is a search dimension (`tMolassesDuration`), and
     `drop = "drop_duration"`.
   - **Imaging** (`OptimiseTweezerImaging`):

     ```python
     w * scores["fidelity"] + (1 - w) * scores["survival"]
     ```

     where `w = fidelity_weight`, an argument. No drop. This replaces the
     `fidelity_floor` rule.
6. **`FindTrapSites` does no drop.**
   - Its calibration shots run with no overrides: no `BackgroundDrop` and no
     tweezer-off VVAs.
   - The calibration image is the mean of the first images, with no
     background frame subtracted; the top-hat filter already removes the
     background.
   - `background_subtracted_mean` is removed if nothing else uses it.

## Non-goals

- **One-image shots.** Every evaluation still takes two images; scores that
  need only the first ignore the second. Taking one image needs a
  script-level way to skip the second, to be done later.
- **Changing the threshold policy.** Loading is still judged against a
  threshold refitted on each batch, and survival against the accepted one.
  The existing TODO stays.
- **Constrained Bayesian optimisation** (approach C, below).

## Constraints

- **Saved parameters and results.** These optimisers still propose through
  `save_as`. They record the same scores as today (`loading`, `survival`,
  `fidelity`, the occupancy and counts), plus the objective.
- **Other optimisers are untouched.** `StagedScan`, `Fit` and the demos keep
  `objective_of(results)`. The two-argument form exists only on
  `TweezerOptimisation`. How it receives the settings is left to the builder;
  the framework's `BayesianOptimisation` already knows each evaluation's
  settings.
- **Tests follow the design.** The existing tests of the old objectives
  (`fidelity_floor`, `loading_floor`) are rewritten or deleted.

## Approaches considered

- **A. Each optimiser hand-writes `measure` and an if/else objective ladder.**
  Explicit, but it repeats the script mapping, the drop overrides and the tier
  arithmetic in every optimiser. The ladder's constants also silently repeat
  the dimension limits.
- **B. Declarative requirements** (`requirements = [AtLeast(...)]`,
  `then = Shortest(...)`), turned into a tiered score by the base class.
  Compact, but a reader must learn new words and trust the base class to
  turn them into a score.
- **C. Constrained optimisation in the framework.** Ruled out by the user.
- **Chosen: custom objective functions in penalty form.** This takes B's
  framework half (generic `measure`, declared drop, settings passed in) and
  A's explicitness: each objective is plain arithmetic in the optimiser.
  - The penalty form is preferred to switching weights at the threshold,
    because it is smooth: the Bayesian optimiser's model gets a slope towards
    the threshold rather than a cliff.
  - The penalty weight is a visible, tunable argument.

## Calls I made (check these)

- **`OptimiseMolassesShims` also minimises molasses time**, with the same
  objective as `OptimiseMolassesLight`, since "the molasses" covers both. If
  the shims optimiser should only meet the two thresholds, its objective is
  the two penalty terms alone.
- **Defaults:**
  - `loading_threshold` 0.55 (from the user);
  - `survival_threshold` 0.8;
  - `shortfall_penalty` 10, so a 10 % shortfall costs as much as the whole
    range of times;
  - `fidelity_weight` 0.5.
- **The MOT load-time parameter is a placeholder:**
  `MOT_LOAD_TIME = "tMOTLoadTime"  # FIXME: placeholder, replace with the
  script's parameter name`.
  - Its dimension limits are also placeholders, `(10, 500)`, in the script's
    units.
  - The molasses time uses `tMolassesDuration` with limits `(1, 20)`.
  - The user should check all of these before running.
- **No search dimension is saved under a placeholder name.** The MOT load
  time's `save_as` is the placeholder constant, so an accepted proposal would
  save under that name until it is replaced.

## Left to the builder

- How `objective_of` gets `settings`, and where `shortfall`/`shortness` live:
  in `experiment/optimisers.py` if `shortness` needs only a `Dimension`, else
  in `optimisers/tweezer_optimisation.py`.
- Spans for the new time dimensions.
- The tests: each objective's arithmetic and the `measure` mapping, using the
  fakes `tests/test_tweezer_array.py` already provides.
- The skill updates: `add-optimiser`, whose rules this changes. The user
  approves them by approving this design.
