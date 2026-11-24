# Framework / user-code split, and readable experiments

## Why

Day to day, physicists edit **experiments, optimisers, analyses (live and
default) and applets**. They know Python from data analysis, but not UI or
networking programming. That code should read plainly, with the Qt, threading,
ZMQ and archive machinery out of sight in framework code underneath.

The repo already does this for experiments: `experiment/` is the framework and
`experiments/` is what people edit. This change applies the same pattern to
analyses:

- A **singular** folder is framework. A **plural** folder is what people edit.

The last code commit on this branch (`9bf55f1`) merged each default analysis's
calculation and Qt view into `analysis/default_analyses/`. Its aim was right:
it took `TweezerHistogramView` out of the generic framework file
`GUI/experiments/analyses.py`. Its side effect, putting Qt into `analysis/`, is
unwanted and is undone here.

## Goals

1. `analysis/` holds framework and shared maths only, with no Qt and no
   scripts people edit.
2. The code people edit sits in plural folders: `experiments/`, `optimisers/`,
   `analyses/live/`, `analyses/default/` and `GUI/viewers/`.
3. A default analysis is a Qt-free calculation in `analyses/default/`, plus a
   short view in `GUI/default_analyses/`. Each analysis has its own module;
   none shares a file with framework code.
4. **Every base drawing lives in `GUI/views/`**: the image viewer, the line
   plot and the histogram plotter. A specialisation of one (an image with trap
   sites marked, say) lives in the applet or default analysis that uses it,
   and moves to `GUI/views/` only once a second panel needs it. Applets and
   default-analysis views never import from one another.
5. In every experiment and optimiser, `prepare`, `run_point` and `finish`
   (and `measure` and `fit` in optimisers) read as a plain sequence of physics
   steps. Helper methods (`saved_trap_sites`, say) may be more involved, as
   long as their code or comments make clear what they do.
6. Numerical work such as top-hat filtering, summing at sites, threshold fits
   and survival is done by functions imported from `analysis/`, never written
   inline in an experiment.

## Non-goals

- A declarative default-analysis API, where an analysis declares its controls
  and returns what to plot and a generic view draws it. That is a separate,
  larger project; views stay hand-written Qt.
- Merging the two cloud-fit outline drawings (the image monitor's and
  `CloudFitView`'s, both from `contour_ellipses`). Each stays a specialisation
  in its own panel; making them one shared drawing is a later design change.
- Running default analyses during an experiment. They are still calculated
  when a file is opened in the Results tab.
- Deleting old code (`analysis_old.py`, `crashit.py` and the like). These files
  move with their kind and are not deleted.

## Anti-goals: this fails even if it runs if…

- **Behaviour changes.** This is a pure restructure. Recorded result names,
  units, shapes and every number an experiment, optimiser or analysis produces
  stay the same. Files recorded before this change must still open in the
  Results tab with the same analyses.
- **Notebook imports break.** Notebooks import
  `pytweezer.analysis.analysis as an`, `cloud_fit`, `trap_sites` and
  `array_loading`, and those paths must keep working.
- **An experiment is made "readable" by burying it.** A one-line `run_point`
  that calls a 60-line `_do_everything()` is not the goal. The core methods
  should show the steps.
- **Hooks are added for tests.** No parameters, globals or fallbacks are added
  to production code so a test can reach it. Tests follow the new layout
  (CLAUDE.md).

## Constraints

- The Analysis Manager server reads its script folder from the Properties tree
  (`Analysis/analysisdir`). On the server PC that is an absolute Windows path to
  `pytweezer/analysis/`. Once the scripts move, that saved value would point at
  the wrong folder.
- Saved Analysis Manager filters store `script` as a filename relative to that
  folder. Filenames must not change, so saved filters keep working.
- Live scripts run as `python <path> <token>`, so each must keep its
  `if __name__ == "__main__":` block.
- Sphinx API docs are generated from the package tree, so the new packages need
  `__init__.py` files with a docstring.

## The approaches considered

**A. Keep the folders, separate the files.** Live scripts and shared maths stay
together in `analysis/`, with the default analyses' calculations in
`analysis/default_analyses/`. This is the least churn and needs no settings
change. But `analysis/` stays a mix of framework and editable code, so it
doesn't follow the `experiment/` vs `experiments/` pattern.

**B. Mirror experiment/experiments.** **Chosen.** `analysis/` keeps only
framework and maths, and the editable code moves to `analyses/`. The folder
name alone says whether something is framework or editable. The cost is a
one-off change to where the Analysis Manager looks.

**C. A declarative default-analysis API.** This would keep Qt out of editable
code entirely. It was deferred as too big for this change (see Non-goals).

## Target layout

```
pytweezer/
  analysis/                    FRAMEWORK + shared maths (no Qt, no process scripts)
    analysis_base.py           ImageAnalysis / DataAnalysis / run_analysis
    analysis.py                legacy notebook library (path unchanged)
    analysis_old.py, floating_point_arithmetics.py   unchanged
    array_loading.py           site photons/counts, top-hat, loading statistics
    trap_sites.py              site geometry and placement
    cloud_fit.py               fit_cloud, fit_frames… (CloudFitAnalysis process moves out)
  analyses/                    EDITED DAY TO DAY
    __init__.py
    live/                      Analysis Manager scripts, same filenames as today:
                               data_average, experiment_timer, find_tweezer_atoms,
                               gaussian_fit, image_stats, imageselector, projections,
                               random_number, roi_slice, rotate_image, shot_background,
                               substract, tweezer_counts, tweezer_loading,
                               tweezer_occupancy_counts, crashit,
                               cloud_fit (the CloudFitAnalysis process, importing
                               the fit from analysis/cloud_fit.py)
    default/                   default analyses' calculations, Qt-free, one module each
      tweezer_histogram.py     HistogramSettings, source, needed_results,
                               point_groups, tweezer_histogram
      cloud.py                 cloud_cameras … cloud_temperature, TofFit, GravityFit
  experiments/, optimisers/    unchanged location; readability pass (below)
  GUI/
    experiments/analyses.py    FRAMEWORK only: AnalysisView, ComputedAnalysisView,
                               AnalysisEntry, declared_analyses
    default_analyses/          the views; EDITED DAY TO DAY
      __init__.py              ANALYSES = tweezer_histogram.ANALYSES | cloud.ANALYSES
      tweezer_histogram.py     TweezerHistogramView + its ANALYSES entries
      cloud.py                 CloudFitView, CloudTemperatureView + entries
    views/                     base drawings, plus specialisations once shared
      image.py                 image viewer (phase 2, out of ImageDisplay)
      line_plot.py             line plot (phase 2, out of LivePlot's PlotPanel)
      histogram.py             histogram plotter (out of TweezerLoadingPlots)
      tweezer_loading.py       TweezerLoadingPlots: a specialisation, here
                               because the applet and a default analysis share it
    viewers/                   applets: properties, subscriptions, any
                               specialised drawing only they use
    applet.py                  applet FRAMEWORK
```

`analysis/default_analyses/` is removed. The Results tab imports `ANALYSES` from
`pytweezer.GUI.default_analyses`.

### Analysis Manager

- The live-script folder becomes a constant in
  `pytweezer/configuration/paths.py`, `<tweezerpath>/pytweezer/analyses/live/`.
  The server stops reading `analysisdir` from Properties.
- Remove `analysisdir` from `configuration/properties/properties.default.json`.
  A PC's saved `properties.json` may still contain it; it is simply ignored.
- The GUI keeps taking the folder from the server's response, as it does now.

## Views

A **base view** is a general-purpose widget that shows data it is handed:
the image viewer, the line plot and the histogram plotter. It knows nothing of
streams, Properties, devices, the archive or threads, and nothing specific to
one measurement. Base views live in `GUI/views/`.

A **specialisation** builds on a base view for one purpose. Examples are an
image with trap sites marked, a histogram with a two-Gaussian fit and a
threshold, or a cloud image with fit outlines. It lives beside the one applet
or default analysis that uses it. Once a second panel needs it, it moves to
`GUI/views/`, as `TweezerLoadingPlots` does now.

An **applet** (in `GUI/viewers/`) owns its properties and subscriptions and
hands each message to its views. A **default-analysis view** (in
`GUI/default_analyses/`) owns its controls and the compute wiring
(`ComputedAnalysisView`) and hands each result to its views. Controls that talk
to a device, like `camera_sidebar.py`, are not views and stay with their
applet.

The work comes in two phases. Each gets its own plan, and phase 1 ships before
phase 2 starts.

- **Phase 1**, with the analysis split:
  - Create `GUI/views/` with a histogram plotter pulled out of
    `TweezerLoadingPlots`'s histogram panels.
  - `TweezerLoadingPlots` moves there as a shared specialisation built on that
    plotter.
  - The cloud-fit and temperature drawings stay in
    `GUI/default_analyses/cloud.py` as specialisations.
- **Phase 2, the applets.** Today the drawing is fused into the applet classes
  by inheritance:
  - `CameraApplet`, `ImageProjectionDisplay` and `TweezerDisplay` subclass the
    `ImageDisplay` applet.
  - `StackedPlotter` subclasses `LivePlot`.

  Phase 2 pulls the base image viewer (image, colour bar, ROIs, overlays) out
  of `ImageDisplay` and the line plot out of `LivePlot`'s `PlotPanel`. Each
  applet then holds a view, base or specialised, rather than inheriting from
  another applet. A specialised view may subclass its base view.
  - `TweezerDisplay`'s site overlay and `ImageProjectionDisplay`'s projections
    become image-viewer specialisations in their own applets.
  - `slm_monitor`'s `ImagePanel` and `scalar_history` are rebuilt on the base
    views.

  This is a refactor, not a move. The camera applet is 850 lines, so applets
  must look and behave exactly as before. Check each one in the real GUI
  (`run-pytweezer` skill), not only in tests. Saved applet Properties keep
  their keys.

## Readable experiments: what changes

The rule (goal 5) applies to every module in `experiments/` and `optimisers/`.
These are the known cases:

- **`LoadTweezers`**
  - `prepare`: split the camera set-up block (stop live, binning, ROI, trigger,
    EM gain, start live) and the trap-site loading into named helpers, so
    `prepare` reads as "set up the HamCam, then load and record the trap
    sites".
  - `record_data`: the top-hat → photons → threshold → occupancy chain becomes
    one call to an `analysis/` function.
- **`TweezerOptimisation.score_batch`**: replace the
  `an.get_array_loading_statistics(..., binning=60, show_histogram=False,
  threshold_detection=True, verbose=False)` and
  `an.extract_survival_probability` calls with clean functions in
  `analysis/array_loading.py`. These wrap the same calls with the same
  arguments, so the numbers do not change. The TODO about loading using a
  per-batch threshold while survival uses the calibrated one stays, wherever
  that logic ends up.
- **`FindTrapSites.fit`**: same treatment for the background-subtracted average,
  the top-hat and the threshold fit.
- **`RbMotMasterRunner.run_point`**: move the frame-count and restart checks
  into a named helper.
- The other experiments and optimisers are already readable or close. The
  builder checks each one against the rule and leaves alone any that pass.

Where `LoadTweezers.saved_trap_sites` and the live `tweezer_loading.py` both
turn saved full-sensor sites into in-frame sites and check the summing windows
fit, the shared part may move to `analysis/trap_sites.py`. That is up to the
builder.

## Docs, skills, tests

- Update the skills that name these paths: `add-analysis-script`, `add-applet`,
  `add-experiment`, `add-optimiser` and `pytweezer-architecture`. They also
  change their rules:
  - live analyses go in `analyses/live/`;
  - a default analysis is a calculation in `analyses/default/` plus a view in
    `GUI/default_analyses/`;
  - base drawings go in `GUI/views/`, and specialisations stay with the panel
    that uses them until they are shared;
  - singular means framework, plural means editable.

  Approving this design is the go-ahead for those skill changes.
- Add one line to CLAUDE.md's "Architecture in brief" stating the
  singular/plural convention.
- Update `docs/architecture.md` where it names moved modules.
- Tests: update imports and paths only. The suite must pass, along with
  `ruff check .`, `ruff format --check .` and the Sphinx build.
- Commits go on the current branch, `claude/code-restructuring-clarity-c889qv`,
  as new commits on top of `9bf55f1`. History is not rewritten.

## Breaking changes to report to the user (notebooks)

These modules move: `pytweezer.analysis.<live script>` becomes
`pytweezer.analyses.live.<live script>` (for example `gaussian_fit`,
`roi_slice`), and `CloudFitAnalysis` moves out of `pytweezer.analysis.cloud_fit`.
The library modules named under Anti-goals do not move.
`analysis/default_analyses/` existed only on this branch.

## Left to the builder

- Names and signatures of the new `analysis/` helper functions, and of the
  helper methods in experiments, provided the core methods read as physics
  steps.
- Whether a plain `git mv` or a split commit works best for each move, keeping
  history readable.
- The order of the work within each phase and how it is split into commits.
- The base views' exact boundaries and APIs, as long as each is general-purpose
  and an applet holds its views rather than inheriting from another applet.
