# Framework / User-Code Split, Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate framework code from the code physicists edit day to day.
After this plan, `analysis/` holds only framework and maths, with no Qt. Live
and default analyses move to `analyses/`. Default-analysis views move to
`GUI/default_analyses/`, and base drawings start in `GUI/views/`. The tweezer
experiments' core methods read as physics steps.

**Architecture:** This is a pure restructure: no recorded result, number or
saved setting changes. The tweezer scoring that experiments do inline moves
behind one `TrapSites` object in `analysis/array_loading.py`. Modules move with
`git mv`, so history follows them. The Analysis Manager's script folder becomes
a path constant instead of a saved Property.

**Tech Stack:** Python 3, PyQt6 + pyqtgraph, numpy/scipy, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-23-framework-and-user-code-split-design.md`

## Global Constraints

- **Pure restructure.** Recorded result names, units, shapes and every number
  an experiment, optimiser or analysis produces stay the same. Files recorded
  before this change must still open in the Results tab with the same
  analyses.
- **Notebook imports keep working.** Notebooks import
  `pytweezer.analysis.analysis as an`, `cloud_fit`, `trap_sites` and
  `array_loading`; those paths must still work.
- **Live script filenames do not change.** Saved Analysis Manager filters
  store the bare filename.
- **No hooks for tests.** Don't add parameters, globals or fallbacks to
  production code so a test can reach it. Tests follow the new layout
  (CLAUDE.md).
- **Readability means visible steps.** `prepare`, `run_point`, `finish`,
  `measure` and `fit` read as a plain sequence of physics steps. Don't hide
  them behind a single catch-all helper.
- **Follow CLAUDE.md's commit procedure.** Run `uv run ruff check --fix` and
  `uv run ruff format` on the changed files, `git add` them whole, then
  commit. End each message with the session's `Co-Authored-By` and
  `Claude-Session` lines. Never use `--no-verify`.
- **Run Qt tests offscreen.** Set `QT_QPA_PLATFORM=offscreen` for any test or
  script that imports Qt.
- **Documentation style.** Use Google-style docstrings and UK English. Comment
  only what is surprising.

## Review Focus

1. **The server PC's saved `Analysis/analysisdir`** still points at the old
   folder. The server must ignore it and list `analyses/live/`. Task 3 pins
   this.
2. **Saved filters naming scripts by filename** (for example the default tree's
   `image_stats.py`) must still resolve. Task 3 checks every filter in
   `properties.default.json` against the new folder.
3. **Old measurement files** declare the analyses `tweezer_histogram`,
   `tweezer_photon_histogram`, `cloud_fit` and `cloud_temperature`; all four
   names must stay in `ANALYSES`. Task 5 pins this.
4. **`analysis/` must import no Qt**, even through a helper. Task 5 pins this
   for every module in the package.
5. **`FindTrapSites`' threshold.** It still fits its threshold from its own
   start value (`initial_threshold`), not from a saved threshold. Task 2 keeps
   the existing `tests/test_tweezer_array.py` FindTrapSites tests green and
   unedited.

---

### Task 1: `TrapSites`: tweezer scoring as one analysis object

**Files:**
- Modify: `pytweezer/analysis/array_loading.py`
- Test: `tests/test_array_loading.py` (create)

**Interfaces:**
- Produces, in `pytweezer.analysis.array_loading`:
  - `top_hat_image(image: np.ndarray, feature_size: int) -> np.ndarray`: one
    image through `an.morphological_tophat_high_pass` as `float`. Make the
    existing `top_hat(frames, feature_size)` use it.
  - `background_subtracted_mean(images, backgrounds) -> np.ndarray`: the mean
    of `images` minus the mean background, the value
    `an.tweezer_show_bg_subtracted(images, backgrounds, show=False)` returns.
    Call that function.
  - `@dataclass(frozen=True) class BatchLoading` with fields
    `photons: np.ndarray` (`(shots, sites)`), `loading: float`,
    `threshold: float` and `fidelity: float`.
  - `@dataclass(frozen=True) class TrapSites` with fields
    `positions: dict[int, tuple[int, int]]` (`{index: (row, col)}` in frame
    pixels), `window_size: int`, `feature_size: int` and `threshold: float`
    (photons). Its methods all take raw frames:
    - `counts(frames) -> np.ndarray`: `site_counts(frames, positions, window_size)`.
    - `photons(frames) -> np.ndarray`:
      `photons_at_sites(top_hat(frames, feature_size), positions, window_size)`.
    - `occupied(frames) -> np.ndarray`: `photons(frames) > threshold`, as
      bool `(frames, sites)`.
    - `fit_loading(frames) -> BatchLoading`. Calls
      `an.get_array_loading_statistics(top_hat(frames, feature_size), positions,
      threshold=threshold, window_size=window_size, binning=60,
      show_histogram=False, threshold_detection=True, verbose=False)`, which
      returns `(photons, loading, threshold, fidelity)`. Return
      `BatchLoading(np.asarray(photons), float(np.mean(loading)),
      float(threshold), float(fidelity))`.
    - `survival(firsts, seconds) -> float`. Calls
      `an.extract_survival_probability(top_hat(firsts, …), top_hat(seconds, …),
      positions, threshold=threshold, window_size=window_size)`.
- Keep every existing function in the module unchanged; other code uses them.

- [ ] **Step 1: Write the failing tests** in `tests/test_array_loading.py`. Build
  frames with bright 3×3 spots of known amplitude at chosen sites on a flat
  background. Use the photon-conversion monkeypatch that
  `tests/test_tweezer_array.py` uses for `ExpParameterManager`, so the
  conversion is known.
  - `test_occupied_marks_the_sites_with_an_atom_in_each_frame`: two frames, a
    spot at site 0 in both and at site 1 in the first only. Expect
    `occupied(frames)` to equal `[[True, True], [True, False]]`.
  - `test_counts_are_raw_pixel_sums_at_each_site`: `counts(frames)[0, 0]`
    equals the sum of the `window_size` box around site 0 in the unfiltered
    first frame.
  - `test_fit_loading_matches_the_notebook_statistics`: `fit_loading(frames)`
    equals, field by field, what `an.get_array_loading_statistics` returns
    with the arguments listed above.
- [ ] **Step 2: Check they fail.** Run
  `uv run pytest tests/test_array_loading.py -q`. Expect an ImportError
  (`TrapSites` is not defined).
- [ ] **Step 3: Implement** the Interfaces above in `array_loading.py`, and
  add a line to the module docstring naming `TrapSites`.
- [ ] **Step 4: Check they pass.** Run
  `uv run pytest tests/test_array_loading.py tests/test_tweezer_histogram.py tests/test_analysis.py -q`.
  Expect all to pass.
- [ ] **Step 5: Commit**: "TrapSites scores a camera's frames at the saved trap sites".

### Task 2: Tweezer experiments and optimisers read as physics steps

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py`,
  `pytweezer/optimisers/tweezer_optimisation.py`,
  `pytweezer/optimisers/find_trap_sites.py`,
  `pytweezer/experiments/rb_motmaster_runner.py`
- Test: `tests/test_tweezer_array.py`, `tests/test_optimisers.py`,
  `tests/test_motmaster_experiment.py`. These are behavioural tests and should
  pass unedited. Edit one only where it reaches into a renamed attribute, and
  say which in the commit message.

**Interfaces:**
- Consumes: `TrapSites`, `BatchLoading`, `top_hat_image` and
  `background_subtracted_mean` from Task 1.
- Produces: `LoadTweezers.trap_sites: TrapSites | None`. This replaces the
  separate `trap_sites` dict and the `trap_window_size`, `trap_feature_size`
  and `trap_threshold` attributes. `LoadTweezers.saved_trap_sites(saved) ->
  tuple[TrapSites | None, str]` keeps its contract: `(None, "")` means the
  experiment wants none, as `FindTrapSites` overrides it to say.

- [ ] **Step 1: Check the baseline.** Run
  `uv run pytest tests/test_tweezer_array.py tests/test_optimisers.py tests/test_motmaster_experiment.py -q`
  and expect all to pass. They are the safety net for this task.
- [ ] **Step 2: Rewrite `LoadTweezers`.**
  - `prepare` becomes `super().prepare()`, then `self.set_up_hamcam()`, then
    `self.load_trap_sites()`.
    - `set_up_hamcam()` is today's camera block, in the same order.
    - `load_trap_sites()` takes over the saved-parameter fetch, the
      `saved_trap_sites` call, the raise/log logic, setting
      `self.trap_sites` and `self.no_trap_sites_reason`, and recording
      `trap_sites`, `trap_window_size`, `trap_feature_size` and `threshold`.
      The record names, units and values stay unchanged.
  - `record_data` uses `self.trap_sites.occupied(frames)` and
    `self.trap_sites.counts(frames)`. It no longer imports `top_hat`,
    `photons_at_sites` or `site_counts`.
  - `saved_trap_sites` returns a `TrapSites`.
- [ ] **Step 3: Rewrite `TweezerOptimisation.score_batch`** to record the
  counts with `self.trap_sites.counts(...)`, then use `loading =
  self.trap_sites.fit_loading(firsts)` and `survival =
  self.trap_sites.survival(firsts, seconds)`. Record `first_occupied` as
  `loading.photons > self.trap_sites.threshold` and `second_occupied` as
  `self.trap_sites.occupied(seconds)`. Return the same three-key dict. Keep
  the comment on which threshold judges what, and its TODO, verbatim. Remove
  `import analysis as an`.
- [ ] **Step 4: Rewrite `FindTrapSites.fit`.**
  - Use `background_subtracted_mean`, then
    `fit_trap_sites(top_hat_image(average, self.feature_size), …)`.
  - Calibrate with `TrapSites(whole_pixels, self.window_size,
    self.feature_size, self.initial_threshold).fit_loading(firsts)`, where
    `whole_pixels` is `{index: (round(y), round(x))}` as today.
  - Record and propose exactly what it does today.
  - Remove `import analysis as an`.
- [ ] **Step 5: Rewrite `RbMotMasterRunner.run_point`** so that each camera's
  checked frames come from a helper,
  `read_run_frames(self, name: str, camera, mark: int) -> np.ndarray`, which
  holds today's restart, no-frames and frame-count checks and messages.
  `run_point` reads: start shots, run sequence, settle, record each camera's
  frames.
- [ ] **Step 6: Review the rest.** Read every other module in `experiments/`
  and `optimisers/` against the Global Constraint on readable core methods.
  Change only those that fail it, and list in the commit message the ones you
  checked and left alone.
- [ ] **Step 7: Run the tests.** Repeat Step 1's command plus
  `uv run pytest tests/ -q`. Expect everything to pass.
- [ ] **Step 8: Commit**: "Tweezer experiments score sites through TrapSites; core methods read as steps".

### Task 3: Live analyses move to `analyses/live/`

**Files:**
- Create: `pytweezer/analyses/__init__.py`, `pytweezer/analyses/live/__init__.py`.
  Each gets a docstring:
  - `analyses/__init__.py`: the analyses people edit; `pytweezer.analysis` is
    their framework and maths.
  - `analyses/live/__init__.py`: the scripts the Analysis Manager lists, each
    run as `python <file> <name>`.
- Move (`git mv pytweezer/analysis/<f> pytweezer/analyses/live/<f>`):
  `crashit, data_average, experiment_timer, find_tweezer_atoms, gaussian_fit,
  image_stats, imageselector, projections, random_number, roi_slice,
  rotate_image, shot_background, substract, tweezer_counts, tweezer_loading,
  tweezer_occupancy_counts` (`.py` each).
- Create: `pytweezer/analyses/live/cloud_fit.py`, holding `CloudFitAnalysis`,
  `_PASS_THROUGH` and the `__main__` block, cut from
  `pytweezer/analysis/cloud_fit.py`. Move the "Run as an analysis"
  Input/Output/Properties part of that module's docstring with it. The fit
  functions stay where they are.
- Modify: `pytweezer/configuration/paths.py`,
  `pytweezer/servers/analysis_manager.py:30-32`,
  `pytweezer/GUI/analysismanager.py` (the `analysisdir` default near line 340
  and the "runs a script from pytweezer/analysis" tooltip near line 190),
  `configuration/properties/properties.default.json` (delete
  `Analysis.analysisdir`).
- Test: update imports in `tests/test_analysis.py`, `tests/test_image_monitor.py`
  and `tests/test_tweezer_histogram.py`. Create `tests/test_analysis_manager_scripts.py`.

**Interfaces:**
- Produces: `pytweezer.configuration.paths.live_analyses_path: str`, equal to
  `tweezerpath + "/pytweezer/analyses/live/"` (lower case, like `icon_path`).
  The server sets `self.analysisdir = live_analyses_path` and never reads the
  Property.

- [ ] **Step 1: Write the failing tests** in `tests/test_analysis_manager_scripts.py`:
  - `test_the_manager_lists_the_live_analyses`:
    `AddFilterSheet._scan_scripts(live_analyses_path)` (a classmethod, no Qt
    instance needed) has `"tweezer_loading.py"` and `"cloud_fit.py"` under
    `"Image"` and `"gaussian_fit.py"` under `"Data"`.
  - `test_every_default_filter_names_a_live_script`: every `script` under
    `Analysis/Image` and `Analysis/Data` in `properties.default.json` exists in
    `live_analyses_path`.
  - `test_the_server_ignores_a_saved_script_folder`: construct the server
    class from `pytweezer/servers/analysis_manager.py` with its `Properties`
    patched to return an `analysisdir` of `"C:\\old\\pytweezer/analysis/"`.
    Patch only the socket bind, as `tests/test_servers.py` does for other
    servers; check that file first. Expect `analysisdir ==
    live_analyses_path`.
- [ ] **Step 2: Check they fail.** Run
  `uv run pytest tests/test_analysis_manager_scripts.py -q` and expect a
  failure.
- [ ] **Step 3: Do the moves and code changes** listed under Files. Update
  each moved script's imports only where it imported a sibling script. Run
  `grep -rn "pytweezer.analysis\.\(crashit\|data_average\|experiment_timer\|find_tweezer_atoms\|gaussian_fit\|image_stats\|imageselector\|projections\|random_number\|roi_slice\|rotate_image\|shot_background\|substract\|tweezer_counts\|tweezer_loading\|tweezer_occupancy_counts\)" pytweezer bin tests`
  until it returns nothing.
- [ ] **Step 4: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_analysis_manager_scripts.py tests/test_analysis.py tests/test_image_monitor.py tests/test_tweezer_histogram.py tests/test_cloud_fit.py -q`
  and expect a pass.
- [ ] **Step 5: Commit**: "Live analyses move to analyses/live; the Analysis Manager lists them from a fixed path".

### Task 4: `GUI/views/`: a histogram plotter, and `TweezerLoadingPlots` on it

**Files:**
- Create: `pytweezer/GUI/views/__init__.py`, with a docstring. Base drawings
  live here: general-purpose widgets that show data they are handed, knowing
  nothing of streams, Properties, devices, the archive or threads. A
  specialisation also lives here once a second panel uses it.
- Create: `pytweezer/GUI/views/histogram.py`
- Create: `pytweezer/GUI/views/tweezer_loading.py`, holding
  `TweezerLoadingPlots` and its colour constants, `_translucent` and
  `_gaussian`, moved out of
  `pytweezer/analysis/default_analyses/tweezer_histogram.py`.
- Modify: `pytweezer/GUI/viewers/tweezer_loading.py:22` and
  `pytweezer/analysis/default_analyses/tweezer_histogram.py`, both importing
  `TweezerLoadingPlots` from `GUI/views/tweezer_loading.py`.
- Test: `tests/test_histogram_plot.py` (create). Change only the import in
  `tests/test_tweezer_loading_plots.py`.

**Interfaces:**
- Produces, in `pytweezer.GUI.views.histogram`:
  `class HistogramPlot(pg.PlotItem)`, a probability-density histogram plot
  styled like every pytweezer plot.
  - `__init__(self, title: str, x_label: str = "", parent=None)` applies the
    title with `plot_title_style()`, `style_plot`, the grid at alpha 0.15, the
    left label "probability density" and no SI prefix on the left axis.
  - `add_histogram(self, colour, *, filled: bool = False, alpha: int = 255,
    width: float = 1, name: str | None = None) -> pg.PlotDataItem` adds a
    step curve, `stepMode="center"`. Filled curves use `fillLevel=0` and a
    translucent brush.
  - `draw(curve: pg.PlotDataItem, values: np.ndarray, bins: int,
    value_range: tuple[float, float]) -> None` is a static method:
    `np.histogram(..., density=True)`, then `curve.setData(edges, density)`.
  - `clear_histograms(self) -> None` removes only the curves `add_histogram`
    made.
- `TweezerLoadingPlots` keeps its public API: `show_result(head, photons)`,
  `show_message(text)`, `plot_items`, `bins`, `site_bins` and
  `show_site_labels`. Its `site_plot` and `total_plot` become `HistogramPlot`s,
  added to the `GraphicsLayoutWidget` with `win.addItem`. It draws with
  `add_histogram`/`draw`/`clear_histograms` and keeps the two-Gaussian fit,
  threshold line and loading map as its own specialisation.

- [ ] **Step 1: Write the failing tests** in `tests/test_histogram_plot.py`
  (offscreen):
  - `test_a_histogram_is_a_normalised_density`: `draw` of
    `[0.5, 1.5, 1.5, 3.5]` with 4 bins over `(0, 4)` gives a curve whose
    `getData()` y values are `[0.25, 0.5, 0, 0.25]` and whose x values are the
    edges `[0, 1, 2, 3, 4]`.
  - `test_clearing_removes_only_the_histograms`: after two `add_histogram`
    calls and one `plot()` line added directly, `clear_histograms()` leaves
    exactly one data item on the plot.
- [ ] **Step 2: Check they fail.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_histogram_plot.py -q`
  and expect an ImportError.
- [ ] **Step 3: Implement `HistogramPlot`**, then move and rebase
  `TweezerLoadingPlots` onto it as described under Interfaces.
- [ ] **Step 4: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_histogram_plot.py tests/test_tweezer_loading_plots.py tests/test_results_analyses.py -q`
  and expect a pass.
- [ ] **Step 5: Look at it.** Render a `TweezerLoadingPlots` offscreen with a
  synthetic result, before and after the change (`git stash` to get the
  before). Save both to the scratchpad at deviceScaleFactor 1 and compare
  them by eye. They should look the same.
- [ ] **Step 6: Commit**: "GUI/views: HistogramPlot, and TweezerLoadingPlots built on it".

### Task 5: Default analyses: Qt-free calculations in `analyses/default/`, views in `GUI/default_analyses/`

**Files:**
- Create: `pytweezer/analyses/default/__init__.py`. Adapt the docstring of
  today's `analysis/default_analyses/__init__.py`: a default analysis is
  offered for an experiment's files when the experiment names it in
  `Experiment.analyses`. This package holds the calculations; the views and the
  `ANALYSES` registry are in `pytweezer.GUI.default_analyses`.
- Create: `pytweezer/analyses/default/tweezer_histogram.py`.
  `git mv analysis/default_analyses/tweezer_histogram.py` here, then cut it to
  the calculation: `COUNTS`, `PHOTONS`, `FRAMES`, `RESULTS`,
  `HistogramSettings`, `source`, `needed_results`, `point_groups` and
  `tweezer_histogram`. `_filtered` is replaced by `array_loading.top_hat`.
- Create: `pytweezer/analyses/default/cloud.py`.
  `git mv analysis/default_analyses/cloud.py` here, then cut it to lines 1–383
  of today's file (`cloud_cameras` … `cloud_temperature`, `TofFit`,
  `GravityFit`). Rewrite its last docstring paragraph to point at the views'
  new module.
- Create: `pytweezer/GUI/default_analyses/__init__.py` (`ANALYSES =
  tweezer_histogram.ANALYSES | cloud.ANALYSES`, docstring as above),
  `GUI/default_analyses/tweezer_histogram.py` (`TweezerHistogramView` and its
  `ANALYSES`) and `GUI/default_analyses/cloud.py` (everything from today's
  `POINT_INDEX` onwards, including `CloudFitDrawing` and
  `TemperatureDrawing`, which stay here as specialisations).
- Delete: `pytweezer/analysis/default_analyses/`.
- Modify: `pytweezer/GUI/experiments/results.py:50` (import `ANALYSES` from
  `pytweezer.GUI.default_analyses`) and the `GUI/experiments/analyses.py`
  module docstring (point at the new homes).
- Test: change imports in `tests/test_results_analyses.py` and
  `tests/test_cloud_results.py` (calculation names come from
  `analyses.default.*`, views from `GUI.default_analyses.*`). Create
  `tests/test_layout.py`.

**Interfaces:**
- Consumes: `TweezerLoadingPlots` from `pytweezer.GUI.views.tweezer_loading`
  (Task 4).
- Produces: `pytweezer.GUI.default_analyses.ANALYSES: dict[str,
  AnalysisEntry]`, with exactly the keys `tweezer_histogram`,
  `tweezer_photon_histogram`, `cloud_fit` and `cloud_temperature`.

- [ ] **Step 1: Write the failing tests** in `tests/test_layout.py`:
  - `test_files_recorded_before_the_move_keep_their_analyses`: `set(ANALYSES)`
    equals those four names.
  - `test_framework_analysis_imports_no_qt`: for every module under
    `pytweezer.analysis` (found with `pkgutil.iter_modules`), run a subprocess
    with `MPLBACKEND=Agg` that imports it and prints
    `"PyQt6" in sys.modules`. Expect `False` every time.
  - `test_default_analysis_calculations_import_no_qt`: the same check for
    every module under `pytweezer.analyses.default`.
- [ ] **Step 2: Check they fail.** Run
  `uv run pytest tests/test_layout.py -q` and expect a failure
  (`analysis.default_analyses` imports Qt).
- [ ] **Step 3: Do the moves and splits** listed under Files.
- [ ] **Step 4: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_layout.py tests/test_results_analyses.py tests/test_cloud_results.py tests/test_tweezer_histogram.py tests/test_results_browser.py tests/test_experiment_gui.py -q`
  and expect a pass.
- [ ] **Step 5: Commit**: "Default analyses: calculations in analyses/default, views in GUI/default_analyses".

### Task 6: Skills, CLAUDE.md and docs follow the layout; full verification

**Files:**
- Modify: `.claude/skills/add-analysis-script/SKILL.md`,
  `.claude/skills/add-applet/SKILL.md`,
  `.claude/skills/add-experiment/SKILL.md`,
  `.claude/skills/add-optimiser/SKILL.md`,
  `.claude/skills/pytweezer-architecture/SKILL.md`, `CLAUDE.md` and
  `docs/architecture.md`.

The user approved these rule changes along with the design:

- [ ] **Step 1: Update `add-analysis-script`.**
  - A live analysis is a script in `pytweezer/analyses/live/`, built on
    `pytweezer.analysis.analysis_base`.
  - Shared maths goes in `pytweezer/analysis/`, with no Qt.
  - A default analysis is a Qt-free calculation in `pytweezer/analyses/default/`
    plus a view in `pytweezer/GUI/default_analyses/`, registered in that
    package's `ANALYSES`.
  - Replace the `default_analyses/tweezer_histogram.py` example paths.
- [ ] **Step 2: Update `add-applet`.**
  - Base drawings (image viewer, line plot, histogram plotter) live in
    `pytweezer/GUI/views/`.
  - A specialisation stays in the applet or default-analysis view that uses
    it, and moves to `GUI/views/` once a second panel needs it.
  - Applets and default-analysis views never import from one another.
  - `TweezerLoadingPlots` in `GUI/views/tweezer_loading.py` is the shared
    example.
  - The image viewer and line plot are added to `GUI/views/` in phase 2; until
    then, say the applets' drawing still sits in `image_monitor.py`/`live_plot.py`.
- [ ] **Step 3: Update the other three skills.** In `add-experiment`, the
  default-analysis section and the paths at lines ~229–240, ~451 and ~568
  change. In `add-optimiser`, `TweezerOptimisation`/`FindTrapSites` now score
  through `TrapSites`. In `pytweezer-architecture`, add the singular/plural
  convention, wherever it orients a reader in the package tree.
- [ ] **Step 4: Update `CLAUDE.md`.** Add one line to "Architecture in brief": a
  singular package is framework (`experiment/`, `analysis/`) and a plural one
  is what people edit day to day (`experiments/`, `optimisers/`, `analyses/`,
  `GUI/viewers/`, `GUI/default_analyses/`).
- [ ] **Step 5: Fix stale paths.** Run
  `grep -rn "analysis/default_analyses\|analysis\.default_analyses\|pytweezer/analysis/[a-z_]*\.py" .claude/skills docs/architecture.md CLAUDE.md`
  and fix every hit that names a moved module. Leave `docs/superpowers/` alone:
  those are dated records.
- [ ] **Step 6: Verify everything.**
  - `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q`: all pass.
  - `uv run ruff check .` and `uv run ruff format --check .`: clean.
  - `uv sync --group docs && cd docs && uv run make html`: builds without new
    warnings about missing modules.
- [ ] **Step 7: Commit**, staging only these docs and skills: "Skills and docs follow the framework/user-code layout".
- [ ] **Step 8: Push** with `git push -u origin claude/code-restructuring-clarity-c889qv`.
- [ ] **Step 9: Report** to the user the notebook-breaking moves:
  - `pytweezer.analysis.<live script>` → `pytweezer.analyses.live.<live script>`;
  - `CloudFitAnalysis` now lives in `pytweezer.analyses.live.cloud_fit`;
  - `LoadTweezers.trap_window_size`, `trap_feature_size` and `trap_threshold`
    are now fields of `LoadTweezers.trap_sites`.
