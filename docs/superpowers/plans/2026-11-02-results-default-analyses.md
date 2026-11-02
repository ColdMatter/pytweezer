# Default Analyses in the Results Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiments declare default analyses. The Results tab offers them in a dropdown next to the universal "Plot against" quick plot. The first one is a per-file tweezer loading histogram.

**Architecture:** An analysis has two halves. The compute half is numpy only, lives in `pytweezer/analysis/`, takes a `Measurement`, and is shared with the live `tweezer_loading.py` analysis. The view half is a Qt widget in `pytweezer/GUI/`, and its drawing is shared with the live `TweezerLoadingViewer`. When a file is created, its root attr `analyses` records the names its experiment declares. The Results tab reads only that attr, then looks each name up in a GUI-side registry.

**Tech Stack:** Python 3, numpy/scipy, h5py, PyQt6, pyqtgraph, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-results-default-analyses-design.md`

## Global Constraints

- Branch `results-default-analyses`, built on #44. The PR description must contain `Depends on #44`.
- Results of existing analysis do not change. Anything that computes sites, photons, threshold, fidelity, loading or survival must match `pytweezer/analysis/analysis.py`. That file is the same as on `rearrangement-optimisation-RnD`, which is the authority. A suspected error is reproduced and marked `# TODO: check <what looks wrong>`, never fixed.
- The Results tab never imports experiment modules. Experiment modules never import Qt or pyqtgraph.
- Analyses never write into the h5 file.
- Google-style docstrings, UK English, `get_logger` from `pytweezer/logging_utils.py`, ruff clean. Comment only what is surprising.
- Tests follow the `write-tests` skill. Set `QT_QPA_PLATFORM=offscreen` for widget tests. Run the suite with `uv run pytest tests/ -q`.
- Commit and push after each task.

## Review Focus

1. **A file whose measurement failed after a few points.** `n_done` is less than `n_points`. The histogram must use only the points that ran (`Measurement` already trims), and must not raise when a selected group has no shots. → Task 4 test `test_a_group_with_no_shots_gives_a_header_without_photons`.
2. **A frames file where the site finder can't find `sites` spots** (wrong number typed, empty array). The view must show "Found no N trap sites…" and not raise. → Task 4 test `test_frames_without_the_sites_give_a_header_without_photons`, Task 7 test `test_view_shows_the_compute_error_instead_of_raising`.
3. **Clicking quickly from one file to another while a histogram computes.** The result of the older file must never be drawn over the newer one. → Task 7 test `test_a_stale_result_is_discarded`.
4. **The Experiment Manager going away mid-load.** `ArchiveUnavailable` on the background thread must appear as a message in the view, not as a traceback. → Task 7 test `test_view_shows_the_compute_error_instead_of_raising`, which covers both errors.
5. **A file declaring an analysis this GUI doesn't know** (newer code on the manager's PC). The dropdown shows it disabled, and nothing raises. → Task 8 test `test_unknown_analysis_is_listed_disabled`.

---

### Task 1: Shared loading statistics

Move the numerics that both the live analysis and the per-file histogram need into one GUI-free module. Make them match `analysis.get_array_loading_statistics`.

**Files:**
- Create: `pytweezer/analysis/array_loading.py`
- Modify: `pytweezer/analysis/tweezer_loading.py` (`find_sites` moves out; `LoadingStatistics.summary` calls the new functions)
- Test: `tests/test_analysis.py` (the `# tweezer_loading` section; reuse `tweezer_frame`, `patterned_frame`, `SITE_CENTRES`, `LOADED_IN_TEN`)

**Interfaces:**
- Produces:
  - `find_sites(image: np.ndarray, n_sites: int, step: float) -> dict[int, tuple[int, int]] | None`. Moved verbatim from `tweezer_loading.py`, which re-imports it.
  - `photons_at_sites(frames: Iterable[np.ndarray], sites: dict[int, tuple[int, int]], window_size: int) -> np.ndarray`. Shape `(shots, sites)`, float64. Uses `an.sum_pixel_values` then `an.convert_counts_to_photons`, exactly as `get_array_loading_statistics(method="box")` does. Frames must already be top-hat filtered.
  - `loading_statistics(photons: np.ndarray, sites: dict[int, tuple[int, int]], fallback_threshold: float) -> dict`. Returns the live header without `shots`/`sites_from`/`rid`: `sites`, `threshold`, `fidelity`, `loading`, `loading_spread`, `site_loading`, `site_rows`, `site_cols`, `background`, `signal`.

- [ ] **Step 1: Write the failing equivalence tests**

```python
def test_loading_statistics_match_the_notebook_analysis(monkeypatch):
    from pytweezer.analysis import analysis as an
    from pytweezer.analysis.array_loading import (
        find_sites, loading_statistics, photons_at_sites,
    )

    frames = [
        an.morphological_tophat_high_pass(patterned_frame(s).astype(float), feature_size=10)
        for s in range(40)
    ]
    sites = find_sites(np.mean(frames, axis=0), len(SITE_CENTRES), 10)
    photons = photons_at_sites(frames, sites, window_size=5)
    head = loading_statistics(photons, sites, fallback_threshold=0.0)

    expected_photons, expected_loading, expected_threshold, expected_fidelity = (
        an.get_array_loading_statistics(
            frames, sites, threshold=0.0, window_size=5, binning=60,
            show_histogram=False, threshold_detection=True, verbose=False,
        )
    )
    assert photons == pytest.approx(expected_photons)
    assert head["threshold"] == pytest.approx(expected_threshold)
    assert head["fidelity"] == pytest.approx(expected_fidelity)
    assert head["site_loading"] == pytest.approx(list(expected_loading))


def test_loading_statistics_fall_back_like_the_notebook_analysis(monkeypatch):
    # Same as above, but with an.detect_loading_threshold monkeypatched to raise
    # ValueError and fallback_threshold set to the midpoint of the photons. Assert
    # threshold == fallback_threshold, background is None and signal is None, and
    # fidelity == pytest.approx(...) using the fourth return value of
    # get_array_loading_statistics with the same patch and threshold.
```

Write the second test in full, following the comment. The existing `tests/test_analysis.py` tweezer_loading tests stay and must keep passing.

- [ ] **Step 2: Run them and see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_analysis.py -q -k "notebook_analysis"`
Expected: FAIL with `ModuleNotFoundError: pytweezer.analysis.array_loading`.

- [ ] **Step 3: Implement `array_loading.py` and make `LoadingStatistics.summary` use it**

`loading_statistics` fits with `an.detect_loading_threshold`, catching `(ValueError, TypeError, StopIteration, RuntimeError)`, and computes fidelity as `get_array_loading_statistics` does. On a failed fit, use `fallback_threshold` and the same split-means fallback as `get_array_loading_statistics`. That fallback looks wrong: it adds the two error probabilities and calls the sum the fidelity. Reproduce it with `# TODO: check the fallback fidelity: this is the error probability, not 1 minus it (as in analysis.get_array_loading_statistics)`. A non-finite fidelity becomes `None`, as the live code already does. `LoadingStatistics.summary` keeps its signature and output keys, and becomes `photons_at_sites` + `loading_statistics` plus `shots`. The live analysis's fallback fidelity changes from `None` to the notebook value; that change is intended.

- [ ] **Step 4: Run the tweezer_loading tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_analysis.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/analysis/array_loading.py pytweezer/analysis/tweezer_loading.py tests/test_analysis.py
git commit -m "Share the tweezer loading statistics between the live and file analyses"
```

---

### Task 2: Experiments declare their analyses, and files record them

**Files:**
- Modify: `pytweezer/experiment/experiment.py` (class attribute `analyses`; add it to `schema()`)
- Modify: `pytweezer/experiment/runner.py:64-72` (`open_writer` header)
- Modify: `pytweezer/experiments/load_tweezers.py`, `pytweezer/experiments/tweezer_array.py` (`TweezerArrayMeasurement`)
- Test: `tests/test_experiment_runner.py`

**Interfaces:**
- Produces: `Experiment.analyses: ClassVar[tuple[str, ...]] = ()`. The file root attr `"analyses"` is a JSON list string, so `json.loads(measurement.attrs["analyses"])` gives a list of names. A file written before this change has no such attr. `schema()` gains `"analyses": list(cls.analyses)`. The name `"tweezer_histogram"` is used by Tasks 4 and 7.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_file_records_the_analyses_its_class_declares(tmp_path):
    class Declares(Experiment):
        analyses = ("tweezer_histogram",)
        def run_point(self): self.record("y", 1.0)

    class Inherits(Declares): ...

    for cls in (Declares, Inherits):
        m = run_local(cls, Scan(), path=tmp_path / f"{cls.__name__}.h5")
        assert json.loads(m.attrs["analyses"]) == ["tweezer_histogram"]


def test_a_class_declaring_nothing_records_an_empty_list():
    # run_local an Experiment subclass with no analyses; json.loads(attrs["analyses"]) == []
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_experiment_runner.py -q -k analyses`
Expected: FAIL with `KeyError: 'analyses'`.

- [ ] **Step 3: Implement**

`open_writer` adds `"analyses": json.dumps(list(experiment_cls.analyses))` to the header before `**header`. `LoadTweezers` and `TweezerArrayMeasurement` declare `analyses = ("tweezer_histogram",)`.

- [ ] **Step 4: Run the experiment tests**

Run: `uv run pytest tests/test_experiment_runner.py tests/test_experiment_storage.py tests/test_tweezer_array.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/experiment.py pytweezer/experiment/runner.py pytweezer/experiments/load_tweezers.py pytweezer/experiments/tweezer_array.py tests/test_experiment_runner.py
git commit -m "Record each experiment's default analyses in its files"
```

---

### Task 3: Tweezer measurements record their per-site photons

**Files:**
- Modify: `pytweezer/experiments/tweezer_array.py` (`score_batch`)
- Test: `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: `photons_at_sites` (Task 1).
- Produces: every point of `TweezerImaging`, `MolassesRecapture` and their optimisers records the results `first_photons` and `second_photons`, both float `(shots, sites)`. `trap_sites` (constant, `(sites, 2)` int) and `threshold` (constant) are already recorded.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("cls", [TweezerImaging, MolassesRecapture])
def test_each_point_records_every_shots_photons_at_every_site(rig, cls):
    measurement = run_local(cls, **FAST)
    shots, sites = FAST["shots"], len(SITES)
    assert measurement.results["first_photons"].shape == (1, shots, sites)
    assert measurement.results["second_photons"].shape == (1, shots, sites)
    loaded = (
        measurement.results["first_photons"][0] > measurement.constants["threshold"]
    )
    assert loaded.mean() == pytest.approx(measurement.results["loading"][0], abs=0.05)
```

Adjust `FAST`/`SITES` to the names the file actually uses. Also run one optimiser (`OptimiseTweezerImaging` with its fast settings, as the file's existing optimiser tests do) and assert `first_photons` has one row per point.

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_tweezer_array.py -q -k photons`
Expected: FAIL with `KeyError: 'first_photons'`.

- [ ] **Step 3: Implement**

In `score_batch`, keep the first return value of `get_array_loading_statistics` as the first-image photons (it is the same box-sum photon array). Take the second-image photons from `photons_at_sites(seconds, self.trap_sites, self.window_size)`, then `self.record` both. `score_batch` runs inside `run_point` for the measurements and the optimisers alike, so the results land per point, and `measure()`'s return dict is unchanged. Replace the comment "occupancy is always judged against the calibration's threshold" with a correct one, plus `# TODO: check loading is judged against a per-batch refit while survival uses the calibration threshold (as on rearrangement-optimisation-RnD)`.

- [ ] **Step 4: Run the tweezer tests**

Run: `uv run pytest tests/test_tweezer_array.py tests/test_optimisers.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiments/tweezer_array.py tests/test_tweezer_array.py
git commit -m "Record each tweezer point's photons at every site"
```

---

### Task 4: Tweezer histogram of a file

**Files:**
- Create: `pytweezer/analysis/tweezer_histogram.py`
- Test: `tests/test_tweezer_histogram.py`

**Interfaces:**
- Consumes: `find_sites`, `photons_at_sites`, `loading_statistics` (Task 1). Results `first_image`/`second_image` (`LoadTweezers`), or `first_photons`/`second_photons` plus constants `trap_sites`/`threshold` (Task 3).
- Produces:

```python
@dataclass(frozen=True)
class HistogramSettings:
    image: int = 0             # 0 first image, 1 second
    group: int | None = None   # index into point_groups(); None = all points
    sites: int = 64            # frames files only, like the three below
    feature_size: int = 10
    window_size: int = 5
    detection_step: float = 100.0
    fallback_threshold: float = 100.0   # used when the file has no threshold constant

def source(measurement: Measurement) -> str | None   # "photons" | "frames" | None
def needed_results(measurement: Measurement) -> list[str]
def point_groups(measurement: Measurement) -> list[tuple[str, np.ndarray]]
def tweezer_histogram(measurement: Measurement, settings: HistogramSettings) -> tuple[dict, np.ndarray | None]
```

  - `source` and `needed_results` decide from `measurement.result_shapes` alone, which is filled even when no results are loaded. "photons" wins when both kinds are present.
  - `point_groups` returns one `(label, bool mask over the run points)` per distinct combination of the scan's axis values, in first-seen order. For example the label is `"load_time = 0.5 ms"`, built with `summary.display_value`. With no scan axes it returns one group per point, labelled `"point 3"`.
  - `tweezer_histogram` returns the live analysis's header (`shots`, `sites`, then the `loading_statistics` keys) plus `"source"`, with photons `(shots, sites)`. If the sites can't be found or the group has no shots, it returns `(head, None)` with `head["error"]` set to a sentence. Missing results raise `KeyError`.
  - For frames, the sites are found once in the mean of every filtered **first** image in the file, and only the shots of the selected image and group are summed. For photons, the sites are the `trap_sites` constant, and `fallback_threshold` is the file's `threshold` constant.

- [ ] **Step 1: Write the failing tests**

Build files with `run_local`:
- **Frames:** a small `Experiment` subclass in the test that records `first_image`/`second_image` from `tests.test_analysis.patterned_frame`. Move `tweezer_frame`/`patterned_frame`/`SITE_CENTRES`/`LOADED_IN_TEN` to `tests/helpers.py` so both test files use them.
- **Photons:** the Task 3 measurement via the `rig` fixture. Move `FakeSequencer`/`FakeHamCam`/`FakeRig` and the fixture to `tests/conftest.py`, or import them from `tests.test_tweezer_array`.

```python
def test_frames_file_gives_the_live_analysis_result():
    m = frames_file(shots=40, repetitions=1)       # one point per shot, no axes
    head, photons = tweezer_histogram(m, HistogramSettings(sites=len(SITE_CENTRES), detection_step=10))
    expected_head, expected_photons = loading_window(patterned_frame(s) for s in range(40))
    assert photons == pytest.approx(expected_photons)
    assert head["threshold"] == pytest.approx(expected_head["threshold"])
    assert head["source"] == "frames"

def test_photons_file_uses_the_recorded_sites_and_photons(rig): ...
    # head["site_rows"] / head["site_cols"] equal the trap_sites constant; photons
    # equal the stacked first_photons; image=1 gives second_photons.

def test_a_scan_value_pools_its_repetitions():
    # frames file with ListAxis over a dummy argument with 2 values and 3 repetitions:
    # point_groups gives 2 groups; selecting group 0 gives shots == 3.

def test_without_scan_axes_each_point_is_a_group(): ...
def test_a_group_with_no_shots_gives_a_header_without_photons(): ...
    # a mask with no True (e.g. a file with n_done < n_points whose group only ran later):
    # photons is None and "error" in head.
def test_frames_without_the_sites_give_a_header_without_photons(): ...
    # sites=len(SITE_CENTRES) + 5: photons is None and "error" in head.
def test_needed_results_names_only_what_the_source_uses(): ...
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_tweezer_histogram.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pytweezer/analysis/tweezer_histogram.py`**

Filter frames with `an.morphological_tophat_high_pass(frame.astype(np.float64), feature_size=...)`, as `LoadingStatistics.add` does.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tweezer_histogram.py tests/test_analysis.py tests/test_tweezer_array.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/analysis/tweezer_histogram.py tests/test_tweezer_histogram.py tests/helpers.py tests/conftest.py tests/test_analysis.py tests/test_tweezer_array.py
git commit -m "Histogram a tweezer measurement file like the live loading analysis"
```

---

### Task 5: Reusable tweezer loading plots

Pull the drawing out of `TweezerLoadingViewer` so the Results tab can embed it.

**Files:**
- Create: `pytweezer/GUI/tweezer_loading_plots.py`
- Modify: `pytweezer/GUI/viewers/tweezer_loading.py` (the applet becomes subscribe, poll, `plots.show`)
- Test: `tests/test_tweezer_loading_plots.py`

**Interfaces:**
- Produces:

```python
class TweezerLoadingPlots(QtWidgets.QWidget):
    def __init__(self, parent=None, *, bins=60, site_bins=40, show_site_labels=False): ...
    bins: int; site_bins: int; show_site_labels: bool   # plain attributes, redraw on next show()
    def show_result(self, head: dict, photons: np.ndarray | None) -> None
    def show_message(self, text: str) -> None           # clears the plots, puts text in the readout
    readout: QtWidgets.QLabel
    plot_items: tuple[pg.PlotItem, ...]                 # so the applet can add its menu actions
```

  - `show_result` draws as `_draw` does today, and writes the readout as `_show_status(head)` does when `site_loading` is present. Without photons it clears the plots and shows `head.get("error")`, or the "Looking for N trap sites…" text.
  - Name it `show_result`, not `show`, which `QWidget.show` already uses.
  - The stream-specific text ("Subscribe to…", "Waiting for shots on…") stays in the applet, via `show_message`.

- [ ] **Step 1: Write the failing tests**

```python
def test_plots_draw_a_result_and_report_it(qapp):
    head, photons = loading_window(patterned_frame(s) for s in range(40))
    plots = TweezerLoadingPlots()
    plots.show_result(head, photons)
    assert "loading" in plots.readout.text() and "fidelity" in plots.readout.text()
    assert len(plots.site_map.data) == len(SITE_CENTRES)


def test_plots_show_an_error_without_photons(qapp):
    plots = TweezerLoadingPlots()
    plots.show_result(
        {"shots": 3, "sites": 64, "error": "Found no 64 trap sites"}, None
    )
    assert "Found no 64 trap sites" in plots.readout.text()
```

- [ ] **Step 2: Run them and see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_tweezer_loading_plots.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement and slim the applet**

The applet keeps its `PropertyAttribute`s and copies them onto the widget in `update_subscriptions`. Its behaviour is unchanged.

- [ ] **Step 4: Run the tests, then check the applet still launches**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_tweezer_loading_plots.py tests/test_applet_launcher.py tests/test_analysis_gui.py -q`
Expected: all pass. Then build the applet offscreen once, so construction errors surface: `QT_QPA_PLATFORM=offscreen uv run python -c "from pytweezer.GUI.viewers import tweezer_loading"`.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/GUI/tweezer_loading_plots.py pytweezer/GUI/viewers/tweezer_loading.py tests/test_tweezer_loading_plots.py
git commit -m "Share the tweezer loading plots between the applet and other panels"
```

---

### Task 6: A second archive client for background loads

**Files:**
- Modify: `pytweezer/experiment/archive.py`
- Test: `tests/test_results_archive.py`

**Interfaces:**
- Produces: `ResultsArchive.clone() -> ResultsArchive` (same root) and `RemoteArchive.clone() -> RemoteArchive` (same host, port and timeout, with its own lock and connection).

- [ ] **Step 1: Write the failing test**

```python
def test_a_clone_reads_the_same_archive_over_its_own_connection(tmp_path):
    # ResultsArchive: clone().days() == archive.days(); clone is not archive.
    # RemoteArchive (use the module's existing served-archive fixture): the clone's
    # _lock is not the original's, and clone().runs(day) == archive.runs(day).
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_results_archive.py -q -k clone`
Expected: FAIL with `AttributeError: ... has no attribute 'clone'`.

- [ ] **Step 3: Implement both `clone` methods**

Document in the docstring why a clone exists: a call holds the lock for the whole RPC.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_results_archive.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/archive.py tests/test_results_archive.py
git commit -m "Let a results archive open a second connection for slow loads"
```

---

### Task 7: Analysis registry and the tweezer histogram view

**Files:**
- Create: `pytweezer/GUI/experiments/analyses.py`
- Test: `tests/test_results_analyses.py`

**Interfaces:**
- Consumes: the `tweezer_histogram` module (Task 4), `TweezerLoadingPlots` (Task 5), `archive.clone()` (Task 6).
- Produces:

```python
class AnalysisView(QtWidgets.QWidget):
    """One analysis of the selected measurement, computed off the GUI thread."""
    def set_measurement(self, archive, path: str, measurement: Measurement) -> None
    def clear(self) -> None
    message: QtWidgets.QLabel

@dataclass(frozen=True)
class AnalysisEntry:
    label: str                                          # shown in the dropdown
    view: Callable[[], AnalysisView]

ANALYSES: dict[str, AnalysisEntry] = {
    "tweezer_histogram": AnalysisEntry("Tweezer loading histogram", TweezerHistogramView),
}

def declared_analyses(measurement: Measurement) -> list[str]   # json.loads(attrs.get("analyses", "[]"))
```

  - **`TweezerHistogramView(AnalysisView)`:**
    - **Controls row:** "Image" (`first`/`second`); "Shots from" ("all points" plus each `point_groups` label); and "Sites", "Feature size", "Window" and "Detection step" spin boxes, which are visible only when `source(...) == "frames"`.
    - **Body:** a `TweezerLoadingPlots`.
    - **Compute:** any change computes in a `threading.Thread`. The thread loads with `archive.clone()` (cloned once per archive object and kept) using `needed_results`, then runs `tweezer_histogram` and emits `pyqtSignal(int, object)` carrying `(generation, (head, photons) | Exception)`.
    - **Stale results:** a result whose generation isn't the newest is dropped.
    - **While waiting:** the message shows "Analysing…".
    - **Exceptions:** `ArchiveUnavailable` → "Experiment Manager not reachable."; any other → `f"Could not analyse: {error}"`, logged with `logger.exception`.
    - **Loaded data:** the loaded measurement is cached per path, so changing a control doesn't reload.
  - Wording and layout follow the `pytweezer-gui-design` skill. Read it before building the controls row.

- [ ] **Step 1: Write the failing tests**

Use the frames-file builder from Task 4 and `tests.helpers.until(..., qapp=qapp)`.

```python
def test_view_draws_the_histogram_of_a_frames_file(qapp, tmp_path):
    ...
    # view.set_measurement(ResultsArchive(tmp_path), rel, archive.load(rel, results=[]))
    # until view.plots.readout mentions "loading"; the frames spin boxes are visible.


def test_changing_shots_from_recomputes_on_that_group(qapp, tmp_path): ...


def test_a_stale_result_is_discarded(qapp, tmp_path, monkeypatch):
    ...
    # monkeypatch tweezer_histogram with one that blocks on an Event for the first
    # file. Select file A, then file B, release A: the readout reflects B only.


def test_view_shows_the_compute_error_instead_of_raising(qapp, tmp_path, monkeypatch):
    ...
    # parametrise over ArchiveUnavailable (patch the clone's load) and the
    # "Found no N trap sites" error header (sites far too many): the message or readout
    # shows the sentence and nothing propagates.


def test_declared_analyses_of_an_old_file_is_empty():
    ...
    # a Measurement whose attrs lack "analyses" gives [].
```

- [ ] **Step 2: Run them and see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_results_analyses.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pytweezer/GUI/experiments/analyses.py`**

- [ ] **Step 4: Run the tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_results_analyses.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/GUI/experiments/analyses.py tests/test_results_analyses.py
git commit -m "Add the tweezer histogram as a Results analysis view"
```

---

### Task 8: The Analysis dropdown in the Results tab

**Files:**
- Modify: `pytweezer/GUI/experiments/results.py`
- Modify: `.claude/skills/add-experiment/SKILL.md` (the Running and Files sections: `analyses = (...)`, the `analyses` root attr, the dropdown)
- Modify: `.claude/skills/add-analysis-script/SKILL.md` (one paragraph: how to add a Results analysis, i.e. a compute half in `pytweezer/analysis/`, an `AnalysisView` and an `ANALYSES` entry)
- Test: `tests/test_results_browser.py`

**Interfaces:**
- Consumes: `ANALYSES`, `AnalysisView`, `declared_analyses` (Task 7).
- Produces:
  - `ResultsPanel.analysis_choice: QComboBox`. Item 0 is always `"Plot against"`, with data `None`; each further item is a declared name, with data = that name.
  - `ResultsPanel.analysis_stack: QStackedWidget`. Page 0 is the existing controls and plot.

Behaviour:
- **Layout:** the combo goes at the left of the plot region's header row. The region's title becomes "Analysis".
- **Filling the dropdown:** on `load()`, refill the combo from `declared_analyses(measurement)`. A name missing from `ANALYSES` is added disabled, with the tooltip `"Not known to this GUI; update pytweezer"`.
- **Running measurements:** while the measurement is `running`, every item except "Plot against" is disabled, with the tooltip "Available when the measurement finishes". The existing reload on finish re-enables them.
- **Remembered choice:** the chosen name is kept per `attrs["class_name"]` in a dict for the session. On `load()` it is re-selected if offered and enabled, else "Plot against".
- **Views:** selecting an analysis creates its view once (cached by name), shows it in the stack and calls `set_measurement(self.archive, self.path, self.measurement)`. Selecting another file while an analysis is shown calls that again.
- **Plot against:** the plotting code doesn't change. Live plotting and `replot()` only act while page 0 is shown.

- [ ] **Step 1: Write the failing tests**

```python
def test_dropdown_offers_plot_against_then_the_files_analyses(qapp, tmp_path):
    ...
    # a file of a class with analyses = ("tweezer_histogram",): items are
    # ["Plot against", "Tweezer loading histogram"]; a Sweep file has only "Plot against".


def test_unknown_analysis_is_listed_disabled(qapp, tmp_path):
    ...
    # rewrite a file's attrs["analyses"] = '["nonexistent"]': item 1 is disabled, no raise.


def test_analyses_are_disabled_while_running(qapp, tmp_path): ...


def test_choice_is_kept_for_the_next_file_of_the_same_class(qapp, tmp_path): ...


def test_choosing_an_analysis_shows_its_view(qapp, tmp_path): ...
```

The existing tests in `tests/test_results_browser.py` must pass unchanged.

- [ ] **Step 2: Run them and see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_results_browser.py -q`
Expected: the new tests FAIL with `AttributeError: 'ResultsPanel' object has no attribute 'analysis_choice'`.

- [ ] **Step 3: Implement it in `results.py`, and update both skills**

- [ ] **Step 4: Run the full suite and ruff**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, ruff clean.

- [ ] **Step 5: See it in the real window**

Use the `run-pytweezer` skill to launch `pytweezer-server` simulated on this PC. Do not build the real GUI on PH-BEAST without checking that skill (memory: building the GUI there spawns device servers). Open a `LoadTweezers` file in Results, pick "Tweezer loading histogram", and screenshot it.

- [ ] **Step 6: Commit, push, open the PR**

```bash
git add pytweezer/GUI/experiments/results.py tests/test_results_browser.py .claude/skills/add-experiment/SKILL.md .claude/skills/add-analysis-script/SKILL.md
git commit -m "Offer each measurement's default analyses in the Results tab"
git push -u origin results-default-analyses
gh pr create --base main --title "Default analyses in the Results tab" --body "...Depends on #44..."
```
