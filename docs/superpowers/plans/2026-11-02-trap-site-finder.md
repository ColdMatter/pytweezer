# Trap-site finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the tweezer trap sites once, in a dedicated optimiser fitting the rig's geometry to the camera, store them as an array saved parameter on acceptance, and have every tweezer script and a new tweezer viewer use them.

**Architecture:** Saved parameters gain number arrays (nested lists in JSON, `numpy` arrays to callers). A pure fitting module maps hologram µm to sensor pixels (fixed x mirror + similarity transform). A new `Fit` optimiser kind runs one fit and proposes its keys; `FindTrapSites` is the first. `TweezerArrayMeasurement`, the live loading analysis and the new viewer read the accepted `trap_*` keys, converting full-sensor pixels to their frame's ROI with an origin cameras now put in every frame header.

**Tech Stack:** Python 3.11, numpy, scipy (`ndimage`, `optimize`), pydantic, PyQt6 + pyqtgraph, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-trap-site-finder-design.md`

## Global Constraints

- UK English; Google-style docstrings describing current behaviour only.
- Loggers via `get_logger` from `pytweezer/logging_utils.py`.
- Comments only for the surprising; match surrounding style. Ruff clean.
- Run `uv run pytest tests/ -q` before each commit; commit and push per task on branch `trap-site-finder`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Saved keys and units: `trap_sites` (px), `trap_threshold` (photons), `trap_window_size` (px), `trap_feature_size` (px), `trap_geometry` (µm); group `"Tweezer array"`.
- `trap_sites` rows are `(y, x)` in **full-sensor pixels**; row *i* = geometry site *i*.
- Geometry to pixels: `x_img = -s·x' + x0`, `y_img = s·y' + y0`, `(x', y')` = geometry rotated by θ; start `s = hamcam_px_per_um` (`analysis.py`), θ = −2.5°.
- Default geometry: 8×8, 5 µm, centred on 0, ordered row-major from the image's top-left (so `x_um` *decreases* along a row, because of the mirror).
- Fitted scale outside ±20 % of `hamcam_px_per_um` fails the fit.
- Geometry match tolerance: 0.1 µm per coordinate.

## Review Focus

- A saved site whose summing window falls outside the measurement's ROI: `prepare()` must fail naming the site, not slice a short window (test in Task 5).
- The rig showing a different array from the one the sites were fitted to: measurements refuse (Task 5); the live analysis reports `sites_match: False` (Task 8).
- A proposal read back from h5 as an `ndarray`: shown as "N × 2 array" and accepted as a nested list, not crashing on `.item()` (Task 6).
- An array containing NaN, or a reshaped array for an existing key: refused, nothing stored (Task 1).
- A frame header with no ROI origin (older camera, notebook frame): treated as origin (0, 0) (Tasks 8, 9).

---

### Task 1: Array saved parameters

**Files:**
- Modify: `pytweezer/experiment/saved_parameters.py` (`Value`, `_checked`, `SavedParameterBook.set`, module docstring)
- Modify: `pytweezer/experiment/client.py` (`ExperimentManagerClient.set_saved_parameters`, `get_saved_parameters`, `get_saved_parameter`, module `set_saved_parameters`)
- Modify: `pytweezer/servers/experiment_manager.py` (`_cmd_set_saved_parameters` passes `units`)
- Test: `tests/test_saved_parameters.py`

**Interfaces:**
- Produces: `Value = float | int | bool | str | list` (arrays are nested lists of floats in the model, JSON and log).
- Produces: `SavedParameterBook.set(values, *, reason, who, create=False, group="", unit="", units: dict[str, str] | None = None, rid=None)` — `units[key]` overrides `unit` for a key created here. Same keyword on `ExperimentManagerClient.set_saved_parameters` and module `set_saved_parameters`.
- Produces: `get_saved_parameters()` / `get_saved_parameter()` return array values as `np.ndarray` (float); `set_saved_parameters` accepts `np.ndarray` and numpy scalars (`.tolist()`).

- [ ] **Step 1: Write failing tests** in `tests/test_saved_parameters.py`:
  - `test_an_array_is_stored_as_nested_floats_and_reloads(tmp_path)`: create `sites=np.array([[1, 2], [3, 4]])` via the book, save with `SavedParameterStore`, reload; `value == [[1.0, 2.0], [3.0, 4.0]]`.
  - `test_an_array_key_takes_any_number_of_rows_but_keeps_its_columns`: existing 2×2; setting a 3×2 is stored; setting 2×3, a 1-D array, or a scalar raises `SavedParameterError`; a scalar key refuses a list.
  - `test_an_array_with_a_non_finite_element_is_refused`: `[[1.0, nan]]` refused on create and on update; book unchanged.
  - `test_units_name_each_created_keys_unit`: `set({"a": 1.0, "b": [[1.0]]}, create=True, unit="x", units={"b": "px"})` → units `x`, `px`.
  - `test_numpy_arrays_round_trip_through_the_manager(make_manager)` (copy `test_numpy_scalars_are_stored_as_python_numbers`'s setup): set an `np.ndarray` through the client; `get_saved_parameter` returns an equal `np.ndarray`.
- [ ] **Step 2:** `uv run pytest tests/test_saved_parameters.py -q` — new tests FAIL.
- [ ] **Step 3: Implement.** In `_checked`: a `list`/`np.ndarray` value goes through `np.asarray(value, dtype=float)`; refuse `ndim == 0`, non-finite elements, and (when `existing` is a list) a different `ndim` or `shape[1:]`; return `.tolist()`. Lists vs scalars are distinct types for the "key keeps its type" rule. Client converts `np.ndarray`/`np.generic` with `.tolist()`; getters wrap list values in `np.asarray(..., dtype=float)`. Manager forwards `units`.
- [ ] **Step 4:** `uv run pytest tests/test_saved_parameters.py tests/test_experiment_manager.py -q` — PASS.
- [ ] **Step 5: Commit** "Let saved parameters hold number arrays".

### Task 2: Arrays in the Saved Parameters tab

**Files:**
- Modify: `pytweezer/GUI/saved_parameters.py` (`display_value`, `_TYPE_WORDS`, `_current_changed`/`_show_sheet`)
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: Task 1's list values.
- Produces: `display_value([[...], ...]) -> "64 × 2 array"` (shape joined with " × "); used by the Optimisers tab in Task 6.

- [ ] **Step 1: Failing tests:** `test_an_array_value_shows_its_shape` (`display_value([[1.0, 2.0]] * 64) == "64 × 2 array"`, `display_value([1.0, 2.0]) == "2 array"`), and `test_an_array_parameter_is_shown_but_not_editable(qapp)` using the existing saved-parameters panel fixture: selecting an array key leaves `value_edit`, `apply_button` disabled and the status reads "Arrays are set by accepting an optimiser's proposal".
- [ ] **Step 2:** run them — FAIL.
- [ ] **Step 3: Implement**; history rows use `display_value` already.
- [ ] **Step 4:** `uv run pytest tests/test_experiment_gui.py -q` — PASS.
- [ ] **Step 5: Commit** "Show array saved parameters read-only".

### Task 3: Fitting the array geometry to an image

**Files:**
- Create: `pytweezer/analysis/trap_sites.py`
- Test: `tests/test_trap_sites.py`

**Interfaces:**
- Produces:
  - `default_geometry(rows: int = 8, columns: int = 8, spacing_um: float = 5.0) -> np.ndarray` — `(N, 2)` of `(x_um, y_um)`, ordered per Global Constraints.
  - `geometry_to_pixels(geometry: np.ndarray, scale: float, angle_deg: float, x0: float, y0: float) -> np.ndarray` — `(N, 2)` `(y, x)`.
  - `@dataclass(frozen=True) class TrapSiteFit: sites: np.ndarray; scale: float; angle_deg: float; x0: float; y0: float`.
  - `fit_trap_sites(image: np.ndarray, geometry: np.ndarray, *, px_per_um: float = hamcam_px_per_um, angle_deg: float = -2.5, window_size: int = 5) -> TrapSiteFit` — `sites` in the image's pixels; raises `ValueError` if the fit leaves the image or the scale is beyond ±20 %.
  - `to_roi(sites: np.ndarray, roi_y0: int, roi_x0: int) -> np.ndarray` and `same_geometry(saved: np.ndarray, x_um, y_um, tol_um: float = 0.1) -> bool`.

- [ ] **Step 1: Failing tests** (synthetic image: Gaussian spots, σ≈1.5 px, on Poisson background, built from `geometry_to_pixels` with a known transform):
  - `test_the_default_grid_is_row_major_from_the_top_left_of_the_image`: `geometry_to_pixels(default_geometry(), 2.0, 0, 50, 50)` — row 0 is the smallest `(y, x)`, row 1 is one spacing to its right, row 8 one spacing below.
  - `test_the_fit_recovers_a_known_transform_with_dark_sites`: truth `scale=2.0` (vs start 2.07), `angle=-1.0°`, centre `(60.3, 70.7)` in a 128×160 image, 20 sites dark; every fitted site within 0.5 px of truth, scale within 1 %, angle within 0.3°.
  - `test_a_rig_geometry_keeps_its_own_order`: a shuffled 3×3 geometry; fitted row *i* lands on the spot of geometry row *i*.
  - `test_a_blank_image_or_a_wrong_scale_is_refused`: pure-noise image, and spots drawn at scale 3.0, each raise `ValueError`.
  - `test_to_roi_and_same_geometry`: plain value checks, including a 0.2 µm move and a different count → `False`.
- [ ] **Step 2:** `uv run pytest tests/test_trap_sites.py -q` — FAIL (module missing).
- [ ] **Step 3: Implement.** Filter with `an.morphological_tophat_high_pass` is the caller's job; here smooth with `scipy.ndimage.gaussian_filter(image, 1.0)`. Coarse step: render the template (start scale/angle, centred on 0) as unit points, FFT cross-correlate with the smoothed image, take the peak as `(y0, x0)`. Refine `(scale, angle, x0, y0)` with `scipy.optimize.minimize(method="Nelder-Mead")` maximising the summed `map_coordinates` intensity at the sites. Then check bounds.
- [ ] **Step 4:** run — PASS.
- [ ] **Step 5: Commit** "Fit the tweezer array geometry to a camera image".

### Task 4: A `Fit` optimiser kind

**Files:**
- Modify: `pytweezer/experiment/optimisers.py` (new `Fit`; `Optimiser.proposal_units`, `proposal_group`; schema)
- Test: `tests/test_optimisers.py`

**Interfaces:**
- Produces: `Optimiser.proposal_units: ClassVar[dict[str, str]] = {}`, `Optimiser.proposal_group: ClassVar[str] = ""`; `optimiser_schema()` adds `"units": {key: proposal_units.get(key, "")}` for every proposed key and `"group": proposal_group`.
- Produces: `class Fit(Optimiser)`: `kind = "fit"`; `proposes()` returns `list(cls.proposal_units)`; one point; `run_point()` stores `self.fit()`; `finish()` records `proposed.<key>` for each value (nothing if `fit` never returned), then `super().finish()`. Subclasses implement `fit(self) -> dict[str, Any]`.

- [ ] **Step 1: Failing tests:** a small `DemoFit(Fit, Experiment)` in the test proposing `{"demo_offset": 1.5, "demo_grid": np.ones((3, 2))}` with units `{"demo_offset": "V", "demo_grid": "px"}`:
  - `test_a_fit_runs_one_point_and_proposes_what_it_returns` (`run_local`; constants `proposed.demo_offset == 1.5`, `proposed.demo_grid` shape `(3, 2)`).
  - `test_a_fit_that_raises_proposes_nothing`.
  - `test_the_schema_gives_each_proposals_unit_and_group` (fit kind, `objective is None`, `settings == []`, units dict, group).
- [ ] **Step 2:** run — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** `uv run pytest tests/test_optimisers.py -q` — PASS.
- [ ] **Step 5: Commit** "Add a Fit optimiser kind that proposes the result of one fit".

### Task 5: `FindTrapSites`, and measurements using the accepted sites

**Files:**
- Modify: `pytweezer/experiments/tweezer_array.py`
- Test: `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: Task 3 (`fit_trap_sites`, `default_geometry`, `to_roi`, `same_geometry`), Task 4 (`Fit`).
- Produces: `TweezerArrayShots(LoadTweezers)` — today's `rig`, saved-sequence loading in `prepare()`, `_batch`, `_filtered` (with `feature_size` now an attribute set by subclasses). `TweezerArrayMeasurement(TweezerArrayShots)` — `shots` argument; `prepare()` loads `trap_*` saved parameters; `score_batch` unchanged in behaviour. `FindTrapSites(Fit, TweezerArrayShots)` — arguments `calibration_shots=100`, `initial_threshold=100 photons`, `window_size=5 px`, `feature_size=10 px`, `default_rows=8`, `default_columns=8`, `default_spacing=5 µm`; `proposal_group = "Tweezer array"`, `proposal_units` per Global Constraints.

- [ ] **Step 1: Rework the fake rig in `tests/test_tweezer_array.py`:** `SITES` become `geometry_to_pixels(default_geometry(), 2.0, 0, ...)` rounded (so the default fit applies); `SAVED` gains `trap_sites` (SITES, as float array), `trap_threshold`, `trap_window_size=5`, `trap_feature_size=10`, `trap_geometry=default_geometry()`. `FAST` drops `calibration_shots`.
- [ ] **Step 2: Failing tests:**
  - `test_find_trap_sites_proposes_the_sites_threshold_and_geometry(rig)`: `run_local(FindTrapSites, calibration_shots=30)`; `proposed.trap_sites` within 0.5 px of `SITES` in order; `proposed.trap_geometry == default_geometry()`; `sites_from == "default"`; calibration shots all have tweezers off in the background drop.
  - `test_find_trap_sites_fits_the_rigs_array_when_it_knows_one(rig)`: rig array with `x_um`/`y_um` = a reordered default geometry; proposals follow that order, `sites_from == "rig"`.
  - `test_measurements_use_the_saved_sites_and_take_no_calibration_shots(rig)`: `TweezerImaging` run; `len(rig.goes) == FAST["shots"]`; recorded `trap_sites` equal `SAVED["trap_sites"]` minus the ROI origin, rounded.
  - `test_a_measurement_refuses_without_saved_sites(rig)` (message names `FindTrapSites`), `..._refuses_sites_fitted_to_another_array(rig)` (rig geometry moved 0.5 µm), `..._refuses_a_site_whose_window_leaves_the_roi(rig)` (one saved site at the ROI edge).
  - Replace `test_calibration_finds_as_many_sites...` and the calibration half of `test_only_calibration_turns_the_tweezers_off...` (calibration now lives in `FindTrapSites`).
- [ ] **Step 3:** run — FAIL.
- [ ] **Step 4: Implement.** `FindTrapSites.fit()`: shots as `_calibrate` does today; mean background-subtracted image, top-hat filtered; geometry from the rig (`np.column_stack([x_um, y_um])`) or `default_geometry`; `fit_trap_sites`; add the ROI origin to get full-sensor sites; threshold via `an.get_array_loading_statistics` on ROI-rounded sites; `record` results `calibration_image`, `roi_sites`, `scale`, `angle_deg`, `threshold`, `calibration_fidelity`, constants `sites_from`; return the five proposals. Delete `_calibrate`, the `sites` argument and `detect_trap_sites` use.
- [ ] **Step 5:** `uv run pytest tests/test_tweezer_array.py tests/test_tweezer_histogram.py -q` — PASS.
- [ ] **Step 6: Commit** "Find trap sites in their own optimiser and measure at the accepted ones".

### Task 5b: Every tweezer file records the accepted sites; the histogram uses them

Added at the user's request (2026-11-02): "the default analysis for load
tweezer (histograms) uses the stored array sites".

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py` (`LoadTweezers.prepare` records the sites)
- Modify: `pytweezer/experiments/tweezer_array.py` (measurements reuse what `LoadTweezers` loaded; `FindTrapSites` loads none)
- Modify: `pytweezer/analysis/tweezer_histogram.py` (frames path)
- Test: `tests/test_tweezer_array.py`, `tests/test_tweezer_histogram.py`

**Interfaces:**
- Produces: `LoadTweezers.saved_trap_sites(self) -> tuple[np.ndarray | None, str]` — the accepted sites in this ROI's pixels (rounded ints), or `None` and why (none saved, or fitted to another array than the rig shows). `LoadTweezers.prepare` sets `self.trap_sites` from it and, when not `None`, records constants `trap_sites` (ROI px), `trap_window_size`, `trap_feature_size`; when `None` it logs the reason and records nothing. `FindTrapSites.saved_trap_sites` returns `(None, "finds them")` without loading. `TweezerArrayMeasurement.prepare` raises `RuntimeError(reason)` when `self.trap_sites is None`.
- Produces: `tweezer_histogram` frames path uses the file's `trap_sites`, `trap_window_size`, `trap_feature_size` constants when present; only a file without them falls back to `find_sites`.

- [ ] **Step 1: Failing tests:** `test_load_tweezers_records_the_accepted_sites_in_its_roi(rig)`; `test_load_tweezers_runs_without_saved_sites_and_records_none(rig)`; in `tests/test_tweezer_histogram.py`, `test_a_frames_file_is_scored_at_its_recorded_sites` (sites recorded where `find_sites` would fail, e.g. a mean image with fewer bright spots than sites).
- [ ] **Step 2:** run — FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_tweezer_array.py tests/test_tweezer_histogram.py -q` — PASS.
- [ ] **Step 5: Commit** "Record the accepted trap sites in every tweezer file and histogram at them".

### Task 6: The Optimisers tab for fits and array proposals

**Files:**
- Modify: `pytweezer/GUI/optimisers.py` (`proposals`, `SettingsForm.planned`, `_plotted_names`, `_show_plots`, `accept`, new `_FitPlots`)
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: schema `"units"`, `"group"` (Task 4); `display_value` (Task 2); results `calibration_image`, `roi_sites` and constant `proposed.trap_window_size` (Task 5).
- Produces: `proposals(constants, keys)` returns arrays as nested lists (`.tolist()` for every numpy value).

- [ ] **Step 1: Failing tests** (extend `make_optimisers_panel` fixtures with a `"fit"` schema and file):
  - `test_an_array_proposal_shows_its_shape_and_is_accepted_as_a_list(qapp)`: proposal row reads "64 × 2 array px"; `accept()` sends a nested list.
  - `test_accept_creates_missing_keys_with_the_optimisers_units_and_group(qapp)`: the fake client call records `create=True`, `group="Tweezer array"`, `units` from the schema.
  - `test_a_fit_run_shows_its_image_with_a_square_at_each_site(qapp)`: `_FitPlots` has one image item and `len(roi_sites)` squares.
  - `test_a_fit_plans_one_fit`: `SettingsForm.planned() == "1 fit"`.
  - Update `test_accept_writes_only_ticked_rows_with_the_run` for the new call shape.
- [ ] **Step 2:** run — FAIL.
- [ ] **Step 3: Implement.** `accept()` calls `set_saved_parameters(..., create=True, group=schema["optimiser"]["group"], units=schema["optimiser"]["units"], rid=rid)`. `_FitPlots(key, schema)`: one `pg.PlotWidget` holding a `pg.ImageItem` of `calibration_image` and a `pg.ScatterPlotItem` with `symbol="s"`, size `trap_window_size` in data units (`pxMode=False`), at `roi_sites` (x = column, y = row). `_show_plots` picks `_FitPlots` for kind `"fit"`; `_plotted_names` returns `["calibration_image", "roi_sites"]` for it.
- [ ] **Step 4:** `uv run pytest tests/test_experiment_gui.py -q` — PASS.
- [ ] **Step 5: Commit** "Show trap-site fits in the Optimisers tab and accept array proposals".

### Task 7: ROI origin in camera frame headers

**Files:**
- Modify: `pytweezer/drivers/camera_base.py` (`CameraBase.roi_origin` class attribute `(0, 0)`; `broadcast_image` adds `roi_x0`/`roi_y0` unless the info has them; simulated `set_roi` records it)
- Modify: `set_roi` in `pytweezer/drivers/imagemX2.py`, `orca.py`, `thorcam.py`, `tweezermonitorcam.py` — each sets `self.roi_origin = (int(x0), int(y0))`
- Test: `tests/test_camera_frames.py`

**Interfaces:**
- Produces: every broadcast frame header carries `roi_x0`, `roi_y0` (ints, full-sensor pixels).

- [ ] **Step 1: Failing test** `test_broadcast_frames_say_which_roi_they_were_read_from(camera, clock)` using the simulated camera fixture: after `set_roi(110, 256, 220, 256)`, headers from `acquire_n_frames(..., broadcast=True)` and the live broadcast have `roi_x0 == 110`, `roi_y0 == 220`; before any `set_roi`, `0, 0`.
- [ ] **Step 2:** run — FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/test_camera_frames.py tests/test_camera.py tests/test_hamamatsu.py -q` — PASS.
- [ ] **Step 5: Commit** "Put the ROI origin in every broadcast camera frame".

### Task 8: Live loading analysis at the accepted sites

**Files:**
- Modify: `pytweezer/analysis/tweezer_loading.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `to_roi`, `same_geometry` (Task 3); headers `roi_x0`/`roi_y0` (Task 7, missing → 0).
- Produces: `LoadingStatistics.summary(*, sites: np.ndarray | None, window_size, fallback_threshold)`; output header gains `sites_match` (bool, `True` when the rig does not know its array) and drops `sites_from`. Properties `sites` and `detection_step` go; `feature_size`/`window_size` come from `trap_feature_size`/`trap_window_size`.

- [ ] **Step 1: Failing tests:** patch `tweezer_loading.get_saved_parameters` to return trap keys for `SITE_CENTRES` plus an ROI origin; `test_tweezer_loading_scores_the_saved_sites_in_the_frames_roi` (frame header `roi_x0/roi_y0` shifts; loading as `test_…patterned…` expects), `test_tweezer_loading_publishes_no_photons_without_saved_sites` (replaces the "until it finds the sites" test), `test_tweezer_loading_flags_sites_fitted_to_another_array` (`sites_match is False`).
- [ ] **Step 2:** run — FAIL.
- [ ] **Step 3: Implement.** Fetch saved parameters like the rig: at most every `RIG_REFRESH_S`, and afresh on a new `rid`; keep the last answer when the manager cannot be reached (log a warning once).
- [ ] **Step 4:** `uv run pytest tests/test_analysis.py -q` — PASS.
- [ ] **Step 5: Commit** "Score live loading at the accepted trap sites".

### Task 9: Tweezer viewer

Load the `add-applet` skill first; it says how applets are registered and launched.

**Files:**
- Create: `pytweezer/GUI/viewers/tweezer_monitor.py`
- Modify: the applet registry the `add-applet` skill names
- Test: `tests/test_gui.py` (or where the skill puts applet tests)

**Interfaces:**
- Consumes: `to_roi` (Task 3), headers `roi_x0`/`roi_y0` (Task 7), `ExperimentFeed` (`pytweezer/GUI/experiments/feed.py`) `queue_changed` snapshot's `saved_parameters_version`, `ExperimentManagerClient.saved_parameters()`.
- Produces: `class TweezerDisplay(ImageDisplay)` with property `show_site_labels` (default `False`); `set_sites(sites: np.ndarray | None, window_size: int)`; squares redrawn for each frame's ROI origin.

- [ ] **Step 1: Failing tests** (offscreen, fake image client and saved parameters): `test_the_tweezer_viewer_draws_a_square_at_each_saved_site_in_the_frames_roi` (square centres equal `to_roi(sites, roi_y0, roi_x0)` in (x, y) order, size `trap_window_size`), `test_the_tweezer_viewer_redraws_when_the_saved_parameters_change`, `test_with_no_saved_sites_the_viewer_says_so`.
- [ ] **Step 2:** run — FAIL. **Step 3:** implement (squares as one `pg.ScatterPlotItem`, `symbol="s"`, `pxMode=False`, no fill; labels as `pg.TextItem`s when enabled). **Step 4:** run — PASS.
- [ ] **Step 5:** Screenshot it offscreen per `run-pytweezer`/`add-applet` guidance if the skill offers a way; otherwise rely on the tests.
- [ ] **Step 6: Commit** "Add a tweezer viewer drawing the accepted trap sites".

### Task 10: Finish

- [ ] `uv run pytest tests/ -q` — all pass; `uv run ruff check . && uv run ruff format --check .`.
- [ ] `cd docs && uv run make html` builds.
- [ ] Update the `add-experiment` skill only if it documents `TweezerArrayMeasurement` calibration (grep first).
- [ ] Open a PR from `trap-site-finder` describing the change and the first-time workflow: run FindTrapSites, check it in the Optimisers tab, accept.
