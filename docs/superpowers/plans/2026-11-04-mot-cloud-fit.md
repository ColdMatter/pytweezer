# MOT Cloud Fits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit MOT clouds with a 2D Gaussian live and per experiment point, overlay the fit on image monitors, add projections with 1D fits and a projections monitor, and give MOT experiments default Results analyses for size and time-of-flight temperature.

**Architecture:** One Qt-free module, `pytweezer/analysis/cloud_fit.py`, does all fitting. Live analyses (Analysis Manager, server PC) and experiments both call it. Camera broadcasts gain a pixel scale, so every fit, overlay and plot shares one coordinate system: x is the column, row 0 is at the top, in metres where the scale is known. Results views reuse the tweezer histogram's off-thread loading through a base class extracted from it.

**Tech Stack:** numpy, scipy (`least_squares`), PyQt6/pyqtgraph, ZMQ clients from `pytweezer.servers`, h5 measurements through `pytweezer.experiment`.

**Spec:** `docs/superpowers/specs/2026-11-04-mot-cloud-fit-design.md`

## Global Constraints

- Branch `mot-cloud-fit`; commit after each task (CLAUDE.md "To commit": ruff check --fix, ruff format, `git add` whole files, never `--no-verify`).
- UK English; Google-style docstrings describing current behaviour only; comment only what is surprising.
- Logging through `get_logger` from `pytweezer/logging_utils.py`.
- Qt windows in tests and checks run with `QT_QPA_PLATFORM=offscreen`; never open a GUI on the live display.
- Tests follow `.claude/skills/write-tests/SKILL.md`: new cases in the area's existing test file, ≤10 per module, known-answer checks for numerics.
- Coordinates: pixel column `c`, row `r` of a frame sit at `x = (origin_x + c + 0.5)·res_x`, `y = (origin_y + r + 0.5)·res_y`, where `res = head["_imgresolution"]` and `origin = head["_offset"]` (binned pixels). Without them `res = (1, 1)` and `origin = (0, 0)`, and units are `"px"`; with them, `"m"`. `angle` is that of the major axis, measured from +x towards +y in these coordinates, in (−π/2, π/2].
- The fit field names, exactly: `x0, y0, sigma_x, sigma_y, sigma_major, sigma_minor, angle, amplitude, offset, integral, units, success`.
- Dated notebooks and dated specs/plans are never edited.
- Minor skill updates go in the same commit as their code; new skill sections are proposed to the user (Task 10).

## Review Focus

1. **A frame with no cloud** (MOT off, a background shot, a dark frame). The fit returns `success=False` in under a second, never raises. The overlay clears, and the recorded fields are NaN. Pinned in Task 1 (`test_blank_frame_fails_quietly`) and Task 6 (`test_point_without_a_cloud_records_nan`).
2. **A cloud running off the frame edge**, so its ±4σ window is clipped. The fit still converges on the visible part with the centre right. Pinned in Task 1 (`test_cloud_at_the_edge_is_fitted`).
3. **uint16 frames minus a brighter background.** Subtraction happens in float, with no wrap-around to ~65535. Pinned in Task 3 (`test_background_subtraction_is_signed_float`).
4. **A background that doesn't match** (another rid, or a different shape after an ROI change). The frame passes through unsubtracted rather than crashing the analysis. Pinned in Task 3 (`test_mismatched_background_passes_frame_through`).
5. **A binned, offset ROI.** The overlay ellipse and the recorded centre land on the cloud in the same coordinates as the image. Pinned in Task 1 (`test_fit_reports_in_frame_coordinates`) and Task 4 (`test_overlay_ellipse_centred_on_fit`).

---

### Task 1: The fit module

**Files:**
- Create: `pytweezer/analysis/cloud_fit.py`
- Modify: `pytweezer/analysis/gaussian_fit.py` (its `GaussianFit` may call `fit_profile`; keep its published fields unchanged)
- Test: `tests/test_cloud_fit.py` (new; no existing file covers this area)

**Interfaces:**
- Produces:
  - `FIELDS: tuple[str, ...]`, the ten numeric field names above.
  - `@dataclass(frozen=True) class CloudFit`: the twelve fields; `as_dict() -> dict`.
  - `frame_geometry(head: dict) -> tuple[tuple[float, float] | None, tuple[float, float]]`: `(resolution or None, origin)` from a stream header.
  - `fit_cloud(image: np.ndarray, resolution: tuple[float, float] | None = None, origin: tuple[float, float] = (0.0, 0.0)) -> CloudFit`
  - `@dataclass(frozen=True) class ProfileFit`: `center, sigma, amplitude, offset, success`.
  - `fit_profile(x: np.ndarray, y: np.ndarray) -> ProfileFit`; `profile_curve(x: np.ndarray, fit: ProfileFit) -> np.ndarray`.
  - `contour_ellipses(fit: CloudFit, n: int = 8, points: int = 100) -> list[np.ndarray]`: each `(2, points)` `[x, y]`, at levels `k/(n+1)` of the amplitude, `k = 1..n` (radius `sqrt(-2 ln f)` σ).
  - `fit_frames(frames: np.ndarray, background: np.ndarray | None, resolution, origin) -> tuple[dict[str, float], np.ndarray]`: the fit of the mean background-subtracted image, as `{field: value}` for `FIELDS` (NaN where `success` is False). It adds `<field>_error` for `x0, y0, sigma_x, sigma_y, sigma_major, sigma_minor`: the SEM over the frames that fitted, each fit seeded from the mean fit, NaN with fewer than two. It also returns the mean image as float32.

- [ ] **Step 1: Write the failing tests** in `tests/test_cloud_fit.py`. `gaussian_image(shape, x0, y0, s_major, s_minor, angle, amplitude, offset)` is a helper that builds frames from the model in pixel-centre coordinates, with seeded Poisson-like noise.
  - `test_rotated_cloud_parameters_recovered`: a 200×300 frame, `angle=0.5`, σ 12 and 6 px. The fit recovers centre to 0.1 px, σs to 2 %, angle to 0.02 rad, and `sigma_x == sqrt(s_maj² cos² + s_min² sin²)` to 2 %.
  - `test_fit_reports_in_frame_coordinates`: the same cloud fitted with `resolution=(4e-6, 4e-6), origin=(10, 20)`. `x0 == (10 + c0 + 0.5)·4e-6`, the σs are scaled by 4e-6, and `units == "m"`.
  - `test_blank_frame_fails_quietly`: a noise-only frame gives `success is False`, and the call returns in < 1 s.
  - `test_cloud_at_the_edge_is_fitted`: centre 3 px inside the left edge, σ 10 px. The centre is recovered to 1 px.
  - `test_large_frame_fits_through_window`: a 2048×2048 frame with σ 8 px is recovered to 2 %. This is a correctness check on the window and binning path, not a timing test.
  - `test_profile_fit_recovers_gaussian`: 1D, σ to 2 %.
  - `test_contour_ellipse_lies_on_its_level`: the model evaluated at each ellipse point equals `offset + amplitude·k/(n+1)` to 1e-6 relative.
  - `test_frame_errors_are_sem_of_frame_fits`: 5 frames with known per-frame σ jitter. `sigma_x_error` matches the SEM of the individual fits to 10 %.

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_cloud_fit.py -q` fails with an ImportError.

- [ ] **Step 3: Implement `cloud_fit.py`.** Method, from the spec:
  1. Moments seed on the whole frame: baseline from the 10th percentile, weights clipped at zero, and the second-moment matrix giving σs and angle.
  2. A window of ±4σ_major about the seed, clipped to the frame, binned by `max(1, floor(σ_minor_px / 3))`.
  3. `scipy.optimize.least_squares(method="lm")` with an analytic Jacobian over (A, x0, y0, σ_major, σ_minor, θ, offset), capped at `max_nfev=200`.
  4. On a non-positive moments signal or a fit error, return the seed with `success=False`.

  Then normalise the angle into (−π/2, π/2] and swap the axes so that σ_major ≥ σ_minor. `sigma_x = sqrt(Σxx)`, `sigma_y = sqrt(Σyy)`. `integral = 2π·A·σ_major_px·σ_minor_px` (summed counts). The `run_analysis` entry point is added in Task 3.

- [ ] **Step 4: Run the tests to verify they pass.** `uv run pytest tests/test_cloud_fit.py -q` passes. Also time a 2048² frame by hand with `uv run python -c` and note the ms in the commit message; it should be about 10 ms or less.

- [ ] **Step 5: Commit** with the message "Cloud fit: 2D Gaussian with window, analytic Jacobian, contours, frame errors".

### Task 2: Pixel scale in camera headers; ORCA renamed

**Files:**
- Modify: `pytweezer/configuration/config.py` (`"CaF ORCA"` → `"ORCA"`, stream `orca`, drop the "two entries" comment; add `"object_pixel_m": None` to the ORCA, CaF ThorCam, Rb ThorCam and Rb HamCam entries)
- Modify: `pytweezer/drivers/camera_base.py` (`__init__` gains `object_pixel_m: float | None = None`; `broadcast_image` adds `_imgresolution` and `_offset` when it is set)
- Modify: every camera driver and sim twin whose `__init__` feeds `CameraBase` (`orca.py`, `thorcam.py`, `tweezermonitorcam.py`, `imagemX2.py` and the HamCam/hamamatsu path), so `device_server._config_kwargs` passes the key through
- Modify: `pytweezer/experiments/camera_settings.py`, the `pytweezer/experiment/camera_experiment.py` docstring, `.claude/skills/add-device-driver/SKILL.md` and its `scripts/check_device.py`, `tools/bench_hamcam_rpc.py` (`"CaF ORCA"`/`"Rb ORCA"` → `"ORCA"`)
- Test: `tests/test_camera.py`

**Interfaces:**
- Produces: camera broadcast headers carrying `_imgresolution = [p·b, p·b]` and `_offset = [roi_x0 / b, roi_y0 / b]`, where `p = object_pixel_m` and `b = binning`, only when `object_pixel_m` is set. Also the device name `"ORCA"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_camera.py`:
  - `test_broadcast_header_carries_pixel_scale`: a simulated camera with `object_pixel_m=2e-6`, binning 2 and ROI origin (100, 40) broadcasts `_imgresolution == [4e-6, 4e-6]` and `_offset == [50, 20]`.
  - `test_broadcast_header_without_scale_is_unchanged`: no `_imgresolution` key.
- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_camera.py -q`
- [ ] **Step 3: Implement the changes.** Then check every consumer of camera streams: `roi_slice.py` (reads `_offset`/`_imgresolution`), `CameraROI.copy_as_camera_roi`, `analysis/tweezer_loading.py` (sites by `roi_x0`/`roi_y0`), and `datamgr.py:452`. Each should still behave correctly when the scale is present.
- [ ] **Step 4: Run the whole suite.** `uv run pytest tests/ -q` passes.
- [ ] **Step 5: Commit** with the message "Cameras broadcast their pixel scale; the ORCA is one device, ORCA".

### Task 3: Live analyses

**Files:**
- Modify: `pytweezer/analysis/cloud_fit.py` (add `class CloudFitAnalysis(ImageAnalysis)` and `if __name__ == "__main__": run_analysis(CloudFitAnalysis)`)
- Create: `pytweezer/analysis/shot_background.py`; delete `pytweezer/analysis/subtract_bg.py` and fix its mention in `analysis_base.py`'s docstring
- Modify: `pytweezer/analysis/projections.py`, `pytweezer/analysis/image_stats.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `fit_cloud`, `frame_geometry`, `fit_profile`, `profile_curve` (Task 1).
- Produces:
  - `CloudFitAnalysis` publishes on the Data hub, stream `<name>`, no array. The header is the `CloudFit.as_dict()` fields plus `rid`, `index`, `_imgresolution` and `_offset` from the input.
  - `ShotBackground(ImageAnalysis).process(head, data)`. A frame with `head.get("background")` is stored as that stream's background for its `rid` (frames of one background shot are averaged) and nothing is published. Any other frame is published as `float64(data) − background` when the rid and shape match. Otherwise it is published unchanged as float64.
  - `projections.py` publishes `<name>_x` and `<name>_y` as `[coords, sums, fit_curve]`, a 3×N array. Coordinates are in frame coordinates (constraint above). The header carries `center, sigma, amplitude, offset, success` from `fit_profile`.
  - `image_stats.py` copies `rid` and `index` into its header.

- [ ] **Step 1: Write the failing tests** in `tests/test_analysis.py`, calling `process` directly as existing cases there do:
  - `test_cloud_fit_analysis_publishes_fit_fields_with_geometry`
  - `test_background_frame_is_held_not_published`
  - `test_background_subtraction_is_signed_float`: uint16 frame 10 minus background 20 gives −10.
  - `test_mismatched_background_passes_frame_through`, parametrised over another rid and another shape.
  - `test_projection_fit_curve_matches_sums_of_a_gaussian`: the `_x` row 2 fits row 1, and `center` is at the cloud's x.
- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_analysis.py -q`
- [ ] **Step 3: Implement the four files.** Check `live_plot.py` and `scalar_history.py` read only rows 0–1 of a projection message. If one reads all rows, make it draw rows 0–1 only. `tests/test_live_plot.py` covers this.
- [ ] **Step 4: Run the analysis and live-plot tests.** `uv run pytest tests/test_analysis.py tests/test_live_plot.py -q` passes.
- [ ] **Step 5: Commit** with the message "Live cloud fit, shot background, projections with 1D fits".

### Task 4: Image monitor: orientation, overlays, data panel

**Files:**
- Modify: `pytweezer/GUI/viewers/image_monitor.py`, `pytweezer/GUI/viewers/data_sidebar.py`, `pytweezer/GUI/viewers/tweezer_monitor.py` (drop the "Image Monitor draws them transposed" note, and any transposition workaround that the base now makes redundant)
- Test: `tests/test_image_monitor.py`

**Interfaces:**
- Consumes: `contour_ellipses`, `CloudFit`, `FIELDS` (Task 1).
- Produces, all used by Task 5's subclass:
  - `ImageDisplay` draws row-major with row 0 at the top. Match `tweezer_monitor.py`'s `ImageItem` setup and view-box orientation.
  - Property `overlaystreams` (Data), with a context-menu entry "Overlays".
  - Property `n_contours` (int, default 8).
  - `ImageDisplay.update_overlays()`, called from `poll()`.
  - Module function `overlay_paths(head: dict, n: int) -> list[np.ndarray]`: `[]` when the header lacks the fit fields or `success` is False.
  - `show_sidebar` defaults to True, and the sidebar sits below the image.
  - `DataSidebar` shows the latest message of every subscribed stream, one block per stream, laid out per `pytweezer-gui-design`.

- [ ] **Step 1: Write the failing tests:**
  - `test_overlay_ellipse_centred_on_fit`: a header from `fit_cloud` on a binned, offset frame. The mean of each path's points equals `(x0, y0)`.
  - `test_failed_fit_clears_overlay`: `success=False` gives `[]`.
  - `test_sidebar_keeps_each_streams_latest_message`
  - `test_image_monitor_builds_offscreen`, if no existing case already covers it.
- [ ] **Step 2: Run them to verify they fail.** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_monitor.py -q`
- [ ] **Step 3: Implement.** Draw the ellipses as white `PlotCurveItem`s in the plot, keyed by stream, replaced on each message. Then re-check `camera_roi` (it already handles both axis orders) and `roi_slice.py` against row-major ROIs.
- [ ] **Step 4: Run the viewer tests.** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_monitor.py tests/test_tweezer_monitor.py -q` passes.
- [ ] **Step 5: Commit** with the message "Image monitor: row-major, fit overlays, data panel below and on". The commit body says saved ROIs on lab Image Monitors must be redrawn once.

### Task 5: Projections monitor

**Files:**
- Modify: `pytweezer/GUI/viewers/image_plot_monitor.py`. Rewrite it in place as `ImageProjectionDisplay(ImageDisplay)`. The path stays, because `applet_launcher.py` and `configuration/properties/properties.json` reference it; update their class or name references if they name `ImagePlotDisplay`.
- Modify: `docs/notes/applets.md` (its line for this file)
- Test: `tests/test_image_monitor.py`

**Interfaces:**
- Consumes: `ImageDisplay` (Task 4) and the projection messages `[coords, sums, fit_curve]` (Task 3).
- Produces: property `projectionstreams` (names of projection analyses). `<name>_x` draws as sums plus fit curve in a plot above the image that shares its x-axis. `<name>_y` draws in a plot to its left that shares its y-axis, with values running towards the image.

- [ ] **Step 1: Write the failing test** `test_projection_monitor_builds_and_draws_both_profiles`. Feed one `_x` and one `_y` message through the update method. Each side plot then has a data curve and a fit curve.
- [ ] **Step 2: Run it to verify it fails.**
- [ ] **Step 3: Implement it.** Build the side plots around the parent's plot in `init_gui`, and add their update to `poll()`.
- [ ] **Step 4: Run the test to verify it passes.** Then take an offscreen screenshot following `.claude/skills/run-pytweezer/SKILL.md` and look at it. Simulated frames with a Gaussian are fine: publish one from a test script on a scratch stream.
- [ ] **Step 5: Commit** with the message "Projections monitor is an Image Monitor with linked sum plots".

### Task 6: CameraExperiment: background shots, cloud fits

**Files:**
- Modify: `pytweezer/experiment/camera_experiment.py`
- Test: `tests/test_camera_frames.py` (or whichever existing file covers `take_shot`; read them first)

**Interfaces:**
- Consumes: `fit_frames` (Task 1); CONFIG `object_pixel_m` (Task 2).
- Produces:
  - `take_shot(self, *, background: bool = False, **overrides) -> dict[str, np.ndarray]`. With `background`, every broadcast header also carries `"background": True`.
  - `CameraExperiment.record_cloud_fit(self, name: str, frames: np.ndarray, background: np.ndarray | None = None) -> None`. It records `<name>_<field>` for `FIELDS`, the `_error`s and, when `save_mean_image` is set, `<name>_mean_image`. Resolution comes from the camera's CONFIG `object_pixel_m` × the applied binning (or the CONFIG entry's `binning`, else 1); origin from `roi_origins[name]` ÷ binning.
  - Class attribute `cloud_species: ClassVar[str | None] = None`. When set, `prepare` records the constant `cloud_mass_amu`, using `{"CaF": 59.0, "Rb": 86.909}`. Put those masses in `cloud_fit.py` as `SPECIES_MASS_AMU`.
  - Argument `save_mean_image = Bool(True, group="Camera")`, declared on the experiments that record fits (Task 7), not on the base.

- [ ] **Step 1: Write the failing tests:**
  - `test_background_shot_marks_broadcast_headers`
  - `test_point_records_cloud_fit_fields`: a Gaussian frame through `run_local`, with `sigma_x` recovered.
  - `test_point_without_a_cloud_records_nan`
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement the changes.**
- [ ] **Step 4: Run the experiment tests.** `uv run pytest tests/ -q -k "camera or experiment"` passes.
- [ ] **Step 5: Commit** with the message "CameraExperiment: marked background shots, recorded cloud fits".

### Task 7: CaFFluorescence records the fit; LoadTweezersMOTImage

**Files:**
- Modify: `pytweezer/experiments/caf_fluorescence.py`, `pytweezer/experiments/load_tweezers.py`
- Test: `tests/test_caf_fluorescence.py`, and the existing LoadTweezers test file (find it with `grep -l LoadTweezers tests/`)

**Interfaces:**
- Consumes: Task 6.
- Produces:
  - `CaFFluorescence` takes its background shot with `take_shot(background=True, ...)` and records `thorcam_*` fit fields beside `counts`. It declares `cloud_species = "CaF"`, `save_mean_image` and `analyses = ("cloud_fit", "cloud_temperature")`.
  - `class LoadTweezersMOTImage(LoadTweezers)`, new, in `load_tweezers.py`. It declares `orca = CameraShots("ORCA", frames=1, locked=("frames",))`, `cloud_species = "Rb"`, `save_mean_image`, and `analyses = (*LoadTweezers.analyses, "cloud_fit", "cloud_temperature")`. `run_point` runs `LoadTweezers.run_point`'s shot once and records `orca_*` fit fields from the same shot's ORCA frame, with no background. Refactor `LoadTweezers.run_point` so the subclass reuses the shot's frames rather than taking a second shot. `LoadTweezers` itself does not change behaviour and never touches the ORCA.
  - Subclasses that set `analyses = ()` (the optimisers) keep their own.

- [ ] **Step 1: Write the failing tests:**
  - `test_fluorescence_point_records_thorcam_cloud_fit`, using the existing `FakeCamera` with a Gaussian `frame`.
  - `test_fluorescence_background_shot_is_marked`
  - `test_mot_image_variant_records_orca_fit_from_the_same_shot`: one MOTMaster run per point, with HamCam results and `orca_sigma_x` both recorded.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement the changes.** `measure()` keeps returning only `counts` and `counts_error`, because `OptimiseMOTLasers`' objective reads it. The fit is recorded in `run_point`.
- [ ] **Step 4: Run the tests.** `uv run pytest tests/test_caf_fluorescence.py tests/test_optimise_mot_lasers.py <load-tweezers test file> -q` passes.
- [ ] **Step 5: Commit** with the message "MOT experiments record the cloud fit and declare its analyses".

### Task 8: Results tab: the cloud fit view

**Files:**
- Create: `pytweezer/analysis/cloud_results.py` (Qt-free)
- Modify: `pytweezer/GUI/experiments/analyses.py`. Extract `ComputedAnalysisView(AnalysisView)` from `TweezerHistogramView`, keeping its archive clone, worker thread, generation counter, pending-request coalescing and error messages. `TweezerHistogramView` becomes a subclass. Register `"cloud_fit"` as "MOT Gaussian fit".
- Create: `pytweezer/GUI/experiments/cloud_views.py`
- Test: `tests/test_results_analyses.py`

**Interfaces:**
- Consumes: `contour_ellipses`, `CloudFit`, `FIELDS` (Task 1); recorded `<camera>_*` fields (Task 6); `summary.aggregate`.
- Produces:
  - `ComputedAnalysisView` with hooks:
    - `needed_results(self, measurement) -> list[str]`
    - `settings(self) -> Any`
    - `compute(self, loaded: Measurement, settings) -> Any` (runs off-thread)
    - `show_result(self, result) -> None`
    - `unavailable_reason(self, measurement) -> str | None`
  - `cloud_cameras(measurement) -> list[str]`: camera prefixes with a recorded `_sigma_x`.
  - `scanned_arguments(measurement) -> list[str]`.
  - `cloud_sizes(measurement, camera, axis) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]`: for `sigma_x` and `sigma_y`, `(axis values, mean, error)`. Points sharing a value are pooled through `aggregate`; a value with one point keeps its recorded `_error`.
  - `point_image(measurement, camera, index) -> tuple[np.ndarray | None, CloudFit]`.
  - `CloudFitView(ComputedAnalysisView)`, with controls for camera, plot-against argument and point. It shows the point's mean image with its contour ellipses (row-major, as the Image Monitor) beside σx and σy against the chosen argument with error bars. A file with no mean images shows the plot only.

- [ ] **Step 1: Write the failing tests:**
  - `test_cloud_sizes_pool_repeated_scan_values`: 2 values × 3 repetitions written with `write_run`. Mean and SEM are as computed by hand.
  - `test_single_point_value_keeps_recorded_error`
  - `test_cloud_fit_view_draws_a_fit_file` (offscreen, `until`).
  - The existing `TweezerHistogramView` tests still pass, unchanged, after the extraction.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement the changes.**
- [ ] **Step 4: Run the Results tests.** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_results_analyses.py tests/test_results_browser.py -q` passes.
- [ ] **Step 5: Commit** with the message "Results tab: MOT Gaussian fit view; shared off-thread view base".

### Task 9: Results tab: time-of-flight temperature

**Files:**
- Modify: `pytweezer/analysis/cloud_results.py`, `pytweezer/GUI/experiments/cloud_views.py`, `pytweezer/GUI/experiments/analyses.py` (register `"cloud_temperature"` as "MOT temperature (time of flight)")
- Test: `tests/test_results_analyses.py`

**Interfaces:**
- Consumes: `cloud_sizes`, `scanned_arguments` (Task 8); the constant `cloud_mass_amu` (Task 6).
- Produces:
  - `@dataclass(frozen=True) class TofFit`: `temperature_K, temperature_err_K, sigma0, sigma0_err, success, message`.
  - `tof_fit(t_s: np.ndarray, sigma: np.ndarray, sigma_err: np.ndarray, mass_amu: float) -> TofFit`. It fits σ² = σ₀² + (k_B T/m) t², weighted by `2σ·σ_err` where finite, unweighted otherwise. With fewer than three distinct times it returns `success=False`, with the message saying so.
  - `guess_time_axis(names: list[str]) -> str | None`: the first name containing `FreeExp`, `FreeFall` or `Drop` (case-insensitive).
  - `cloud_temperature(measurement, camera, axis, seconds_per_unit) -> dict`: `TofFit` for x and y, the σ² against t² series, and the centre against t² with a fitted g (`y0 = c + ½ g t²`).
  - `CloudTemperatureView(ComputedAnalysisView)`, with controls for camera, time argument (default from `guess_time_axis`) and time unit. The time unit is a spin box defaulting to 10 µs, the MOTMaster pattern clock tick the notebooks use (`FreeExpTime·1e-5`). It shows σx² and σy² against t² with fits, T_x and T_y ± error in µK, σ₀, and the centre against t² with g. Without `cloud_mass_amu` it says the experiment recorded no species.

- [ ] **Step 1: Write the failing tests:**
  - `test_tof_fit_recovers_temperature`: synthetic σ(t) for Rb at 20 µK, σ₀ = 0.5 mm, with 1 % noise. T is recovered to 5 % and lies within 2 of its stated errors.
  - `test_too_few_times_is_refused`
  - `test_guess_time_axis` (parametrised: `tFreeFallDuration`, `FreeExpTime`, `tDropDuration`, none)
  - `test_cloud_temperature_view_builds_offscreen`
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement the changes.**
- [ ] **Step 4: Run the Results tests.** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_results_analyses.py -q` passes.
- [ ] **Step 5: Commit** with the message "Results tab: MOT time-of-flight temperature".

### Task 10: Skills, docs, full check

**Files:**
- Modify: `.claude/skills/add-analysis-script/SKILL.md`, `.claude/skills/add-applet/SKILL.md`, `.claude/skills/add-experiment/SKILL.md`, `docs/notes/applets.md`

- [ ] **Step 1: Make the minor skill edits** (renamed or new files, table rows, the ORCA name, `take_shot(background=...)`, `LoadTweezersMOTImage` and the new analysis names) and commit them.
- [ ] **Step 2: Propose the significant skill changes to the user and wait for approval.** Those are a new section in `add-applet` on overlays and the projections monitor, and one in `add-experiment` on recording cloud fits. Apply them once approved.
- [ ] **Step 3: Run the full suite and lint.** `uv run pytest tests/ -q` passes. `uv run ruff check . && uv run ruff format --check .` are clean. `cd docs && uv run make html` builds.
- [ ] **Step 4: List for the user** the dated notebooks that use the removed `subtract_bg.py` or the `"CaF ORCA"` name (grep the Box notebook folders), without editing them. Also list the open `object_pixel_m` values still to fill in.
- [ ] **Step 5: Push the branch and open a PR.** The PR body notes the ROI redraw, `object_pixel_m`, and that `"Rb ORCA"`'s deletion in 78d0611 is superseded by the rename.
