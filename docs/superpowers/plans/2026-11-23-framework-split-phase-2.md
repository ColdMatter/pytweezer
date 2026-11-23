# Framework / User-Code Split, Phase 2 Implementation Plan (Applets)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 1 (`docs/superpowers/plans/2026-11-23-framework-split-phase-1.md`)
has merged, so `GUI/views/` and `HistogramPlot` already exist.

**Goal:** Separate the applets' drawing from their wiring. The image viewer
and the line plot become base views in `GUI/views/`. No applet inherits from
another applet any more. Every applet looks and behaves exactly as before.

**Architecture:**
- **Two base views.** `ImageView` (`GUI/views/image.py`) and `LinePlot`
  (`GUI/views/line_plot.py`) draw only what they are handed.
- **Two framework base classes.** The wiring that several applets share
  (subscriptions, masks, overlays, persisted ROIs and levels, the data sidebar;
  the plot grid and experiment feed) moves into `ImageApplet`
  (`GUI/image_applet.py`) and `LivePlotApplet` (`GUI/live_plot_applet.py`).
  Each holds its view.
- **Thin applets.** The applets in `GUI/viewers/` subclass those framework
  classes, not each other, and keep only what is specific to them: the camera
  controls, the site overlay, the projections, the stacked arrangement.
- **Inheriting a framework class is fine.** The spec's rule is that an applet
  must not inherit from another *applet*. Inheriting framework like
  `ImageApplet` is the same as inheriting `Applet` today.

**Tech Stack:** PyQt6, pyqtgraph, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-23-framework-and-user-code-split-design.md`
(the "Views" section).

## Global Constraints

- **Applets look and behave exactly as before.** Check each one in a rendered
  window against the screenshot taken before the change (Task 1), not only in
  tests.
- **Saved Properties keep their keys:**
  - every `PropertyAttribute` name;
  - `<key>/datastreams`, `<key>/fields`, `<key>/experimentresults`;
  - `<stream>/color` and `sidebarstreams`;
  - the ROI paths `<name>/ROI/<region>/{pos,size,color}` and `/Servers/roilist`;
  - `/Camera ROI presets/...`.

  `PropertyAttribute`s stay declared on applet classes, framework or
  user-level, because they resolve against `self._props`.
- **Launcher paths don't change.** Every `script` path in
  `GUI/applet_launcher.py` and `properties.default.json` stays as it is, and
  each viewer module keeps its `main(name)`.
- **Views know nothing of streams, Properties, devices or threads.** Colours
  resolved through `Applet.stream_color` are passed into the views. ROIs that
  persist to Properties (`zmq_ROI`, `CameraROI`) are created by the applet and
  handed to the view as plain `pg.ROI`s.
- **Applets never inherit from another applet,** and default-analysis views
  never import from applets (and vice versa).
- **Leave dead code alone.** Don't fix or remove dead code you find
  (`ImageDisplay._maxlevels`, `_autolevels`, `_imgpath`, `imDataDict`; the
  2-value `self.imgstream.recv()` at `image_monitor.py:213`). List it in the
  final report instead; this is a refactor.
- **No hooks for tests.** Tests follow the new attribute paths.
- **Follow CLAUDE.md's procedures** for commits (ruff, whole-file `git add`,
  trailer lines, never `--no-verify`), `QT_QPA_PLATFORM=offscreen` for Qt
  tests, Google docstrings and UK spelling.

## Review Focus

1. **Autoscale.** When a frame arrives, pyqtgraph emits
   `sigLevelChangeFinished` even though no one dragged anything. Only a real
   drag may turn autoscale off and save `manual_levels`. Task 2 pins this.
2. **Restoring saved manual levels.** An applet opened with
   `autoscale=False` and saved `manual_levels` must show the first frame at
   those levels, not autoscaled. Task 3 pins this.
3. **New-frame detection by identity.** `CameraApplet` and `TweezerDisplay`
   spot a new frame by the head dict's identity. `ImageView.frame()` must
   return the same dict object it was given, not a copy. Task 2 pins this.
4. **Init order.** `ImageProjectionDisplay` and `CameraApplet` used to work
   round `update_subscriptions()` running before their plots existed. Under
   composition, the view must exist before the first subscription. Task 4 pins
   this with an applet constructed with saved projection streams.
5. **Alignment.** The projection side plots and the stacked plotter's panels
   line up through fixed axis widths and stretch factors. The screenshot
   comparison in Tasks 4 and 6 is the check.

---

### Task 1: Before-screenshots of every applet

**Files:**
- Create: `tests/applet_screenshots.py`. This is a script, not a test, and is
  committed so phase 2's reviewers can rerun it.

- [ ] **Step 1: Write the script.** For each applet (`ImageDisplay`,
  `ImageProjectionDisplay`, `TweezerDisplay`, `CameraApplet`, `SlmMonitor`,
  `LivePlot`, `StackedPlotter`, `ScalarHistory`, `TweezerLoadingViewer`), it:
  - constructs the applet offscreen with the same monkeypatches
    `tests/test_image_monitor.py` uses for Properties and the stream clients;
  - feeds one synthetic message (a 128×96 Gaussian frame with a head carrying
    `_imgresolution` and `_offset`, or a data message);
  - saves `widget.grab()` to `<out_dir>/<Applet>.png`. `out_dir` is
    `sys.argv[1]`.

  For `CameraApplet`, use the fake device that
  `tests/test_camera*.py` use. If that needs real hardware, skip
  `CameraApplet` and say so.
- [ ] **Step 2: Take the before screenshots.** Run
  `QT_QPA_PLATFORM=offscreen uv run python tests/applet_screenshots.py <scratchpad>/before`
  and expect one PNG per applet.
- [ ] **Step 3: Commit**: "Script to screenshot every applet with synthetic data".

### Task 2: `ImageView`, the base image viewer

**Files:**
- Create: `pytweezer/GUI/views/image.py`
- Test: `tests/test_image_view.py`

**Interfaces:**
- Produces: `class ImageView(QtWidgets.QWidget)`, with these attributes:
  - `graphics: pg.GraphicsLayoutWidget`;
  - `plot: pg.PlotItem`, with Y inverted, aspect locked and axes in metres;
  - `image_item: pg.ImageItem`, `axisOrder="row-major"`;
  - `lut`: a `HistogramLUTItem`, or a `ColorBarItem` when `colour_bar="bar"`;
  - `mask_item`.

  Its signals:
  - `sig_levels_dragged = pyqtSignal(float, float)`, emitted for user drags
    only;
  - `sig_mouse_moved = pyqtSignal(object)`, the scene position, unthrottled.

  Its methods:
  - `__init__(self, parent=None, *, colour_map: str = "magma",
    invert_colours: bool = False, colour_bar: str | None = "histogram",
    units: str = "m", title: str | None = None)`. `colour_bar` is
    `"histogram"`, `"bar"` or `None`. The colour map goes on `lut.gradient`,
    never `setLookupTable`.
  - `show_image(self, image, head: dict | None = None, *, levels:
    tuple[float, float] | None = None)` places the frame with `frame_rect`.
    `levels=None` autoscales; the `_autoscaling_frame` guard moves in here.
  - `clear_image(self)`
  - `frame(self) -> tuple[np.ndarray | None, dict | None]` returns the
    last-shown image and the identical head object.
  - `set_levels(self, low, high)` and `levels(self) -> tuple[float, float]`
  - `set_colour_map(self, name)` and `set_bar_ticks(self, ticks)`
  - `show_mask(self, mask, head=None)`
  - `show_paths(self, key: str, paths: list[np.ndarray], pen=None)`, which
    replaces and generalises `show_overlay` (`overlay_curves` keyed by `key`)
  - `add_roi(self, roi: pg.ROI)` and `rois(self) -> list[pg.ROI]`
  - `add_side_plot(self, side: str, size: int = 140) -> pg.PlotItem`, where
    `side` is `"top"` or `"left"`. It returns a plot linked to `plot` on the
    shared axis, with that axis's width or height fixed and stretch 1:4, as
    `ImageProjectionDisplay` sets up today.
  - `add_menu_action(self, text, slot, checkable: bool = False) -> QAction`
    adds to the plot's view-box menu.

  Also move `frame_rect` (from `image_monitor.py:67`) into this module as a
  module-level function.

- [ ] **Step 1: Write the failing tests** in `tests/test_image_view.py`:
  - `test_a_frame_is_placed_by_its_resolution_and_offset`: `image_item`'s rect
    equals `frame_rect(head, shape)`.
  - `test_frame_returns_the_head_it_was_given`: `frame()[1] is head`.
  - `test_showing_a_frame_does_not_count_as_a_drag`: connect a spy to
    `sig_levels_dragged`; `show_image(img)` twice with different ranges leaves
    it uncalled.
  - `test_dragging_the_levels_reports_them`:
    `lut.setLevels(1, 2); lut.sigLevelChangeFinished.emit(lut)` calls the spy
    with `(1, 2)`.
  - `test_fixed_levels_are_kept`: `show_image(img, levels=(0, 5))` leaves
    `levels() == (0, 5)`.
  - `test_paths_replace_by_key`: `show_paths("a", [p1, p2])` then
    `show_paths("a", [p1])` leaves one curve under `"a"`.
  - `test_a_side_plot_shares_the_image_axis`: the `"top"` plot's x is linked to
    `plot`; the `"left"` plot's y is linked and inverted.
- [ ] **Step 2: Check they fail.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_view.py -q` and
  expect an ImportError.
- [ ] **Step 3: Implement `ImageView`**, porting the drawing code listed above
  from `ImageDisplay.init_gui`, `poll`, `update_mask` and `show_overlay`.
- [ ] **Step 4: Check they pass.** Run the same command and expect a pass.
- [ ] **Step 5: Commit**: "GUI/views: ImageView, the base image viewer".

### Task 3: `ImageApplet` framework class; `ImageDisplay` holds an `ImageView`

**Files:**
- Create: `pytweezer/GUI/image_applet.py`. `ImageApplet(Applet)` takes all of
  `ImageDisplay`'s wiring:
  - the `PropertyAttribute`s at `image_monitor.py:174-185`;
  - the image, mask and overlay clients and `update_subscriptions`;
  - `image_index` and `index_selector`;
  - the sidebar toggle and `DataSidebar`;
  - the view-box menu actions, `set_autoscale`, and the persisted
    `manual_levels`, which are written on `view.sig_levels_dragged`;
  - `poll`, `update_mask` and `update_overlays`, which hand their data to
    `self.view`.

  It also takes `CameraROI`, `camera_roi` and `overlay_paths` from
  `image_monitor.py`. These are applet-side because ROIs persist to Properties
  and overlays come from a stream.
  - `self.view: ImageView` is created *before* the first
    `update_subscriptions()`.
  - It keeps `panel_layout` and `image_layout`, documented as the places an
    applet adds widgets beside or under the image.
- Move: `git mv pytweezer/GUI/viewers/archive/zmq_ROI.py pytweezer/GUI/zmq_roi.py`.
  It is framework now, and only image applets import it. Update its importers.
- Modify: `pytweezer/GUI/viewers/image_monitor.py` becomes
  `class ImageDisplay(ImageApplet)`, plus `main`. Keep the class name, because
  saved applet setups and the launcher refer to it by script.
- Test: update `tests/test_image_monitor.py` to the new paths only. For
  example, `monitor.image_item` becomes `monitor.view.image_item`,
  `monitor.overlay_curves` becomes `monitor.view.overlay_curves`, and the
  `zmq_ROI` import and its monkeypatch target change.

**Interfaces:**
- Consumes: `ImageView`, `frame_rect` (Task 2).
- Produces: `pytweezer.GUI.image_applet.ImageApplet`, with attributes `view`,
  `panel_layout`, `image_layout`, `sidebar`, `imgstream`, `maskstream` and
  `overlayclient`. Subclasses may override `init_gui` and must call
  `super().init_gui()` first; `self.view` exists from then on.

- [ ] **Step 1: Add the failing test** to `tests/test_image_monitor.py`:
  - `test_saved_manual_levels_apply_to_the_first_frame`: Properties hold
    `autoscale=False` and `manual_levels=[10, 20]`. After the first frame is
    polled, `monitor.view.levels() == (10, 20)`.

  Then update the existing tests' attribute paths.
- [ ] **Step 2: Check it fails.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_monitor.py -q`
  and expect failures on the attribute paths.
- [ ] **Step 3: Implement** the Files list above.
- [ ] **Step 4: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_monitor.py tests/test_image_view.py -q`
  and expect a pass.
- [ ] **Step 5: Compare screenshots.** Run
  `tests/applet_screenshots.py <scratchpad>/after` and compare `ImageDisplay.png`
  with the before shot. They should look the same.
- [ ] **Step 6: Commit**: "ImageApplet holds an ImageView; ImageDisplay is a thin applet on it".

### Task 4: The image applets stop inheriting `ImageDisplay`

**Files:**
- Modify: `pytweezer/GUI/viewers/image_plot_monitor.py`. `ImageProjectionDisplay`
  becomes `(ImageApplet)`. Its two plots come from
  `self.view.add_side_plot("top")` and `("left")`, created in `init_gui` after
  `super().init_gui()`. Remove the `top_plot is None` guard and the
  pre-`super()` setup it needed. The projection curves stay here as its
  specialisation.
- Modify: `pytweezer/GUI/viewers/tweezer_monitor.py`. `TweezerDisplay` becomes
  `(ImageApplet)`, using `SiteOverlay(self.view.plot)`. Change detection uses
  `self.view.frame()[1]`. `SiteOverlay` stays here as its specialisation.
- Modify: `pytweezer/GUI/viewers/camera_applet.py`. `CameraApplet` becomes
  `(ImageApplet)`:
  - mouse readout from `self.view.sig_mouse_moved` and `self.view.image_item`;
  - ROIs from `self.view.rois()` filtered for `CameraROI`;
  - frame state from `self.view.frame()`;
  - the camera switch calls `self.view.clear_image()`.

  Keep the sidebar in `panel_layout`, the status bar in `image_layout`, and
  its `_show_sidebar` default override.
- Test: update paths in `tests/test_image_monitor.py`
  (`top_plot`/`left_plot`/`projection_curves`) and `tests/test_tweezer_monitor.py`.
  Add one test to `tests/test_image_monitor.py`:
  - `test_saved_projection_streams_subscribe_at_start`: Properties hold
    `projectionstreams=["p"]` before construction. Construction does not
    raise, and both projection curves exist.

- [ ] **Step 1: Add the new test and update the paths.** Then run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_image_monitor.py tests/test_tweezer_monitor.py -q`
  and expect failures.
- [ ] **Step 2: Implement** the three ports above.
- [ ] **Step 3: Run the tests.** Repeat Step 1's command and expect a pass.
  Then check
  `grep -rn "(ImageDisplay)\|import ImageDisplay" pytweezer/GUI` returns
  nothing.
- [ ] **Step 4: Compare screenshots** for `ImageProjectionDisplay`,
  `TweezerDisplay` and `CameraApplet` against the before shots. Projection
  plots must align with the image edges.
- [ ] **Step 5: Commit**: "Projection, tweezer and camera applets build on ImageApplet, not ImageDisplay".

### Task 5: `SlmMonitor` on `ImageView`

**Files:**
- Modify: `pytweezer/GUI/viewers/slm_monitor.py`. Replace `ImagePanel` with
  `ImageView(colour_bar="bar", ...)` per panel. Use `set_colour_map`, fixed
  `levels`, and `set_bar_ticks` for the phase panel's π ticks. Remove its
  duplicate of `frame_rect`. Its colour-map lookup and levels policy (the phase
  period, or `max`) stay in the applet.
- Test: update `tests/test_slm.py` only if it reaches `ImagePanel`. Per the
  survey it does not, so expect no change.

- [ ] **Step 1: Implement the port.**
- [ ] **Step 2: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_slm.py tests/test_image_view.py -q`
  and expect a pass.
- [ ] **Step 3: Compare screenshots** of `SlmMonitor`: same colour maps, bars
  and ticks.
- [ ] **Step 4: Commit**: "SlmMonitor draws its panels with ImageView".

### Task 6: `LinePlot` base view, `LivePlotApplet`, and the plot applets

**Files:**
- Create: `pytweezer/GUI/views/line_plot.py`
- Create: `pytweezer/GUI/live_plot_applet.py`. It takes `LivePlotApplet(Applet)`,
  plus `ResultHistory`, `rid_title`, `numeric_fields` and `grid_position`,
  from `viewers/live_plot.py`. `PlotPanel` keeps its wiring (DataClient,
  Properties reads, `poll`, `seen_fields`, deques) and draws through a
  `LinePlot`. `LivePlotApplet` keeps the feed, the grid, `arrange(keys)` (the
  hook), `add_plot`/`remove_plot`, `clear_history`, `poll` and the
  `PropertyAttribute`s at `live_plot.py:303-306`.
- Modify: `pytweezer/GUI/viewers/live_plot.py` becomes `class LivePlot(LivePlotApplet)`
  plus `main`. `viewers/stacked_plotter.py` becomes
  `class StackedPlotter(LivePlotApplet)`, keeping its `arrange` override and
  `_plots` default, and uses `LinePlot.link_x`, `set_y_axis_width` and
  `show_x_values`. `viewers/scalar_history.py` draws on a `LinePlot`, keeping
  its readout label and deques.
- Test: `tests/test_line_plot.py` (create). In `tests/test_live_plot.py`,
  change only the `ResultHistory` import.

**Interfaces:**
- Produces: `class LinePlot(pg.PlotWidget)`, with the grid at alpha 0.15, the
  legend offset at (-10, 10) and `style_plot` applied. Its methods:
  - `__init__(self, parent=None, *, legend: bool = True, x_label: str = "",
    y_label: str = "", title: str | None = None)`
  - `curve(self, key: str, colour, *, markers: bool = False, **style) ->
    pg.PlotDataItem` gets or creates a curve: lines, or `"o"` markers.
  - `set_curve(self, key, x, y)`, `remove_curve(self, key)` and
    `clear_curves(self)`
  - `set_x_label(self, text)` and `set_title(self, text)`
  - `link_x(self, other: "LinePlot | None")`, `set_y_axis_width(self, px:
    int)` and `show_x_values(self, show: bool)`
  - `add_menu_action(self, text, slot) -> QAction`

  Colours always come in from the applet, resolved with `stream_color`.

- [ ] **Step 1: Write the failing tests** in `tests/test_line_plot.py`:
  - `test_curve_is_reused_by_key`: `curve("a", c) is curve("a", c)`.
  - `test_set_curve_draws_the_data`: `getData()` round-trips x and y.
  - `test_markers_draw_points_not_lines`: a markers curve has symbol `"o"` and
    no pen.
  - `test_remove_curve_forgets_the_key`: after `remove_curve("a")`, `curve("a",
    c)` is a new item.
- [ ] **Step 2: Check they fail.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_line_plot.py -q` and
  expect an ImportError.
- [ ] **Step 3: Implement** `LinePlot`, then `LivePlotApplet`, then port the
  three applets.
- [ ] **Step 4: Run the tests.** Run
  `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_line_plot.py tests/test_live_plot.py -q`
  and expect a pass. Then check
  `grep -rn "(LivePlot)\|import LivePlot\b" pytweezer/GUI` returns nothing.
- [ ] **Step 5: Compare screenshots** of `LivePlot`, `StackedPlotter` and
  `ScalarHistory`. Stacked panels must stay aligned.
- [ ] **Step 6: Commit**: "LinePlot base view; LivePlot, StackedPlotter and ScalarHistory build on it".

### Task 7: Skills and docs; full verification

**Files:**
- Modify: `.claude/skills/add-applet/SKILL.md` (around l.242 and l.375–381),
  `docs/notes/applets.md` (the tree at l.36–46) and
  `.claude/skills/pytweezer-gui-internals/SKILL.md`, if it names these classes
  (grep it).

- [ ] **Step 1: Update the add-applet skill.**
  - An image applet subclasses `ImageApplet` and draws through `self.view`
    (`ImageView`). A data-plot applet subclasses `LivePlotApplet` or uses a
    `LinePlot`.
  - Add a widget beside or under the image through `panel_layout` and
    `image_layout`.
  - A persisted ROI is `pytweezer.GUI.zmq_roi.zmq_ROI`, passed to
    `view.add_roi`.
  - Remove the "subclass `ImageDisplay`" guidance and the phase-1 note saying
    the image viewer and line plot would arrive in phase 2.
- [ ] **Step 2: Update `docs/notes/applets.md`'s tree** to the new classes.
- [ ] **Step 3: Verify everything.**
  - `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q`: all pass.
  - `uv run ruff check .` and `uv run ruff format --check .`: clean.
  - `cd docs && uv run make html`: builds.
- [ ] **Step 4: Check the real GUI.** Following the `run-pytweezer` skill,
  launch `pytweezer-server` offscreen, open each applet from the applet
  launcher, and screenshot it. Each must start without error.
- [ ] **Step 5: Commit** only these docs and skills, then push to
  `claude/code-restructuring-clarity-c889qv`.
- [ ] **Step 6: Report** to the user:
  - the screenshot comparisons;
  - the dead code left in place (listed in Global Constraints);
  - that `zmq_ROI` moved to `pytweezer.GUI.zmq_roi`, which breaks any notebook
    importing it from `GUI.viewers.archive`.
