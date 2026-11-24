# Tweezer Layout Applet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An applet to draw a named tweezer layout on the focal plane, stored on the Experiment Manager, and an `SlmTweezerLayout` experiment that puts a named layout on the Rb SLM.

**Architecture:** A Qt-free layout module (`pytweezer/experiment/layouts.py`) holds the model, presets, lattice snapping and a `ModelStore`-backed book. The Experiment Manager serves it like recipes. The rig gains `make_layout`, with `make_array` reduced to rectangle positions plus the same path. The experiments share a base class. The applet is a pure editor that only reads/writes layouts and submits experiments.

**Tech Stack:** Python 3, pydantic, numpy, PyQt6 + pyqtgraph, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-17-tweezer-layout-applet-design.md`

## Global Constraints

- The layout module, the applet and `array_hologram.py` must import without cupy; cupy only inside functions that compute masks.
- The applet never calls the rig or the SLM; it only talks to the Experiment Manager (`layouts`, `save_layout`, `delete_layout`, `last_request`, `submit`).
- `SlmTweezerArray` keeps its name and its arguments `rows`, `columns`, `spacing`, `angle`.
- Coordinates everywhere are µm in the focal plane, relative to the blaze spot. Only the applet's display is turned 180° (both view axes inverted).
- Only **on** sites go to the rig. Site order = layout list order.
- UK spelling, Google-style docstrings, `get_logger` (never `logging.getLogger`), no change-history in docstrings, comment only the surprising.
- Load skills before each area: `add-experiment` (Tasks 2, 4), `pytweezer-device-framework` (Task 3), `add-applet` + `pytweezer-gui-design` + `pytweezer-gui-internals` (Tasks 5, 6), `write-tests` (every task).
- Commit per task: `uv run ruff check --fix <files>`, `uv run ruff format <files>`, `git add` whole files, `git commit`. Never `--no-verify`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push after each task (branch `tweezer-layout-applet`).
- Qt tests use the `qapp` fixture from `tests/conftest.py` (no pytest-qt); tests and smoke runs use `QT_QPA_PLATFORM=offscreen`. Never open a window on the live display.

**Calls made while planning (deviations from the spec, all simplifications):**
- `array_hologram.array_hologram()` is replaced by `layout_hologram()`; the rig computes rectangle positions with `layouts.rectangle`. There's no second caller to keep the wrapper for.
- No `layouts_version` in the manager snapshot: the applet refreshes its list whenever the layout combo opens and after its own saves/deletes. That's enough for a handful of PCs, without subscription plumbing.
- Undo (Ctrl+Z), left to the builder by the spec, is not in v1.
- Focal-grid margin = `2 * min(d, 10 µm) + 3 * dx_f`, where `d` is the median nearest-neighbour distance (0 for one site). Identical to today for rectangles up to 10 µm spacing.

## Review Focus

1. **A layout name typed with stray spaces or the wrong case** in the experiment form. Expected: `get_layout` strips whitespace; a miss raises `KeyError` that names close matches. The test is in Task 2.
2. **A layout with every site switched off** submitted or saved. Expected: the manager refuses to save it ("no site is on"), and `SlmTweezerLayout.prepare` fails the same way if one somehow exists. Tests are in Tasks 2 and 4.
3. **Two sites at the same position** (a freehand double-tap, or a grid tap on an occupied point). Expected: the second tap toggles the existing site and never adds a duplicate. The test is in Task 5.
4. **Applying a preset or opening a layout over unsaved edits.** Expected: the user is asked first, and declining keeps the edits. The test is in Task 6 (`dirty` flag and confirm hook).
5. **Experiment Manager down when the applet starts.** Expected: the applet opens with an empty, editable layout, a status line saying the manager is unreachable, and no reachable-field outline. It must not crash. The test is in Task 6.

---

### Task 1: Layout model, presets, lattice snapping, store

**Files:**
- Create: `pytweezer/experiment/layouts.py`
- Modify: `pytweezer/helper_scripts/phasemask.py` (`generate_kagome_lattice` in all three generator classes, ~lines 286, 1468, 2428)
- Test: `tests/test_tweezer_layouts.py`

**Interfaces:**
- Produces:
  - `class LayoutError(ValueError)`
  - `class Site(BaseModel)`: `x_um: float`, `y_um: float`, `on: bool = True`
  - `class Lattice(BaseModel)`: `kind: Literal["rectangle", "kagome"]`, `spacing_um: float`, `angle_deg: float = 0.0`, `rows: int | None = None`, `columns: int | None = None`, `sites: int | None = None`
  - `class Layout(BaseModel)`: `name: str` (validator strips; empty → error), `sites: list[Site]`, `lattice: Lattice | None = None`, `min_spacing_um: float = 3.0`, `submitter: str = ""`, `saved_at: datetime = Field(default_factory=now)`; property `on_sites -> tuple[np.ndarray, np.ndarray]` (x_um, y_um of on sites, list order)
  - `rectangle(rows: int, columns: int, spacing_um: float, angle_deg: float = 0.0) -> list[Site]`
  - `kagome(sites: int, spacing_um: float, angle_deg: float = 0.0) -> list[Site]`
  - `preset_sites(lattice: Lattice) -> list[Site]` (dispatches on `kind`)
  - `lattice_points(lattice: Lattice, x_range_um: tuple[float, float], y_range_um: tuple[float, float]) -> np.ndarray` shape `(n, 2)`
  - `nearest_lattice_point(lattice: Lattice, x_um: float, y_um: float) -> tuple[float, float]`
  - `close_pairs(sites: list[Site], min_spacing_um: float) -> list[tuple[int, int]]` (i < j, all sites, on or off)
  - `class LayoutBook`: `get(name) -> Layout` (raises `LayoutError`), `save(layout, overwrite=False) -> Layout` (stamps `saved_at`), `delete(name)`, `find() -> list[Layout]` sorted by name
  - `class LayoutState(BaseModel)`: `schema_version: int = 1`, `layouts: list[Layout]`; `class LayoutStore(ModelStore)` with `model = LayoutState`

- [ ] **Step 1: Write the failing tests** in `tests/test_tweezer_layouts.py`:

```python
def test_rectangle_matches_the_generators_positions_and_order():
    sites = rectangle(2, 3, 5.0, angle_deg=2.5)
    # generate_weighted_array: linspace over columns/rows, meshgrid, row-major, then rotate CCW
    xs = np.linspace(-5, 5, 3); ys = np.linspace(-2.5, 2.5, 2)
    X, Y = np.meshgrid(xs, ys); a = np.radians(2.5)
    expected_x = X.ravel() * np.cos(a) - Y.ravel() * np.sin(a)
    expected_y = X.ravel() * np.sin(a) + Y.ravel() * np.cos(a)
    np.testing.assert_allclose([s.x_um for s in sites], expected_x)
    np.testing.assert_allclose([s.y_um for s in sites], expected_y)
    assert all(s.on for s in sites)

def test_kagome_sites_sit_on_the_kagome_lattice_with_nearest_neighbours_one_spacing_apart():
    sites = kagome(30, 4.0)
    # every site's nearest neighbour is 4.0 µm away; and every site is a lattice point
    lattice = Lattice(kind="kagome", spacing_um=4.0)
    for s in sites:
        assert nearest_lattice_point(lattice, s.x_um, s.y_um) == pytest.approx((s.x_um, s.y_um))

@pytest.mark.parametrize("rows, columns", [(2, 3), (3, 3), (4, 2)])
def test_rectangle_preset_sites_are_points_of_its_lattice(rows, columns):
    lattice = Lattice(kind="rectangle", spacing_um=5.0, angle_deg=3.0, rows=rows, columns=columns)
    for s in preset_sites(lattice):
        assert nearest_lattice_point(lattice, s.x_um + 1.0, s.y_um - 1.0) == pytest.approx((s.x_um, s.y_um))

def test_lattice_points_cover_the_window_and_nothing_outside():
    pts = lattice_points(Lattice(kind="rectangle", spacing_um=5.0, rows=3, columns=3), (-11, 11), (-6, 6))
    assert len(pts) == 5 * 3 and np.all(np.abs(pts[:, 0]) <= 11) and np.all(np.abs(pts[:, 1]) <= 6)

def test_close_pairs_finds_only_sites_nearer_than_the_minimum():
    sites = [Site(x_um=0, y_um=0), Site(x_um=2, y_um=0, on=False), Site(x_um=10, y_um=0)]
    assert close_pairs(sites, 3.0) == [(0, 1)]

def test_on_sites_keeps_list_order_and_drops_off_sites(): ...  # Layout with on/off/on → x of sites 0 and 2

def test_a_name_is_stripped_and_cannot_be_empty(): ...  # Layout(name="  a ") → "a"; Layout(name=" ") raises ValidationError

def test_book_refuses_to_overwrite_unless_asked_and_find_sorts_by_name(): ...
    # save "b", save "a"; save "a" again → LayoutError "already exists"; overwrite=True replaces; find() names == ["a", "b"]

def test_store_round_trip(tmp_path): ...  # LayoutStore(tmp_path / "layouts.json").save(state); load() == state
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_tweezer_layouts.py -q`
Expected: collection error (`ModuleNotFoundError: pytweezer.experiment.layouts`).

- [ ] **Step 3: Implement `pytweezer/experiment/layouts.py`**

Mirror `recipes.py` for the book and store (`ModelStore` and `now` from `pytweezer.experiment.queue`). Pure numpy, no cupy.

- `rectangle` reproduces `generate_weighted_array` plus `rotate_coordinates` (counter-clockwise, about the origin), row-major with y slowest.
- `kagome` is the body of `generate_kagome_lattice` (centres `m·a1 + n·a2`, `a1 = (2s, 0)`, `a2 = (s, √3 s)`, basis `(s,0)`, `(s/2, √3 s/2)`, `(−s/2, √3 s/2)`, keep the closed shell nearest the requested count), plus the same rotation. Drop the `print`.
- `lattice_points`, rectangle: points `R(θ)·((i + ox)·s, (j + oy)·s)` for integer i, j, where `ox = 0` if `columns` is odd else `0.5` (and `oy` likewise with `rows`), so the lattice passes through the preset's sites.
- `lattice_points`, kagome: the three basis offsets over the centre lattice, rotated.
- In both cases, enumerate integer indices over the window's bounding box in lattice coordinates (unrotate the window corners), then filter to the window.
- `nearest_lattice_point` takes the nearest point of `lattice_points` in a window of ±2 spacings around the tap.

- [ ] **Step 4: Make the three `generate_kagome_lattice` methods call `layouts.kagome`**

Each returns `x_pos, y_pos` as before. `OptimisationBasedPhasemaskGeneratorGPU3D`'s version, if its signature returns `z` too, returns `np.zeros_like(x_pos)` for it, so the return shape stays unchanged.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tweezer_layouts.py -q`
Expected: all pass.

- [ ] **Step 6: Commit** — `git add pytweezer/experiment/layouts.py pytweezer/helper_scripts/phasemask.py tests/test_tweezer_layouts.py`, message "Add tweezer layout model, presets and store".

---

### Task 2: Layouts on the Experiment Manager and the client API

**Files:**
- Modify: `pytweezer/servers/experiment_manager.py` (store creation ~line 120, commands next to `_cmd_recipes` ~line 297)
- Modify: `pytweezer/experiment/client.py` (client methods next to `recipes()` ~line 182; module functions next to `recipes()` ~line 424)
- Test: `tests/test_experiment_manager.py`

**Interfaces:**
- Consumes: Task 1 `Layout`, `LayoutBook`, `LayoutStore`, `LayoutError`.
- Produces:
  - Manager commands: `layouts` → `{"layouts": [layout json, …]}`; `save_layout` (`layout`, `overwrite`); `delete_layout` (`name`).
  - `ExperimentManagerClient.layouts() -> list[Layout]`, `.save_layout(layout: Layout, overwrite: bool = False) -> None`, `.delete_layout(name: str) -> None`
  - Module functions in `client.py`: `layouts(*, client=None) -> list[Layout]`, `get_layout(name: str, /, *, client=None) -> Layout`, `save_layout(layout: Layout, /, *, overwrite: bool = False, client=None) -> None`, `delete_layout(name: str, /, *, client=None) -> None`

- [ ] **Step 1: Write the failing tests** in `tests/test_experiment_manager.py`, after the recipe tests, using `make_manager` and `in_process_client`:

```python
def test_layouts_round_trip_and_survive_a_restart(make_manager):
    manager = make_manager()
    client = in_process_client(manager)
    save_layout(Layout(name="ring", sites=[Site(x_um=0, y_um=0), Site(x_um=5, y_um=0, on=False)]), client=client)
    [saved] = layouts(client=in_process_client(make_manager()))  # a restarted manager on the same root
    assert saved.name == "ring" and [s.on for s in saved.sites] == [True, False]
    assert "@" in saved.submitter
    with pytest.raises(ManagerError, match="already exists"):
        save_layout(Layout(name="ring", sites=[Site(x_um=1, y_um=1)]), client=client)
    save_layout(Layout(name="ring", sites=[Site(x_um=1, y_um=1)]), overwrite=True, client=client)
    assert get_layout(" ring ", client=client).sites[0].x_um == 1
    delete_layout("ring", client=client)
    assert layouts(client=client) == []

def test_a_layout_with_no_site_on_is_refused(make_manager):
    client = in_process_client(make_manager())
    with pytest.raises(ManagerError, match="no site is on"):
        save_layout(Layout(name="dark", sites=[Site(x_um=0, y_um=0, on=False)]), client=client)

def test_get_layout_names_close_matches(make_manager):
    client = in_process_client(make_manager())
    save_layout(Layout(name="checkerboard", sites=[Site(x_um=0, y_um=0)]), client=client)
    with pytest.raises(KeyError, match="checkerboard"):
        get_layout("checkerbord", client=client)
```

(`make_manager()` called again gives a manager on the same root, as `test_recipes_are_filled_in_published_and_survive_a_restart` relies on.)

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_experiment_manager.py -q -k layout`
Expected: `ImportError` for `save_layout`.

- [ ] **Step 3: Implement the manager side**

- `self.layout_store = LayoutStore(self.root / "layouts.json")` and `self.layouts = LayoutBook(self.layout_store.load())` next to the recipe store.
- `_cmd_save_layout` validates `Layout`, sets `submitter` from the request, and raises `LayoutError("a layout needs at least one site on; no site is on")` when no site is on. It saves via the book and then persists with `self.layout_store.save(self.layouts.state)`, as the recipe path does; follow whatever `_recipes_changed` does for persistence, minus the version bump.
- `LayoutError` must reach the client as a `ManagerError` the same way `RecipeError` does: check how the manager maps `RecipeError` and add `LayoutError` alongside it.
- The client's `save_layout` sends `submitter=submitter_name()`.
- `get_layout` strips the name, fetches all layouts, and raises `KeyError(f"no layout {name!r}; did you mean …?")` using `difflib.get_close_matches`, as `get_saved_parameter` does.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_experiment_manager.py tests/test_tweezer_layouts.py -q`
Expected: all pass.

- [ ] **Step 5: Update the `add-experiment` skill** with a short "Tweezer layouts" line: the four client functions and that `SlmTweezerLayout` (Task 4) takes `layout=`. This is a minor edit, so it goes in this task's commit.

- [ ] **Step 6: Commit** — "Serve tweezer layouts from the Experiment Manager".

---

### Task 3: Holograms for arbitrary layouts; rig `make_layout`

**Files:**
- Modify: `pytweezer/helper_scripts/array_hologram.py`
- Modify: `pytweezer/coordinators/rearrangement.py` (`make_array` ~line 255, `get_array` docstring, module docstring)
- Test: `tests/test_rearrangement.py` (the `fake_gpu` fixture ~line 50, new tests after line 119)

**Interfaces:**
- Consumes: Task 1 `layouts.rectangle`.
- Produces:
  - `array_hologram.layout_hologram(generator, x_um: np.ndarray, y_um: np.ndarray, *, envelope: dict | None, blaze_um: tuple[float, float], fresnel_focal_length_mm: float, zernike: bool, pupil_radius_um: float, pupil_dump_um: tuple[float, float], iterations: int, damping: float, phase_randomness: float) -> dict`. It returns the same keys `array_hologram()` returns today. `array_hologram()` is deleted.
  - `array_hologram.focal_margin_um(x_um, y_um, focal_pixel_um: float) -> float` = `2 * min(d, 10.0) + 3 * focal_pixel_um`, where `d` is the median nearest-neighbour distance, or 0 for one site.
  - `array_hologram.reachable_field_um(saved: dict) -> tuple[float, float, float, float]` = `(x0, y0, width, height)` in µm, relative to the blaze spot. The square of side `λ f / pitch` (from `slm_wavelength`, `slm_focal_length`, `slm_pixel_pitch`) centred on the zeroth order at `(−blaze_x, −blaze_y)` (from `slm_blaze_x/y`, in metres → µm).
  - `Rearrangement.make_layout(x_um: list[float], y_um: list[float], *, name: str, generator_settings, envelope, blaze_um, fresnel_focal_length_mm, zernike, pupil_radius_um, pupil_dump_um, iterations, damping, phase_randomness) -> dict`. It writes the mask, and `get_array()` then carries `sites`, `x_um`, `y_um`, `weights` and `layout=name`.
  - `Rearrangement.make_array(...)`: unchanged signature and return. It now goes through the same private path with positions from `layouts.rectangle`, and still remembers `rows`/`columns`/`spacing_um`/`angle_deg`.

- [ ] **Step 1: Change `fake_gpu` and write the failing tests**

Change `fake_gpu` to patch `array_hologram.layout_hologram` with a fake taking `(generator, x_um, y_um, **optics)`. It returns a mask that depends on the positions (e.g. `np.full((1024, 1024), len(x_um), np.uint8)`), plus `x_um`, `y_um`, `weights=np.ones(len(x_um))`. Then:

```python
def test_rig_puts_a_named_layout_on_the_slm_and_reports_it(rig, fake_gpu):
    hologram = rig.make_layout([0.0, 5.0, 0.0], [0.0, 0.0, 5.0], name="ell", **OPTICS)
    np.testing.assert_array_equal(rig.slm.last_mask, hologram["mask"])
    array = rig.get_array()
    assert array["sites"] == 3 and array["layout"] == "ell"
    assert array["x_um"] == [0.0, 5.0, 0.0] and "rows" not in array

def test_rig_array_sites_are_the_rectangle_preset_positions(rig, fake_gpu):
    rig.make_array(**ARRAY, **OPTICS)
    expected = layouts.rectangle(3, 4, 5.0, 2.5)
    np.testing.assert_allclose(rig.get_array()["x_um"], [s.x_um for s in expected])

def test_focal_margin_is_unchanged_for_a_rectangle_and_capped_for_sparse_sites():
    x, y = np.meshgrid(np.arange(3) * 5.0, np.arange(3) * 5.0)
    assert array_hologram.focal_margin_um(x.ravel(), y.ravel(), 0.9) == pytest.approx(2 * 5.0 + 2.7)
    assert array_hologram.focal_margin_um(np.array([0.0, 300.0]), np.array([0.0, 0.0]), 0.9) == pytest.approx(20.0 + 2.7)
    assert array_hologram.focal_margin_um(np.array([1.0]), np.array([1.0]), 0.9) == pytest.approx(2.7)

def test_reachable_field_is_lambda_f_over_pitch_centred_on_the_zeroth_order():
    saved = {"slm_wavelength": 0.852e-6, "slm_focal_length": 17.3e-3, "slm_pixel_pitch": 17e-6,
             "slm_blaze_x": 48e-6, "slm_blaze_y": 4e-6}
    x0, y0, w, h = array_hologram.reachable_field_um(saved)
    side = 0.852 * 17300 / 17
    assert (w, h) == pytest.approx((side, side))
    assert (x0 + w / 2, y0 + h / 2) == pytest.approx((-48.0, -4.0))
```

Keep the existing three `get_array` tests passing unchanged.

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_rearrangement.py -q`
Expected: the new tests fail (`AttributeError: make_layout` / `focal_margin_um`).

- [ ] **Step 3: Implement**

- `layout_hologram` is today's `array_hologram` body from the optimisation onwards.
- It builds weights as ones, or `envelope_weights(envelope, x + bx, y + by)`, and draws initial phases with `np.random.rand(n) * phase_randomness * 2π`, matching `generate_weighted_array`.
- It passes `[weights, phases, x_um, y_um]` to `superposition_optimization`, and uses `focal_margin_um` for the grid.
- In the rig, factor the generator-cache/settings block and the `update_mask` + `_remember_array` into one private method used by both public ones.
- Update the module docstrings (rig, `array_hologram`) to describe `make_layout`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_rearrangement.py tests/test_slm_tweezer_array.py tests/test_tweezer_layouts.py -q`
Expected: all pass. GPU-only tests may be skipped. If `test_slm_tweezer_array.py`'s GPU tests call `array_hologram.array_hologram`, port them to `layout_hologram` with `layouts.rectangle` positions.

- [ ] **Step 5: Commit** — "Make SLM holograms for any layout; rig make_layout".

---

### Task 4: Shared base class and `SlmTweezerLayout`

**Files:**
- Modify: `pytweezer/experiments/slm_tweezer_array.py`
- Test: `tests/test_slm_tweezer_array.py`

**Interfaces:**
- Consumes: Task 2 `get_layout`; Task 3 `make_layout`.
- Produces:
  - `SlmHologramExperiment(Experiment)`, with no `run_point`, so it stays out of the Experiments tab. It holds `rig`, `compensate_envelope`, the Optics/Pupil/Optimisation arguments, `sleep_time`, `prepare()` (unchanged behaviour), `broadcast()`, and a helper `optics() -> dict` returning the keyword set shared by `make_array`/`make_layout` (generator settings, envelope or `None`, blaze, Fresnel, Zernike, pupil, iterations, damping, phase randomness).
  - `SlmTweezerArray(SlmHologramExperiment)`: Array group `rows`, `columns`, `spacing`, `angle`; `run_point` as today.
  - `SlmTweezerLayout(SlmHologramExperiment)`: `layout = Text("", group="Array", tooltip="name of a saved tweezer layout")`.
    - `prepare()` calls `super().prepare()` and `self.layout_model = get_layout(self.layout)`. It raises `ValueError("layout … has no site on")` if `on_sites` is empty, and records `"layout"` as `self.layout_model.model_dump(mode="json")`.
    - `run_point` calls `self.rig.make_layout(list(x), list(y), name=self.layout_model.name, **self.optics())`, then broadcasts and records `phasemask`, `uniformity`, `min_max_ratio` and `mse` exactly as `SlmTweezerArray` does.

- [ ] **Step 1: Write the failing tests**

Give `RecordingRig` a `make_layout(self, x_um, y_um, *, name, **optics)` that records and returns the same fake hologram as `make_array`. Monkeypatch `slm_tweezer_array.get_layout` in the fixture to look up a dict of `Layout`s. Then:

```python
def test_layout_experiment_puts_the_on_sites_of_the_named_layout_on_the_rig(rig_and_streams, layouts_by_name):
    layouts_by_name["ell"] = Layout(name="ell", sites=[Site(x_um=0, y_um=0), Site(x_um=5, y_um=0, on=False), Site(x_um=0, y_um=5)])
    rig, _ = rig_and_streams
    run_local(SlmTweezerLayout, layout="ell", iterations=5)
    [made] = rig.made
    assert made["x_um"] == [0, 0] and made["y_um"] == [0, 5] and made["name"] == "ell"

def test_layout_experiment_records_the_whole_layout_including_off_sites(rig_and_streams, layouts_by_name): ...
    # measurement["layout"]["sites"] has 3 entries, the middle one on == False

def test_layout_experiment_with_no_site_on_fails_the_run(rig_and_streams, layouts_by_name): ...
    # all sites off → run fails with "no site on"

def test_both_experiments_ask_the_rig_with_the_same_optics(rig_and_streams, layouts_by_name): ...
    # run each once; the made-dicts minus position/geometry keys are equal
```

Also check `runnable_experiments(slm_tweezer_array)` lists exactly `SlmTweezerArray` and `SlmTweezerLayout`.

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_slm_tweezer_array.py -q`
Expected: the new tests fail on import of `SlmTweezerLayout`.

- [ ] **Step 3: Implement** as in Interfaces. Update the module docstring: two experiments, same broadcast.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_slm_tweezer_array.py tests/test_experiment_manager.py -q`
Expected: all pass, including the existing `SlmTweezerArray` tests unchanged.

- [ ] **Step 5: Commit** — "Add SlmTweezerLayout experiment".

---

### Task 5: Qt-free layout editor logic

**Files:**
- Create: `pytweezer/GUI/viewers/tweezer_layout.py` (this task adds only `LayoutEditor` at the top of the module; the widget comes in Task 6)
- Test: `tests/test_tweezer_layout_applet.py`

**Interfaces:**
- Consumes: Task 1 (`Site`, `Lattice`, `Layout`, `preset_sites`, `nearest_lattice_point`, `close_pairs`).
- Produces `class LayoutEditor`. It holds the layout being edited and imports no Qt.
  - Attributes: `sites: list[Site]`, `lattice: Lattice | None`, `mode: Literal["grid", "freehand"]`, `min_spacing_um: float = 3.0`, `name: str | None` (the open layout), `dirty: bool`.
  - `tap(x_um: float, y_um: float, pick_radius_um: float) -> None`
    1. A site within `max(pick_radius_um, min_spacing_um / 2)` toggles (the nearest one).
    2. Otherwise, in grid mode: snap with `nearest_lattice_point`; if a site is already within 1e-6 µm of that point, toggle it, else append a new on site there.
    3. Otherwise, in freehand mode: if any site is within `min_spacing_um`, toggle the nearest; else append a new on site at the tap.
    4. Grid mode with `lattice is None` does nothing.
  - `apply_preset(lattice: Lattice) -> None`: replaces the sites, sets `lattice`, sets `dirty`.
  - `remove_off_sites() -> None`
  - `load(layout: Layout) -> None`: clears `dirty`. `to_layout(name: str) -> Layout`.
  - `close_site_indices() -> set[int]`, `counts() -> tuple[int, int]` (on, off).

- [ ] **Step 1: Write the failing tests**

```python
def test_tapping_a_site_toggles_it_and_marks_the_layout_changed(): ...
def test_grid_tap_snaps_to_the_lattice_and_a_second_tap_there_toggles_instead_of_duplicating(): ...
def test_freehand_tap_lands_where_tapped(): ...
def test_freehand_tap_within_min_spacing_toggles_the_nearest_site_instead_of_adding(): ...
def test_grid_mode_without_a_lattice_ignores_taps(): ...
def test_remove_off_sites_keeps_on_sites_in_order(): ...
def test_round_trip_through_a_layout_keeps_lattice_min_spacing_and_on_flags(): ...
def test_close_sites_are_reported_for_highlighting(): ...
```

Use concrete numbers: a rectangle lattice of spacing 5 µm, a tap at (2.4, 0.3) snapping to (2.5, 0) on a 2×2 preset; min spacing 3 µm.

- [ ] **Step 2: Run to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_tweezer_layout_applet.py -q`
Expected: import error.

- [ ] **Step 3: Implement `LayoutEditor`.**

- [ ] **Step 4: Run tests.** Expected: pass.

- [ ] **Step 5: Commit** — "Add tweezer layout editor logic".

---

### Task 6: The Tweezer Layout applet

**Files:**
- Modify: `pytweezer/GUI/viewers/tweezer_layout.py` (add `TweezerLayoutApplet(Applet)`, `main`, `__main__`)
- Modify: `pytweezer/GUI/applet_launcher.py` (`DEFAULT_APPLETS`: `{"name": "Tweezer Layout", "script": "pytweezer/GUI/viewers/tweezer_layout.py", "description": "Draw a tweezer array and save it as a named layout"}`)
- Test: `tests/test_tweezer_layout_applet.py`

**Interfaces:**
- Consumes: Task 5 `LayoutEditor`. Task 2 `layouts`, `save_layout`, `delete_layout`. `ExperimentManagerClient().last_request(module, class_name)`. `submit("pytweezer.experiments.slm_tweezer_array:SlmTweezerLayout", **args)`. Task 3 `reachable_field_um`. `get_saved_parameters`.
- Produces: `TweezerLayoutApplet`, with `stream_category` unused and `poll_interval = 0`. The class has two hooks that tests override: `confirm(question: str) -> bool` (a `QMessageBox` in production) and `ask_name(current: str) -> str | None` (a `QInputDialog`).

Layout follows the spec's sketch. Read `pytweezer-gui-design` for wording, colours from `theme.py`, button rules and the status line.

- **Canvas.** A `pg.PlotWidget` with aspect locked, labels "x" / "y" in µm, and `invertX(True)` plus `invertY(True)` on the view box (the 180° display turn). Sites are one `ScatterPlotItem`: filled brush for on, no brush for off, red pen for `close_site_indices()`. The lattice dots in view (Grid mode) are a second faint scatter, refreshed on `sigRangeChanged`. The reachable field is a dashed `QGraphicsRectItem`.
- **Taps.** Connect `scene().sigMouseClicked`. Map with `vb.mapSceneToView` and ignore drags. `pick_radius_um` is 12 screen px converted with `vb.viewPixelSize()`. Then call `editor.tap` and redraw.
- **Toolbar.** Layout combo: refreshed from `layouts()` in its `showPopup`, and opening an entry asks if dirty. Save, Save as…, Delete. Preset kind combo with stacked rectangle (rows, columns, spacing, angle) and kagome (sites, spacing, angle) fields, plus Apply (asks if dirty). Grid/Freehand radio buttons. Min spacing spin box in µm, enabled in Freehand. On/off counts. Zoom −, +, Fit.
- **Buttons.** "Remove off sites" and "Put on SLM".
- **Saving.** Save with close pairs asks "N pairs of sites are closer than the minimum spacing. Save anyway?". A manager error goes to the status line, never as an exception. The window title is `name` with `*` when dirty.
- **Put on SLM.** If dirty, asks to save first, and aborts if declined.
  - Arguments come from `last_request(module, "SlmTweezerLayout")`; otherwise the args of `last_request(module, "SlmTweezerArray")` that `SlmTweezerLayout`'s schema also declares; otherwise none.
  - It then sets `layout=name`, calls `submit`, and shows "Queued run <rid>" in the status line.
- **Manager down at start.** If `layouts()` or `get_saved_parameters()` raises, the applet logs it with `get_logger`, puts "Experiment Manager unreachable: layouts can't be listed or saved" in the status line, leaves out the field outline, and stays editable.

- [ ] **Step 1: Write the failing tests** (offscreen; monkeypatch the client functions in the applet module; `Properties` as other applet tests do; see `tests/test_tweezer_monitor.py` for the pattern):

```python
def test_applet_opens_editable_when_the_manager_is_down(qapp, monkeypatch): ...
    # layouts and get_saved_parameters raise; applet constructs; status text contains "unreachable"; no field item

def test_applying_a_preset_over_unsaved_edits_asks_first_and_keeps_them_on_no(qapp, monkeypatch): ...
    # make dirty, confirm → False, apply preset → sites unchanged

def test_put_on_slm_submits_the_layout_with_the_last_runs_settings(qapp, monkeypatch): ...
    # last_request(SlmTweezerLayout) None; last_request(SlmTweezerArray) has args {rows:3, zernike:True, iterations:50};
    # captured submit args == {zernike:True, iterations:50, layout:"ring", ...} and contain no "rows"

def test_view_is_turned_180_degrees(qapp, monkeypatch): ...
    # vb.state["xInverted"] and vb.state["yInverted"] are both True
```

- [ ] **Step 2: Run to see them fail.** Expected: `AttributeError`/import of `TweezerLayoutApplet`.

- [ ] **Step 3: Implement the applet and the launcher entry.**

- [ ] **Step 4: Run tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_tweezer_layout_applet.py tests/test_applet_launcher.py -q`
Expected: pass.

- [ ] **Step 5: Offscreen smoke run and screenshot** (following the `run-pytweezer` skill, offscreen only)

Construct the applet against simulated defaults, apply a 6×6 rectangle, toggle a few sites, `grab()` to a PNG in the scratchpad. Check it reads clearly against `pytweezer-gui-design`, then iterate.

- [ ] **Step 6: Full suite and lint**

Run: `uv run pytest tests/ -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, clean.

- [ ] **Step 7: Commit and push** — "Add the Tweezer Layout applet". Then open a PR from `tweezer-layout-applet` to `main`. The description summarises the feature, notes that touch pinch-zoom is future work, and ends with the Claude Code attribution line.
