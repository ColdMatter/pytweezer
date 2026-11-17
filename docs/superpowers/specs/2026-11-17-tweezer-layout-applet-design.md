# Tweezer Layout applet — design

## Intent

Draw a tweezer array by clicking (or tapping) on a picture of the focal plane,
save it under a name, and have an experiment put that named array on the Rb SLM.
Today the only way to get an array onto the SLM is `SlmTweezerArray`, which makes
a rotated N×M rectangle; anything else (a missing site, a pattern, a kagome
lattice) means editing code.

## Goals

- An applet that edits a **layout**: a named list of trap sites in the focal
  plane, each on or off.
- Two ways of placing sites: **Grid** (taps snap to a lattice) and **Freehand**
  (taps land where tapped, subject to a minimum spacing).
- Start from presets: **N×M rectangle** and **kagome**, and from any saved layout.
- Layouts saved by name, shared by every PC, and loadable by name from an
  experiment.
- A new experiment, `SlmTweezerLayout`, that puts a named layout on the SLM.
- Usable on a touchscreen as far as taps go.

## Non-goals

- Saving phase masks. The layout is saved; the mask is computed from the
  *current* optics saved parameters every time a layout is put on the SLM, so a
  recalibrated blaze/Zernike/Fresnel never leaves a saved array stale.
- Drawing over the live camera image (needs a camera-pixel → focal-plane-µm
  calibration the code does not have).
- Per-site intensity weights in the editor. Sites are on or off; trap weights
  still come from the measured depth envelope when `compensate_envelope` is set.
- Pinch-to-zoom (pyqtgraph has no gesture support; zoom buttons instead).
  Future work.
- Rearrangement targets. The layout model doesn't preclude it later.

## Anti-goals

- The applet must never write the SLM or call the rig directly. The rig is a
  reserved device; a write behind the queue would trample a running experiment
  and make `get_array()` report an array no experiment put there.
- `SlmTweezerArray` must keep working unchanged for its existing recipes and
  notebooks.

## Approaches considered

1. **Applet as a pure layout editor; layouts stored on the Experiment Manager;
   masks made by an experiment through the queue.** *Chosen.* Shared by every
   PC, respects device reservations, reuses the existing SLM Monitor for
   previews.
2. Applet calls the rig to compute and preview masks live. Rejected: bypasses
   reservations (see anti-goals).
3. Layouts as saved parameters. Rejected: no new storage, but clutters the
   Saved Parameters tab with things that are not calibrations proposed by
   optimisers and accepted by hand.
4. Layout JSON files on the applet's PC. Rejected: an experiment on another
   PC can't read them.

## Design

### Coordinates

Everything in the layout is in µm in the focal plane, relative to the blaze
spot — the same frame `array_hologram` and `Rearrangement.get_array()` already
use for `x_um`/`y_um`.

The canvas is **displayed as the Tweezer Monitor shows the HamCam image**: the
camera mirrors x and the monitors invert y, so the canvas is the focal plane
turned 180° (both plot axes inverted), and a pattern drawn on it looks the way
the atoms will. This is a display transform only — stored coordinates stay
focal-plane µm, and taps are mapped back through it. The camera's small
(~2.5°) rotation is a fitted calibration, not a fixed convention, and is not
applied. Check the flip against `pytweezer/analysis/trap_sites.py` and the
Tweezer Monitor's view settings rather than trusting this paragraph.

### Layout model — `pytweezer/experiment/layouts.py` (new)

Pure Python/numpy (no cupy), importable anywhere. Pydantic models, mirroring
`recipes.py`:

```python
class Site(BaseModel):
    x_um: float
    y_um: float
    on: bool = True

class Lattice(BaseModel):
    kind: Literal["rectangle", "kagome"]
    spacing_um: float
    angle_deg: float = 0.0
    rows: int | None = None       # rectangle
    columns: int | None = None    # rectangle
    sites: int | None = None      # kagome: requested site count

class Layout(BaseModel):
    name: str                     # stripped, non-empty
    sites: list[Site]
    lattice: Lattice | None       # the preset/grid it was drawn on; None if freehand-only
    min_spacing_um: float = 3.0
    submitter: str = ""           # whoever saved it
    saved_at: datetime
```

Also in this module:

- `rectangle(rows, columns, spacing_um, angle_deg) -> list[Site]` — same
  positions and ordering as `generate_weighted_array` (centred, rotated by
  `rotate_coordinates`), so a rectangle preset with every site on reproduces
  `SlmTweezerArray`'s array exactly.
- `kagome(sites, spacing_um, angle_deg) -> list[Site]` — the algorithm of
  `OptimisationBasedPhasemaskGeneratorGPU.generate_kagome_lattice`, moved here
  as a plain function (plus rotation). The method exists on all three
  generator classes in `phasemask.py` (`…GPU`, `…GPU3D` which also returns
  `z`, `…GPUSeidel`); each calls the shared function (`…GPU3D` adds its zero
  `z`).
- `lattice_points(lattice, x_range_um, y_range_um)` — every lattice point in a
  window, for Grid-mode snapping and drawing the faint grid dots.
- **Site order** (*my call*): the on sites go to the rig in list order. Presets
  produce sites in the generator's order (row-major for a rectangle, so it
  matches `SlmTweezerArray`); sites added by tapping are appended; toggling
  never reorders. Site indices in tweezer analyses follow this order, so any
  edit to a layout's on sites means re-running FindTrapSites, as changing the
  rectangle does today.
- `close_pairs(sites, min_spacing_um)` — indices of sites closer than the
  minimum spacing.
- `LayoutBook` / `LayoutStore` — save (create or overwrite by name), find, delete;
  persisted as `layouts.json` with a `schema_version`, exactly as
  `RecipeBook`/`RecipeStore` do (`ModelStore`).

### Experiment Manager

- Owns a `LayoutStore` at `self.root / "layouts.json"` beside
  `recipes.json`, and a `_layouts_version` counter.
- Commands `layouts` (all, with version), `save_layout`, `delete_layout`,
  following `_cmd_recipes`/`_cmd_save_recipe`. `save_layout` refuses an empty
  layout or one with no site on, and refuses to overwrite an existing name
  unless `overwrite=True`.
- The version is added to whatever the manager already publishes for
  `recipes_version`, so the applet can notice another PC's save.

### Client API — `pytweezer/experiment/client.py`

`layouts()`, `get_layout(name)` (KeyError naming close matches, as
`get_saved_parameter` does), `save_layout(layout, *, overwrite=False)`,
`delete_layout(name)` — named like `recipes()`/`save_recipe()`/
`delete_recipe()`, in `client.py` beside them.

### Hologram — `pytweezer/helper_scripts/array_hologram.py`

Split `array_hologram()` into:

- `layout_hologram(generator, x_um, y_um, *, envelope, blaze_um, …,
  phase_randomness)` — everything after the positions are known: weights
  (ones, or the envelope), random initial phases, optimisation, layering,
  pupil, simulation. The focal-grid margin, currently `2 * spacing_um + 3 *
  dx_f`, becomes `2 * d + 3 * dx_f` where `d` is the median nearest-neighbour
  distance of the sites (0 for one site) — identical to today for a rectangle
  — capped at `2 * min_spacing`-style bound so a sparse freehand layout can't
  blow up the simulated grid (*builder's call* on the exact cap).
- `array_hologram(generator, *, rows, columns, spacing_um, angle_deg, …)` —
  generates the rectangle positions and calls `layout_hologram`. Behaviour
  unchanged.

### Rig — `pytweezer/coordinators/rearrangement.py`

- New `make_layout(x_um, y_um, *, name, generator_settings, envelope, blaze_um,
  fresnel_focal_length_mm, zernike, pupil_radius_um, pupil_dump_um,
  iterations, damping, phase_randomness)`: positions of the **on** sites only.
  Writes the mask, remembers the array with `layout=name` as its geometry.
- `make_array` becomes a wrapper sharing the generator-cache/settings code with
  `make_layout`, still remembering `rows`/`columns`/`spacing_um`/`angle_deg`.
- `get_array()` docstring updated: an array from `make_layout` carries
  `layout` instead of the rectangle geometry. FindTrapSites already uses only
  `x_um`/`y_um`, so it works on drawn layouts unchanged.

### Experiments — `pytweezer/experiments/slm_tweezer_array.py`

- A base class `SlmHologramExperiment` holding everything the two share: the
  rig device, the Optics/Pupil/Optimisation argument groups,
  `compensate_envelope`, `sleep_time`, `prepare()` (saved parameters,
  generator settings, envelope, image stream) and `broadcast()`.
- `SlmTweezerArray(SlmHologramExperiment)` — keeps its name and its Array
  arguments (`rows`, `columns`, `spacing`, `angle`); `run_point` calls
  `make_array`. Existing recipes keep working.
- `SlmTweezerLayout(SlmHologramExperiment)` — `layout = Text("", group="Array")`
  (a `Choice` can't be used: its options are fixed at class definition, and
  layouts are added at runtime). `prepare()` fetches it with `get_layout`
  (failing with close-match suggestions on a bad name) and records the whole
  layout (all sites, on and off, plus lattice) as `layout` in the h5, so a run
  stays reproducible after the layout is edited. `run_point` calls
  `make_layout` with the on sites.
- The experiment manager discovers experiments by class; the base class must not
  appear in the Experiments tab (follow whatever the existing abstract-base
  convention is — `introspect.runnable_experiments` lists only classes that
  define their own `run_point`, so a base without one is hidden already).

### The applet — `pytweezer/GUI/viewers/tweezer_layout.py` (new)

An `Applet` subclass run with `run_applet`, listed by the Applet Launcher like
the other viewers. It subscribes to no streams.

```
┌ Tweezer Layout ───────────────────────────────────────────┐
│ Layout [checkerboard-10 ▾]  Save  Save as…  Delete         │
│ Start from: [Rectangle ▾] 10×10, 5 µm, 2.5°  [Apply]       │
│ Mode: (•) Grid  ( ) Freehand   Min spacing [3.0 µm]        │
│ 84 on / 16 off                    [−] [+] [Fit]            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │  · ● ● ○ ● ·          ● on   ○ off   · empty grid pt │   │
│ │  · ● ○ ● ● ·                                         │   │
│ │  · ● ● ● ○ ·        - - SLM reachable field - -      │   │
│ └──────────────────────────────────────────── x (µm) ──┘   │
│ [Remove off sites]                         [Put on SLM]    │
└───────────────────────────────────────────────────────────┘
```

**Canvas.** A pyqtgraph plot, aspect locked, axes in µm. Sites drawn as a
`ScatterPlotItem`: filled for on, hollow for off; red outline for any site in a
close pair. In Grid mode the lattice points in view are faint dots. The SLM's
reachable field — the focal-plane extent `λ f / pitch` in x and y, centred on
the zeroth order at `−blaze`, from the saved parameters — is a dashed
rectangle. If the saved parameters can't be read (manager down), the field is
omitted and the status line says so; editing still works.

**Interaction** (single tap/click only, so it works by touch):

- Tap a site → toggle on/off. "On a site" means within half the minimum
  spacing, or within a fixed screen-pixel radius, whichever is larger, so taps
  are forgiving when zoomed out.
- Tap empty space in **Grid** mode → add a site at the nearest lattice point
  (no-op if that point already has a site, which then toggles instead).
- Tap empty space in **Freehand** mode → add a site exactly there, unless it is
  within the minimum spacing of an existing site, in which case that site
  toggles.
- Drag pans (pyqtgraph default); mouse wheel zooms; `−`/`+`/`Fit` buttons zoom
  for touch.
- "Remove off sites" deletes every off site.
- Ctrl+Z undoes the last edit (a short in-memory history). *Builder's call
  whether this is worth it in v1.*

**Presets.** "Start from" offers Rectangle (rows, columns, spacing, angle) and
Kagome (site count, spacing, angle). Apply replaces the current sites and sets
the Grid lattice, asking first if there are unsaved changes. The lattice of
the last applied preset (or of the opened layout) is what Grid mode snaps to;
a layout with no lattice can still be edited in Freehand mode, and Grid mode is
disabled until a preset is applied.

**Saved layouts.** The combo lists layouts from `get_layouts()`, refreshed when
the manager's layout version changes. Opening one asks first if there are
unsaved changes. Save overwrites the open layout; Save as… prompts for a name
and asks before overwriting another. Delete asks first. Saving with close
pairs shows a warning naming how many and lets the user save anyway. The
window title shows `*` for unsaved changes.

**Put on SLM.** Saves first if there are unsaved changes (asking), then
submits `SlmTweezerLayout` with `layout=name` and every other argument taken
from the last `SlmTweezerLayout` request (`client.last_request`), falling back
to the last `SlmTweezerArray` request's shared arguments, then to class
defaults — so it uses whatever Zernike/iteration settings the lab last ran
with (*my call*). Uses `submit("pytweezer.experiments.slm_tweezer_array:SlmTweezerLayout",
layout=name, ...)`. Shows the run id in the status line; the result appears in
the SLM Monitor applet.

**Touch.** Qt synthesises mouse events from taps, so taps work with no extra
code. Verify the tap → click path on the lab touchscreen; if synthesised
clicks don't reach pyqtgraph's `sigClicked`, handle `QEvent.TouchBegin/End` on
the view widget. Pinch zoom is future work.

## Constraints

- Mask computation needs cupy and a CUDA GPU, which only PH-BEAST has; the
  applet and layout module must import without cupy.
- Follow `pytweezer-gui-design` for look and wording, `add-applet` for applet
  plumbing, `add-experiment` for the experiments and the client API.
- Experiment Manager state files live under its data root; `layouts.json` must
  survive restarts and be written atomically like `recipes.json`.
- UK spelling, Google-style docstrings, `get_logger`.

## Skills to update

- `add-experiment`: the new layout client functions and `SlmTweezerLayout`
  (minor: a line/table row each).
- `add-applet` / `pytweezer-gui-design`: only if a new convention emerges.

## Left to the builder

- Exact tap radius in screen pixels, colours (from `theme.py`), widget layout
  details within the sketch.
- Whether undo makes v1.
- Layout ordering in the combo (alphabetical is fine).
- Note: `SimulatedTweezerArrayCamera` always images a fixed 13×13 grid, so in
  simulation a drawn layout never appears on the camera; check it through the
  SLM Monitor's simulated trap plane instead.
- Test coverage, per the `write-tests` skill: the layout model/book, presets
  (rectangle reproducing `generate_weighted_array` positions), the manager
  commands, and `SlmTweezerLayout` against the simulated rig as far as the
  existing `test_slm_tweezer_array.py` already goes.
