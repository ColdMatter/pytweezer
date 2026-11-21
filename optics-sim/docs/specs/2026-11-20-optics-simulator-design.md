# Optics bench simulator — design

## Intent

A click-and-drag optics bench in the spirit of GaussianBeam.exe, for the CaF and
Rb tweezer experiments. You place a light source and a sequence of optics on an
axis — real catalogue lenses, ideal lenses, an SLM — drag them or type exact
numbers, and see the light propagate live. It answers two questions the lab asks
repeatedly:

1. **Chromatic focal shift in imaging.** Image a point source (on- or off-axis)
   through real lenses and read off how far the focus moves between wavelengths,
   and what the off-axis image looks like.
2. **Beam sizing.** Find the lens positions that give a desired Gaussian beam
   size (or waist, or collimation) at a given place.

It is a standalone tool, not part of the lab control software. It lives in
`optics-sim/` on the `claude/optics-light-simulator-rd87a7` branch of pytweezer
for now and is to move to its own repository later, so it must not import or
depend on anything in pytweezer.

## Goals

- **2D, unfolded-axis view.** Horizontal is distance along the optical axis
  (`z`, mm), vertical is height off the axis (`y`, mm). Mirrors and folds are not
  modelled; the beam always runs left to right.
- **Real catalogue lenses**, traced surface by surface with Snell's law and the
  glass index at each wavelength, so chromatic and off-axis aberrations come out
  of the physics, not a model of them.
- **Ideal thin lenses** (focal length + aperture) for quick sketches. They are
  achromatic.
- **Off-axis sources and decentred / tilted elements.**
- **Multiple wavelengths at once**, each drawn in its own colour.
- **Gaussian beams** propagated through the paraxial (ABCD) behaviour of the real
  surfaces at each wavelength, drawn as a ±w(z) envelope, with waist positions
  and sizes read out — the GaussianBeam view, but with real lenses.
- **SLM as a phase element** acting on rays: a 1D phase profile across its
  aperture bends each ray by the local phase gradient.
- **Lens-position solver** for beam-size targets.
- **Live:** dragging an element re-traces and redraws every frame without lag.
- **Two equal ways to edit:** drag on the bench, or type numbers in a table /
  properties panel; each updates the other.
- **Save / load** a whole setup as a JSON file. The format is documented so a
  later Python reader can open the files.
- **Zero install:** the built app is a single self-contained HTML file that
  opens from disk in any modern browser (no server, no Python, nothing for
  Windows Smart App Control to block).

## Non-goals (for now)

- Wave optics: interference, diffraction patterns, holograms, multi-spot SLM
  patterns. (A wave-propagation view may come later.)
- Breadboard / top-down geometry with mirrors, beamsplitters and folds.
- 3D or skew rays. Tracing is **meridional only**, so for off-axis points the
  app shows the tangential behaviour; sagittal focus and astigmatism between the
  two are not computed. The UI should say so where it reports off-axis results.
- Non-sequential effects (ghosts, back-reflections), polarisation, coatings and
  transmission losses, tolerancing.
- Running from a Jupyter notebook. A Python reader for saved files may follow.
- Any connection to pytweezer, its saved parameters or devices.

## Anti-goals (would make it a failure even if it "works")

- **Precise-looking numbers that are wrong.** The engine must be validated
  against vendor data (see Validation) before the readouts are trusted.
- **Laggy dragging.** If the bench doesn't feel live it loses its point over
  typing numbers into Zemax.
- **Needing installation or a server** to use it.

## Constraints

- TypeScript, built with Vite. Lives entirely in `optics-sim/` with its own
  `package.json`; `node_modules/` and build output are gitignored.
- Work stays on the branch `claude/optics-light-simulator-rd87a7`, as the user
  asked, with no pull request into pytweezer `main` (a TypeScript app doesn't
  belong in the lab-control repo; it moves to its own repository later).
- pytweezer's CI (ruff on `.py`, pytest on `tests/`, Sphinx docs) must keep
  passing; it does not test this folder, and no CI job is added for it here.
  The app's own tests run locally with `npm test` (Vitest); CI comes with its
  own repository.
- pytweezer's GitHub Pages already serves its docs, so hosting the app online
  waits for its own repository. Until then the deliverable is the single-file
  build.
- Units: lengths in mm, wavelengths in nm, angles in degrees in the UI
  (radians internally).
- UK English in the UI and docs.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **Chosen** | **Own 2D ray tracer in TypeScript, browser app** | No install, smooth canvas dragging, shareable single file. The engine (~1–2k lines) is ours to write and test. The group mostly writes Python, so extending it is a little harder for them. |
| | Own 2D tracer in Python + PyQt6/pyqtgraph | Familiar to the group and scriptable from notebooks, but needs Python/uv installed, and a packaged `.exe` is ~100 MB and unsigned (Smart App Control blocks it). Notebook use was judged unnecessary. |
| | Build on `rayoptics` (Python) | Reads `.zmx` and traces already, but its model is a lens prescription, not draggable objects on an axis; rebuilding it on every drag is slow and awkward. |
| | C++/Qt or Rust native app | Single fast `.exe`, but performance isn't the bottleneck for a 2D trace and few in the group could maintain it. |

Placement: inside pytweezer was considered; standalone was chosen because the
tool is unrelated to lab control, shouldn't drag in pytweezer's hardware
dependencies, and should be shareable beyond the group. It sits in a folder on
this branch only because this session can't create a repository.

## What was decided

### Elements

All elements sit at an axial position `z` (mm) and are traced in increasing `z`
(sequential). Every element has a clear aperture; rays outside it are blocked
and drawn stopping there.

A setup may hold **several sources at once** (e.g. an on-axis and an off-axis
point source to compare their images, or a Gaussian beam alongside a ray
source). *(Builder's-check call.)* Each source carries its own wavelength list;
there is no global list. Ray sources are drawn as rays; Gaussian sources as an
envelope plus their centre ray. The ray / Gaussian / both toggle shows or hides
those two kinds. Readouts are grouped per source and wavelength, and a solver
target names the source and wavelength it applies to.

- **Point source** — position `z`, height `y`; list of wavelengths; cone
  half-angle (or "fill the first element's aperture"); ray count.
- **Collimated beam** — height, full width, field angle; wavelengths; ray count.
- **Gaussian beam** — waist radius `w0`, waist position, wavelengths, M².
  Propagated by complex `q` through each element's ABCD matrix at each
  wavelength.
- **Catalogue lens** — a multi-surface prescription (curvatures, thicknesses,
  glasses, aperture, conic / even-asphere terms) placed by its first vertex.
  Can be flipped (reversed), decentred in `y` and tilted.
- **Ideal lens** — focal length, aperture; decentre and tilt.
- **SLM** — aperture, pixel pitch, design wavelength `λ0`, and a phase profile
  `φ0(y)` built from any sum of: Fresnel lens (focal length at `λ0`), blaze
  (deflection angle at `λ0`), 1D Zernike-like polynomial terms in `y`, and a
  profile loaded from a CSV (`y_mm, phase_rad`). Reflective SLMs are treated as
  transmissive phase plates (unfolded axis).
- **Aperture / iris** — diameter, decentre.
- **Screen** — a plane that reports what arrives there.

### Physics

- **Ray tracing:** exact intersection with spherical, conic and even-asphere
  surfaces in 2D; vector Snell's law; total internal reflection or missed
  surfaces terminate the ray. Decentre and tilt by transforming into the
  element's local frame.
- **Dispersion:** glass index from the catalogue's dispersion formula (Schott /
  Sellmeier and the other AGF formulas as needed) evaluated at each wavelength.
  Glass data is bundled as JSON, not pulled from `opticalglass` (it drags in
  PySide6 and pandas and is Python anyway).
- **SLM:** first-order diffraction of the designed phase: the ray's direction
  sine changes by `(λ / 2π) · dφ0/dy`. This gives the correct chromatic
  behaviour of diffractive elements (a Fresnel lens designed for `f0` at `λ0`
  focuses at `f0·λ0/λ`; a blaze deflects by `λ/Λ`). Other orders are ignored.
  Phase wrapping does not enter the first-order model. *(Builder's-check call.)*
  Pixel pitch `p` limits it: where the local phase period `2π / |dφ0/dy|` is
  shorter than two pixels the SLM cannot produce that deflection, so rays
  landing there are blocked and that part of the SLM is drawn highlighted as a
  warning.
- **Gaussian beams:** for each element, the paraxial ABCD at each wavelength
  (for catalogue lenses, built from its surfaces' refraction and transfer
  matrices with `n(λ)`; for the SLM, from the paraxial part of `φ0`). The
  envelope is drawn about the beam's centre ray, which is really traced, so
  decentres steer it. The ABCD matrices are the elements' on-axis ones in
  their own frame; tilt does not change the Gaussian focusing (it would only
  matter at large tilts, which this tool isn't for). *(Builder's-check call.)*
- **Gaussian validity warnings** *(builder's-check call)*: the Gaussian readouts
  are marked unreliable, not hidden, when the beam is clipped (2w at an element
  exceeds 0.8 of its clear aperture, say) or when the centre ray meets an
  element tilted by more than a few degrees.

### Readouts

- **Focus per wavelength** for a point source: paraxial image position and
  best-focus position (minimum RMS ray height in the meridional fan), and the
  differences between wavelengths. A plot of focus position vs wavelength over a
  chosen range.
- **At a screen:** ray intercept heights per wavelength, RMS spot size, centroid.
- **Gaussian:** waist position and radius after every element; `w(z)` under the
  cursor anywhere on the bench.

### Solver

Choose one or more targets — beam radius `w` at a position, waist radius,
waist position, collimation (waist far away) — and one or more free variables
(element positions, each optionally bounded). A bounded local optimiser
(Nelder–Mead or similar) minimises the target error; the result is shown and can
be accepted or undone.

### Lens catalogue

- Thorlabs and Edmund publish a Zemax `.zmx` file per lens. The app parses the
  surface data from `.zmx` (sequential `SURF` blocks: `CURV`, `DISZ`, `GLAS`,
  `DIAM`, `CONI`, `PARM` for even aspheres) into its own prescription JSON.
- A starter set of common parts is bundled: Thorlabs Ø1" and Ø2" achromats
  (`AC254-…`, `AC508-…`, -A and -B coatings) and N-BK7 plano-convex singlets
  (`LA1…`) across the usual focal lengths. Vendor `.zmx` files can't be
  downloaded from the build environment, so these prescriptions are
  transcribed from the vendors' published drawings (radii, centre thicknesses,
  glasses).
- **Accuracy target (decided by the user):** what must be right is the
  *difference* in focus between wavelengths, which is insensitive to the
  rounding on the drawings. Absolute focus positions from a drawing-derived lens
  are labelled approximate in the readouts; the label goes away once the real
  `.zmx` for that part is imported. Each lens records its source (`drawing` or
  `zmx`) so the UI can tell.
- The user can import any further `.zmx` through a file picker; imported lenses
  are saved into the setup file so it stays self-contained.
- A lens using a glass that isn't in the bundled data is flagged, and the user
  picks a substitute glass or types an index.

### Interface

```
┌ Bench (canvas) ──────────────────────────────────────────────────┐
│  source      L1 (f=100)        L2 (AC254-250-B)    SLM    screen │
│    ●═══════════╪════════════════════╪═════════════════▌──────│   │
│     rays / Gaussian envelope, one colour per wavelength          │
├ Elements table ─────────────────┬ Properties (selected) ─────────┤
│ # type   z (mm)  part / f   y   │ every parameter as a number    │
├ Readouts ───────────────────────┴ Solver ────────────────────────┤
│ focus per λ, Δfocus, spot sizes, waists │ targets, variables, Run│
└──────────────────────────────────────────────────────────────────┘
```

- Bench: pan and zoom, with **independent vertical zoom** (heights are mm, axis
  lengths hundreds of mm). Drag an element along `z`; drag with a modifier to
  change its `y` (source height or decentre). Selecting on the bench selects in
  the table and vice versa.
- Wavelength colours follow the visible spectrum; infrared wavelengths get
  distinct, labelled colours.
- Toggle ray view / Gaussian view / both.
- Undo / redo for every edit, including accepting a solver result.
- Autosave the current setup to browser storage; explicit save / load to JSON
  files.

### Save format

One JSON file per setup: a format version, the global wavelength list, and an
ordered list of elements with their parameters (imported lens prescriptions and
glass data included inline). Documented in the folder's README so a Python
reader can follow later.

## Validation

The engine's tests must include at least:

- N-BK7 index at 587.6 nm = 1.5168 (and a couple of other glasses at known
  wavelengths).
- Thin-lens imaging and Gaussian `q` propagation against closed-form results.
- A Thorlabs achromat's focal-shift curve (focus relative to its design
  wavelength) against Thorlabs' published curve, to within a few µm across the
  coating band; its back focal length within the drawing's rounding (≈0.1%).
- A plano-convex singlet's focal length at two wavelengths against the
  lensmaker's equation with the bundled `n(λ)`.
- The SLM Fresnel lens focusing at `f0·λ0/λ`.
- Parsing a real `.zmx` file into the expected prescription.

## Left to the builder

- UI framework (none, Preact, Solid, …) and component layout details.
- Canvas 2D vs SVG for the bench (canvas expected, for drag performance).
- Exact list of bundled lenses and glasses.
- Default ray counts, colours, keyboard shortcuts, snapping.
- Optimiser choice, provided it respects bounds and finishes in well under a
  second for a few variables.
