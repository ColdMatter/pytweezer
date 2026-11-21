# Optics Bench Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser-based 2D optics bench (single self-contained HTML file) where sources, real catalogue lenses, ideal lenses, SLMs, apertures and screens are placed on an unfolded axis, dragged or typed, and rays plus Gaussian beams are traced live, with chromatic focus readouts and a lens-position solver.

**Architecture:** A pure TypeScript engine (`src/engine/`, no DOM) does glass dispersion, exact 2D meridional ray tracing, paraxial ABCD / Gaussian `q` propagation, focus readouts, `.zmx` parsing, setup I/O and the solver; it is unit-tested with Vitest. A vanilla-TypeScript UI (`src/ui/`, no framework) holds the setup in an undoable store and renders a canvas bench, elements table, properties panel, readouts and solver panel. Vite builds everything into one `dist/index.html`.

**Tech Stack:** TypeScript 5, Vite, `vite-plugin-singlefile`, Vitest (+ `jsdom` for UI unit tests), the pre-installed Playwright/Chromium for data gathering and the end-to-end smoke test. No UI framework, no runtime dependencies.

**Spec:** `optics-sim/docs/specs/2026-11-20-optics-simulator-design.md` — read it alongside this plan.

## Global Constraints

- Everything lives in `optics-sim/`; nothing imports from or depends on pytweezer. Do not add any `.py` file (pytweezer's CI runs ruff on `.py`, pytest on `tests/`).
- Branch `claude/optics-light-simulator-rd87a7`; no pull request; push after every task. Commit messages end with the session's attribution lines.
- `node_modules/` and `dist/` are gitignored. The deliverable is `npm run build` → a single `dist/index.html` that opens from disk (`file://`) with no server.
- Units: lengths mm, wavelengths nm, angles degrees in UI and saved files; radians only inside the engine.
- UK English in UI text and docs.
- Tracing is **meridional only**; any readout for an off-axis source shows the note "Tangential (meridional) rays only — sagittal focus not computed".
- Gaussian readouts are marked unreliable (not hidden) when `2w > 0.8 ×` an element's clear aperture or the centre ray meets an element tilted by more than 3°.
- Lenses carry `source: 'zmx' | 'drawing'`; absolute focus positions involving any `drawing` lens are labelled "approx." in readouts. Focal *shifts* are not labelled.
- Glass data comes from the CC0 refractiveindex.info database (`https://github.com/polyanskiy/refractiveindex.info-database`), converted into `src/data/glasses.json`; not from `opticalglass`.
- Dragging must stay live: `traceSetup` for 3 sources × 3 wavelengths × 31 rays × 6 lenses runs in under 8 ms.
- The solver finishes in under 1 s for up to 4 variables.

## Review Focus

1. **Elements overlapping or out of order** (a lens body overlapping the next, a source placed after a lens, two optics at the same z): tracing still works on z-sorted optics; the table marks overlapping rows; the solver penalises overlap so it never proposes interpenetrating lenses. → tests in Task 3 (`overlap warning`) and Task 8 (`solver avoids overlap`).
2. **No light gets through** (aperture closed, source aimed away, every ray clipped): readouts say "No rays reach this point" instead of NaN, and nothing throws. → tests in Task 3 (`all rays blocked`) and Task 5 (`no-rays status`).
3. **Collimated output** (waist at effectively infinite distance, parallel rays after the last element): Gaussian waist reads "collimated", focus reads "no focus (collimated)"; never 1e15 mm. → tests in Task 4 (`collimated waist`) and Task 5 (`collimated status`).
4. **Wavelength outside a glass's data range or an unknown glass** (e.g. 1550 nm through a glass tabulated to 1.0 µm; a `.zmx` naming a glass we don't bundle): extrapolated indices are flagged in readouts; unknown glass blocks rays at that lens with a warning naming the glass, until a substitute is chosen. → tests in Task 1 (`inRange`) and Task 3 (`unknown glass`).
5. **Bad files** (hand-edited or newer-version setup JSON; `.zmx` with `COORDBRK`, `TOROIDAL`, `MIRR`, inch units, UTF-16): either parsed correctly (inch, UTF-16) or rejected with a message naming the problem, leaving the current setup untouched. → tests in Task 6 (`zmx rejects`) and Task 7 (`parseSetup rejects`).

---

## File structure

```
optics-sim/
  package.json, tsconfig.json, vite.config.ts, index.html, .gitignore, README.md
  scripts/build-glasses.mjs        # refractiveindex.info YAML -> src/data/glasses.json
  scripts/fetch-thorlabs.mjs       # Playwright: lens prescriptions + focal-shift data
  src/data/glasses.json            # generated, committed
  src/data/lenses.json             # bundled catalogue, committed
  src/data/thorlabs-focal-shift/   # vendor focal-shift data used only by tests
  src/engine/
    types.ts       # every shared type (below)
    glass.ts       # dispersion formulas, lookup
    surface.ts     # sag, ray–surface intersection, refraction
    complex.ts     # minimal complex arithmetic
    lens.ts        # catalogue lens: trace, ABCD, flip, thickness
    slm.ts         # SLM phase profile, trace, ABCD
    optic.ts       # dispatch over Optic kinds (ideal, aperture, screen inline)
    trace.ts       # launch rays, sequential trace, warnings
    gaussian.ts    # q propagation, waists, radiusAt, validity
    focus.ts       # focus per wavelength, focal-shift curve, screen stats
    zmx.ts         # .zmx decode + parse
    io.ts          # setup JSON (de)serialisation + validation
    solver.ts      # bounded Nelder–Mead over optic positions
  src/ui/
    main.ts, app.ts, store.ts, colours.ts, defaultSetup.ts
    table.ts, properties.ts, readouts.ts, solverPanel.ts, catalogue.ts
    bench/view.ts, bench/draw.ts, bench/interact.ts
  src/styles.css
  tests/*.test.ts                  # Vitest, one file per engine module + ui units
  e2e/smoke.mjs                    # Playwright smoke test against dist/index.html
```

---

### Task 1: Scaffold, glass data and refractive index

**Files:**
- Create: `optics-sim/package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `.gitignore`, `README.md` (stub: what it is, `npm install`, `npm run dev|build|test|e2e`)
- Create: `optics-sim/scripts/build-glasses.mjs`, `optics-sim/src/data/glasses.json`
- Create: `optics-sim/src/engine/types.ts` (glass types only for now), `optics-sim/src/engine/glass.ts`
- Test: `optics-sim/tests/glass.test.ts`

**Interfaces:**
- Produces (in `types.ts`):
  ```ts
  export interface GlassData {
    name: string;                       // catalogue name, e.g. "N-BK7"
    formula: 1 | 2 | 3 | 'constant';    // refractiveindex.info formula number
    coefficients: number[];             // as in the YAML, wavelengths in µm
    rangeUm: [number, number];
  }
  export type GlassLookup = (name: string) => GlassData | undefined;
  ```
- Produces (in `glass.ts`): `refractiveIndex(glass: GlassData, wavelengthNm: number): number`, `inRange(glass: GlassData, wavelengthNm: number): boolean`, `makeGlassLookup(extra?: Record<string, GlassData>): GlassLookup` (bundled JSON, `extra` takes precedence; name matching is case-insensitive). `'constant'` uses `coefficients[0]` as `n`.

- [ ] **Step 1:** Scaffold: `npm init`, devDependencies `typescript vite vite-plugin-singlefile vitest jsdom yaml`; scripts `dev`, `build` (`tsc --noEmit && vite build`), `test` (`vitest run`), `e2e` (`npm run build && node e2e/smoke.mjs`), `glasses` (`node scripts/build-glasses.mjs <path-to-rii-clone>`). `tsconfig` strict, `"noUncheckedIndexedAccess": true`. `vite.config.ts` uses `viteSingleFile()`. `.gitignore`: `node_modules/`, `dist/`.
- [ ] **Step 2:** Write `scripts/build-glasses.mjs`: shallow-clone or take a path to the refractiveindex.info database, read `database/data/specs/{schott,ohara,cdgm,hoya,sumita}/optical/*.yml`, keep the first `DATA` entry whose `type` is `formula 1|2|3`, and write `{ [name]: GlassData }` to `src/data/glasses.json`. Include every Schott, Ohara, CDGM and Hoya glass (file size is fine; it keeps imported lenses working). Where two catalogues share a name, Schott wins. Run it and commit the JSON.
- [ ] **Step 3: Write the failing tests** in `tests/glass.test.ts`:
  ```ts
  const g = makeGlassLookup();
  test('N-BK7 nd', () => expect(refractiveIndex(g('N-BK7')!, 587.56)).toBeCloseTo(1.51680, 4));
  test('N-SF6 nd', () => expect(refractiveIndex(g('N-SF6')!, 587.56)).toBeCloseTo(1.80518, 4));
  test('N-BAF10 nd', () => expect(refractiveIndex(g('N-BAF10')!, 587.56)).toBeCloseTo(1.67003, 4));
  test('N-BK7 at 1064 nm', () => expect(refractiveIndex(g('N-BK7')!, 1064)).toBeCloseTo(1.50669, 4));
  test('case-insensitive lookup', () => expect(g('n-bk7')?.name).toBe('N-BK7'));
  test('extra overrides bundled', () => expect(refractiveIndex(makeGlassLookup({ 'N-BK7': { name: 'N-BK7', formula: 'constant', coefficients: [1.6], rangeUm: [0, 99] } })('N-BK7')!, 800)).toBe(1.6));
  test('inRange', () => { expect(inRange(g('N-BK7')!, 780)).toBe(true); expect(inRange(g('N-BK7')!, 3000)).toBe(false); });
  ```
- [ ] **Step 4:** Run `npm test` → FAIL (module not found).
- [ ] **Step 5:** Implement `glass.ts`. Formulas (λ in µm), from refractiveindex.info: 1 `n²−1 = C0 + Σ Bi λ²/(λ² − Ci²)`; 2 `n²−1 = C0 + Σ Bi λ²/(λ² − Ci)`; 3 `n² = C0 + Σ Ci λ^Ci+1` (pairs coefficient, exponent).
- [ ] **Step 6:** `npm test` → PASS. `npm run build` succeeds (blank page is fine).
- [ ] **Step 7:** Commit and push.

---

### Task 2: Surfaces — sag, intersection, refraction

**Files:**
- Modify: `optics-sim/src/engine/types.ts`
- Create: `optics-sim/src/engine/surface.ts`
- Test: `optics-sim/tests/surface.test.ts`

**Interfaces:**
- Produces (in `types.ts`):
  ```ts
  export interface Ray { z: number; y: number; dz: number; dy: number; wavelengthNm: number } // (dz,dy) unit, dz > 0 in normal use
  export interface SurfaceShape { curvature: number; conic: number; asphere: number[]; semiDiameterMm: number } // asphere = [A4, A6, …], sag += Σ A(2k+2) y^(2k+2)
  ```
- Produces (in `surface.ts`): `sag(s: SurfaceShape, y: number): number`; `intersect(s: SurfaceShape, vertexZ: number, ray: Ray): { z: number; y: number; normalZ: number; normalY: number } | null` (null if the ray misses the surface or hits outside `semiDiameterMm`); `refract(ray: Ray, hit, n1: number, n2: number): Ray | null` (null on total internal reflection; normal orientation handled internally).

- [ ] **Step 1: Write the failing tests:**
  ```ts
  test('sphere sag', () => expect(sag({ curvature: 1/50, conic: 0, asphere: [], semiDiameterMm: 12.7 }, 10)).toBeCloseTo(1.010205, 6));
  test('flat sag is zero', ...);           // curvature 0 → 0
  test('parabola sag', ...);              // conic −1, c = 1/50, y = 10 → c y²/2 = 1.0
  test('asphere term adds', ...);          // A4 = 1e-5, flat, y = 10 → 0.1
  test('axial ray hits vertex', ...);      // ray z=0,y=0 → hit z = vertexZ exactly
  test('hit outside semi-diameter → null', ...);
  test('Snell at flat face', () => { /* 30° in air onto n=1.5 → angle asin(0.5/1.5) = 19.47122° */ });
  test('TIR returns null', ...);           // 45° from n=1.5 into n=1
  test('normal incidence on sphere is undeviated', ...); // ray through centre of curvature
  ```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Intersection: start from the vertex-plane intersection and Newton-iterate `f(t) = z(t) − vertexZ − sag(y(t))` (≤ 20 iterations, tolerance 1e-10 mm); return null when the sag square root goes imaginary or iterations don't converge. Refraction in vector form: `t = η d + (η cosθi − cosθt) n̂` with `η = n1/n2`, normal oriented against the ray.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 3: Elements, sources and the sequential tracer

**Files:**
- Modify: `optics-sim/src/engine/types.ts`
- Create: `optics-sim/src/engine/lens.ts`, `optics-sim/src/engine/optic.ts`, `optics-sim/src/engine/trace.ts`
- Test: `optics-sim/tests/trace.test.ts`, `optics-sim/tests/lens.test.ts`

**Interfaces:**
- Produces (in `types.ts`) — these are also the saved-file schema, so keep field names exactly:
  ```ts
  export type Id = string;
  interface SourceBase { id: Id; label: string; z: number; y: number; wavelengthsNm: number[] }
  export interface PointSource extends SourceBase { kind: 'point'; coneHalfAngleDeg: number | 'fill'; rayCount: number }
  export interface CollimatedSource extends SourceBase { kind: 'collimated'; widthMm: number; angleDeg: number; rayCount: number }
  export interface GaussianSource extends SourceBase { kind: 'gaussian'; angleDeg: number; waistRadiusMm: number; waistZ: number; m2: number }
  export type Source = PointSource | CollimatedSource | GaussianSource;

  export interface PrescriptionSurface extends SurfaceShape { thicknessToNextMm: number; glassAfter: string | null } // null = air; last surface has thickness 0, glass null
  export interface LensPrescription { partNumber: string; vendor: string; source: 'zmx' | 'drawing'; sourceRef: string; surfaces: PrescriptionSurface[] }

  interface Placed { id: Id; label: string; z: number; decentreMm: number; tiltDeg: number } // z = first vertex (lens) or plane
  export interface CatalogueLens extends Placed { kind: 'lens'; prescription: LensPrescription; flipped: boolean }
  export interface IdealLens extends Placed { kind: 'ideal'; focalLengthMm: number; apertureMm: number } // aperture = diameter
  export type PhaseTerm =
    | { kind: 'fresnel'; focalLengthMm: number }        // at design wavelength
    | { kind: 'blaze'; angleDeg: number }               // deflection at design wavelength
    | { kind: 'polynomial'; coefficientsRad: number[] } // Σ a_k (y/R)^k, k = 1…, R = aperture/2
    | { kind: 'table'; yMm: number[]; phaseRad: number[] };
  export interface Slm extends Placed { kind: 'slm'; apertureMm: number; pixelPitchUm: number; designWavelengthNm: number; phase: PhaseTerm[] }
  export interface Aperture { kind: 'aperture'; id: Id; label: string; z: number; decentreMm: number; diameterMm: number }
  export interface Screen { kind: 'screen'; id: Id; label: string; z: number }
  export type Optic = CatalogueLens | IdealLens | Slm | Aperture | Screen;

  export interface Setup { version: 1; sources: Source[]; optics: Optic[]; glasses: Record<string, GlassData> }

  export interface RayPath { wavelengthNm: number; points: [number, number][]; blockedBy: Id | null; final: Ray | null } // final = ray leaving the last optic
  export type TraceWarning =
    | { kind: 'overlap'; ids: [Id, Id] }
    | { kind: 'unknown-glass'; opticId: Id; glass: string }
    | { kind: 'glass-extrapolated'; glass: string; wavelengthNm: number }
    | { kind: 'slm-pitch-limit'; opticId: Id };
  export interface SourceTrace { sourceId: Id; byWavelength: { wavelengthNm: number; rays: RayPath[] }[] }
  export interface SetupTrace { sources: SourceTrace[]; warnings: TraceWarning[]; endZ: number }
  ```
- Produces (in `lens.ts`): `lensThickness(p: LensPrescription): number`; `flipPrescription(p: LensPrescription): LensPrescription`; `traceLens(lens: CatalogueLens, ray: Ray, glasses: GlassLookup): { ray: Ray | null; points: [number, number][]; warning?: TraceWarning }`.
- Produces (in `optic.ts`): `opticExtent(o: Optic): [number, number]` (axial start/end), `interactOptic(o: Optic, ray: Ray, glasses: GlassLookup): { ray: Ray | null; points: [number, number][]; warning?: TraceWarning }`. SLM delegates to `slm.ts` (Task 4 creates it; until then `interactOptic` throws for `'slm'`).
- Produces (in `trace.ts`): `launchRays(source: Source, optics: Optic[], wavelengthNm: number): Ray[]`; `traceRay(ray: Ray, optics: Optic[], glasses: GlassLookup, endZ: number): RayPath`; `traceSetup(setup: Setup, endZ?: number): SetupTrace` (default `endZ` = last optic end + 30 % of the layout length, at least 50 mm; builds its own lookup from `setup.glasses`).

- [ ] **Step 1: Write the failing tests** (`trace.test.ts`, `lens.test.ts`):
  - `ideal lens images a point perfectly`: point source z=0, y=1, cone 5°, 21 rays, 780 nm; ideal f=100 at z=200. Every `final` ray, extended to z=400, has y = −1 within 1e-9.
  - `ideal lens uses tangent form`: a ray at height h, slope 0 leaves with `dy/dz = −h/f` exactly.
  - `aperture clips`: collimated width 10 through aperture diameter 4 → rays with |y| > 2 have `blockedBy` = aperture id.
  - `all rays blocked`: aperture diameter 0.01 off-axis at decentre 5 → every ray blocked, `traceSetup` doesn't throw.
  - `plano-convex focal length`: N-BK7 surfaces `[{curvature: 1/51.5, …, thickness 3.6, glass 'N-BK7'}, {curvature 0, …}]`, collimated paraxial ray at h = 0.01 → crosses axis at distance from the second vertex equal to `51.5/(n−1) − 3.6/n` within 1e-4 mm, for both 587.56 nm and 1064 nm.
  - `flip reverses order and signs`: `flipPrescription` on a 3-surface achromat; flipping twice returns the original; thickness unchanged.
  - `decentred ideal lens steers`: collimated axial ray, ideal f=100 decentred by +1 → crosses y = +1 at z = lens + 100.
  - `tilted flat plate shifts but doesn't deviate`: N-BK7 plate 10 mm thick tilted 10° — output direction equals input; lateral shift matches `t sinθ (1 − cosθ/√(n² − sin²θ))`.
  - `overlap warning`: two lenses whose extents overlap → one `overlap` warning naming both ids.
  - `unknown glass`: a lens with `glassAfter: 'XYZ-1'` → rays blocked at that lens and an `unknown-glass` warning.
  - `point source fill`: `coneHalfAngleDeg: 'fill'` aims the outermost rays at the first optic's aperture edges (within 1e-9 rad).
  - `trace performance`: 3 sources × 3 wavelengths × 31 rays × 6 achromats (3 surfaces each): median of 20 `traceSetup` calls < 8 ms.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Optics are traced in ascending `z` (stable sort). Placement transform: translate by `(z, decentreMm)` then rotate by `tiltDeg` about the element's axial midpoint; trace in the local frame; transform back. Ideal lens: thin plane, `tanθ' = tanθ − h_local/f`, blocked outside aperture. Aperture: block |y − decentre| > d/2 at its plane. Screen: records a point, never blocks. Air is n = 1 (catalogue indices are relative to air). A ray that misses a lens surface is blocked by that lens. Rays continue to `endZ` after the last optic. Collect `glass-extrapolated` once per (glass, wavelength).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 4: SLM, paraxial ABCD and Gaussian beams

**Files:**
- Create: `optics-sim/src/engine/slm.ts`, `optics-sim/src/engine/complex.ts`, `optics-sim/src/engine/gaussian.ts`
- Modify: `optics-sim/src/engine/optic.ts`, `optics-sim/src/engine/lens.ts`, `optics-sim/src/engine/types.ts`
- Test: `optics-sim/tests/slm.test.ts`, `optics-sim/tests/gaussian.test.ts`

**Interfaces:**
- Produces (in `types.ts`):
  ```ts
  export type Abcd = [number, number, number, number]; // A, B (mm), C (1/mm), D
  export interface Waist { afterOpticId: Id | null; z: number; radiusMm: number; collimated: boolean } // collimated: Rayleigh range > 1e5 mm; readouts then ignore z
  export type GaussianWarning = { kind: 'clipped'; opticId: Id } | { kind: 'tilted'; opticId: Id };
  export interface GaussianSegment { zStart: number; zEnd: number; qStart: { re: number; im: number }; centre: [number, number][] }
  export interface GaussianTrace { sourceId: Id; wavelengthNm: number; centre: RayPath; segments: GaussianSegment[]; waists: Waist[]; warnings: GaussianWarning[] }
  ```
- Produces (in `slm.ts`): `slmPhaseGradient(slm: Slm, y: number): number` (rad/mm, design phase φ0), `slmPhaseCurvature(slm: Slm, y: number): number` (rad/mm²), `traceSlm(slm: Slm, ray: Ray): { ray: Ray | null; points; warning? }`, `slmAbcd(slm: Slm, yHit: number, wavelengthNm: number): Abcd`.
- Produces (in `lens.ts`): `lensAbcd(lens: CatalogueLens, wavelengthNm: number, glasses: GlassLookup): Abcd` (vertex to vertex, air both sides).
- Produces (in `optic.ts`): `opticAbcd(o: Optic, wavelengthNm: number, yHit: number, glasses: GlassLookup): Abcd` (identity for aperture/screen; ideal `[1,0,−1/f,1]`).
- Produces (in `gaussian.ts`): `traceGaussian(source: GaussianSource, optics: Optic[], wavelengthNm: number, glasses: GlassLookup, endZ: number): GaussianTrace`; `radiusAt(trace: GaussianTrace, z: number): number | null` (null inside glass or beyond the trace); `traceGaussians(setup: Setup, endZ: number): GaussianTrace[]` (every Gaussian source × wavelength).

- [ ] **Step 1: Write the failing tests:**
  - `slm fresnel chromatic focus`: SLM at z=0, Fresnel f0 = 200 mm at λ0 = 780 nm, aperture 15 mm, pitch 8 µm; collimated width 1 mm at 852 nm → `final` rays cross the axis at z = 200·780/852 = 183.10 mm (±0.02).
  - `slm blaze deflects λ/Λ`: blaze 1° at 780 nm, ray at 852 nm → `sinθ' = (852/780)·sin 1°` (±1e-9).
  - `slm table gradient`: table with φ = 2y² samples → gradient at y = 1 equals 4 rad/mm (±1e-6).
  - `slm pitch limit blocks`: Fresnel f0 = 5 mm, pitch 8 µm, aperture 15 mm → rays where `|dφ0/dy|·p > π` are blocked and an `slm-pitch-limit` warning is raised; rays nearer the axis pass.
  - `slm ABCD`: Fresnel term gives `C = −(λ/λ0)/f0`; blaze alone gives identity.
  - `free-space Gaussian`: w0 = 1 mm, 1000 nm, waist at z=0, no optics → `radiusAt(z = πw0²/λ = 3141.59 mm)` = √2 (±1e-6).
  - `M² scales divergence`: same with m2 = 2 → radius at 1570.80 mm is √2.
  - `thin-lens waist`: w0 = 1 mm at an ideal f = 100 lens, 780 nm → waist at `f/(1+(f/zR)²)` behind it with radius `w0·(f/zR)/√(1+(f/zR)²)`, matching to 1e-9 relative.
  - `thick-lens ABCD`: `lensAbcd` of the plano-convex from Task 3 has `det = 1` (±1e-12) and `−1/C = 51.5/(n−1)` (±1e-6 relative).
  - `collimated waist`: w0 = 5 mm at 780 nm, no optics (zR ≈ 1.007e5 mm) → `collimated: true`; w0 = 1 mm → `collimated: false`.
  - `4f relay`: ideal f1 = 50, f2 = 150 at separation 200, input waist at the front focal plane of lens 1 → final waist at the back focal plane of lens 2 with radius 3 × w0 (±1e-9 relative).
  - `clipped warning`: w = 3 mm at an ideal lens with aperture 5 mm → `clipped` warning.
  - `tilted warning`: lens tilted 5° in the beam → `tilted` warning; 2° → none.
  - `decentre steers the envelope`: decentred ideal lens → the Gaussian `centre` path matches the traced ray.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. SLM: sum terms for `φ0(y)` in the local frame (Fresnel `−π y²/(λ0 f0)`, blaze `2π y sin α0/λ0`, polynomial, table with linear interpolation and central-difference derivatives), then `sinθ' = sinθ + (λ/2π)·dφ0/dy` (λ in mm), block where `|dφ0/dy|·pitch > π` or outside the aperture. `slmAbcd` uses `C = (λ/2π)·slmPhaseCurvature(yHit)`. Lens ABCD: refraction `[1, 0, (n1−n2)c/n2, n1/n2]` and transfer `[1, d, 0, 1]`, multiplied in order; with air both sides the result has unit determinant. Gaussian: trace the centre ray with `traceRay`; propagate `q` in air with `λ_eff = M² λ`; at each optic apply `opticAbcd` (lens: from first to last vertex, so the segment skips the glass) at the centre ray's local height; record the waist after each optic as `z = exit − Re(q)`, `w0 = √(Im(q)·λ_eff/π)`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 5: Focus readouts

**Files:**
- Create: `optics-sim/src/engine/focus.ts`
- Test: `optics-sim/tests/focus.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface FocusResult { wavelengthNm: number; status: 'ok' | 'collimated' | 'no-rays'; paraxialZ: number | null; bestZ: number | null; rmsAtBestMm: number | null; imageHeightMm: number | null; approximate: boolean }
  export function focusPerWavelength(setup: Setup, sourceId: Id): FocusResult[];           // point and collimated sources
  export function focusShiftCurve(setup: Setup, sourceId: Id, minNm: number, maxNm: number, steps: number): { wavelengthNm: number; paraxialZ: number | null }[];
  export interface ScreenSpot { wavelengthNm: number; heightsMm: number[]; centroidMm: number | null; rmsMm: number | null }
  export function screenStats(trace: SourceTrace, screenId: Id): ScreenSpot[];
  ```
  `approximate` is true when any lens the light passed through has `source: 'drawing'`.

- [ ] **Step 1: Write the failing tests:**
  - `ideal imaging focus`: Task 3's ideal-lens setup → `paraxialZ` and `bestZ` = 400 (±1e-6), `rmsAtBestMm` < 1e-9, `imageHeightMm` = −1.
  - `chromatic shift of a singlet`: collimated source (780 and 1064 nm) through the plano-convex → `paraxialZ(1064) − paraxialZ(780)` equals the difference of `51.5/(n−1) − 3.6/n` at the two wavelengths (±1e-4 mm).
  - `spherical aberration moves best focus`: same singlet with width 20 mm → `bestZ < paraxialZ` and `rmsAtBestMm > 0`.
  - `collimated status`: collimated source, no optics → `status: 'collimated'`, both z null.
  - `no-rays status`: everything blocked → `status: 'no-rays'`, nulls, no throw.
  - `approximate flag`: a `drawing` lens in the path → `approximate: true`; `zmx` → false.
  - `screen stats`: three rays at known heights on a screen → centroid and RMS as computed by hand.
  - `focus shift curve`: returns `steps` entries spanning min..max.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Use the surviving rays' `final` segments (straight lines). Best focus has a closed form: with `y_i(z) = y_i0 + m_i(z − z0)`, the variance is quadratic in z, so `z* = z0 − Cov(y0, m)/Var(m)`. Paraxial focus: trace two extra rays at ±1e-4 rad about the source's central ray (point) or ±1e-3 mm about the central ray (collimated) and intersect them. If `Var(m)` (or the paraxial pair's slope difference) < 1e-12, the status is `collimated`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 6: `.zmx` parser and the bundled lens catalogue

**Files:**
- Create: `optics-sim/src/engine/zmx.ts`, `optics-sim/scripts/fetch-thorlabs.mjs`, `optics-sim/src/data/lenses.json`, `optics-sim/src/data/thorlabs-focal-shift/*.json`
- Test: `optics-sim/tests/zmx.test.ts`, `optics-sim/tests/catalogue.test.ts`

**Interfaces:**
- Produces (in `zmx.ts`): `decodeZmx(bytes: Uint8Array): string` (UTF-16LE/BE with BOM, or UTF-8/Latin-1); `parseZmx(text: string, vendor?: string): LensPrescription` (throws `ZmxError` with a message naming the problem). `source: 'zmx'`, `sourceRef` = the file name if given, `partNumber` from `NAME`.
- Produces (`lenses.json`): `LensPrescription[]`.

- [ ] **Step 1: Gather the data.** Write `scripts/fetch-thorlabs.mjs` using the pre-installed Playwright (`require('playwright')`, `chromium.launch()`; do not install Playwright). For each part, open the Thorlabs product page and (a) download the Zemax file if the page offers one, otherwise read the drawing's radii, centre thicknesses, diameter and glasses; (b) download the focal-shift data where offered. Parts: `AC254-{030,040,050,060,075,100,125,150,200,250,300,400,500}-{A,B}`, `AC508-{075,100,150,200,250,300,400,500,750,1000}-{A,B}`, N-BK7 plano-convex `LA1xxx` at f = 25, 30, 35, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000 mm (Ø1"). Every entry records `source` and `sourceRef` (URL). Save vendor files under `scripts/vendor-cache/` (gitignored) and the focal-shift data for at least AC254-100-B and AC254-200-A as `{ wavelengthNm, shiftUm }[]` JSON. **If the pages cannot be read even with Chromium, stop and ask the user** to download the Zemax files (or drawings) and focal-shift data for these parts into a folder; do not transcribe prescriptions from memory.
- [ ] **Step 2: Write the failing tests** (`zmx.test.ts` uses one real downloaded `.zmx` copied to `tests/fixtures/` plus inline synthetic files):
  - `parses a Thorlabs achromat`: the fixture → 3 surfaces, curvatures/thicknesses/glasses matching the file, `glassAfter` of the last surface null.
  - `UTF-16 decode`: the same text encoded UTF-16LE with BOM parses identically.
  - `inch units converted`: `UNIT IN` file → mm values ×25.4.
  - `even asphere`: `TYPE EVENASPH` with `PARM 2` → `asphere[0]` = that value (A4); a nonzero `PARM 1` (r² term) is folded into curvature or rejected with a message.
  - `zmx rejects`: `COORDBRK`, `TOROIDAL`, `MIRR` and a file with no glass surfaces each throw `ZmxError` whose message names the cause.
  - `catalogue.test.ts`: every `lenses.json` entry has ≥ 2 surfaces, all glasses resolve in `makeGlassLookup()`, `sourceRef` non-empty; paraxial EFL at the coating-band centre (A: 535 nm, B: 855 nm) is within 1 % of the part's nominal focal length.
  - `matches Thorlabs focal shift`: for each focal-shift file, `focusShiftCurve` for a collimated source through that lens, minus its value at the file's reference (zero-shift) wavelength, matches the vendor shift within 5 µm across the coating band (A: 400–700 nm, B: 650–1050 nm).
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Implement `zmx.ts`. Lens surfaces run from the first `SURF` with a `GLAS` to the surface after the last glassy one; ignore object/image/dummy air surfaces outside that range. `DIAM` is the semi-diameter; `CURV` 1/length; `DISZ` thickness to next; `CONI` conic; `UNIT` sets the scale. Then generate `lenses.json` from the cache (zmx where available, else drawing values).
- [ ] **Step 5:** Run → PASS. If the focal-shift comparison fails, investigate before relaxing anything (glass catalogue edition, reference wavelength, paraxial vs best focus); report a persistent mismatch to the user rather than widening the tolerance.
- [ ] **Step 6:** Commit and push (not the vendor cache).

---

### Task 7: Setup save/load

**Files:**
- Create: `optics-sim/src/engine/io.ts`
- Modify: `optics-sim/README.md` (document the JSON format: version, every field from `Setup` in Task 3 with units, an example)
- Test: `optics-sim/tests/io.test.ts`

**Interfaces:**
- Produces: `serialiseSetup(setup: Setup): string` (pretty JSON); `parseSetup(text: string): Setup` (throws `SetupFormatError` with a path-qualified message, e.g. `optics[2].focalLengthMm: expected a non-zero number`); `newId(): Id`.

- [ ] **Step 1: Write the failing tests:**
  - `round trip`: a setup containing every source and optic kind, an imported glass and a lens prescription serialises and parses back deep-equal.
  - `parseSetup rejects`: invalid JSON; `version: 2` ("made by a newer version"); missing `sources`; unknown optic `kind`; `focalLengthMm: 0`; negative `apertureMm`; empty `wavelengthsNm`; duplicate ids — each throws `SetupFormatError` naming the path.
  - `ideal lens may be negative`: `focalLengthMm: −50` accepted.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with a small hand-written validator (no schema library).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 8: Lens-position solver

**Files:**
- Create: `optics-sim/src/engine/solver.ts`
- Test: `optics-sim/tests/solver.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export type Target =
    | { kind: 'radiusAt'; sourceId: Id; wavelengthNm: number; z: number; radiusMm: number }
    | { kind: 'waistRadius'; sourceId: Id; wavelengthNm: number; radiusMm: number }   // final waist
    | { kind: 'waistPosition'; sourceId: Id; wavelengthNm: number; z: number }
    | { kind: 'collimated'; sourceId: Id; wavelengthNm: number };
  export interface Variable { opticId: Id; minZ: number; maxZ: number }
  export interface SolveResult { setup: Setup; cost: number; residuals: number[]; converged: boolean }
  export function solve(setup: Setup, targets: Target[], variables: Variable[]): SolveResult;
  ```
  Residuals: radius errors as a fraction of the target; position errors in mm/1 mm; collimation as far-field half-angle in mrad. An overlap between any two optics adds a residual of (overlap mm)×100.

- [ ] **Step 1: Write the failing tests:**
  - `expands a beam`: Gaussian w0 = 0.5 mm at z=0, 780 nm; ideal f=50 (z ∈ [20, 200]) and f=150 (z ∈ [150, 500]); targets radius 1.5 mm at z=700 and at z=1000 → `cost < 1e-6`, lens separation within 200 ± 10 mm (near-afocal ×3).
  - `respects bounds`: a target only reachable outside the bounds → returned positions within bounds, `converged: false`.
  - `solver avoids overlap`: two catalogue lenses whose unconstrained optimum overlaps → the result has no `overlap` warning.
  - `fast`: four variables finish in < 1 s.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement Nelder–Mead on unbounded variables mapped to bounds by `z = min + (max − min)·(1 + sin u)/2`; start from the current positions plus 4 random restarts (seeded PRNG so tests are deterministic); stop at cost < 1e-12, 2000 evaluations or 900 ms. Returns a new `Setup`; the input is not mutated.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 9: UI shell — store, layout, elements table, properties panel

**Files:**
- Create: `optics-sim/src/ui/main.ts`, `app.ts`, `store.ts`, `table.ts`, `properties.ts`, `defaultSetup.ts`, `optics-sim/src/styles.css`; modify `index.html`
- Test: `optics-sim/tests/store.test.ts`, `optics-sim/tests/properties.test.ts` (jsdom)

**Interfaces:**
- Produces (in `store.ts`):
  ```ts
  export interface AppState { setup: Setup; selectedId: Id | null }
  export interface Store {
    get(): AppState;
    apply(update: (s: Setup) => Setup, label: string, opts?: { coalesce?: string }): void; // coalesce: consecutive updates with the same key (one drag) form one undo step
    select(id: Id | null): void;
    undo(): void; redo(): void; canUndo(): boolean; canRedo(): boolean;
    subscribe(fn: (s: AppState) => void): () => void;
  }
  export function createStore(initial: Setup, storage?: Storage | null): Store; // autosave key 'optics-sim/setup/v1'
  export function loadAutosave(storage?: Storage | null): Setup | null;
  ```
- Produces (in `properties.ts`): `fieldsFor(item: Source | Optic): FieldSpec[]` where `FieldSpec = { key: string; label: string; unit: string; validate: (v: number) => string | null }`.
- Layout follows the spec's sketch: toolbar (add source ▾, add optic ▾, open, save, undo, redo, ray/Gaussian toggles), bench canvas, elements table + properties side by side, readouts + solver panel below.

- [ ] **Step 1: Write the failing tests:**
  - `undo/redo`: two applies, undo twice restores the initial setup, redo once re-applies the first.
  - `coalesced drag is one step`: ten applies with the same `coalesce` key → one undo returns to the start.
  - `autosave survives throwing storage`: a `Storage` whose `setItem`/`getItem` throw → no exception; `loadAutosave` returns null.
  - `autosave round trip`: `createStore` then `loadAutosave` on the same storage returns the setup.
  - `field validation`: ideal lens focal length 0 → error message; −50 → null; aperture −1 → error; wavelength 0 → error.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Table rows: type, label, z, part/f, y/decentre, with a warning marker on rows named in any `TraceWarning`; editing a cell or a properties field validates and calls `store.apply`; selecting a row selects the element. Properties panel shows every field of the selected element (including SLM phase terms, add/remove terms, loading a `table` term from a CSV of `y_mm, phase_rad`, and the lens's flip and source label "from drawing"/"from Zemax file"). Open/save use a file input and a Blob download of `serialiseSetup`; a `parseSetup` error is shown in a dialog and leaves the state untouched. `defaultSetup.ts`: an on-axis point source (780, 852 nm) imaged by two catalogue achromats onto a screen, plus a Gaussian source (780 nm) through two ideal lenses offset in y so both are visible.
- [ ] **Step 4:** Run → PASS. `npm run dev` shows the shell with table and properties working (bench may be empty).
- [ ] **Step 5:** Commit and push.

---

### Task 10: Bench canvas — view, drawing and interaction

**Files:**
- Create: `optics-sim/src/ui/bench/view.ts`, `draw.ts`, `interact.ts`, `optics-sim/src/ui/colours.ts`
- Test: `optics-sim/tests/view.test.ts`, `optics-sim/tests/colours.test.ts`

**Interfaces:**
- Produces (in `view.ts`): `interface View { zMin; zMax; yMin; yMax; widthPx; heightPx }`, `toScreen(v, z, y): [number, number]`, `toWorld(v, px, py): [number, number]`, `zoomAbout(v, px, py, factor, axis: 'both' | 'y'): View`, `pan(v, dxPx, dyPx): View`, `fitAll(setup, trace, widthPx, heightPx): View`.
- Produces (in `colours.ts`): `wavelengthColour(nm: number): string` (visible 380–750 nm follows the spectrum; above 750 nm a fixed distinct palette keyed by rounded wavelength, labelled in the legend as IR).
- Produces (in `draw.ts`): `drawBench(ctx, view, state: AppState, trace: SetupTrace, gaussians: GaussianTrace[], opts: { showRays: boolean; showGaussian: boolean; hoverZ: number | null }): void`.
- Produces (in `interact.ts`): `attachBenchInteraction(canvas, store, getView, setView): void`; `hitTest(view, setup, px, py): Id | null`.

- [ ] **Step 1: Write the failing tests:**
  - `zoom keeps the cursor point fixed`: `toWorld` of the cursor is unchanged by `zoomAbout`, for both axes and y-only.
  - `y-only zoom leaves z range`: `zMin/zMax` unchanged.
  - `round trip`: `toWorld(toScreen(p)) = p`.
  - `colours`: 532 nm is green-dominant; 780, 852 and 1064 nm return three different colours; the same input always returns the same colour.
  - `hitTest`: a click within 6 px of a lens outline returns its id; empty space returns null.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Draw: axis, lens cross-sections from their real surface sags (decentred, tilted, flipped), ideal lenses as double-arrows, SLM as a hatched bar with the pitch-limited region highlighted, apertures as jaws, screens as dashed lines; rays as polylines in their wavelength colour, blocked rays ending at the blocker; Gaussian ±w envelope as a translucent band around the centre path; legend of wavelengths; a scale bar per axis. Interaction: wheel zooms both axes about the cursor, Shift+wheel zooms y only; drag on empty space pans; drag an element moves `z` (Alt-drag moves `y`/decentre) through `store.apply(…, { coalesce: 'drag:<id>' })`; click selects; Delete removes the selection; hover shows `w(z)` and z, y in a tooltip. Re-trace and redraw on store change, at most once per animation frame. Wire into `app.ts`.
- [ ] **Step 4:** Run → PASS. In `npm run dev`, dragging an achromat updates rays live with no visible lag.
- [ ] **Step 5:** Commit and push.

---

### Task 11: Readouts, solver panel and catalogue browser

**Files:**
- Create: `optics-sim/src/ui/readouts.ts`, `solverPanel.ts`, `catalogue.ts`
- Test: `optics-sim/tests/readouts.test.ts` (jsdom; formatting only)

**Interfaces:**
- Consumes: `focusPerWavelength`, `focusShiftCurve`, `screenStats` (Task 5), `traceGaussians`, `radiusAt` (Task 4), `solve` (Task 8), `parseZmx`/`decodeZmx` (Task 6), `lenses.json`.
- Produces (in `readouts.ts`): `formatFocus(results: FocusResult[], offAxis: boolean): string[]` (lines shown in the panel) and `renderReadouts(el, state, trace, gaussians): void`.

- [ ] **Step 1: Write the failing tests** for `formatFocus` and its Gaussian counterpart:
  - two wavelengths → a line per wavelength and one `Δfocus 780→852 nm: +0.410 mm` line (sign = longer minus shorter).
  - `approximate: true` → absolute positions carry "approx."; the Δ line does not.
  - `collimated` → "no focus (collimated)"; `no-rays` → "No rays reach this point".
  - off-axis source → the meridional note from Global Constraints is present.
  - Gaussian collimated waist → "collimated"; a `clipped` warning → "unreliable: beam clipped at <label>".
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Readouts panel: per source, per wavelength focus/waist lines, screen spot stats, warnings; a small focal-shift plot (canvas) for the selected point/collimated source over a user-set range. Solver panel: add targets (kind, source, wavelength, values) and variables (optic, min/max z, defaulting to ±50 mm around the current position); Run shows cost and proposed positions; Accept applies them as one undo step; Discard leaves the setup alone. Catalogue browser: searchable list of `lenses.json` by part number/focal length; Add places the lens after the selected element; Import .zmx reads a file through `decodeZmx`/`parseZmx`, shows a `ZmxError` message on failure, and for unknown glasses offers a dropdown of bundled glasses or a typed constant index, stored in `setup.glasses`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit and push.

---

### Task 12: Single-file build, smoke test, README

**Files:**
- Create: `optics-sim/e2e/smoke.mjs`
- Modify: `optics-sim/README.md`

- [ ] **Step 1: Write the smoke test** (`e2e/smoke.mjs`, plain Node using the pre-installed `playwright`): open `file://…/dist/index.html` in Chromium at 1400×900; assert no console errors; the bench canvas has non-background pixels; the readouts contain "Δfocus"; drag the first achromat 60 px right on the canvas and assert its z in the table increased; press Ctrl+Z and assert z is restored; save the setup, reload the page, open the saved file and assert the table matches; take a screenshot to `e2e/screenshot.png` (gitignored) at deviceScaleFactor 1.
- [ ] **Step 2:** Run `npm run e2e` → PASS. Confirm `dist/` contains only `index.html` and that it works when copied elsewhere.
- [ ] **Step 3:** README: what the app does and doesn't do (from the spec's goals / non-goals, including meridional-only and "drawing" lenses being approximate in absolute position), how to open the built file, controls (wheel / Shift+wheel / drag / Alt-drag / Delete / Ctrl+Z), how to import `.zmx`, the save format (from Task 7), how to regenerate `glasses.json` and the catalogue.
- [ ] **Step 4:** Run `npm test` and `npm run e2e` one last time; both pass. Run `uv run ruff check .` from the pytweezer root to confirm nothing Python changed.
- [ ] **Step 5:** Commit and push.
