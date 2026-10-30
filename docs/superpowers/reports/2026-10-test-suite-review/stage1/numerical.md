# Stage 1: Numerical and coordinators

## pytweezer/coordinators/base.py

No entries: the only logic is the role lookup in `require_role`, and a missing role raises a `KeyError` at construction, which is loud.

## pytweezer/coordinators/__init__.py

No entries: it only re-exports `Coordinator`.

## pytweezer/coordinators/rearrangement.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| numerical/rearrangement#1 | With no detector, `_extract_occupancy` (top-hat, window sums, then threshold) marks exactly the grid sites that hold a bright spot, in `grid_positions` key order. | 3 | unit | quiet: wrong atoms are paired and moved, and the array fills with holes nobody can trace back | a synthetic image with Gaussian spots on a known subset of sites; the mask must equal that subset |
| numerical/rearrangement#2 | `_pairing_record` accounts for every atom: `occupied_idx` = `moving_idx` ∪ `discarded_idx` (disjoint), and `unfilled_target_idx` lists exactly the targets missing from `final_idx`, whether there are more atoms than targets or fewer (parametrised). | 3 | unit | quiet: per-atom survival analysis blames the wrong sites | conservation of atom number: n_occupied = n_moving + n_discarded and n_targets = n_moving + n_unfilled, on a hand-built plan |
| numerical/rearrangement#3 | `_play_sequence_pipelined` hands every generated frame to `slm.update_mask` exactly once, in generation order, and returns that count (parametrised over Rearrangement and Transport, which carry identical copies). | 4 | unit | quiet: a dropped or reordered frame makes the atoms jump, and the loss looks like a physics problem | — |
| numerical/rearrangement#4 | `_preload_sequence_pipelined` stores frame k in on-board slot k for every k. | 4 | unit | quiet: the hardware-clocked sequence plays out of order | — |
| numerical/rearrangement#5 | Successive `update_mask` calls on the display path are at least `panel_period_s` apart, including the first pair (parametrised over the plain and `_timings` paths). | 4 | unit | quiet: frames overwrite each other before the panel refreshes, so the atoms see larger jumps than `d0` asks for | — |
| numerical/rearrangement#6 | If the SLM write fails partway through a sequence longer than `UPLOAD_QUEUE_DEPTH`, the error reaches the caller and the call returns instead of hanging. | 4 | unit | quiet if swallowed (a truncated move is reported as a success); a hang freezes the whole composite server | — |
| numerical/rearrangement#7 | `arm_rearrangement_preload_trigger` arms auto-increment for n frames and asks the pulser for n−1 pulses at `period_us`. | 4 | unit | quiet: one trigger too many wraps the SLM back to frame 0, i.e. the initial array | — |
| numerical/rearrangement#8 | After a pulser failure, `arm_rearrangement_preload_trigger` still stops auto-increment and switches wait-for-trigger off. | 4 | unit | quiet: the SLM keeps advancing on stray triggers, or later writes block until timeout | — |
| numerical/rearrangement#9 | `shutdown` clears the initialised state even when `camera.stop_acquisition` raises. | 4 | unit | loud: an exception during server teardown | — |
| numerical/rearrangement#10 | `arm_rearrangement` passes the occupancy mask it measured, and `d0`/`profile` from `initialise`, to `iter_rearrangement_sequence`. *Added from stage 2; covered by `tests/test_rearrangement.py::test_arm_streams_sequence_to_slm` (deleted in PR #12 because it failed; this entry now has no cover).* | 4 | unit | quiet: atoms move at the wrong speed, or to and from the wrong sites | — |

## pytweezer/coordinators/transport.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| numerical/transport#1 | `arm_transport_pattern` plays the rearrangement frames and then the transport frames to the SLM, with one camera image before, one between and one after, and returns the three images in that order. | 4 | unit | quiet: images are labelled with the wrong stage, so analysis compares the wrong pair | — |
| numerical/transport#2 | The rearrangement stage gets `d0`, and the transport stage gets `d0_transport`, `d0_z_transport` and `step_norm`, from `initialise`. | 4 | unit | quiet: atoms move at the wrong speed and are lost, with no error | — |
| numerical/transport#3 | `path_fn="linear"` passes `linear_path` and `path_fn="lift"` passes `lift_path(d_lift, smooth_corners, only_moving)` to the transport sequence (parametrised). | 4 | unit | quiet: atoms are moved in-plane through occupied sites instead of lifted over them | — |

## pytweezer/optimisation.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| numerical/optimisation#1 | `DetectResonances` stores the VCO with the most background-subtracted counts as `cool_vco_resonance` and `rep_vco_resonance` (parametrised over the two scans). | 1 | unit | quiet: the imaging resonance is saved off-peak, and every later image is dimmer | a fake camera whose counts peak at a known VCO value |
| numerical/optimisation#2 | `OptimiseMOTNumber` stores each scan's peak value under that scan's own parameter (`tMOTLoadCoolVCO`, `tMOTLoadCoolVVA`, `tMOTLoadRepVCO`, `tMOTLoadRepVVA`; parametrised). | 1 | unit | quiet: a MOT setting is written to the wrong key and the MOT runs off-optimum | a fake camera with a known peak per scanned key |
| numerical/optimisation#3 | At each `OptimiseMOTNumber` scan point, the scanned key carries the scan value and every other key carries its current stored value, so later scans run at the earlier optima (parametrised over the four scans). | 1 | unit | quiet: scans run at stale settings and the "optimum" is not optimal | — |
| numerical/optimisation#4 | After `OptimiseMOTShimParameters`, ExpParams holds, under `tMOTShimX/Y/ZccValue`, the evaluated shim triple whose loading probability was highest, as MotMaster received it. | 1 | unit | quiet: swapped axes or a non-best point are saved as the new shims | a fake loading probability that depends on the shim values MotMaster receives |
| numerical/optimisation#5 | In `OptimiseMolassesShimParameters`, a setting with loading ≥ 0.5 always beats one below 0.5, and among those, higher survival wins. | 3 | unit | quiet: the optimiser climbs towards poor loading or poor survival | the ordering defined by the piecewise objective, with fake loading and survival values |
| numerical/optimisation#6 | `DetectMOTPosition` returns the pixel centre of the background-subtracted mean image. | 3 | unit | quiet: every scan sums counts in a window that is off the MOT | a fake camera with a Gaussian at a known pixel on a uniform background frame |

## pytweezer/phasemask.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| numerical/phasemask#1 | `get_jv_pairing_lap` returns the assignment with the smallest total squared distance, and every target gets at most one atom, for fewer atoms than targets, equal numbers, and more atoms (parametrised). | 3 | unit | quiet: longer moves, more loss, and possibly two atoms sent to one site | exhaustive search over all assignments for N, M ≤ 6 |
| numerical/phasemask#2 | The last frame of `iter_rearrangement_sequence` (all sites loaded) equals the hologram of the target array: weights, phases and positions of `terms2`. | 3 | unit | quiet: the atoms stop short of, or past, their target sites | the endpoint s(1) = 1 of both profiles; compare with `generate_phasemask(terms2)` on the same background, within 1 LSB |
| numerical/phasemask#3 | No trap moves more than `d0` in any frame, for both `minimum_jerk` and `linear` (parametrised). | 3 | unit | quiet: atoms are dragged faster than the trap can hold them | the peak of s′(τ) = 30τ²(1−τ)² is 1.875, so n = ⌈1.875·d_max/d0⌉ (or ⌈d_max/d0⌉ for linear) keeps max step·d_max ≤ d0 |
| numerical/phasemask#4 | The trap phase interpolates the short way round the circle: from π−0.1 to −π+0.1 it changes by +0.2 in total, not −(2π−0.2). | 3 | unit | quiet: trap intensities flicker in the middle of the move | analytic case |
| numerical/phasemask#5 | Traps with no atom, and surplus atoms left out of the assignment, have zero weight in every frame, while assigned traps ramp from w1 to w2. | 3 | unit | quiet: power goes to empty traps, the real ones get shallower, or a discarded atom stays trapped | the occupancy mask and assignment chosen by hand |
| numerical/phasemask#6 | For a single trap at (x0, y0), `generate_phasemask` gives a focal-plane peak at (x0/dx_f, y0/dy_f) pixels from the centre, with the sign right, in `simulate_focal_plane`. | 3 | unit | quiet: the array is scaled or mirrored and the sites no longer match the camera's grid | the FFT focal-plane model and dx_f = λf/L |
| numerical/phasemask#7 | `superposition_optimization` on a uniform 3×3 target gives trap intensities from `evaluate_traps` that are equal to within 2%. | 3 | unit | quiet: non-uniform trap depths and a lower loading fidelity | the target's own uniformity, and the 4-fold symmetry of a square grid |
| numerical/phasemask#8 | `get_zernike_polynomial` modes 1–13 are orthonormal on the unit disk: RMS 1, and pairwise inner products 0. | 3 | unit | quiet: aberration correction is applied with the wrong strength or mode | Zernike orthonormality, by numerical quadrature on a fine polar grid |
| numerical/phasemask#9 | `transform_phase_8bit(superimpose(masks))` maps phase onto 0–255 linearly modulo 2π, with no value outside the uint8 range. | 3 | unit | quiet: part of the phase range is clipped, and every hologram on the SLM is distorted | analytic mapping: phase φ goes to round(255·(φ mod 2π)/2π), up to a constant offset |
| numerical/phasemask#10 | `iter_rearrangement_sequence`, and `plan_rearrangement` followed by `iter_rearrangement_sequence_timings`, leave the caller's float32 `terms1` unchanged (parametrised over the two paths). | 4 | unit | quiet: the second rearrangement starts from the target positions with zeroed weights | — |

## Suspected bugs

- pytweezer/coordinators/rearrangement.py, also transport.py: `from pytweezer import analysis as an` binds the empty package `pytweezer/analysis/__init__.py`, which shadows `pytweezer/analysis.py`. So the default (no-detector) occupancy path raises `AttributeError: module 'pytweezer.analysis' has no attribute 'morphological_tophat_high_pass'`. Smallest input: `_extract_occupancy(np.zeros((10, 10)), 0)` with `_state={"detector": None, "grid_positions": {0: (5, 5)}, "window_size": 3, "feature_size": 2}`.
- pytweezer/coordinators/rearrangement.py, also transport.py: `initialise` uses `self.pm_gen`, taken from role `"phasemask_generator"`. The configured "Rb Rearrangement Rig" has no such sub-device and nothing else builds one, though the module docstring says `initialise` builds it. So `initialise` fails with `'NoneType' object has no attribute 'generate_phasemask'`. Smallest input: `Rearrangement({"slm": SimulatedSLM(), "camera": cam}, {}).initialise(...)` on a GPU host.
- pytweezer/coordinators/rearrangement.py, also transport.py: the panel pacing waits *after* each `update_mask`, so the first two writes are back to back. Frame 0 is shown only for one write time, which breaks the documented minimum interval. The `_play_sequence_pipelined_timings` path has the same bug. Smallest input: `panel_period_s=0.1` and `_play_sequence_pipelined([np.zeros((2, 2), np.uint8)] * 3)` with an SLM stub that records `perf_counter` per call. The gaps come out ≈ `[0.0, 0.1]`, not `[0.1, 0.1]`.
- pytweezer/coordinators/rearrangement.py, also transport.py: `shutdown` calls `print(..., exc_info=True)`, which raises `TypeError`. Smallest input: an initialised coordinator whose `camera.stop_acquisition` raises.
- pytweezer/coordinators/rearrangement.py, also transport.py: `arm_rearrangement_timings` runs the `_pairing_record`/`ttff_ms`/`med` block twice. The extra device syncs are harmless to the numbers, but it is a copy-paste duplicate. Smallest input: any `arm_rearrangement_timings()` call.
- pytweezer/coordinators/transport.py: it imports `linear_path, lift_path` from `pytweezer.phasemask`, and neither is defined there. The module therefore fails to import on every machine, which contradicts its docstring ("this module imports and status works on any machine"). Smallest input: `import pytweezer.coordinators.transport`.
- pytweezer/coordinators/transport.py: `arm_transport_pattern` cannot run against either generator in `phasemask.py`, for three reasons:
  - It passes `init_refine_iters`/`init_refine_damping`, which `iter_rearrangement_sequence` does not accept.
  - It passes 5-element `(w, phi, x, y, z)` terms where the 2D class unpacks 4.
  - It calls `PM.iter_transport_sequence`, which neither class defines.

  Smallest input: any `arm_transport_pattern()` call after `initialise`.
- pytweezer/coordinators/transport.py: `arm_rearrangement_timings` reads `s["num_images"]`, which `Transport.initialise` never stores. This raises `KeyError` only after the whole sequence has played; it is raised inside the `try` and again in the `except`. `arm_rearrangement_timings` and `arm_rearrangement_preload_trigger` would also hit the 4-vs-5 term unpack. `d0_z` and `fps` are stored but never used. Smallest input: `arm_rearrangement_timings()` after `initialise`.
- pytweezer/phasemask.py: `plan_rearrangement` builds `curr_w/phi/x/y` with `cp.asarray(..., dtype=float32)` and no `.copy()`. The fix and its comment are in `iter_rearrangement_sequence`. So, for float32 cupy `terms1`:
  - `curr_w[off_mask] = 0` zeroes the caller's weights at plan time.
  - `iter_rearrangement_sequence_timings` walks the caller's positions to the targets in place.

  The second `arm_rearrangement_timings` then starts from the wrong array. Smallest input: float32 cupy `terms1` with one empty site; after `plan_rearrangement(terms1, terms2, occ)` and consuming the timings generator, `terms1[2]` equals the target x and `terms1[0]` has a zero.
- pytweezer/phasemask.py: `superimpose` returns phase in [0, 2π), but `transform_phase_8bit` assumes [−π, π). It adds π, so half the range lands at 255–382 before `astype(uint8)`. Out-of-range float→uint8 is undefined: if the cast wraps, the result is a harmless global π offset; if it saturates, which CUDA's `cvt` does by default, half the phase range is clipped to 255. Smallest input: `transform_phase_8bit(superimpose([cp.full((1, 1), 1.5 * cp.pi)]))` gives 318.75 before the cast.
- pytweezer/phasemask.py: when every atom already sits on its target (`max_dist == 0`), `n_steps` is 0. `dw` is then divided by zero and no frames are yielded, and `arm_rearrangement_preload_trigger` goes on to call `start_auto_increment(0)` and `pulser.send_pulses(-1)`. An all-empty `occ_mask` instead raises on `.max()` of an empty array, and also on `1.0 / len(moving_idx)`. Smallest input: `terms1 == terms2` with `occ_mask` all True.
- pytweezer/phasemask.py: at import, the module sets `os.environ['CUDA_PATH']` to a hard-coded Windows path on every machine. It also imports `cupy`/`lap` and compiles `cp.ElementwiseKernel`. Without a GPU stack, even the pure-numpy helpers (`get_zernike_polynomial`, `rotate_*`, `ZernikeCalibrator`) therefore cannot be imported, so none of the entries above can run on CI. Smallest input: `import pytweezer.phasemask` without cupy.
- pytweezer/phasemask.py: `OptimisationBasedPhasemaskGeneratorGPU3D.generate_rearrangement_sequence` has two faults:
  - `n_steps` comes from the in-plane distance only, so a pure-z move gives 0 frames.
  - `z2[final_idx]` indexes the numpy `Zn` returned by `generate_weighted_array` with a cupy index, which raises `TypeError`.

  Its terms also carry `array_shape`, which the 2D-style unpack expects. Smallest input: terms from `generate_weighted_array` for both arrays, with a full `occ_mask`.
- pytweezer/phasemask.py: `ZernikeCalibrator.get_zernike_coeff` ignores `polynomial_degree` and always evaluates the 10-term cubic. A degree-2 file (6 coefficients) raises `IndexError`; a degree-4 file is silently truncated. Smallest input: a JSON with `{"polynomial_degree": 2, "coefficients": [0, 0, 0, 0, 0, 0]}`.
- pytweezer/optimisation.py: when `an.fit_gaussian` raises, `DetectMOTPosition` prints "Could not detect MOT" and then fails at `return` with `UnboundLocalError` on `centre_x`. Every caller dies on the same path. Smallest input: blank frames from the camera, so the fit fails.
- pytweezer/optimisation.py: `OptimiseMOTShimParameters` and `OptimiseMolassesShimParameters` divide by `n_calls - 1` in the xi schedule. Smallest input: `n_calls=1` gives `ZeroDivisionError`.
