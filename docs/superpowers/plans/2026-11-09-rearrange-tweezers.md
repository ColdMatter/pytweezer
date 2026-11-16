# Rearrange Tweezers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A queued `RearrangeTweezers` experiment that runs one rearrangement shot per point through the `Rb Rearrangement Rig` and MOTMaster, and records occupancy, pairing, target fill and timings.

**Architecture:** The phase-mask generator learns to build move frames on the displayed mask's own layers. The rig keeps the initial array's optimised terms from `make_array`, builds the target terms in `prepare_rearrangement`, and runs one shot in `rearrange`. `RearrangeTweezers(LoadTweezers)` prepares the rig, then runs `rig.rearrange` and MOTMaster's `Go` together with `run_parallel`.

**Tech Stack:** Python, cupy (GPU host only), sipyco RPC, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-09-rearrange-tweezers-design.md`

## Global Constraints

- Three HamCam frames per shot: occupancy, then two follow-up frames. One 3-frame snap acquisition, read with `acquire_n_frames(1)` then `acquire_n_frames(2, start_frame=1)`.
- Script `RbTweezerBasic2026_2_Assemble`, with `LoadTweezers.rb.saved` as its bindings.
- Existing `initialise`, `arm_rearrangement*` and the generator's existing methods keep their behaviour: the RnD notebooks use them.
- Nothing crossing RPC is a cupy array. The rig keeps terms as numpy; `cp.asarray` happens inside the generator.
- `coordinators/rearrangement.py` must keep importing and serving `status()`/`test()` without a GPU. Its module-level `cp` is `None` there, so the rig never calls `cp` on the `rearrange` path. Use `.get()` when present, otherwise `np.asarray`.
- Occupancy and target fill use the same scoring: top-hat box sum in photons above the `FindTrapSites` threshold (`trap_threshold`).
- Sites are `(y, x)` ROI pixels, in the initial array's order, as `trap_sites` are.
- Defaults: kagome, 70 sites, 6 × 6 square, 5 µm, 2.5°, 5000 iterations, damping 0.5, phase randomness 1.0, `d0` 0.5 µm, linear profile, `arm_delay` 0.1 s.
- UK English, Google-style docstrings, `get_logger`, ruff clean, the suite green before each push. Branch: `ccr-bb840753-gv9frj`.
- One addition to the spec: `array_hologram` returns `optimised_weights` beside `phases`. A move must start from the optimised terms, as the notebooks' `terms1` did, and `weights` stays the requested envelope weights that `SlmTweezerArray` displays.

## Review Focus

1. **The SLM is rewritten between `prepare` and a shot** (someone runs `SlmTweezerArray`): `rearrange` must refuse, not move atoms with stale terms. Test in Task 2.
2. **Reading the follow-up frames fails after MOTMaster has run:** the acquisition is still stopped and the initial mask restored. Test in Task 2.
3. **More atoms than targets:** the surplus gets `destination` −1, and no target index appears twice. Test in Task 2.
4. **A scanned `hamcam_gain`:** it is re-sent before each shot whose gain changed. Test in Task 3.
5. **Saved `trap_sites` in a different order from the rig's array** (same geometry, so `LoadTweezers`' check passes): the affine residual refuses the task. Test in Task 3.

---

### Task 1: Generator and `array_hologram` build move frames on the mask's layers

**Files:**
- Modify: `pytweezer/helper_scripts/phasemask.py` (`OptimisationBasedPhasemaskGeneratorGPU.plan_rearrangement`, new `iter_planned_sequence`)
- Modify: `pytweezer/helper_scripts/array_hologram.py`
- Test: `tests/test_slm_tweezer_array.py` (has the GPU-gated `generator` fixture)

**Interfaces:**
- Produces:
  - `array_hologram.background_phase(generator, *, blaze_um, fresnel_focal_length_mm, zernike) -> cupy array`: `superimpose` of the Fresnel lens, the blaze and (if `zernike`) `generator.zernike`. `array_hologram` builds its mask as `superimpose([array_phase, background_phase(...)])`.
  - `array_hologram.pupil_arrays(generator, radius_um, dump_dx_um, dump_dy_um) -> (inside, dump)`. `apply_pupil` becomes `cp.where(inside, phase, dump)` using it.
  - `array_hologram.optimise_terms(generator, weights, phases, x_um, y_um, *, iterations, damping) -> dict`, with keys `array_phase` (cupy), `optimised_weights`, `phases`, `x_um`, `y_um` (numpy) and `uniformity`, `min_max_ratio`, `mse` (float). `array_hologram` uses it.
  - `array_hologram.array_hologram(...)` result gains `phases` and `optimised_weights` (numpy).
  - `array_hologram.target_terms(generator, *, kind, rows, columns, sites, spacing_um, angle_deg, envelope, blaze_um, iterations, damping, phase_randomness) -> dict`, with the keys of `optimise_terms` minus `array_phase`, plus `weights` (the requested weights):
    - `"square"` uses `generate_weighted_array(np.ones((rows, columns)), spacing_um, ...)`.
    - `"kagome"` uses `generate_kagome_lattice(sites, spacing_um)`, then `rotate_coordinates(..., angle_deg)`, with random phases scaled by `phase_randomness`.
    - Weights are `envelope_weights(envelope, x + blaze_x, y + blaze_y)`, or ones if `envelope` is `None`.
    - Any other `kind` raises `ValueError`.
  - `plan_rearrangement(terms1, terms2, occ_mask, d0=0.5, profile="minimum_jerk", background=None)`: `background=None` keeps `superimpose([fresnel, blaze, zernike])`.
  - `iter_planned_sequence(plan, *, to_host=False, pupil=None)` yields one uint8 frame per step, as `iter_rearrangement_sequence_timings` does but without events. `pupil=(inside, dump)` applies `cp.where(inside, composite, dump)` before `transform_phase_8bit`.

- [ ] **Step 1: Write the failing GPU-only test** in `tests/test_slm_tweezer_array.py`

```python
def test_the_first_move_frame_matches_the_mask_the_array_was_shown_with(generator):
    optics = dict(blaze_um=(30.0, -12.0), fresnel_focal_length_mm=1125.0, zernike=True)
    hologram = array_hologram.array_hologram(
        generator, rows=2, columns=2, spacing_um=5.0, angle_deg=0.0, envelope=None,
        pupil_radius_um=8000.0, pupil_dump_um=(100.0, 0.0), iterations=20,
        damping=0.5, phase_randomness=1.0, **optics,
    )
    terms = [hologram[k] for k in ("optimised_weights", "phases", "x_um", "y_um")]
    nudged = [terms[0], terms[1], terms[2] + 1e-3, terms[3]]
    plan = generator.plan_rearrangement(
        terms, nudged, np.ones(4, bool), d0=1.0,
        background=array_hologram.background_phase(generator, **optics),
    )
    frames = list(generator.iter_planned_sequence(
        plan, to_host=True, pupil=array_hologram.pupil_arrays(generator, 8000.0, 100.0, 0.0)
    ))
    assert len(frames) == 1
    assert np.mean(frames[0] != hologram["mask"]) < 0.01
```

- [ ] **Step 2:** `uv run pytest tests/test_slm_tweezer_array.py -q -rs`. Expected: SKIPPED without CUDA. On the GPU host it fails with `unexpected keyword 'background'`.
- [ ] **Step 3: Implement** the interfaces above. Keep the existing callers of `array_hologram` and `apply_pupil` working, including `SlmTweezerArray`. Extend `fake_gpu`'s fake `hologram` in `tests/test_rearrangement.py` to return `phases` and `optimised_weights` (zeros and ones).
- [ ] **Step 4:** `uv run pytest tests/test_slm_tweezer_array.py tests/test_rearrangement.py -q`. Expected: everything passes, with the GPU tests skipped here.
- [ ] **Step 5:** ruff, then commit: "Build rearrangement frames on the displayed mask's layers".

### Task 2: The rig prepares and runs one rearrangement shot

**Files:**
- Modify: `pytweezer/coordinators/rearrangement.py`
- Modify: `pytweezer/analysis/array_loading.py`
- Test: `tests/test_rearrangement.py`, `tests/test_tweezer_array.py`

**Interfaces:**
- Consumes: Task 1's `target_terms`, `background_phase`, `pupil_arrays`, `plan_rearrangement(background=)`, `iter_planned_sequence(pupil=)`, and the `array_hologram` result keys.
- Produces:
  - `array_loading.occupied_sites(frame, sites, *, window_size, feature_size, threshold) -> np.ndarray[bool]`. `sites` is `(N, 2)` `(y, x)`. It returns `photons_at_sites([top-hat filtered frame], dict(enumerate(sites)), window_size)[0] > threshold`.
  - `rearrangement.destinations(n_sites, moving_idx, final_idx) -> np.ndarray[int]`: −1 everywhere except `out[moving_idx] = final_idx`.
  - `make_array` also stores `(optimised_weights, phases, x_um, y_um)`, the generator, and its layer settings (envelope, blaze, Fresnel, Zernike, pupil).
  - `Rearrangement.prepare_rearrangement(*, kind, rows, columns, sites, spacing_um, angle_deg, iterations, damping, phase_randomness, occupancy_sites, threshold_photons, window_size, feature_size, follow_up_frames=2) -> dict`. It returns `x_um`, `y_um`, `weights`, `uniformity`, `min_max_ratio`, `mse`. It raises `RuntimeError` when there is no GPU or `get_array()` is `None`.
  - `Rearrangement.rearrange(*, d0_um, profile, header) -> dict`. It returns:
    - `frames` `(3, H, W)`;
    - `occupied` (bool per site) and `destination` (int per site);
    - `n_frames`;
    - `occupancy_ms`, `calculation_and_upload_ms` and `total_rearrangement_ms`.

    It raises `RuntimeError` if `prepare_rearrangement` hasn't run, or if `get_array()` is `None`. A frame-0 failure re-raises as `RuntimeError("no occupancy frame: MOTMaster may have fired before the camera was armed (arm_delay): <error>")`.

Tests use `fake_gpu` plus:
- `fake_gpu` gains `target_terms` (a 3-site target), `background_phase` and `pupil_arrays` stand-ins.
- `make_generator` returns a `FakeGenerator`. Its `plan_rearrangement` records its arguments and returns `{"moving_idx": ..., "final_idx": ...}` from the test's choice. Its `iter_planned_sequence` yields `n` zero uint8 frames of shape `(1024, 1024)`.
- `rig.camera` is replaced by a fake that returns given frames, records `start_acquisition`/`acquire_n_frames(n, start_frame=..., broadcast=..., header=...)`/`stop_acquisition`, and can raise on a chosen read.

- [ ] **Step 1: Write the failing tests** in `tests/test_rearrangement.py`:
  - `test_a_shot_moves_the_occupied_atoms_and_returns_three_frames`:
    - frame 0 is lit at sites 0 and 2 of 4; the plan returns `moving_idx=[0, 2]`, `final_idx=[1, 0]`;
    - asserts `occupied == [T, F, T, F]`, `destination == [1, -1, 0, -1]` and `frames.shape[0] == 3`;
    - asserts the camera reads were `(1, 0)` then `(2, 1)`, broadcast with the header;
    - asserts `rig.slm.last_mask` is the array's mask afterwards.
  - `test_a_shot_with_no_atoms_plans_nothing_and_still_reads_the_follow_up_frames`: `n_frames == 0`, the plan was never called, and both reads happened.
  - `test_destinations_send_surplus_atoms_nowhere`: `destinations(5, [0, 3], [1, 0]) == [1, -1, -1, 0, -1]`, and no target index repeats.
  - `test_a_failed_occupancy_read_names_arm_delay_and_stops_the_camera`: it raises with `"arm_delay"` in the message, `stop_acquisition` was called, and the mask is restored.
  - `test_a_failed_follow_up_read_still_stops_the_camera_and_restores_the_mask`.
  - `test_a_shot_refuses_once_something_else_has_written_the_slm`: after `prepare_rearrangement`, a `rig.slm.update_mask(zeros)`, then `rearrange` raises `RuntimeError` and the camera was never started.
  - `test_prepare_rearrangement_refuses_without_an_array_on_the_slm`.
  - `test_prepare_rearrangement_refuses_without_a_gpu` (no `fake_gpu`: `_HAS_CUPY` is false here; the error is the existing "needs cupy/lap and a CUDA GPU").
  - `test_plan_is_built_on_the_layers_make_array_used`: the fake plan received the `background_phase` built from `OPTICS`' blaze, Fresnel and Zernike.
  - `test_occupied_sites_counts_only_sites_above_threshold`, in `tests/test_tweezer_array.py` (it has `FakeHamCam._frame` and `midpoint_threshold()`): a frame with every other site of `SITES` loaded gives exactly that pattern.
- [ ] **Step 2:** `uv run pytest tests/test_rearrangement.py -q`. Expected: FAIL (`prepare_rearrangement` is not defined).
- [ ] **Step 3: Implement** the interfaces.
  - In `rearrange`: check the array and the prepared state, show the initial mask, start the acquisition, read frame 0, score it, plan (skipped if nothing is occupied), and play the frames through the existing `_play_sequence_pipelined`.
  - Read frames 1 and 2. In `finally`, stop the acquisition and show the initial mask.
  - Time with `time.perf_counter()` as `arm_rearrangement` does.
- [ ] **Step 4:** `uv run pytest tests/test_rearrangement.py tests/test_tweezer_array.py -q`. Expected: PASS.
- [ ] **Step 5:** ruff, then commit: "Rig prepares and runs single rearrangement shots".

### Task 3: `RearrangeTweezers`

**Files:**
- Create: `pytweezer/experiments/rearrange_tweezers.py`
- Modify: `pytweezer/experiments/load_tweezers.py`: move the gain re-send out of `take_shot` into `update_gain(self) -> None`, which `take_shot` calls.
- Test: `tests/test_rearrange_tweezers.py`

**Interfaces:**
- Consumes: Task 2's `prepare_rearrangement`, `rearrange` and `occupied_sites`; `run_parallel` and `after` from `pytweezer.helper_scripts.parallel`; `LoadTweezers.update_gain`.
- Produces:
  - `fit_affine(from_xy, to_xy) -> tuple[np.ndarray, float]`: a `(3, 2)` matrix from `np.linalg.lstsq` on `[x, y, 1]`, and the RMS residual in pixels.
  - `apply_affine(matrix, xy) -> np.ndarray`.
  - The class, its arguments and its records exactly as in the spec's "The experiment" section:
    - `target_spacing` and `d0` are `Number(..., unit="um", scale=1e-6)`;
    - `target_angle` is in radians, shown in degrees (`scale=math.pi / 180`), as in `SlmTweezerArray`;
    - `arm_delay` is `Number(0.1, unit="ms", scale=1e-3, min=0)`;
    - `target_kind` is `Choice(["kagome", "square"])` and `profile` is `Choice(["linear", "minimum_jerk"])`.

Fixture: as `tests/test_tweezer_array.py`'s `rig`, importing `SAVED`, `SITES`, `FakeSequencer`, `FakeHamCam` and `FakeORCA` from it, with a `FakeRearrangementRig`:
- `get_array()` returns `{"sites": 64, "x_um": ..., "y_um": ...}` from `default_geometry()`, which maps to `SITES` by `px = 2 * µm + 60`.
- `prepare_rearrangement` records its kwargs and returns a target of `default_geometry(3, 3, 5.0)`.
- `rearrange` records `time.monotonic()` and returns frames lit with `FakeHamCam`'s Gaussian spots at chosen pixels.

`FakeSequencer` also records `time.monotonic()` at each `start_motmaster_experiment`.

- [ ] **Step 1: Write the failing tests:**
  - `test_fit_affine_recovers_a_known_map` (scale 2, a 3° rotation, an offset; residual < 1e-9);
  - `test_a_shot_records_occupancy_destination_fill_and_timings`:
    - frame 0 is lit at `SITES[:10]`, frames 1 and 2 at the first 5 and 4 targets;
    - asserts `occupied[:10].all()`, `loading == 10/64`, `target_filled_1 == [T]*5 + [F]*4` and `fill_fraction_2 == 4/9`;
    - asserts the shapes `rearrangement_images (1, 3, 128, 128)`, `initial_counts (1, 64)` and `target_counts (1, 2, 9)`;
  - `test_motmaster_fires_arm_delay_after_the_rig_is_asked_to_rearrange` (`arm_delay=0.05`; gap ≥ 0.04 s);
  - `test_target_sites_are_placed_by_the_fitted_map`: the constant `target_sites == (2 * y_um + 60, 2 * x_um + 60)`;
  - `test_it_refuses_without_an_array_on_the_slm`;
  - `test_it_refuses_saved_sites_in_another_order_than_the_array` (permuted `trap_sites`);
  - `test_it_refuses_a_target_whose_window_leaves_the_roi` (a target at `x_um = 200`);
  - `test_the_target_array_cannot_be_scanned`, parametrised over `target_spacing` and `target_kind`; `run_local` raises `ValueError`;
  - `test_a_scanned_gain_is_resent_before_each_shot_that_changes_it`;
  - `test_a_failed_go_fails_the_task`;
  - `test_the_rig_is_given_the_saved_sites_threshold_and_target`: the recorded kwargs carry `trap_sites` as `(64, 2)`, `threshold_photons == SAVED["trap_threshold"]` and `kind == "kagome"`.
- [ ] **Step 2:** `uv run pytest tests/test_rearrange_tweezers.py -q`. Expected: FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Implement** the interfaces.
  - `prepare` follows the spec's five steps. Refusal reasons come from `no_trap_sites_reason`, "no array on the SLM", "the saved trap sites do not fit the rig's array (residual N px)", and "target sites [i] leave the camera's ROI".
  - `planned_points` refuses every `target_*` axis, then defers to `LoadTweezers`.
  - `run_point` calls `update_gain()`, then the `run_parallel` call, then the records.
- [ ] **Step 4:** `uv run pytest tests/test_rearrange_tweezers.py tests/test_tweezer_array.py -q`, then the full suite. Expected: PASS.
- [ ] **Step 5:** ruff, then commit: "RearrangeTweezers: queued rearrangement shots". Push.

### Task 4: Skills and notes

**Files:**
- Modify: `.claude/skills/add-experiment/SKILL.md` ("Tweezer array measurements": add step 5, `RearrangeTweezers`, covering its session order, records and `arm_delay`)
- Modify: `.claude/skills/pytweezer-device-framework/SKILL.md` (the rig's `prepare_rearrangement`/`rearrange`, next to the coordinator example)
- Modify: `docs/notes/rearrangement_coordinator.md` (the queued path, beside `initialise`/`arm_rearrangement`)

- [ ] **Step 1:** Make the three edits, naming only methods and records that Tasks 1–3 define.
- [ ] **Step 2:** `cd docs && uv run --group docs make html`. Expected: "build succeeded".
- [ ] **Step 3:** Commit: "Skills and notes: queued rearrangement". Push.
