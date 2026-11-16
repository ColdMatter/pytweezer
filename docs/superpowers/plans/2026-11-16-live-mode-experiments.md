# Live Mode Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cameras stay live through MOTMaster experiments; experiments tag and read their shots from the running acquisition; `ContinuousAcquisition` goes.

**Architecture:** The `Camera` base gains shot tags: `start_shot(header)` marks the frame count, the live thread adds the header and a shot-relative `index` to frames from the mark on, `wait_frames(mark, n)` returns a shot's frames. `LoadTweezers` (and subclasses) and `RbMotMasterRunner` use them instead of snap acquisitions and their own broadcasts.

**Tech Stack:** Python, pylablib camera backends, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-live-mode-experiments-design.md`

## Global Constraints

- Branch `experiments-use-live-mode`, off `camera-applet` (PR #63); PR says `Depends on #63`.
- No new tests (user's standing rule). Existing tests whose fakes model snap acquisition are changed to the new design; `tests/test_camera_continuous.py` is deleted.
- UK spelling; Google-style docstrings describing current behaviour only; comments only for the surprising.
- Ruff check and format clean; `uv run pytest tests/ -q` passes before each commit.

## Review Focus

1. The next shot starts before the live thread has broadcast the last frames of the previous one: those frames keep the previous shot's tag.
2. The acquisition restarts mid-task (a gain change restarts live): old tags are dropped; a shot started after the restart is tagged from frame 0.
3. A shot's frames never arrive (MOTMaster failed, trigger cable out): `wait_frames` raises `TimeoutError` after the camera's `timeout`, the task fails, and `finish` still calls `end_shots()`.
4. `start_shot` on a camera that is not acquiring: `RuntimeError` saying to start live, not a silent mark of 0.
5. Idle frames after a task (external trigger fired by a notebook): untagged, running `index`, no `rid`.

These are checked by hand on the simulated camera (Task 1, Step 3), not by new tests.

---

### Task 1: Shot tags in the camera driver

**Files:**
- Modify: `pytweezer/drivers/camera_base.py` (`Camera.__init__`, `arm`, `stop_live`, `_broadcast_live`; new `start_shot`, `end_shots`, `wait_frames`)
- Modify: `.claude/skills/add-device-driver/SKILL.md` (one line naming the three methods where live mode is described)

**Interfaces:**
- Produces:
  - `Camera.start_shot(header: dict) -> int` — the shot's mark (frame number of its first frame).
  - `Camera.end_shots() -> None`
  - `Camera.wait_frames(mark: int, n: int, timeout: float | None = None) -> np.ndarray` — `(n, H, W)`; `timeout` defaults to `self.timeout`.

- [ ] **Step 1: Implement.**
  - State: `self._shot_tags: list[tuple[int, dict | None]]` (mark, header), ascending marks, under `self._lock`.
  - `start_shot`: under the lock, `RuntimeError` if `not self._acquisition_in_progress()` ("not acquiring; start live first"); mark = `self._count_frames()`; append `(mark, dict(header))`.
  - `end_shots`: append `(count, None)` while acquiring, else clear.
  - `arm` clears the tags when it starts a new acquisition; `stop_live` clears them.
  - `_broadcast_live`: for frame `sent`, the tag is the last entry with mark ≤ `sent`. With a header: `info.update(header)` and `info["index"] = sent - mark`. Without: unchanged (`index` = `sent`). After each batch drop entries superseded by a later entry with mark ≤ `sent`.
  - `wait_frames`: poll `frames_acquired()` every 5 ms until ≥ `mark + n`, then `read_frames_from(mark)[:n]`; `RuntimeError` if the count falls below `mark` (acquisition restarted); `TimeoutError` past the deadline.
- [ ] **Step 2: Run** `uv run pytest tests/test_camera.py tests/test_camera_frames.py -q` — passes.
- [ ] **Step 3: Hand check on `SimulatedCamera`** with a scratchpad script (not committed): live with a capturing image client; `start_shot({"rid": 7})`, `wait_frames(mark, 2)`; second shot started before the poll catches up; `end_shots()`; restart live. Check the five Review Focus cases from the captured headers.
- [ ] **Step 4: Ruff, commit** "Cameras tag a shot's live frames with the header an experiment hands them".

### Task 2: LoadTweezers family on live mode

**Files:**
- Modify: `pytweezer/experiments/load_tweezers.py` (`LoadTweezers.prepare`, `finish`, `take_shot`; `LoadTweezersNImages.take_shot`; `LoadTweezersMOTImage.prepare`, `finish`, `run_point`)
- Modify: `tests/test_tweezer_array.py` (fakes and the tests below)
- Modify: `.claude/skills/add-experiment/SKILL.md` where it describes how LoadTweezers takes frames

**Interfaces:**
- Consumes: Task 1's `start_shot`, `end_shots`, `wait_frames`.

- [ ] **Step 1: Implement.**
  - `prepare`: `stop_live()`, the same setters as now, then `start_live()`. Drop `setup_acquisition(2, mode="snap")` and `hamcam_was_live`.
  - `take_shot`: a changed gain → `stop_live()`, `set_gain`, `start_live()`; then `mark = hamcam.start_shot({"rid": self.rid})`, `run_sequences(**sequence)`, return `hamcam.wait_frames(mark, 2)`. No `start/stop_acquisition`, no broadcast.
  - `finish`: `hamcam.end_shots()` in a `try` before `super().finish()`; the camera stays live.
  - `LoadTweezersNImages.take_shot`: the same with 10 frames; the `hamcam_n_images` stream goes.
  - `LoadTweezersMOTImage`: `prepare` → ORCA `stop_live()`, `set_trigger_source("ext")`, `start_live()`; `run_point` → `orca_mark = orca.start_shot({"rid": self.rid})` before `take_shot()`, then `orca.wait_frames(orca_mark, 1)`; `finish` → `orca.end_shots()`.
- [ ] **Step 2: Update the tests to the design.**
  - `FakeHamCam`/`FakeORCA`: `start_shot(header)` records the header and returns 0; `wait_frames(mark, n)` returns what `acquire_n_frames` returned. Drop `acquire_n_frames`.
  - `test_every_shot_broadcasts_its_frames_tagged_with_the_task` → asserts every `start_shot` header is `{"rid": -1}`.
  - `test_load_tweezers_sets_the_hamcam_up_for_two_triggered_frames` → expected calls end `("set_gain", (600,)), ("start_live", ())`.
  - `test_a_scanned_gain_is_sent_before_each_shot_that_changes_it`: unchanged.
  - `test_a_failed_shot_stops_the_cameras_and_puts_live_ones_back_in_live` → renamed `test_a_failed_shot_ends_the_cameras_shots`; the last call is `end_shots` and there is no `stop_live` after the last `start_live`.
- [ ] **Step 3: Run** `uv run pytest tests/test_tweezer_array.py tests/test_trap_sites.py tests/test_tweezer_histogram.py -q` — passes.
- [ ] **Step 4: Ruff, commit** "LoadTweezers takes its shots from the live HamCam and ORCA".

### Task 3: RbMotMasterRunner tags its runs

**Files:**
- Modify: `pytweezer/experiments/rb_motmaster_runner.py` (`run_point`, new `finish`, class docstring)

- [ ] **Step 1: Implement.** `run_point` takes `marks = {name: camera.start_shot({"rid": self.rid})}` in place of `frames_acquired()`; settling and reading are unchanged. `finish` calls `end_shots()` on each ticked camera (in a `try`), then `super().finish()`.
- [ ] **Step 2: Run** `uv run pytest tests/ -q -k "motmaster or runner"` — passes.
- [ ] **Step 3: Ruff, commit** "RbMotMasterRunner tags its runs' live frames with the task".

### Task 4: Remove ContinuousAcquisition

**Files:**
- Delete: `pytweezer/experiments/camera_continuous.py`, `tests/test_camera_continuous.py`
- Modify: `.claude/skills/notebook-rig/SKILL.md:79` (drop the `ContinuousAcquisition` mention; the camera is set up by the Camera applet or its `CameraSettings` experiment)

- [ ] **Step 1:** Delete; `git grep -n "ContinuousAcquisition\|camera_continuous" -- . ':!docs/superpowers'` finds nothing.
- [ ] **Step 2: Run** `uv run pytest tests/ -q` — passes. `cd docs && uv run make html` — no new warnings.
- [ ] **Step 3: Commit** "Remove ContinuousAcquisition; the Camera applet and CameraSettings replace it".

### Task 5: Push and PR

- [ ] Push `experiments-use-live-mode`; open a PR against `main` with `Depends on #63`, listing the breaking changes: experiments leave cameras live on external trigger; `hamcam_n_images` stream removed; `ContinuousAcquisition` removed; experiment frames now broadcast by the live thread (tagged `rid`, shot `index`). Hardware checklist: LoadTweezers on the HamCam gives two frames per shot with the applet showing them; LoadTweezersMOTImage's ORCA frame; a scanned gain; RbMotMasterRunner.
