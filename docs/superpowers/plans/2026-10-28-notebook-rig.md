# Notebook Rig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lab-book notebooks on ph-bonesaw run MOTMaster shots (linked or not), scans and lock auto-tuning through pytweezer devices, get every camera's frames back in memory, and keep today's MOTMaster zips readable by `analysis3.readFromZip`.

**Architecture:** A new `pytweezer.notebook` package holds `Rig` (talks to one MOTMaster and any number of cameras by `get_device`, or to in-process objects in tests) and `zips` (finds the zip MOTMaster just wrote and injects frames into it, plus sibling zips for other species). Cameras gain a small frame-index protocol on `Camera` (`arm`, `frames_acquired`, `read_frames_from`, `settings`, live broadcast), so the rig collects "every frame since the mark" without knowing frames per shot. The ThorCam driver gains the CaF settings (serial, bulb trigger, binning, gain, live streaming) and a `CaF ThorCam` CONFIG entry.

**Tech Stack:** Python 3.11, sipyco RPC, pylablib cameras, numpy, `zipfile`, `tifffile`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-notebook-rig-design.md`

## Global Constraints

- No change to EDMSuite (`analysis3.py`, MOTMaster C#).
- A CaF zip holds only CaF frames; every other camera key goes to a sibling zip.
- Zip names: MOTMaster's `{element}{ddMMMyy}{batch:02d}_{n:03d}.zip` in `{data_root}\{yyyy}\{MM}{Mon}\{dd}\` (e.g. `...\2026\10Oct\27\CaF27Oct2600_012.zip`); a sibling replaces the leading element with the camera key (`Rb27Oct2600_012.zip`).
- Injected entry names must not end in `parameters.txt` or `hardwareReport.txt` unless they are copies of MOTMaster's (that is how `readFromZip` recognises them).
- Saved frames are uint16 TIFFs; `Shot.images[key]` is a float64 `(n, H, W)` stack, as `readFromZip` returns.
- Users never write classes or touch the queue for routine work.
- Use `get_logger`, Google-style docstrings, UK spelling, ruff-clean, comments only for the surprising.

## Review Focus

1. **Camera frames still arriving after `Go()` returns** (USB/transfer lag): the rig must wait until each camera's count has been stable for `frame_settle_s` before reading, not read immediately. Test in Task 4: a fake camera that delivers its last frame on the first post-`Go` poll still gives the full stack.
2. **`save=True` but MOTMaster wrote no new zip** (save toggle overridden, MOTMaster error swallowed): raise `RuntimeError` naming the day folder, never inject into an older zip. Test in Task 3.
3. **A run that crosses midnight**: MOTMaster names the folder at save time, so the rig snapshots both the start-day and end-day folders. Test in Task 3 (`find_new_zip` with the zip appearing in the next day's folder).
4. **KeyboardInterrupt mid-scan** (users interrupt cells): shots already taken stay in `rig.last_shots`. Test in Task 5.
5. **A camera returning a different frame count from the previous shot of the same script** (missed trigger): `warnings.warn` naming camera, script, old and new counts; never silent. Test in Task 4.

---

### Task 1: Camera frame-index protocol and live broadcast

**Files:**
- Modify: `pytweezer/drivers/camera_base.py`
- Modify: `pytweezer/drivers/thorcam.py` (only the `arm` override, below)
- Test: `tests/test_camera_frames.py` (new)

**Interfaces:**
- Produces, on `Camera` (all RPC-callable, pyon-serialisable returns):
  - `arm(buffer_frames: int = 1000) -> None` — start a continuous ring-buffer acquisition unless `_acquisition_in_progress()`; base calls `setup_acquisition("sequence", buffer_frames)` then `start_acquisition()`.
  - `frames_acquired() -> int` — frames acquired since the acquisition started; base returns `self._backend.get_frames_status()[0]` (pylablib).
  - `read_frames_from(start: int) -> np.ndarray` — frames `start .. frames_acquired()-1` as `(n, H, W)`; `n == 0` gives shape `(0, 0, 0)` without touching the hardware. Base reads with `self._backend.read_multiple_images((start, stop), peek=True)` under `self._lock`. Raise `RuntimeError` if frames before `stop` have been overwritten (pylablib returns fewer than requested).
  - `settings() -> dict` — JSON-safe dict for the zip; base returns `{}`.
  - `start_live(poll_s: float = 0.05) -> None` / `stop_live() -> None` — a daemon thread that broadcasts every new frame on the image stream (info `{"timestamp", "index"}`); `close()` calls `stop_live()`.
  - `_acquisition_in_progress() -> bool` hook; base returns `self._backend.acquisition_in_progress()`.
  - `self._lock: threading.RLock`, created in `Camera.__init__` before `_connect()`; every backend call in the four methods above holds it.
- `ThorCam.arm(buffer_frames=1000)` overrides to call its own `setup_acquisition(buffer_frames)` (its signature differs; leave that signature alone).
- `SimulatedCamera`: `start_acquisition` records `self._started = time.monotonic()`; `frames_acquired()` returns `int((now - started) * sim_frame_rate)` while running (class attribute `sim_frame_rate = 5.0`), 0 when stopped; `read_frames_from` generates that many frames with `_generate_frame`; `_acquisition_in_progress` returns `self._running`; `settings()` returns `{"shape": list(self._shape)}`.

- [ ] **Step 1: Write the failing tests** in `tests/test_camera_frames.py`, using `SimulatedCamera` with `sim_frame_rate` monkeypatched and `time.monotonic` patched to a controllable clock:
  - `test_arm_starts_a_stopped_camera_once` — after `arm()`, `_running` is True; a second `arm()` does not call `setup_acquisition` again (spy).
  - `test_frames_since_a_mark_are_exactly_the_new_ones` — mark = `frames_acquired()`, advance clock by 0.6 s at rate 5 → `read_frames_from(mark).shape[0] == 3`.
  - `test_reading_with_no_new_frames_gives_an_empty_stack` — `read_frames_from(frames_acquired()).shape == (0, 0, 0)`.
  - `test_live_broadcast_sends_each_new_frame_once` — replace `image_client` with a recorder, `start_live(poll_s=0.01)`, advance clock to make 3 frames, wait until 3 sends, `stop_live()`; indices sent are `[0, 1, 2]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_camera_frames.py -q` — expect failures (`AttributeError: arm`).
- [ ] **Step 3: Implement** the interfaces above in `camera_base.py` and the `ThorCam.arm` override.
- [ ] **Step 4: Run** `uv run pytest tests/test_camera_frames.py tests/test_simulation_mode.py tests/test_rearrangement.py -q` — all pass.
- [ ] **Step 5: Commit** `git add pytweezer/drivers/camera_base.py pytweezer/drivers/thorcam.py tests/test_camera_frames.py && git commit -m "Add a frame-index protocol and live broadcast to cameras"` and push.

### Task 2: CaF ThorCam device

**Files:**
- Modify: `pytweezer/drivers/thorcam.py`
- Modify: `pytweezer/configuration/config.py` (new `"CaF ThorCam"` entry after `"CaF ORCA"`)
- Test: `tests/test_camera_frames.py`

**Interfaces:**
- `ThorCam.__init__(self, image_dir=None, timeout=5.0, stream_name=None, serial_number: str | None = "32148", trigger_mode: str | None = None, trigger_polarity: str = "rise", binning: int | None = None, gain_db: float | None = None, buffer_frames: int = 1000, live: bool = False)`. `serial_number=None` opens the first camera `list_cameras_tlcam()` returns. After connecting, apply in this order when not `None`: `set_trigger_mode(trigger_mode)` (`"bulb"` for CaF), `setup_ext_trigger(trigger_polarity)`, `set_roi(hbin=binning, vbin=binning)` over the full sensor, `set_gain(gain_db)`. If `live`: `arm(buffer_frames)` then `start_live()`. Defaults leave the Rb ThorCam's behaviour unchanged.
- `ThorCam.settings()` returns `{"serial_number", "trigger_mode", "trigger_polarity", "binning", "gain_db", "exposure_s", "roi"}` read back from the backend.
- CONFIG entry:
  ```python
  "CaF ThorCam": {
      "active": True,
      "class": "pytweezer.drivers.thorcam:ThorCam",
      "sim_class": "pytweezer.drivers.thorcam:SimulatedThorLabsCamera",
      "teardown": "close",
      "host": HOSTS["ph-bonesaw"],
      "port": get_next_port(),
      "simulate": SIMULATING,
      "serial_number": None,
      "trigger_mode": "bulb",
      "trigger_polarity": "rise",
      "binning": 16,
      "gain_db": 12.0,
      "live": True,
      "stream_name": "caf_thorcam",
      "timeout": 5.0,
  },
  ```
  `"rise"` is pylablib's name for pycaf's `active_high`. Simulated builds drop the extra kwargs (`_config_kwargs` filters by signature); `SimulatedCamera.__init__` gains `live: bool = False` and starts its own live broadcast when true, so the GUI viewer works in simulation too.

- [ ] **Step 1: Write the failing test** `test_caf_thorcam_builds_simulated_and_streams_live` — `device_server.build_spec("CaF ThorCam", conf | {"simulate": True})` (conf from `get_config()["Devices"]["CaF ThorCam"]`) returns a camera whose `_acquisition_in_progress()` is True and whose live thread is alive; `close()` stops it.
- [ ] **Step 2: Run** it; expect `KeyError: 'CaF ThorCam'`.
- [ ] **Step 3: Implement** the ThorCam kwargs, `settings()`, the `live` flag on `SimulatedCamera`, and the CONFIG entry.
- [ ] **Step 4: Run** `uv run pytest tests -q -k "camera or simulation or servers"` — pass.
- [ ] **Step 5: Commit and push** ("Add the CaF ThorCam device").

### Task 3: Zip discovery and frame injection

**Files:**
- Create: `pytweezer/notebook/__init__.py` (exports `Rig`, `Shot` once Task 4 lands; empty docstring module now)
- Create: `pytweezer/notebook/zips.py`
- Test: `tests/test_notebook_zips.py`

**Interfaces (all in `pytweezer.notebook.zips`):**
- `day_folder(data_root: Path, day: datetime.date) -> Path` — `data_root / f"{yyyy}" / f"{MM}{Mon}" / f"{dd}"`, e.g. `2026/10Oct/27`.
- `snapshot(folders: Iterable[Path], element: str) -> set[Path]` — existing `{element}*.zip` files.
- `find_new_zip(before: set[Path], folders: Iterable[Path], element: str) -> Path` — the one zip present now and not in `before`; `RuntimeError` naming the folders if none, and if more than one.
- `inject_frames(zip_path: Path, key: str, frames: np.ndarray, settings: dict) -> None` — append `{key}_{i:04d}.tif` (uint16, `tifffile`) per frame and `{key}_camera_settings.json`; mode `"a"`.
- `write_sibling(main_zip: Path, element: str, key: str, frames: np.ndarray, settings: dict) -> Path` — new zip named `main_zip.name` with the leading `element` replaced by `key`, in the same folder; copies MOTMaster's entries ending `parameters.txt` and `hardwareReport.txt` from `main_zip`, then the frames and settings as `inject_frames` does. Refuses (`FileExistsError`) to overwrite.
- `read_zip_like_analysis3(path: Path) -> tuple[np.ndarray, dict]` lives in the **test file**, not the package: a faithful copy of `analysis3.readFromZip`'s logic (natural-sorted `.tif` entries as float arrays; `parameters.txt` lines `name\tvalue` to `{name: float}`), so tests prove compatibility without importing EDMSuite.

- [ ] **Step 1: Write the failing tests:**
  - `test_day_folder_matches_motmaster` — `day_folder(root, date(2026, 10, 27)) == root / "2026" / "10Oct" / "27"`.
  - `test_injected_caf_zip_reads_back_only_caf_frames` — build a fake MOTMaster zip with `CaF27Oct2600_012_parameters.txt` (`"dummy\t0\n"`) and `_hardwareReport.txt`; inject 3 frames for key `"CaF"`; `read_zip_like_analysis3` gives shape `(3, H, W)`, float dtype, frame order preserved (frame i filled with value i), params `{"dummy": 0.0}`.
  - `test_sibling_zip_holds_its_frames_and_motmaster_parameters` — `write_sibling(main, "CaF", "Rb", frames2, {})` creates `Rb27Oct2600_012.zip`; reading it gives the 2 Rb frames and the same params; the main zip still has no `Rb_` entries.
  - `test_no_new_zip_is_an_error` and `test_two_new_zips_is_an_error` (`pytest.raises(RuntimeError, match=folder name)`).
  - `test_a_zip_written_after_midnight_is_found` — `before` snapshot of day 27 and 28 folders, zip appears in 28 → found.
  - `test_settings_file_is_not_mistaken_for_parameters` — injected settings entry name does not end with `parameters.txt`.
- [ ] **Step 2: Run** `uv run pytest tests/test_notebook_zips.py -q` — fail on import.
- [ ] **Step 3: Implement** `zips.py`.
- [ ] **Step 4: Run** the tests — pass.
- [ ] **Step 5: Commit and push** ("Find and extend MOTMaster run zips from notebooks").

### Task 4: `Rig.run` and `Shot`

**Files:**
- Create: `pytweezer/notebook/rig.py`
- Modify: `pytweezer/notebook/__init__.py` (export `Rig`, `Shot`)
- Modify: `pytweezer/configuration/caf_mm_config.json` (`data_root_path` → `C:\\Users\\cafmot\\Box\\Working folder\\cafmot\\datasync\\mot_master_data`, matching EDMSuite's `PHBonesawFileSystem.MOTMasterDataPath`)
- Modify: `pytweezer/drivers/motmaster.py` (`SimulatedMotMasterInterface`: setters `set_iterations`, `set_save_toggle`, `set_trigger_mode`, `set_run_until_stopped` record their values; `start_motmaster_experiment` sleeps `sim_run_s = 0.5` so simulated cameras catch frames)
- Test: `tests/test_notebook_rig.py`

**Interfaces:**
- `@dataclass class Shot`: `script: str`, `params: dict[str, Any]`, `images: dict[str, np.ndarray]`, `zips: dict[str, Path]` (empty when unsaved), `t_start: float`, `t_end: float`. `Shot.zip_number -> int | None` from the main zip's `_nnn` suffix.
- `class Rig`:
  - `__init__(self, motmaster: str | Any = "CaF MotMaster", cameras: dict[str, str | Any] | None = None, *, linked: bool = False, element: str | None = None, data_root: Path | str | None = None, buffer_frames: int = 1000, frame_settle_s: float = 0.3, frame_timeout_s: float = 5.0)`. Strings go through `get_device`; objects are used as given (tests, in-process). `element`/`data_root` default to `data_prefix`/`data_root_path` of the MOTMaster's `config_file` JSON (read locally via `get_device_config(name)["config_file"]` and `pytweezer.drivers.motmaster.CONFIG_DIR`); required when `motmaster` is an object. Calls `cam.arm(buffer_frames)` on every camera.
  - `linked: bool` attribute; `run` sets `mm.set_linked(self.linked)` when it differs from what it last set.
  - `run(self, script: str, *, save: bool = True, iterations: int = 1, **params) -> Shot` — per call: `set_motmaster_experiment(script)`, `set_run_until_stopped(False)`, `set_trigger_mode(False)`, `set_iterations(iterations)`, `set_save_toggle(save)`; marks `frames_acquired()` per camera; if `save`, snapshots today's and tomorrow's folders; `start_motmaster_experiment(params)`; settles; `read_frames_from(mark)` per camera; `params = {**mm.get_params(), **params}`; if `save`, `find_new_zip`, `inject_frames` for the key equal to `element`, `write_sibling` for every other key, each with `cam.settings()`. Appends the shot to `self.last_shots` and returns it.
  - Settling: poll every camera's `frames_acquired()` every 0.05 s until no count has changed for `frame_settle_s`, or `frame_timeout_s` since `Go` returned.
  - Count check: `self._last_counts[(script, key)]`; a different count from the previous shot of that script → `warnings.warn(f"{key}: {new} frames from {script!r}, {old} last time")`.
  - `last_shots: list[Shot]` — cleared at the start of each `run`/scan call, appended per shot.
  - `close(self) -> None` — `set_linked(False)` if this rig turned it on, then `close_rpc()` on clients it created.

- [ ] **Step 1: Write the failing tests** in `tests/test_notebook_rig.py` with in-process fakes:
  - `FakeCamera(frames_per_shot, lag_polls=0)`: `arm`, `frames_acquired`, `read_frames_from`, `settings`; `trigger(n)` adds frames, optionally only after `lag_polls` more `frames_acquired()` calls.
  - `FakeMotMaster(tmp_root, cameras)`: the setter methods, `get_params()` → `{"dummy": 0.0, "tLoad": 100}`, `start_motmaster_experiment(params)` triggers each camera with its `frames_per_shot × iterations` and, if save is on, writes `CaF{ddMMMyy}00_{n:03d}.zip` with parameters/hardwareReport entries into today's folder.
  - `test_run_returns_each_cameras_frames_and_parameters` — `cameras={"CaF": cam3, "Rb": cam2}`, `run("S", dummy=1.5)` → `images["CaF"].shape[0] == 3`, `images["Rb"].shape[0] == 2`, float dtype, `params["dummy"] == 1.5`, `params["tLoad"] == 100`.
  - `test_saved_run_puts_caf_frames_in_motmasters_zip_and_rb_in_a_sibling` — reading both zips with `read_zip_like_analysis3` (import from `tests.test_notebook_zips`) gives exactly 3 and 2 frames.
  - `test_iterations_land_in_one_zip` — `iterations=3`, 3 frames per iteration → one zip, 9 CaF frames.
  - `test_unsaved_run_writes_no_zip` — `save=False`: `shot.zips == {}`, no zips on disk, `set_save_toggle(False)` was called.
  - `test_frames_arriving_after_go_are_collected` — `lag_polls=1` camera still yields all frames (Review Focus 1).
  - `test_changed_frame_count_warns` — second shot with one fewer frame → `pytest.warns(UserWarning, match="CaF: 2 frames")` (Review Focus 5).
  - `test_linked_mode_is_set_and_released` — `rig.linked = True; rig.run(...)` → `set_linked(True)`; `rig.close()` → `set_linked(False)`.
- [ ] **Step 2: Run** `uv run pytest tests/test_notebook_rig.py -q` — fail on import.
- [ ] **Step 3: Implement** `rig.py`, the exports, the JSON path fix and the simulated MOTMaster setters/sleep.
- [ ] **Step 4: Run** `uv run pytest tests/test_notebook_rig.py tests/test_notebook_zips.py tests/test_motmaster_experiment.py -q` — pass.
- [ ] **Step 5: Commit and push** ("Add Rig.run for notebook shots").

### Task 5: Scans

**Files:**
- Modify: `pytweezer/notebook/rig.py`
- Modify: `pytweezer/drivers/motmaster.py` (`MotMasterInterface` and `SimulatedMotMasterInterface`)
- Test: `tests/test_notebook_rig.py`

**Interfaces:**
- Driver: `laser_set_points_wml(self) -> dict[str, float]` (`{laser: slave set point}` over `config["lasers"]`); `ramp_laser_set_point_wml(self, laser: str, target: float, step: float = 1e-5, pause_s: float = 0.05) -> None` (steps from the current set point to `target` in `step` THz, as `scan_wm_laser_set_points` does today, then sets `target` exactly); `set_laser_set_point_wml(self, laser: str, value: float) -> None`. Simulated: a dict of set points seeded from `SIMULATED_LASER_SET_POINTS = {"v0": 494.431..., ...}` (any plausible values), same three methods.
- Rig:
  - `laser_set_points(self) -> dict[str, float]`.
  - `scan(self, script: str, parameter: str, values: Sequence[float], *, save: bool = True, iterations: int = 1, **params) -> list[Shot]` — one `run` per value with `params | {parameter: value}`.
  - `scan_laser(self, script: str, laser: str, values: Sequence[float], *, save: bool = True, iterations: int = 1, settle_s: float = 0.0, **params) -> list[Shot]` — `ramp_laser_set_point_wml(laser, values[0])`, `time.sleep(settle_s)`, then per value `set_laser_set_point_wml(laser, value)` and `run`; each shot's `params[laser]` holds the set point.
  - Both clear `last_shots` once at the start and let `run` append, so an interrupted scan leaves the completed shots there.

- [ ] **Step 1: Write the failing tests** (extend the fakes with set-point methods):
  - `test_scan_runs_once_per_value_with_that_value` — `[s.params["dummy"] for s in rig.scan("S", "dummy", [1, 2, 3])] == [1, 2, 3]`, three zips.
  - `test_scan_laser_ramps_then_steps_through_the_values` — fake records calls: one ramp to `values[0]` before any `Go`, then `set` per value in order; `shots[i].params["BX"] == values[i]`.
  - `test_interrupted_scan_keeps_completed_shots` — fake MOTMaster raises `KeyboardInterrupt` on the third `Go`; after `pytest.raises(KeyboardInterrupt)`, `len(rig.last_shots) == 2` (Review Focus 4).
  - `test_simulated_laser_ramp_reaches_its_target` on `SimulatedMotMasterInterface`.
- [ ] **Step 2: Run** — fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_notebook_rig.py tests/test_motmaster_experiment.py -q` — pass.
- [ ] **Step 5: Commit and push** ("Add parameter and laser scans to Rig").

### Task 6: Auto-tune on the rig

**Files:**
- Modify: `pytweezer/notebook/rig.py`
- Modify: `pytweezer/drivers/motmaster.py` (remove `MotMasterInterface.auto_mot`, `_auto_mot`; move `plot_auto_mot_results` to `pytweezer/notebook/rig.py`)
- Modify: `tests/test_motmaster_experiment.py` (move `test_auto_mot_error_bars_are_standard_errors_of_the_shots` to `tests/test_notebook_rig.py`, rewritten against the rig; delete its `FakeCamera`)
- Test: `tests/test_notebook_rig.py`

**Interfaces:**
- `auto_mot(self, script: str, scan_ranges: dict[str, Sequence[float]], *, camera: str | None = None, iterations: int = 1, field_parameter: str = "MOTCoilsCurrentValue", bg_field_value: float = 0.0, mot_field_value: float = 1.0, save: bool = False, display: bool = False) -> dict[str, dict[str, Any]]`. `camera` defaults to the key equal to `element`. One background shot at `bg_field_value`; for each laser in order, `scan_laser(..., save=save, iterations=iterations, **{field_parameter: mot_field_value})`; per point, counts = per-frame sum of `images - mean(background frames)`; `numbers` = mean, `errors` = SEM over frames (NaN for one frame); park the laser on the best point with `set_laser_set_point_wml` before the next laser. Result per laser: `{"scan_range", "numbers", "errors", "best", "start"}` (same keys as the driver's today, so `plot_auto_mot_results` is unchanged). Unknown laser → `KeyError` listing `laser_set_points()` keys.

- [ ] **Step 1: Write the failing tests:**
  - `test_auto_mot_parks_each_laser_on_its_brightest_point` — fake camera frame values depend on the current fake set point (peak at the middle value) → `results["v0"]["best"] == values[1]` and the last `set` call for `v0` is `values[1]`.
  - `test_auto_mot_error_bars_are_standard_errors_of_the_frames` — parametrised as the old test: one frame → NaN, frames `[0, 2]` → 1.0.
  - `test_auto_mot_does_not_save_by_default` — no zips written.
- [ ] **Step 2: Run** — fail.
- [ ] **Step 3: Implement**; remove the driver's auto-tune and update `tests/test_motmaster_experiment.py`.
- [ ] **Step 4: Run** `uv run pytest tests -q` — whole suite passes.
- [ ] **Step 5: Commit and push** ("Move MOT auto-tuning onto Rig").

### Task 7: Documentation, simulated end-to-end check, PR

**Files:**
- Create: `.claude/skills/notebook-rig/SKILL.md` — when to use (lab-book notebooks, `Rig`, zips, cameras), the 27Oct26-style cells rewritten on the rig (single shot + `get_recapture_rate_three_image` looping over `shot.images["CaF"]`, BX `scan_laser`, `auto_mot`), the zip/sibling rules, `caftweeznb` reinstall after config changes, and the live view (Applet Launcher → image viewer on `caf_thorcam`).
- Modify: `CLAUDE.md` — one row in the skills table: "Running shots, scans or auto-tune from a lab-book notebook (`Rig`)" → `notebook-rig`.
- Modify: `.claude/skills/pytweezer-device-framework/SKILL.md` or `add-device-driver` — one paragraph on the camera frame-index protocol (`arm`/`frames_acquired`/`read_frames_from`/`settings`/`start_live`), wherever cameras are described.

- [ ] **Step 1: Simulated end-to-end:** a scratch script (in the scratchpad, not the repo) builds `SimulatedMotMasterInterface` and two `SimulatedThorLabsCamera(live=False)` in-process, `Rig(mm, {"CaF": a, "Rb": b}, element="CaF", data_root=<tmp>)`, monkeypatches the simulated MOTMaster to write a zip as `FakeMotMaster` does, runs `run`, `scan`, `scan_laser`, `auto_mot`; expect non-empty stacks and readable zips.
- [ ] **Step 2: Write the docs** above.
- [ ] **Step 3: Build docs** `cd docs && uv run make html` — no new warnings from `pytweezer.notebook`.
- [ ] **Step 4: Run** `uv run ruff check . && uv run ruff format --check . && uv run pytest tests -q` — clean and passing.
- [ ] **Step 5: Commit, push, open the PR** against `main` (`gh pr create`), body summarising the feature, the deployment step (reinstall pytweezer in `caftweeznb`; restart device servers on ph-bonesaw so `CaF ThorCam` runs; stop pycaf's grabber first since the device owns the camera), and that hardware verification and the combined linked script are still to do.
