# Camera Applet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Camera applet (live image plus an HCImage-style settings sidebar) for every camera in CONFIG, driven by control lists each camera driver declares, with every applied setting stored in the saved parameters; the `CameraSettings` experiments rebuilt on the same control lists.

**Architecture:** Each `Camera` driver declares a static tuple of `Control`s and keeps the values its setters read back; three new RPC methods (`controls`, `read_controls`, `apply_controls`) expose them. A new applet subclasses the Image Monitor, builds its sidebar from `controls()`, makes every device call on one worker thread, goes read-only while a running task's reservation covers the camera, and stores each applied change with `set_saved_parameters`. `CameraSettings` generates its form arguments from the same `Control`s at import.

**Tech Stack:** Python 3, PyQt6, pyqtgraph, sipyco RPC, pylablib (DCAM, Thorlabs TLCam), uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-12-camera-applet-design.md`. Read it first; this plan argues from it.

## Global Constraints

- No new tests and no new test files (the user's standing rule: tests only when asked). Existing tests that the design breaks are changed to follow the design (CLAUDE.md: "Tests follow the design"). Hand-checks go in the scratchpad, never in the repo.
- `uv run pytest tests/ -q` passes at the end of every task; `main` stays deployable.
- Ruff before every commit: `uv run ruff check --fix <files>` and `uv run ruff format <files>`; `git add` whole files; never `--no-verify`.
- Commit and push after each task, on a feature branch `camera-applet` with a PR to `main`.
- Google-style docstrings, UK English, current behaviour only; comment only the surprising.
- `get_logger(name)` from `pytweezer/logging_utils.py`, never `logging.getLogger`.
- Values a driver keeps, returns and stores are SI (exposure in seconds); display units and scale live on the `Control`.
- Saved-parameter keys: `f"{camera_key(name)}_{control.name}"`; group = the camera's CONFIG name; reason `"Camera applet"` from the applet, `f"{ClassName}, rid {rid}"` from a `CameraSettings` task.
- GUI checks stay offscreen (`QT_QPA_PLATFORM=offscreen`, `.claude/skills/add-applet/scripts/preview_applet.py`); never open a window on the user's display.
- Load the `add-device-driver` skill for Tasks 2-3, `add-experiment` for Task 4, `add-applet`, `pytweezer-gui-design` and `pytweezer-gui-internals` for Tasks 5-6. Skill edits that change a rule need the user's go-ahead (Task 8); minor ones go in the same commit as the code.

## Review Focus

1. **A setting changed while the camera is live.** Expected: live stops, the setting applies, live restarts, frames keep arriving; nothing deadlocks on the camera's `_lock` (the live thread takes it). Checked in Task 1 Step 4.
2. **A value the camera refuses or rounds.** Expected: the field shows the camera's value afterwards (refused: the old value and the error; rounded: the rounded one), and only what the camera took is stored. Checked in Task 1 Step 4 and Task 6 Step 4.
3. **A camera that is inactive or whose server is down.** Expected: the applet stays up, says so in the pill, sidebar empty; choosing another camera works. Checked in Task 6 Step 4.
4. **The manager unreachable, then back with a task holding the camera.** Expected: editable with a "can't tell whether a task holds the camera" note, then read-only as soon as the feed reports the task. Checked in Task 6 Step 4.
5. **One ROI field edited on its own.** Expected: `set_roi` gets the other three from the kept values, and all four show what the camera took. Checked in Task 1 Step 4.

---

### Task 1: Controls on the `Camera` base, simulated cameras, camera catalogue

**Files:**
- Modify: `pytweezer/drivers/camera_base.py`
- Create: `pytweezer/drivers/camera_catalogue.py`
- Modify: `pytweezer/experiments/camera_continuous.py` (import `camera_names` from the catalogue; delete its own)
- Modify: `tests/test_camera_continuous.py`, `tests/test_camera_settings.py` (import path of `camera_names` only)

**Interfaces:**
- Produces, in `camera_base.py`:
  - `@dataclass(frozen=True) class Control` with fields `name: str`, `label: str`, `section: str`, `kind: str` (`"number" | "integer" | "choice" | "bool"`), `default: Any`, `unit: str = ""` (display), `scale: float = 1.0` (SI per display unit), `si_unit: str = ""` (stored), `minimum: float | None = None`, `maximum: float | None = None`, `choices: tuple[str, ...] = ()`; method `describe() -> dict` (JSON-safe; `choices` as a list).
  - Shared constants: `EXPOSURE = Control("exposure", "Exposure", "Exposure & trigger", "number", 10e-3, unit="ms", scale=1e-3, si_unit="s", minimum=1e-6)`, `TRIGGER_POLARITY` (choice `("rise", "fall")`, section "Exposure & trigger"), `BINNING` (integer, min 1, section "Binning & ROI"), `ROI = (roi_x0, roi_width, roi_y0, roi_height)` (integer, unit "px", si_unit "px", section "Binning & ROI"); factories `trigger_source(choices: tuple[str, ...]) -> Control` and `gain(label: str, kind: str, unit: str = "", minimum=None, maximum=None, default=0) -> Control` (section "Gain").
  - `Camera.CONTROLS: tuple[Control, ...] = ()` (class attribute).
  - `Camera.set_trigger_polarity(polarity: str) -> None` (base raises `NotImplementedError`, like `set_gain`).
  - `Camera.controls() -> list[dict]`: each `Control.describe()` plus `"value"` from the kept values. No hardware access.
  - `Camera.read_controls() -> list[dict]`: `self._kept.update(self._read_control_values())`, then `controls()`.
  - `Camera.apply_controls(values: dict[str, Any]) -> list[dict]`: SI values by control name; unknown names raise `KeyError` naming the camera's controls.
  - Protected: `Camera._kept: dict[str, Any]` (set in `__init__` before `_connect`), `Camera._keep(**values)`, `Camera._read_control_values() -> dict[str, Any]` (base returns `{}`), `Camera._apply_control(control: Control, value) -> None`.
  - `camera_catalogue.py`: `camera_names() -> list[str]` (moved verbatim from `camera_continuous.py`, inactive cameras included), `camera_driver(name: str) -> type[Camera]`, `camera_conf(name: str) -> dict` (its CONFIG entry, composites' sub-device entries included), `camera_key(name: str) -> str` (`re.sub(r"\W+", "_", name).strip("_").lower()`: `"Rb HamCam"` → `"rb_hamcam"`).

- [ ] **Step 1: Add `Control`, the shared constants and the `Camera` methods above to `camera_base.py`.**
  `apply_controls` order: `binning`, then the four ROI fields together (missing ones from `_kept`) through `set_roi`, `exposure`, `exposure_mode`, `trigger_source`, `trigger_polarity`, then the rest in `CONTROLS` order with `gain` last. Before applying: if `is_live()`, `stop_live()`; in a `finally`, `read_controls()` then `start_live()` if it was live. Do **not** hold `self._lock` across `stop_live` (the live thread needs it to finish). `_apply_control` default: `bool` kind → `getattr(self, setter)(bool(value))`, otherwise the setter with the value; the setter name is `{"exposure": "set_exposure_time", "trigger_source": "set_trigger_source", "trigger_polarity": "set_trigger_polarity", "gain": "set_gain", "binning": "set_binning"}` for shared controls, `f"set_{name}"` for the rest (drivers override `_apply_control` where that doesn't fit). Base setters stay abstract/raising; their docstrings say a driver keeps what the camera took with `_keep`.

- [ ] **Step 2: Simulated cameras.**
  `SimulatedCamera`: `_read_control_values()` returns `{c.name: self._sim_values.get(c.name, c.default) for c in self.CONTROLS}`; `_apply_control` calls the (possibly stubbed) setter then sets `self._sim_values[control.name] = value`; its existing setters (`set_exposure_time`, `set_gain`, `set_binning`, `set_roi`, `set_trigger_source`) also `_keep` their value; add `set_trigger_polarity`. `simulated_camera_for` copies `real_cls.CONTROLS` onto the class it returns (as it already copies `image_prefix`).

- [ ] **Step 3: Create `camera_catalogue.py` and move `camera_names`** (`device_server.device_index()` / `device_server._load` as now). Update the two test imports. Run `uv run pytest tests/ -q`: PASS.

- [ ] **Step 4: Hand-check in the scratchpad** (not in the repo): build `simulated_camera_for` of a throwaway `Camera` subclass whose `CONTROLS = (EXPOSURE, BINNING, *ROI)`, `stream_name=None`; `start_live()`; `apply_controls({"exposure": 0.02})` returns within 1 s with exposure `0.02` and `is_live()` true; `apply_controls({"roi_x0": 8})` keeps the other three ROI values; `apply_controls({"nope": 1})` raises `KeyError`. Delete nothing in the repo; leave the script in the scratchpad.

- [ ] **Step 5: Ruff, commit, push** — `git add pytweezer/drivers/camera_base.py pytweezer/drivers/camera_catalogue.py pytweezer/experiments/camera_continuous.py tests/test_camera_continuous.py tests/test_camera_settings.py`, message "Camera controls: declared per driver, read back and applied over RPC".

---

### Task 2: DCAM drivers (ImagEM X2, ORCA)

**Files:**
- Modify: `pytweezer/drivers/hamamatsu.py`, `pytweezer/drivers/imagemX2.py`, `pytweezer/drivers/orca.py`
- Modify (only if broken by the change): `tests/test_camera.py`, `tests/test_hamamatsu.py`

**Interfaces:**
- Consumes: Task 1's `Control`, shared constants, `_keep`, `_read_control_values`, `_apply_control`.
- Produces:
  - `hamamatsu.get_dcam_exposure_mode(backend) -> str` (`"internal"` when trigger mode is `"int"`, else `"edge"`/`"level"` from the `trigger_active` attribute, 1/2, as `set_dcam_exposure_mode` writes them).
  - `set_trigger_polarity(polarity: str)` on both: `self._backend.setup_ext_trigger(invert=(polarity == "fall"))`; read back from `get_ext_trigger_parameters()` (check the field name in `.venv`'s pylablib `DCAM.py`).
  - `ImagEMX2Camera.CONTROLS = (EXPOSURE, Control("exposure_mode", …choices ("internal","edge","level")), trigger_source(("int","ext","software")), TRIGGER_POLARITY, gain("Sensitivity", "integer", minimum=0), Control("em_gain", "EM gain", "Gain", "bool", True), Control("direct_em_gain", "Direct EM gain", "Gain", "bool", False), BINNING, *ROI)`.
  - `ORCACamera.CONTROLS = (EXPOSURE, exposure_mode as above, trigger_source(("int","ext","software")), TRIGGER_POLARITY, gain("Contrast gain", "integer", minimum=0), Control("readout_speed", "Readout speed", "Readout", "integer", 1, minimum=1), BINNING, *ROI)`.

- [ ] **Step 1: `get_dcam_exposure_mode` in `hamamatsu.py`.** `set_dcam_exposure_mode` itself is unchanged (it still forces active-low for edge/level: spec, "stays as it is").
- [ ] **Step 2: `_read_control_values()` on both drivers** from the backend: `get_exposure()`, `get_dcam_exposure_mode`, `get_trigger_mode()`, the polarity read-back, `get_roi()` (x0 = hstart, width = hend − hstart, …, binning = hbin), and the attributes `sensitivity`, `ccd_mode` (2 → on), `direct_em_gain_mode` (2 → on), `contrast_gain`, `readout_speed`.
- [ ] **Step 3: Every setter keeps what the camera took.** E.g. `set_exposure_time` → `self._keep(exposure=self._backend.set_exposure(exposure))`; `set_roi`/`set_binning` keep all four ROI fields and binning from the returned ROI; `set_exposure_mode` keeps `exposure_mode`, `trigger_source` and `trigger_polarity` (they are coupled); `enable_em_gain` / `enable_direct_em_gain` keep `em_gain` / `direct_em_gain`. `__init__` ends with `self.read_controls()` after any `external_exposure` set-up. `_apply_control`: `em_gain` → `enable_em_gain`, `direct_em_gain` → `enable_direct_em_gain`; the rest use the base mapping.
- [ ] **Step 4: Run `uv run pytest tests/ -q`.** Fake backends in `tests/test_camera.py` that lack a getter now called in `__init__`: give the fake the method (the test follows the design). PASS.
- [ ] **Step 5: Ruff, commit, push** — "DCAM cameras declare their controls and keep what the camera took".

---

### Task 3: Thorlabs drivers (`thorcam.ThorCam`, `tweezermonitorcam.ThorCam`)

**Files:**
- Modify: `pytweezer/drivers/thorcam.py`, `pytweezer/drivers/tweezermonitorcam.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `CONTROLS = (EXPOSURE, trigger_source(("int","ext","bulb")), TRIGGER_POLARITY, gain("Gain", "number", unit="dB", minimum=0), BINNING, *ROI)` on both; `tweezermonitorcam.ThorCam` gains `set_trigger_polarity` and `settings()` as in `thorcam.ThorCam`.

- [ ] **Step 1: `_read_control_values()`** from `get_exposure()`, `get_trigger_mode()`, `get_ext_trigger_parameters()` (polarity), `get_gain()`, `get_roi()`.
- [ ] **Step 2: Setters keep what the camera took**, as in Task 2 Step 3; `__init__` ends with `self.read_controls()`.
- [ ] **Step 3: `uv run pytest tests/ -q`**: PASS (fix fakes as in Task 2 Step 4).
- [ ] **Step 4: Ruff, commit, push** — "Thorlabs cameras declare their controls and keep what the camera took".

---

### Task 4: Stored settings, and `CameraSettings` rebuilt on the controls

**Files:**
- Create: `pytweezer/experiment/camera_store.py`
- Modify: `pytweezer/experiment/camera_settings.py` (rewrite), `pytweezer/experiments/camera_settings.py`
- Modify: `tests/test_camera_settings.py` (follows the new design; no new cases)

**Interfaces:**
- Consumes: Task 1's `Control`, `camera_driver`, `camera_key`, `Camera.apply_controls`.
- Produces:
  - `camera_store.store_camera_settings(camera: str, values: dict[str, Any], reason: str) -> None`: `set_saved_parameters({f"{camera_key(camera)}_{n}": v …}, reason, create=True, group=camera, units={key: control.si_unit …})`, values coerced to the control's kind (`int` for integer, `float` for number, `bool`, `str`) so a key never changes type. Lets `ManagerUnavailable` / `ManagerError` propagate; callers decide.
  - `CameraSettings.__init_subclass__`: when the subclass's own `vars` hold `camera = Device(name)`, it adds arguments from `camera_driver(name).CONTROLS` with `setattr` and calls each argument's `__set_name__`: number/integer → `set_<name> = Bool(False, group=section)` and `<name> = Number|Integer(default, unit, scale, min, max, group=section)`; the four ROI controls share one `set_roi` Bool; choice → `Choice([UNCHANGED, *choices])`; bool → `Choice([UNCHANGED, "on", "off"])`. Defaults come from the subclass's `control_defaults: dict[str, Any]` (SI) or else `Control.default`.
  - `CameraSettings.run_point`: `values` = the ticked / not-unchanged ones, in SI (`"on"` → `True`); `result = self.camera.apply_controls(values)` (stop/restart live is now inside it); records `camera_settings` (`self.camera.settings()`) and, when the ROI was set, `roi_origin = [roi_x0, roi_y0]` from `result`; then `store_camera_settings(name, {n: value from result for n in values}, f"{type(self).__name__}, rid {self.rid}")`. A store failure logs a warning and does not fail the task.
  - `pytweezer/experiments/camera_settings.py`: the five classes keep their names and `camera = Device(...)`; per-camera defaults move to `control_defaults` (Rb HamCam: `roi_x0=110, roi_width=256, roi_y0=220, roi_height=256`; CaF ThorCam: `gain=12.0, binning=16`).
  - `HamamatsuSettings`, `ImagEMSettings`, `ORCASettings`, `ThorCamSettings` are deleted. Field renames: ThorCam `trigger_mode` → `trigger_source`; ImagEM `gain` keeps its name (labelled Sensitivity). Say in the commit message that saved recipes of these experiments using a renamed field must be re-saved.

- [ ] **Step 1: Write `camera_store.py`.**
- [ ] **Step 2: Rewrite `CameraSettings` as above.** Module docstring keeps the description of what these experiments are for.
- [ ] **Step 3: Update `tests/test_camera_settings.py` to the design.** `RecordingCamera` records `apply_controls(values)` and returns a control list; monkeypatch `pytweezer.experiment.camera_settings.store_camera_settings` to record its calls. The existing cases keep their intent with new assertions: untouched form → `apply_controls({})` only, nothing stored; the ThorCam case → one `apply_controls` with `{"binning": 4, "roi_x0": 8, "roi_width": 64, "roi_y0": 4, "roi_height": 32, "exposure": 2e-3, "trigger_source": "bulb", "gain": 6.0}` and `roi_origin` from the returned list; the driver-specific parametrisation → the matching key in the `values` dict (`{"exposure_mode": "edge"}`, `{"em_gain": True, "direct_em_gain": False}`, `{"exposure_mode": "internal", "readout_speed": 2}`, `{"trigger_polarity": "fall"}`). Drop the ordering and live assertions there (ordering and live now belong to `Camera.apply_controls`). `test_every_configured_camera_has_one_settings_experiment` and the DCAM exposure-mode test stay.
- [ ] **Step 4: `uv run pytest tests/ -q`**: PASS. Then in the scratchpad, `run_local(cs.RbHamCamSettings, set_exposure=True, exposure=0.02, simulate=True)` with `store_camera_settings` patched to print: status `completed`, printed key `rb_hamcam_exposure` = `0.02`.
- [ ] **Step 5: Update `add-experiment`'s mention of camera settings experiments if it describes the old classes (minor edit).** Ruff, commit, push — "CameraSettings built from the drivers' controls; settings stored as saved parameters".

---

### Task 5: Image Monitor layout hook and the settings sidebar widget

**Files:**
- Modify: `pytweezer/GUI/viewers/image_monitor.py`
- Create: `pytweezer/GUI/viewers/camera_sidebar.py`

**Interfaces:**
- Produces:
  - `ImageDisplay.panel_layout: QHBoxLayout` — the applet's top-level layout, holding one widget that contains what `init_gui` builds today, so a subclass can `insertWidget(0, …)` on the left. No visible change to the Image Monitor.
  - `camera_sidebar.CameraSidebar(QWidget)`:
    - signals `camera_chosen(str)`, `apply_requested(dict)` (SI values by control name), `live_toggled(bool)`, `snap_requested()`, `roi_from_drawn_requested()`, `full_sensor_requested()`;
    - `set_cameras(names: list[str], unavailable: dict[str, str])` (name → reason shown greyed);
    - `set_controls(controls: list[dict])` (rebuilds the sections, one collapsible group per `section`, driver-only sections tagged with the camera's model);
    - `show_values(controls: list[dict])` (updates values without rebuilding; never overwrites a field being edited);
    - `set_state(state: str, detail: str = "")` with states `"idle"`, `"live"`, `"held"`, `"unknown_hold"`, `"unavailable"`; `"held"` disables every control and Live/Snap;
    - `set_busy(names: set[str])`, `show_error(name: str, message: str)`, `set_fps(fps: float | None)`.
  - Fields emit on Enter / focus-out / selection only, converting display → SI with the control's `scale`. Snap is enabled only when not live and `trigger_source == "int"`.

- [ ] **Step 1: Add `panel_layout` to `ImageDisplay.init_gui`.** `uv run pytest tests/test_image_monitor.py -q`: PASS.
- [ ] **Step 2: Write `CameraSidebar`** to the mockup in the spec's brainstorm (`layout.html`): picker and status pill on top, Live/Snap, then sections; colours and fonts from `pytweezer/GUI/theme.py` only (per `pytweezer-gui-design`), numeric fields monospace.
- [ ] **Step 3: Render it offscreen in the scratchpad** with the ImagEM's `controls()` list from a simulated camera, in states idle and held; read the PNG: sections, units and the model tag readable, held state visibly greyed.
- [ ] **Step 4: `uv run pytest tests/ -q`**: PASS. Ruff, commit, push — "Camera settings sidebar widget; Image Monitor exposes its top-level layout".

---

### Task 6: The Camera applet

**Files:**
- Create: `pytweezer/GUI/viewers/camera_applet.py`
- Modify: `pytweezer/GUI/applet_launcher.py` (`DEFAULT_APPLETS` entry "Camera", description "Live camera image with its settings")

**Interfaces:**
- Consumes: Task 1 (`camera_names`, `camera_conf`, `camera_driver`, RPC `controls`/`read_controls`/`apply_controls`/`start_live`/`stop_live`/`is_live`/`acquire_single_frame`), Task 4 (`store_camera_settings`), Task 5 (`panel_layout`, `CameraSidebar`), `ExperimentFeed` (`GUI/experiments/feed.py`), `get_device` (`servers/device_client.py`), `camera_roi` (`image_monitor.py`).
- Produces:
  - `class CameraLink(QObject)` in `camera_applet.py`: one daemon worker thread and one `queue.Queue`; it alone owns the `get_device(name)` client (sipyco clients are not thread-safe, so `BackgroundReader`'s thread-per-read does not fit). `call(method: str, *args, callback)` queues a call; `callback(outcome)` runs on the GUI thread with the return value or the exception. A connection error closes the client and the next call reconnects. `close()` stops the thread.
  - `class CameraApplet(ImageDisplay)`, property `_camera = PropertyAttribute("camera", "")`; `main(name)` and `__main__` as every applet.

- [ ] **Step 1: Camera choice.** On start and on `camera_chosen`: store the property; set `imagestreams` to `[camera_conf(name)["stream_name"]]` and re-subscribe; open a new `CameraLink`; `read_controls` → `set_controls`. Unreachable server → `set_state("unavailable", "server not running")`; inactive in CONFIG → `"unavailable", "inactive in CONFIG"`.
- [ ] **Step 2: Holds.** On `ExperimentFeed.queue_changed`, the camera is held when its name is in any running task's `devices`; pill text `f"Held by {class_name} (rid {rid})"`. While held, a 3 s timer calls `controls` (kept values, no hardware) → `show_values`. On release, one `read_controls` then re-enable. While `feed.connected` is false: `set_state("unknown_hold", "Can't tell whether a task holds the camera")`, controls stay enabled.
- [ ] **Step 3: Edits.** `apply_requested(values)` → `set_busy`, `apply_controls(values)`; on success `show_values(result)` and `store_camera_settings(name, {n: value from result for n in values}, "Camera applet")` on the link's thread too (a failure shows "applied, not stored" on those fields); on error `show_error` and `read_controls`. Live → `start_live`/`stop_live`; Snap → `acquire_single_frame(broadcast=True)`; "From drawn ROI" → the four values from `camera_roi` of the first ROI; "Full sensor" → `roi_x0 = roi_y0 = 0` with the width/height `maximum`s from the controls (when a driver gives none, the button is disabled). Status bar under the image: cursor pixel and value, frame max, `index` from the header, `"{h}×{w}, {binning}×{binning}"`; fps from successive header `timestamp`s.
- [ ] **Step 4: Check offscreen, all in simulation** (`PYTWEEZER_ROLE=server` set before launch so `config.py` forces simulation on this PC, see `pytweezer-architecture`; hubs and Properties as `run-pytweezer` describes; the HamCam's server is its composite, `uv run pytweezer-device "Rb Rearrangement Rig"`, in the background and killed afterwards, then `uv run pytweezer-kill-stale`): `preview_applet.py pytweezer/GUI/viewers/camera_applet.py --prop 'camera="Rb HamCam"'`, read the PNG. Then in a scratchpad script, against the same server: an exposure edit while live is shown and stored (patch `store_camera_settings` to print); an out-of-range value shows its error and reverts; picking an inactive camera shows "inactive in CONFIG" and picking back works; a fake feed snapshot with `{"running": [{"rid": 7, "class_name": "LoadTweezers", "devices": ["Rb HamCam"]}]}` greys the sidebar, and an empty one re-enables it; `feed.connected = False` shows the unknown-hold note.
- [ ] **Step 5: `uv run pytest tests/ -q`**: PASS. Ruff, commit, push — "Camera applet: live image with the camera's settings in a sidebar".

---

### Task 7: Rearrangement coordinator and a camera left live

**Files:**
- Modify (only if needed): `pytweezer/coordinators/rearrangement.py`

- [ ] **Step 1: Read where the coordinator starts acquiring** (`start_acquisition`, `acquire_n_frames`, around lines 540-885) and decide whether a camera left live by the applet breaks it (the live thread's `read_frames_from` racing its reads, `arm()` returning early).
- [ ] **Step 2: If it does, have it `stop_live()` before setting the camera up, as the experiments do**, and say so in the PR. If not, record why in the PR description. `uv run pytest tests/test_rearrangement.py -q`: PASS. Commit, push if changed.

---

### Task 8: Skills, docs build, PR

- [ ] **Step 1: Propose the rule changes to the user and wait for the go-ahead** (CLAUDE.md, "Significant"): `add-device-driver` — a camera driver declares `CONTROLS`, keeps read-back values with `_keep`, implements `_read_control_values`; `add-applet` — the Camera applet as the exception to "applets only display" and to "controls go in the context menu"; `notebook-rig` / `pytweezer-device-framework` only if they describe the camera RPC surface. Apply what the user approves.
- [ ] **Step 2: `cd docs && uv run make html`** builds without new warnings from the new modules.
- [ ] **Step 3: Open the PR** from `camera-applet` to `main`, listing: the field rename (`trigger_mode` → `trigger_source`), Task 7's outcome, and that no new tests were written (the user's rule).
