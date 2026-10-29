# Experiment Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Experiment Manager run tasks from different named pipelines concurrently, one worker per running task, never letting two running tasks reserve the same device.

**Architecture:** Each experiment class's declared devices are expanded (composites → members, linked MOTMaster → its partner) into a reservation set, computed in the catalogue's introspection subprocess and stored on the `Task`. The pure-data `ExperimentQueue` owns the scheduling rule (per-pipeline order, priority between pipelines, device clashes) and reports why a pipeline's next task is waiting; the manager just starts what `startable()` returns, one worker each. Workers and `run_local()` refuse devices outside the reservation.

**Tech Stack:** Python 3, pydantic models, ZMQ REQ/REP + sipyco sync_struct notifier, PyQt6, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-10-29-experiment-pipelines-design.md`

## Global Constraints

- Work on branch `experiment-pipelines`; commit and push after each task; main stays deployable. Stage only the files the task touched; leave unrelated changes (e.g. `.vscode/settings.json`) alone.
- End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; end the PR body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Run `uv run pytest tests/ -q` and `uv run ruff check . && uv run ruff format --check .` before every commit; all must pass.
- Docstrings describe current behaviour only (no change history); Google style for new code; UK English.
- Use `get_logger(name)` from `pytweezer/logging_utils.py`, never `logging.getLogger`.
- Comment only what is surprising; no explanatory comment where a name would do.
- `ExperimentQueue` (`pytweezer/experiment/queue.py`) stays free of sockets, threads and processes.
- Default pipeline name is exactly `"main"`.
- Tests follow the `write-tests` skill; set `QT_QPA_PLATFORM=offscreen` for GUI tests; never open GUI windows on the live display.

## Review Focus

1. **Upgrading with a task running:** an old `queue_state.json` with a single `"worker"` record and a running task without `devices` loads, the worker is adopted, and nothing else starts until it ends (it is treated as holding every device). Tests: Task 3 Step 1 (`test_old_state_file_migrates_worker_and_pipeline`, `test_a_running_task_with_unknown_devices_blocks_everything`).
2. **Same device, different spelling:** `Device("rb hamcam")` and `Device("Rb HamCam")` clash. Test: Task 1 Step 1 (`test_names_are_canonicalised`).
3. **Submitted while the catalogue is still busy:** the task queues with `devices=None`, is not started, gets its reservation on a later tick, then starts; if the class turns out to be missing it fails with a message instead of waiting forever. Test: Task 4 Step 1 (`test_reservation_resolved_later_when_catalogue_was_busy`, `test_unresolvable_reservation_fails_the_task`).
4. **Two pipelines' heads both want the same free device in one tick:** only the higher-priority one starts; the other shows `waiting_for`. Test: Task 3 Step 1 (`test_one_tick_never_double_books_a_device`).
5. **Abort with two tasks running:** only the aborted task's worker is killed and only it ends `aborted`. Test: Task 4 Step 1 (`test_abort_kills_only_that_worker`).

---

### Task 1: Device reservation sets

**Files:**
- Create: `pytweezer/experiment/reservations.py`
- Modify: `pytweezer/experiment/arguments.py` (class `Device`, ~line 188)
- Modify: `pytweezer/experiment/motmaster.py` (class `MotMaster`, ~line 38; module docstring's linked-mode paragraph)
- Modify: `pytweezer/experiment/experiment.py` (class `Experiment`)
- Modify: `pytweezer/experiment/introspect.py` (`describe`)
- Modify: `pytweezer/configuration/config.py` (`"CaF MotMaster"` entry, ~line 151)
- Test: `tests/test_experiment_reservations.py` (new)

**Interfaces:**
- Produces:
  - `reservations.ReservationError(ValueError)`
  - `reservations.canonical_device(name: str) -> str` — `device_server.resolve_address(name).name`; unknown name → `ReservationError` carrying the `KeyError` text.
  - `reservations.device_reservation(name: str) -> set[str]` — `{canonical}`, plus every sub-device name if it is a composite; `ReservationError` if the owning CONFIG entry has `"active": False`.
  - `Device.reserves(self) -> set[str]` — `device_reservation(self.device_name)`.
  - `MotMaster.reserves(self) -> set[str]` — base set, plus `device_reservation(partner)` when `self.linked`, where `partner = resolve_address(self.device_name).conf.get("linked_partner")`; missing partner → `ReservationError(f"{device_name!r} is linked but its CONFIG entry has no 'linked_partner'")`.
  - `Experiment.reserved_devices(cls) -> list[str]` (classmethod) — sorted union of `reserves()` over `cls.devices()`.
  - Each class schema from `introspect.describe()` gains `"reserves": list[str] | None` and `"reserve_error": str | None`.

`reservations.py` reaches CONFIG only through `pytweezer.servers.device_server` (`resolve_address`, `device_index`), so tests fake it with `monkeypatch.setattr(device_server, "get_config", ...)` as `tests/test_servers.py::fake_config` does.

- [ ] **Step 1: Write the failing tests** in `tests/test_experiment_reservations.py` with a fake `CONFIG["Devices"]` holding: `"Cam"`, `"Old Cam"` (`"active": False`), `"Rig"` (composite with sub-devices `"Rig Cam"`, `"Rig Dac"`), `"CaF MotMaster"` (`"linked_partner": "Rb MotMaster"`), `"Rb MotMaster"`, `"K MotMaster"` (no partner).

```python
def test_plain_device_reserves_itself(fake_config):
    assert device_reservation("Cam") == {"Cam"}

def test_names_are_canonicalised(fake_config):
    assert device_reservation(" cam ") == {"Cam"}

def test_composite_reserves_its_members_but_a_member_only_itself(fake_config):
    assert device_reservation("Rig") == {"Rig", "Rig Cam", "Rig Dac"}
    assert device_reservation("Rig Cam") == {"Rig Cam"}

def test_unknown_and_inactive_devices_are_refused(fake_config):
    with pytest.raises(ReservationError, match="not found"):
        device_reservation("Nope")
    with pytest.raises(ReservationError, match="inactive"):
        device_reservation("Old Cam")

def test_linked_motmaster_reserves_its_partner(fake_config):
    class Linked(MotMasterExperiment):
        mm = MotMaster("CaF MotMaster", script="S", linked=True)
        def run_point(self): pass
    assert Linked.reserved_devices() == ["CaF MotMaster", "Rb MotMaster"]

def test_unlinked_motmaster_reserves_only_itself(fake_config): ...  # == ["CaF MotMaster"]

def test_linked_motmaster_without_partner_is_refused(fake_config):
    # MotMaster("K MotMaster", linked=True) -> ReservationError match "linked_partner"

def test_describe_reports_reservations_and_errors(fake_config, monkeypatch): ...
    # describe() on a module with one valid and one bad class: schema["reserves"]
    # is the sorted list / None, schema["reserve_error"] None / the message.
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_reservations.py -q`
Expected: FAIL with `ModuleNotFoundError: pytweezer.experiment.reservations`

- [ ] **Step 3: Implement** the interfaces above. Add `"linked_partner": "Rb MotMaster"` to the `"CaF MotMaster"` CONFIG entry only: `EDMSuite/DAQ/MoleculeMOTHardware.cs` makes CaF the primary and RbTweezer the secondary. Update the linked-mode paragraph in `motmaster.py`'s module docstring so it says the secondary is reserved through `linked_partner`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_experiment_reservations.py tests/test_motmaster_experiment.py tests/test_experiment_arguments.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/reservations.py pytweezer/experiment/arguments.py pytweezer/experiment/motmaster.py pytweezer/experiment/experiment.py pytweezer/experiment/introspect.py pytweezer/configuration/config.py tests/test_experiment_reservations.py
git commit -m "Expand declared devices into reservation sets"
```

---

### Task 2: Refuse undeclared devices

**Files:**
- Modify: `pytweezer/servers/device_client.py` (`_resolve_endpoint`, new `restrict_devices`)
- Modify: `pytweezer/experiment/experiment.py` (`__init__`, `device`)
- Modify: `pytweezer/experiment/runner.py` (`open_writer`, `run_local`)
- Modify: existing tests that open devices not declared or not in CONFIG (`tests/test_experiment_runner.py`, `tests/test_motmaster_experiment.py`, others found by the suite run)
- Test: `tests/test_experiment_reservations.py`

**Interfaces:**
- Consumes: `canonical_device`, `Experiment.reserved_devices()` (Task 1).
- Produces:
  - `device_client.UndeclaredDeviceError(PermissionError)`
  - `device_client.restrict_devices(names: Iterable[str] | None) -> None` — sets a module-level allow-list of canonical names; `None` lifts it. `_resolve_endpoint` (so both `get_device` and `get_device_async`) raises `UndeclaredDeviceError` for a name whose `resolve_address(name).name` is outside it.
  - `Experiment.allowed_devices: frozenset[str] | None` (instance attribute, default `None` = unrestricted).
  - `Experiment.device(name, ...)` raises `UndeclaredDeviceError` before opening anything, real or simulated, `fresh=True` included, when `allowed_devices` is set and `canonical_device(name)` is not in it. Message: `f"{type(self).__name__} uses device {name!r} without declaring it; declare it as a class attribute, e.g. camera = Device({name!r})"`.
  - `open_writer(..., allowed_devices: Iterable[str] | None = None)` sets `experiment.allowed_devices`.
  - `run_local` passes `experiment_cls.reserved_devices()`; it does **not** call `restrict_devices`.

- [ ] **Step 1: Write the failing tests**

```python
def test_undeclared_device_is_refused_in_run_local(fake_config, fake_devices):
    class Sneaky(Experiment):
        cam = Device("Cam")
        def run_point(self):
            self.device("Rig Dac")
    m = run_local(Sneaky)
    assert m.status == "failed"
    assert "without declaring it" in m.attrs["error"]
    assert [c.name for c in fake_devices] == []

def test_declared_and_composite_member_devices_are_allowed(fake_config, fake_devices):
    # Experiment declaring Device("Rig") may call self.device("Rig Cam") and
    # self.device("cam", fresh=True); run_local completes.

def test_undeclared_device_is_refused_in_simulation(fake_config):
    # run_local(Sneaky, simulate=True) fails with the same message.

def test_restrict_devices_blocks_direct_get_device(fake_config):
    restrict_devices({"Cam"})
    try:
        with pytest.raises(UndeclaredDeviceError):
            device_client._resolve_endpoint("Rig Dac", None, None, AutoTarget)
        device_client._resolve_endpoint("cam", None, None, AutoTarget)  # no raise
    finally:
        restrict_devices(None)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_reservations.py -q`
Expected: the four new tests FAIL (`ImportError: cannot import name 'restrict_devices'`)

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run the whole suite and fix the existing tests that broke** by declaring the devices they open and faking CONFIG for invented names (`"Cam"`, `"Rig Dac"`, `"K MotMaster"`). Do not weaken the check to make a test pass.

Run: `uv run pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add <this task's Files> && git commit -m "Refuse devices an experiment did not declare"
```

---

### Task 3: Pipelines and the scheduling rule in the queue

**Files:**
- Modify: `pytweezer/experiment/task.py` (`TaskRequest`, `Task`)
- Modify: `pytweezer/experiment/queue.py` (`QueueState`, `ExperimentQueue`)
- Modify: `pytweezer/experiment/recipes.py` (`Recipe.replay`)
- Modify: `pytweezer/experiment/client.py` (`submit`, `save_recipe`, `submit_recipe`, `ExperimentManagerClient.submit_recipe`)
- Modify: `pytweezer/servers/experiment_manager.py` (`_cmd_submit_recipe` passes `pipeline`; replace uses of `queue.running`/`next_due`/`state.worker` just enough to keep one-at-a-time behaviour green until Task 4)
- Modify: `pytweezer/GUI/experiments/queue_view.py` (`set_snapshot`, `queue_summary`), `pytweezer/GUI/experiments/results.py` (`_queue_changed`) — only to read `snapshot["running"]` as a list
- Test: `tests/test_experiment_queue.py`, `tests/test_experiment_recipes.py`

**Interfaces:**
- Produces:
  - `TaskRequest.pipeline: str = "main"`, validator: strip whitespace, empty → `"main"`. `Recipe` inherits it.
  - `Task.devices: list[str] | None = None` (canonical names; `None` = not yet resolved).
  - `QueueState.schema_version = 2`; `QueueState.workers: list[WorkerRecord]` replaces `worker`. A `model_validator(mode="before")` turns an old `"worker": {...} | None` into `"workers"`.
  - `ExperimentQueue.submit(request, at=None, devices: list[str] | None = None) -> Task`
  - `ExperimentQueue.set_devices(rid: int, devices: list[str]) -> None` (waiting tasks only)
  - `ExperimentQueue.running -> list[Task]` (RUNNING or PAUSED, by rid)
  - `ExperimentQueue.worker(rid: int) -> WorkerRecord | None`
  - `ExperimentQueue.heads(at=None) -> list[Task]` — for each pipeline with nothing running or paused, its first task in `ordered()` that is QUEUED and due; sorted by `(-priority, rid)`.
  - `ExperimentQueue.startable(at=None) -> list[Task]` — the scheduling rule below.
  - `ExperimentQueue.waiting_for(task: Task) -> dict[str, int]` — `{device: holding rid}` for a head whose devices are held by running tasks; `{}` otherwise. A running task with `devices is None` holds everything: return `{"(unknown)": rid}` for it.
  - `snapshot()["running"]` is a list; each `"queue"` entry carries `"waiting_for"` (from `waiting_for`).
  - `mark_started` raises `QueueError` if the task's pipeline already has a running task, its devices are `None`, or any is held. It appends the worker record.
  - `finish` removes that rid's worker record and carries the `TODO` below.
  - `action_for`, `pause`, `resume`, `terminate`, `mark_progress`, `mark_paused`, `_running` work on any running task by rid.
  - `next_due` and `state.worker` are removed.
  - `Recipe.replay(..., pipeline: str | None = None)` (`None` = the recipe's own). `client.submit(..., pipeline="main")`, `client.save_recipe(..., pipeline="main")`, `client.submit_recipe(..., pipeline=None)`; the manager's `submit_recipe` command takes an optional `"pipeline"`.

Scheduling rule (`startable`):

```python
claimed = held devices of running tasks   # any running task with devices None -> return []
started = []
for head in self.heads(at):               # already in (-priority, rid) order
    if head.devices is None:
        continue
    if claimed.isdisjoint(head.devices):
        started.append(head)
        claimed |= set(head.devices)
return started
```

TODO to place in `finish`, verbatim:

```python
# TODO: devices are released however the task ended. After an abort a device
# call in flight still completes, a killed linked primary can leave its
# secondary claimed, and a timed-out follower waits for its trigger, so another
# pipeline may start on a sequencer that is not yet idle.
```

- [ ] **Step 1: Write the failing tests** (rework `test_only_one_task_runs_at_a_time` and the `queue.running.status` / `state.worker` assertions to the new shapes; `request()` gains a `devices` helper passed to `submit`).

```python
def test_pipelines_run_side_by_side_but_each_runs_one_task():
    q = ExperimentQueue()
    q.submit(request(pipeline="CaF"), devices=["CaF MotMaster"])
    q.submit(request(pipeline="CaF"), devices=["Rb HamCam"])
    q.submit(request(pipeline="Rb"), devices=["Rb MotMaster"])
    assert [t.rid for t in q.startable()] == [1, 3]

def test_a_shared_device_blocks_across_pipelines_and_says_why():
    # rid 1 (A, ["X"]) started; rid 2 (B, ["X","Y"]) -> startable() == [],
    # waiting_for(rid 2) == {"X": 1}; snapshot()["queue"][0]["waiting_for"] == {"X": 1}

def test_a_task_waiting_behind_its_own_pipeline_has_no_waiting_for(): ...

def test_lower_priority_in_another_pipeline_overtakes_a_blocked_task():
    # running 1 (A, ["X"]); 2 (B, ["X"], priority 5); 3 (C, ["Y"], priority 0)
    assert [t.rid for t in q.startable()] == [3]

def test_a_blocked_head_is_not_overtaken_within_its_pipeline():
    # running 1 (A, ["X"]); 2 (B, ["X"], priority 5); 3 (B, ["Y"]) -> startable() == []

def test_one_tick_never_double_books_a_device():
    # 1 (A, ["X"], priority 0), 2 (B, ["X"], priority 3): startable() == [2]

def test_raising_priority_changes_who_gets_a_freed_device():
    # 1 running (A, ["X"]); 2 (B, ["X"]), 3 (C, ["X"]); set_priority(3, 9);
    # finish(1) -> startable() == [3]

def test_unresolved_devices_never_start_until_set():
    # submit(devices=None) -> startable() == []; set_devices(rid, ["X"]) -> [rid]

def test_a_running_task_with_unknown_devices_blocks_everything():
    # state with a RUNNING task devices=None; another pipeline's head ["Y"] -> []

def test_mark_started_refuses_a_clash_or_a_busy_pipeline(): ...

def test_pipeline_name_is_stripped_and_defaults_to_main():
    assert request(pipeline="  ").pipeline == "main"
    assert request(pipeline=" CaF ").pipeline == "CaF"

def test_old_state_file_migrates_worker_and_pipeline(tmp_path):
    # write a schema_version 1 JSON with "worker": {...} and an active task
    # without "pipeline"/"devices"; QueueStore.load() -> workers == [record],
    # task.pipeline == "main", task.devices is None

def test_finish_releases_only_that_tasks_worker(): ...
```

In `tests/test_experiment_recipes.py`: `test_replay_keeps_the_recipes_pipeline_unless_overridden`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_queue.py tests/test_experiment_recipes.py -q`
Expected: FAIL (`unexpected keyword argument 'devices'`, missing `startable`)

- [ ] **Step 3: Implement** the interfaces above. In the manager, keep starting at most one worker for now: start `startable()[0]` when no worker is running, and pass `devices=` nowhere yet (Task 4 resolves them). Until Task 4, `_cmd_submit` stores `devices=[]` so existing manager tests still start tasks. In the GUI, only adapt to the list shape.

- [ ] **Step 4: Run the suite**

Run: `uv run pytest tests/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add <this task's Files> && git commit -m "Schedule queue tasks by pipeline and device reservation"
```

---

### Task 4: One worker per running task in the manager

**Files:**
- Modify: `pytweezer/servers/experiment_manager.py`
- Modify: `pytweezer/experiment/worker.py` (`run`)
- Modify: `pytweezer/GUI/experiments/feed.py` (`poll`, `points`)
- Test: `tests/test_experiment_manager.py`, `tests/test_experiment_worker.py`, `tests/test_experiment_gui.py` (`test_real_feed_signals_queue_changes_and_points`)

**Interfaces:**
- Consumes: `ExperimentQueue.startable/heads/worker/set_devices/submit(devices=)` (Task 3); schema `"reserves"`/`"reserve_error"` (Task 1); `restrict_devices`, `open_writer(allowed_devices=)` (Task 2).
- Produces:
  - `ExperimentManager.processes: dict[int, psutil.Process]` replaces `process`; `_abort_requested: set[int]`.
  - `_reservation(module: str, class_name: str) -> list[str] | None` — the catalogue schema's `"reserves"`; `None` if the class is missing while `catalogue.busy`; `RecipeError` with `reserve_error` text, or the existing "not in the catalogue" text, otherwise.
  - `_cmd_submit` and `_cmd_submit_recipe` call `_reservation` before `queue.submit(..., devices=...)`, so a bad declaration is refused with `{"ok": False, "error": ...}` and nothing is queued.
  - `_resolve_devices()` (each tick, before starting): for waiting tasks with `devices is None`, set them from `_reservation`; on `RecipeError` finish the task `FAILED` with that message.
  - `_start_due()` starts a worker for every task in `queue.startable()`.
  - `_reap_workers()` and `_reconcile_workers()` handle every worker record (adopt each still-alive one by pid + create time; settle the rest as now).
  - `_cmd_abort` kills `processes[rid]` only.
  - Notifier `"points"` becomes `{str(rid): [row, ...]}`. Starting a task sets its key to `[]` and deletes the keys of finished tasks in the same pipeline. `_add_point` creates the key if missing (adopted worker).
  - `ExperimentFeed.points(rid)` returns `state["points"].get(str(rid), [])`; `poll` emits `point_received` for appends at path `["points", key]` with `rid=int(key)`. Its public signals and methods are otherwise unchanged, so `results.py` needs nothing more.
  - Worker `run`: after loading the task, `restrict_devices(task.devices)` and `open_writer(..., allowed_devices=task.devices)`; a task with `devices is None` finishes `FAILED` with "task has no device reservation".

- [ ] **Step 1: Write the failing tests** (the `StaticCatalogue` demo schema gains `"reserves"`; add a second demo schema in another pipeline with different devices).

```python
def test_two_pipelines_start_two_workers(make_manager, monkeypatch): ...
    # patch _spawn to fake processes; submit A and B with disjoint reserves;
    # tick() -> len(manager.processes) == 2, both tasks running

def test_clashing_tasks_start_one_after_the_other(make_manager, monkeypatch): ...

def test_submit_refuses_a_bad_declaration(make_manager):
    # schema with reserves=None, reserve_error="... linked_partner" ->
    # handle({"command": "submit", ...}) == {"ok": False, "error": <contains linked_partner>}
    # and the queue stays empty

def test_reservation_resolved_later_when_catalogue_was_busy(make_manager): ...
    # catalogue.busy = True and class absent -> submit ok, task.devices None,
    # tick() starts nothing; add the class, busy = False, tick() -> started

def test_unresolvable_reservation_fails_the_task(make_manager): ...
    # devices None, class absent, busy False -> tick() finishes it FAILED
    # with "not in the catalogue"

def test_abort_kills_only_that_worker(make_manager, monkeypatch): ...

def test_restart_adopts_every_live_worker(make_manager, monkeypatch): ...

def test_points_are_kept_per_task(make_manager): ...
    # two running tasks report points; notifier["points"] has both keys

def test_worker_restricts_devices_to_the_reservation(...):  # in test_experiment_worker.py
```

Update `test_real_feed_signals_queue_changes_and_points` to the per-rid points shape.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_manager.py tests/test_experiment_worker.py -q`
Expected: FAIL (`AttributeError: 'ExperimentManager' object has no attribute 'processes'`)

- [ ] **Step 3: Implement** the interfaces above, and remove the Task 3 stop-gaps (`devices=[]` at submit, one-worker limit).

- [ ] **Step 4: Run the suite, including the end-to-end test**

Run: `uv run pytest tests/ -q`
Expected: PASS (`tests/test_experiment_e2e.py` included)

- [ ] **Step 5: Commit**

```bash
git add <this task's Files> && git commit -m "Run one worker per task and reserve devices across pipelines"
```

---

### Task 5: Pipelines in the queue view

**Files:**
- Modify: `pytweezer/GUI/experiments/queue_view.py`
- Test: `tests/test_experiment_gui.py`

Follow the `pytweezer-gui-design` skill for styling (theme colours from `pytweezer/GUI/theme.py`, no hard-coded new palette).

**Interfaces:**
- Consumes: snapshot shape from Tasks 3–4 (`running` list, `pipeline`, `devices`, `waiting_for` on queue entries).
- Produces:
  - `COLUMNS = ["RID", "Pipeline", "Experiment", "Label", "Status", "Progress", "Priority", "Time", "Submitter"]`. The Status column's index is looked up from `COLUMNS`, not hard-coded as `3`.
  - `PipelineChips(QWidget)` in `queue_view.py`: `set_pipelines(counts: dict[str, tuple[bool, int]])` (name → (anything running, number waiting)), `selected() -> str | None` (`None` = All), signal `selection_changed(object)`. One chip is checked at a time; "All" is first and the default. A chip's text is the name, then ` ●` if running, then ` {n}` if n > 0 waiting. If the selected pipeline disappears, selection returns to All.
  - `QueueView.set_snapshot` builds the chips from the pipelines of every task in the snapshot, and hides rows (`setRowHidden`) whose pipeline isn't the selected one.
  - The Status cell of a task with a non-empty `waiting_for` reads `"waiting: " + ", ".join(f"{device} (RID {rid})" ...)`, with a `"blocked"` entry in `_ROW_SHADES` (amber, from the theme's `"starting"` state colour). Otherwise the text is unchanged.
  - Rows are ordered: running/paused (by rid), then queued, then history.
  - The details dialog adds `Pipeline: <name>` and `Devices: <comma-separated devices or "not yet known">`.
  - `queue_summary(snapshot)` → e.g. `"2 running (41, 43), 3 waiting"` / `"nothing running, nothing waiting"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_queue_view_shows_pipeline_and_why_a_task_waits(qapp):
    # snapshot: running [41 CaF, 43 background], queue [44 Rb with
    # waiting_for {"Rb MotMaster": 41}, 45 CaF]
    # -> Pipeline column texts; row for 44 Status == "waiting: Rb MotMaster (RID 41)";
    #    row for 45 Status == "queued"; first two rows are 41 and 43

def test_pipeline_chips_count_and_filter(qapp):
    # chip texts == ["All", "background ●", "CaF ● 1", "Rb 1"];
    # select "CaF" -> only rows 41 and 45 visible; next snapshot without CaF -> All

def test_details_list_pipeline_and_devices(qapp): ...

def test_queue_summary_counts_running_tasks():
    assert queue_summary({"running": [{"rid": 41}, {"rid": 43}], "queue": [{}] * 3}) == "2 running (41, 43), 3 waiting"
```

Update `test_queue_view_rows_and_buttons` for the new column order.

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q`
Expected: FAIL

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run the GUI tests, then grab an offscreen screenshot** of a `QueueView` filled with the snapshot from the first test (`widget.grab().save(...)` into the scratchpad) and look at it: chips readable, amber row distinct, nothing truncated.

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add <this task's Files> && git commit -m "Show pipelines, filter chips and waiting reasons in the queue view"
```

---

### Task 6: Pipeline box on the submit form

**Files:**
- Modify: `pytweezer/GUI/experiments/arg_editor.py` (queue form ~line 400, `load_request` ~line 549, request building ~line 735)
- Modify: `pytweezer/GUI/experiments/panel.py` (pass known pipelines from each snapshot to the editor)
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Produces:
  - `ArgEditor.pipeline: QComboBox` (editable), placed in the queue form directly after "Priority", default text `"main"`, tooltip `"tasks in one pipeline run one at a time, in order"`.
  - `ArgEditor.set_pipelines(names: list[str]) -> None` — sets the drop-down items (sorted, `"main"` always present), keeping the current text.
  - The built `TaskRequest` carries `pipeline=self.pipeline.currentText()`; `load_request` (also used for recipes and "Edit as new") sets it from the request.
  - Saving a recipe from the form stores the pipeline (it is part of the request).

- [ ] **Step 1: Write the failing tests**

```python
def test_form_submits_the_chosen_or_typed_pipeline(editor):
    assert editor.request().pipeline == "main"
    editor.pipeline.setCurrentText("CaF")
    assert editor.request().pipeline == "CaF"

def test_load_request_restores_the_pipeline(editor): ...

def test_panel_offers_pipelines_seen_in_the_queue(qapp): ...
```

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q`
Expected: FAIL

- [ ] **Step 3: Implement** the interfaces above.

- [ ] **Step 4: Run the GUI tests**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add <this task's Files> && git commit -m "Choose a task's pipeline on the submit form"
```

---

### Task 7: Skills, docs and a final check

**Files:**
- Modify: `.claude/skills/add-experiment/SKILL.md` ("The manager and its workers", device declaration section)
- Modify: `.claude/skills/pytweezer-architecture/SKILL.md` if it describes the manager as running one task at a time
- Modify: `.claude/skills/notebook-rig/SKILL.md` only if it mentions `run_local` device use

- [ ] **Step 1: Update the skills** to say: every device must be declared (undeclared use raises `UndeclaredDeviceError`, in workers also for direct `get_device`); what a pipeline is, the default `"main"`, and the scheduling rule; how reservations expand (composite members, `linked_partner`); devices are released as soon as a task ends (the `TODO`); and that `Rig`/notebooks are not covered by reservations.

- [ ] **Step 2: Full check**

Run: `uv run pytest tests/ -q && uv run ruff check . && uv run ruff format --check . && (cd docs && uv run make html)`
Expected: all pass; the docs build has no new warnings.

- [ ] **Step 3: Commit, push, open the PR**

```bash
git add <this task's Files> && git commit -m "Document pipelines and device reservations in the skills"
git push
gh pr create --title "Experiment pipelines with device reservations" --body "<summary of the spec, link to it, test plan>"
```
