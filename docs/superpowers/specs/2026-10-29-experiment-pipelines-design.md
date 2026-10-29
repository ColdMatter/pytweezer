# Experiment pipelines: running tasks in parallel without hardware clashes

Date: 2026-10-29. Status: design, agreed in conversation; not yet planned.

## Intent

The Experiment Manager (`pytweezer/servers/experiment_manager.py`) runs one
task at a time. We want several tasks running at once, without two of them
ever driving the same hardware. Two concrete cases, both real:

- a CaF sequence and an Rb sequence, each on its own hardware;
- a slow background job (e.g. a laser calibration) running during a main
  measurement.

Two mechanisms, with separate jobs:

- **Pipelines** give a defined order of execution. A pipeline is a named
  lane of the queue; within it tasks run strictly one at a time. Tasks that
  share no hardware but must run in sequence go in the same pipeline.
- **Device reservations** give safety. A task starts only if none of the
  devices it reserves is held by a task running in another pipeline. This
  holds even if two tasks were put in different pipelines by mistake.

The name "pipeline" follows ARTIQ, which the experiment framework is modelled
on and which the lab knows. Unlike ARTIQ, there is no overlap of one task's
`prepare` with the previous task's run.

## Goals

1. Tasks in different pipelines run concurrently, each in its own worker
   process.
2. A task never starts while another running task reserves any device it
   reserves.
3. Experiments must declare every device they use; using an undeclared
   device is an error.
4. The queue view shows pipelines, lets you filter by them, and says why a
   waiting task is not starting.

## Non-goals

- **Protection outside the queue.** Notebooks using `Rig`, `run_local()`, and
  GUI panels can still drive any device at any time. The reservation check
  covers only tasks run by the Experiment Manager. This is deliberate for now.
- **Shared/read-only access.** Any device two tasks both reserve means they
  never overlap, even if both only read from it (e.g. the wavemeter). Kept
  simple on purpose; may be revisited.
- **Device reservation fairness.** A waiting task does not reserve devices
  ahead of time (see Scheduling).
- **Overlapping `prepare`** with a running task, ARTIQ-style.

## Anti-goals (would make this a failure even if it "works")

- Two running tasks touching the same physical sequencer, camera or SLM,
  including via a linked MOTMaster's secondary or a composite's sub-device.
- A waiting task that silently never starts, with nothing in the GUI saying
  why.
- An experiment that opens a device it did not declare and gets away with it.

## Behaviour

### Pipelines

- `TaskRequest` gains `pipeline: str = "main"`. Because `Recipe` subclasses
  `TaskRequest`, saved recipes remember their pipeline automatically.
- Pipelines are created on the fly: submitting with a new name creates it.
  There is no fixed list in CONFIG.
- Pipeline names are compared exactly after stripping whitespace; an empty
  name means `"main"`.
- At most one task per pipeline is running or paused at any time.

### Scheduling

On every tick the manager considers each pipeline whose current task (if any)
has finished:

1. The pipeline's candidate is its first waiting task in the existing order:
   highest priority first, then lowest RID; held tasks and tasks whose
   `due_time` is still in the future are skipped, as now.
2. Candidates from all pipelines are taken in priority order (then lowest
   RID). Each one starts if none of its reserved devices is held by a running
   or paused task. Devices claimed by a candidate started earlier in the same
   tick count as held.
3. A candidate that cannot start leaves its pipeline waiting. A lower-priority
   task in the *same* pipeline does not overtake it (the pipeline defines
   order). Tasks in *other* pipelines can start if their devices are free,
   even if their priority is lower. **The user explicitly chose this.**
   It keeps the hardware busy, at the cost that a stream of small jobs can
   delay a big one.
4. The remedy for a task stuck behind others is raising its priority. That
   already exists: `ExperimentQueue.set_priority`, `client.set_priority`, and
   the Priority +/− buttons in `pytweezer/GUI/experiments/queue_view.py`. It
   must keep working for queued tasks, and a change must affect the
   scheduling between pipelines on the next tick.

A paused task keeps its devices reserved: it is mid-sequence.

### Device reservations

Each declared device attribute reports the set of CONFIG device names it
reserves:

- a plain `Device("X")`: `{X}`;
- a composite device (a CONFIG entry with a `"devices"` sub-dict, e.g.
  `"Rb Rearrangement Rig"`): itself plus all its sub-devices. A sub-device
  declared on its own (e.g. `"Rb HamCam"`) reserves just itself, so it
  clashes with any task reserving the composite;
- `MotMaster(..., linked=True)`: itself plus its linked secondary. Today
  pytweezer does not know which MOTMaster a linked primary takes over; the
  pairing lives on the C# side. Record it in CONFIG as a new key on the
  MOTMaster device entry, e.g. `"linked_partner": "Rb MotMaster"` on
  `"CaF MotMaster"`, and vice versa if either can be primary. Declaring
  `linked=True` on a MOTMaster with no `linked_partner` is an error at submit.

Put this behind one overridable method on `Device` (e.g.
`reserves(devices_config) -> set[str]`) so future devices with hidden
dependencies need only override it.

Names are canonicalised the way `get_device()` matches them (whitespace- and
case-insensitively, `device_client.py`), so `"rb hamcam"` and `"Rb HamCam"`
are the same reservation. A declared name that is not in `CONFIG["Devices"]`
fails the submission.

The reservation set is computed **at submit time** from the experiment
class's declarations and stored on the `Task` (new field, e.g.
`devices: list[str]`). The scheduler and the GUI use the stored set, so it is
stable even if the experiment's source changes before the task runs.

### Enforcing declarations

- `Experiment.device(name)` raises a clear error if `name` (canonicalised) is
  not among the task's declared devices, real or simulated. This includes
  `fresh=True` calls. The error names the device and says to declare it as a
  class attribute.
- In a worker, the allowed set is the task's stored reservation set. An
  experiment edited after submission to use a new device therefore fails
  loudly rather than clashing.
- `run_local()` enforces the same rule (allowed set = the class's
  declarations, expanded as above), so mistakes surface before queueing.
- Existing experiments in `pytweezer/experiments/` and
  `pytweezer/experiment/motmaster.py` must still pass. The `MotMaster`
  followers are already opened through declared attributes; a linked
  primary's secondary is reached by the C# side, not via `self.device`, so it
  needs no client.

### Manager and workers

- One worker process per running task. The single `self.process` /
  `QueueState.worker` become per-RID collections.
- Queue state on disk gains the new fields; bump `QueueState.schema_version`
  and load old files (one worker record, tasks without `pipeline` or
  `devices`) by treating them as pipeline `"main"`. Tasks already queued from
  an old file have no stored reservation: recompute it when they are loaded
  or before they start.
- On manager restart, adopt every still-running worker, not just one, using
  the existing token/pid/create-time check.
- Pause, resume, terminate, hold, release, delete and set-priority are
  already per RID and keep their meaning. **Abort** currently assumes a single
  worker (`_cmd_abort` kills `self.process`); it must kill the worker for
  that RID only.
- `ExperimentQueue.running`, `next_due`, `action_for`, `mark_started` and
  `_running` all assume one running task and need reworking. The pure-data
  queue should own the scheduling rule above (it has no sockets or processes,
  so it is easy to test), and the manager should just start what it returns.
- Simulation mode: each worker already builds its own in-process simulated
  devices, so this needs no change beyond running several workers.
- `snapshot()` changes shape: `"running"` becomes a list. Update all
  consumers (the GUI queue view, `client.py`'s waiting helpers, notebooks via
  the client).
- A waiting task's snapshot entry includes why it is waiting, when that is a
  device clash: the device(s) and the RID(s) holding them. Compute this in the
  queue with the same function the scheduler uses, so the GUI never disagrees
  with the scheduler.

### GUI (Experiments tab)

As agreed from the mockups:

1. **Pipeline column** in the queue table.
2. **Filter chips** above the table: "All" plus one chip per pipeline that
   currently has any active task, or has history in the current snapshot
   (builder's call). Each chip shows whether something in it is running (a
   dot) and how many tasks are waiting. One chip is selected at a time;
   "All" is the default.
3. **Why a task is waiting**: the Status cell of a task blocked by a device
   reads e.g. `waiting: Rb MotMaster (RID 41)`, with a distinct (amber)
   shade. A task waiting only because an earlier task in its own pipeline is
   running shows plain `queued`.
4. **All running tasks at the top**, then waiting, queued and history, as the
   table orders things now.
5. **Reserved devices in the details dialog** (double-click a row), listing
   the full expanded set, including a linked secondary or composite members.
6. **Pipeline box on the submit form**: an editable drop-down next to
   Priority, listing existing pipelines, defaulting to `"main"`. Recipes
   carry their pipeline into the form when loaded.

Follow the `pytweezer-gui-design` and `pytweezer-gui-internals` skills for
styling and plumbing.

## Constraints

- Branch per the repo conventions (feature branch + PR); main stays
  deployable; the test suite and the docs build must pass.
- Keep `ExperimentQueue` free of sockets, threads and processes.
- Docstrings describe current behaviour only; UK English.
- Update the `add-experiment` skill (and `pytweezer-architecture` if it
  describes the single-worker manager) to say devices must be declared, what
  a pipeline is, and how reservations are expanded.

## Left to the builder

- Exact field names (`pipeline`, `devices`, `linked_partner`) and the
  method name for reservation expansion.
- How the manager loads the experiment class's declarations at submit time
  (the existing `_schema`/catalogue path already introspects classes;
  `Device.to_schema()` and `MotMaster.to_schema()` already carry the device
  name and `linked`).
- Whether the "why waiting" text also appears as a tooltip, and the precise
  chip styling.
- Which tests to add (per the `write-tests` skill): the scheduling rule and
  reservation expansion are the obvious pure-logic targets.
