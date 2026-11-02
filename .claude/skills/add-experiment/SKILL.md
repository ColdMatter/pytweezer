---
name: add-experiment
description: Write, run, queue and read back pytweezer experiments — an Experiment subclass in pytweezer/experiments/ with declared arguments and devices, run per scan point by the Experiment Manager (or in-process with run_local), saving one h5 file per measurement — and the saved parameters they read. Use whenever the user wants to write, add or change an experiment or measurement sequence (a scan, a MOTMaster sequence plus camera readout, a calibration, tweezer loading or trap sites), submit or queue one from a notebook (submit/wait), scan an argument, record data from a run, load a measurement file, read or set saved parameters, give an experiment a Results-tab analysis, or work on the Experiment Manager, its workers, pipelines, device reservations or queue, or the Experiments/Results/Saved parameters tabs — even if they only describe the physics ("scan the tweezer depth and count atoms") without saying "experiment".
---

# Experiments

An experiment is a class in a module under `pytweezer/experiments/`. The
Experiment Manager (a `CONFIG["Servers"]` process on the server PC) queues
tasks and runs each one in a **fresh worker subprocess** that imports the
module, makes one instance, and writes **one h5 file per measurement**, flushed
after every point.

## Writing one

```python
# pytweezer/experiments/loading.py
import numpy as np

from pytweezer.experiment import Device, Experiment, Integer, Number


class LoadingCurve(Experiment):
    """Atom number against MOT loading time."""

    load_time = Number(0.5, unit="ms", scale=1e-3, min=0)
    shots = Integer(1, min=1)
    camera = Device("Rb ThorCam")

    def prepare(self):
        self.camera.set_roi(...)  # set everything you rely on: device
        self.record("background", ...)  # state persists between tasks

    def run_point(self):
        # self.load_time holds this point's value (SI), self.point its index
        frames = self.camera.acquire_n_frames(self.shots)
        self.record("image", frames[0])
        self.record("atom_number", float(frames.sum()), unit="counts")

    def finish(self): ...  # always runs, also after a failure or terminate
```

Rules that matter:

- **Arguments are class attributes** (`Number`, `Integer`, `Bool`, `Choice`,
  `Text` in `pytweezer/experiment/arguments.py`). Values are SI; `unit`/`scale`
  only control display (`Number(5e6, unit="MHz", scale=1e6)` shows "5 MHz").
  `group=` and `tooltip=` lay out the form. Defaults are validated at import.
  Submitted values are checked by the GUI
  form and by `submit(ExperimentClass, ...)`; anything else (a
  `"module:Class"` string) is only checked when the task starts.
- **No hardware at module import or class definition.** The manager imports
  every module in a subprocess to list its arguments for the GUI.
- `Device("<CONFIG name>")` gives a lazily created RPC client, closed when the
  task ends. For `run_parallel`, take extra clients with
  `self.device(name, fresh=True)` — one sipyco client per thread.
- **Declare every device you use.** `self.device(name)` (also `fresh=True`)
  raises `UndeclaredDeviceError` for a name outside the task's reserved set
  (a member of a declared composite and a linked partner are in it), and a
  queued task's worker also refuses a bare `get_device()` of anything outside
  it (`device_client.restrict_devices`). The declarations are what the
  manager reserves (see "Pipelines and device reservations").
- `record(name, value, unit="")`: in `run_point` it stores into
  `/results/<name>` row `point.index`; in `prepare`/`finish` into
  `/constants/<name>`. Each name's shape and dtype are fixed by its first
  record (later values may only safely widen, e.g. int→float); a mismatch
  raises and fails the task. Strings and dicts (as JSON) are allowed. A point
  that doesn't record a name gets NaN (floats) and `/recorded/<name>` False.
- An exception in any hook fails the task (status `failed`, traceback in the
  file and the queue). Don't catch-and-print.
- A class only appears in the GUI if it is defined in that module and
  overrides `run_point`, so intermediate base classes stay hidden; a base that
  inherits a concrete `run_point` hides with `run_point = Experiment.run_point`
  (`TweezerArrayShots`).
- `self.rid` is the task's run id (-1 under `run_local`), e.g. to tag broadcast
  frames (`acquire_n_frames(..., broadcast=True, header={"rid": self.rid})`).
- An experiment that chooses its own points overrides
  `planned_points(scan)`, raising `ValueError` for a scan it can't honour.

## MOTMaster sequences

```python
from pytweezer.experiment import Device
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterInteger,
    MotMasterNumber,
)


class Tof(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
    caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad")
    tof = MotMasterInteger(100, device="rb", parameter="tDelay1")  # always shown
    camera = Device("Rb ThorCam")

    def run_point(self):
        self.camera.start_acquisition()
        self.run_sequences()  # blocks until every Go() returns
        self.record("image", self.camera.acquire_n_frames(1)[0])
```

`master=True` goes on exactly one `MotMaster` of several (a lone one is the
master). `MotMasterInteger` vs `MotMasterNumber` must match the script's
parameter type (.NET `Int32` vs `Double`). Any other script parameter is set by
dotted name, `submit(Tof, **{"rb.tPulse": 3e-5})` or scanned with
`ListAxis(argument="rb.tPulse", ...)`; the Experiments form adds these from its
search box. Declared parameters are not searchable, Int32 ones need whole
values, and unselected ones keep the script's defaults (not recorded in the
file). `prepare()` sets script, iterations, save toggle and trigger mode on
every task; repeats come from `Scan(repetitions=...)`, one stored point per
shot; `run_sequences(**{"rb.tPulse": 3e-5})` overrides parameters for that one
run (`TweezerArrayShots` runs batches this way). `prepare()` puts followers in trigger mode; `run_sequences()` starts each
follower in a thread, then the master `follower_arm_delay` later
(`MotMasterExperiment` class attribute, default 0.5 s). A follower still running
`follower_timeout` (a `MotMaster(...)` keyword, default 60 s) after the master
finishes fails the task, and its device server stays blocked in the triggered
`Go()` (Abort does not free it), so restart that device server or trigger it
before the next task. Camera APIs differ
between drivers — check the driver, not old notebooks.

**Linked mode** (one script driving PH-BONESAW and RbTweezer together, fired by
TTL from the CaF side): declare only the primary,
`MotMaster("CaF MotMaster", script=..., linked=True)`, and no other `MotMaster`.
Every parameter of both machines, `SecondaryTriggerDelay` included, is a
parameter of that one script. The link lasts one task: `prepare()` sets linked
mode on or off on every sequencer (unlinking first) and `close_devices()`
releases it, even after a failure. While linked, the Rb MOTMaster refuses its
own `SetScriptPath`/`Go`, so an Rb-only task fails if someone has ticked Linked
by hand on PH-BONESAW. Any linked failure, either machine, raises from the task.
`experiments/linked_scope_test.py` is the minimal example; the MOTMaster side
is EDMSuite's `motmaster-linked-mode` skill.

## Running

From the GUI: **Experiments** tab → pick the class → edit, toggle "Scan" per
argument → Submit; MOTMaster parameters beyond the declared ones are added from
the search box. **Results** tab lists files as they appear (no refresh button)
and quick-plots a scalar against a scan axis; "Resubmit" reloads its
arguments. A running measurement plots live from the manager's published points,
not by re-reading its file; see "The manager and its workers".

Its **Analysis** dropdown offers "Plot against" for every file, then the
analyses the experiment declares as a class attribute (inherited), e.g.
`LoadTweezers`' `analyses = ("tweezer_histogram", "tweezer_photon_histogram")`;
an optimiser declares none. Each file records the names (root attr
`analyses`), so the tab never imports the experiment. Names map to views in
`pytweezer/GUI/experiments/analyses.py` (`ANALYSES`; a new one is a Qt-free
computation in `pytweezer/analysis/` plus an `AnalysisView` registered there).
A view reads only what the file recorded, so record what it needs (the tweezer
histograms take `first_counts`/`second_counts` with `trap_sites` and
`threshold`, else per-site photons, else frames: `analysis/tweezer_histogram.py`).
A name this GUI doesn't know is listed disabled, and every analysis is disabled
until the measurement finishes. An analysis only reads the file, through its
own archive connection, off the GUI thread. The tab remembers the choice per
experiment class for the session.

From a notebook or script:

```python
from pytweezer.experiment import LinearAxis, ListAxis, Scan, run_local
from pytweezer.experiment.archive import fetch_measurement
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait

scan = Scan(
    axes=[LinearAxis(argument="load_time", start=0, stop=2e-3, n=21)],
    repetitions=3,
    order="shuffle",
)  # nested | snake | shuffle
rid = submit(LoadingCurve, scan, shots=2, label="after realignment", pipeline="main")
task = wait(rid)  # Task: status, error, h5_path
m = fetch_measurement(rid)  # any PC; results=[...] to load a subset
m.results["atom_number"], m.points["load_time"], m.arguments, m.constants

m = run_local(LoadingCurve, scan)  # in-process, no queue, in memory
m.status, m.attrs["error"]  # a failure is recorded, not raised
```

`submit` needs a class importable by the manager — **not one defined in a
notebook**; use `run_local` for those (pass `path=` to keep a file). A plain
`submit()` is refused at once if the class is not in the catalogue, its module
fails to import, or its devices can't be reserved (`ReservationError`: unknown
or inactive device, or `linked=True` on a MOTMaster with no `linked_partner`;
`run_local` raises the same before running). While the catalogue
is still being read the task is queued and resolved later, failing if it then
can't be.
`Scan(repeat="scan")` repeats the whole grid rather than each point.
`ExperimentManagerClient` has `get/pause/resume/terminate/abort/hold/release/
delete/set_priority/snapshot/history/last_request/catalogue/describe_again/
recipes/save_recipe/delete_recipe/submit_recipe/saved_parameters/
set_saved_parameters/saved_parameter_history`.

For analysis, `Dataset` (`pytweezer/experiment/dataset.py`) reads a file by
rid (or path) and passes attribute access through to its `Measurement`:

```python
from pytweezer.experiment import Dataset

# Works on any PC; root=... for another data dir, results=[...] to load a subset.
ds = Dataset(rid)
ds.summarise()  # same text as the Results tab's metadata pane
# "dataframe" (default) | "dict" | "array"; or get_dataframe/get_dict/get_array
df = ds.get_data()
ax = ds.plot_scan("atom_number")  # mean ± SEM over repetitions; x, series, ax optional
```

`plot_scan` stamps the axes top-right with `ds.git_label` (`git 0a575d0`,
`-dirty` if the run had uncommitted changes), so a plot always names the code
that took it.

Rows are the points run, in run order: `index`, `repetition`, scanned
arguments, `t_start`, `t_end`, then results (SI). The DataFrame drops
array results (images); `get_dict` stacks them; `get_array` is a structured
array of the numeric scalars. `aggregate`/`describe` live in
`pytweezer/experiment/summary.py`, shared with the Results tab.

## Recipes

A recipe is a named, saved form — arguments, scan, priority and label — kept by
the Experiment Manager in `{data_root}/recipes.json`, so every PC sees the same
ones. In the **Experiments** tab recipes sit under their experiment in the
tree: click one to load it into the form (the header names it), **Save as
recipe…** stores the form (asks before replacing a name), right-click → *Submit
now* queues it unchanged, *Delete…* removes it for everyone.

```python
from pytweezer.experiment.client import (
    delete_recipe,
    recipes,
    save_recipe,
    submit_recipe,
)

save_recipe(LoadingCurve, "nightly", scan, shots=2, label="nightly check")
rid = submit_recipe(LoadingCurve, "nightly", shots=3)  # overrides fixed arguments
save_recipe(LoadingCurve, "nightly", shots=4, overwrite=True)  # a taken name is refused
recipes(LoadingCurve)
delete_recipe(LoadingCurve, "nightly")
```

A recipe stores every argument value, not just changed ones, except those its
scan sweeps. Submitting a
recipe that uses an argument the experiment no longer declares is refused with
the names; load it into the form, check it and save it again. An override may
not fix an argument the recipe scans. MOTMaster `attribute.parameter` names are
checked when the task runs, as for any submission.

## Saved parameters

Named calibrated values (float, int, bool, string, or an array of numbers) kept
by the Experiment Manager in `{data_root}/saved_parameters.json`, with every
change appended to `saved_parameters_log.jsonl`, so every PC and every task sees
the same ones (`pytweezer/experiment/saved_parameters.py`).

```python
from pytweezer.experiment.client import (
    get_saved_parameter,
    get_saved_parameters,
    saved_parameter_history,
    set_saved_parameters,
)

get_saved_parameter("tImgCoolVCO")  # KeyError names close matches
get_saved_parameters()  # {key: value}; arrays as numpy arrays
set_saved_parameters({"level": 0.3}, "realigned the probe")  # reason and user logged
set_saved_parameters({"new_key": 1.0}, "first calibration", create=True, group="MOT", unit="V")
saved_parameter_history("level", limit=10)  # newest first; rid= for one run's
```

- A set is all or none. Unknown keys are refused unless `create=True`; a key
  keeps its type (int accepted for a float), and an array keeps its number of
  dimensions and every axis length past the first.
- Read them in `prepare()`/`run_point()` for the value at run time, or as an
  argument default, `level = Number(get_saved_parameter("level"))` — read at
  import, so the module contacts the manager when imported and the form's
  **Defaults** button (which has the manager describe the experiment again)
  re-reads it.
- The **Saved parameters** tab edits values by hand with a reason and shows the
  history; arrays are read-only there. Keys are created only by
  `create=True`, an accepted `Fit`, or `import_parameters_json(path)` in
  `saved_parameters.py` (each top-level category of a `parameters.json`
  becomes a group).
- A simulating manager keeps its own under `<data_root>/simulated/`, apart
  from the lab's.

## Optimisers

An optimiser (`StagedScan`, `BayesianOptimisation`, `Fit`) runs a measurement
experiment and proposes saved-parameter values for someone to accept. It is an
`Experiment`, but writing, running and accepting one is covered by the
add-optimiser skill; a measurement it runs only needs a
`measure(**settings) -> dict[str, float]` method.

## Tweezer array measurements

The Rb tweezer experiments, in the order a session uses them:

1. `SlmTweezerArray` (`slm_array.py`) puts an array on the SLM through
   `Rb Rearrangement Rig` (`rig.make_array(...)`, one mask per point, shown on
   the `slm` image stream for the SLM Monitor applet). The rig remembers it:
   `rig.get_array()` gives `sites` and trap `x_um`/`y_um` (`None` if the rig
   didn't write the mask).
2. The `FindTrapSites` optimiser (add-optimiser) fits the array on the camera
   and proposes the `trap_*` saved parameters.
3. `LoadTweezers` (`load_tweezers.py`) and everything built on it read them in
   `prepare()` (`saved_trap_sites()`), moved into this ROI by the origin
   `camera.set_roi` returned. They are refused when none are saved, when
   `trap_geometry` is not the rig's array (`same_geometry`; an unreachable rig
   is not checked), or when a summing window leaves the ROI. With sites, each
   point records `first_occupied`/`second_occupied` (bool per site) and
   `loading`, plus raw `first_counts`/`second_counts` with
   `save_tweezer_counts`; frames are recorded only with `save_raw_images`.
   `LoadTweezers` runs without sites only if it records frames;
   `TweezerArrayMeasurement` always needs them.
4. `TweezerArrayShots` (`tweezer_array.py`) runs every shot at the saved
   sequence parameters (`SAVED_SEQUENCE_PARAMETERS`: script parameter → saved
   key; a missing key fails `prepare()`), a script parameter set in the form
   winning; the values used are the constant `saved_sequence`. Each point is a
   batch of shots, so its `first_occupied`/`second_occupied`/`*_counts` are
   `(shots, sites)`, and `save_raw_images` (off by default here) records
   `first_images`/`second_images`.

Shots broadcast their HamCam frames with `header={"rid": self.rid}`; the live
tweezer loading analysis (`analysis/tweezer_loading.py`) starts afresh whenever
the rid changes and places the saved sites by each frame's
`roi_x0`/`roi_y0` header.

## Files

`{data_root}/YYYY/MM/DD/{rid:06d}_{Class}.h5`. Root attrs: rid, experiment,
class_name, label, submitter, host, status, error, times, n_points, n_done, git
commit/dirty, analyses (JSON list of the class's `analyses`; absent in older
files). Groups: `/arguments` (effective values; `__schema__` attr),
`/scan` (spec JSON), `/points/{index,repetition,<axis>,t_start,t_end}`,
`/results`, `/recorded`, `/constants`, `/source` (the module's code). Rows are
planned points in execution order; only the first `n_done` ran. Files are
never overwritten (mode `"x"`) and readable mid-run (`locking=False`).

`data_root` = `$PYTWEEZER_DATA_DIR`; else
`CONFIG["Servers"]["Experiment Manager"]["data_root"]`; else `<repo>/data`.
The queue state lives in `{data_root}/queue_state.json`.

When a task finishes, the manager copies its file to the same path under
`CONFIG[...]["copy_root"]` (the router's USB drive, `\\10.59.0.1\g\pytweezer-data`;
`<copy_root>/simulated` when simulating) on a background thread
(`pytweezer/experiment/copier.py`). Runs never write to the network: a failed
copy is retried every minute, and the paths still to copy are kept in
`{data_root}/pending_copies.json` across restarts. Only a manager that binds its
sockets copies, so tests building one with `bind=False` never touch the drive.

Other PCs never read the manager's disk: it serves the files as sipyco RPC
target `"results"` on `results_port` (`pytweezer/experiment/archive.py`).
`ResultsArchive(root)` reads files directly and `RemoteArchive` is the same
interface over RPC (`days`, `runs(day)`, `find(rid)`, `load(path, results)`,
`planned_points(path)`, paths relative to `data_root`). `default_archive()` is
local on the manager's PC or with `PYTWEEZER_DATA_DIR` set, remote elsewhere;
the Results tab, `Dataset(rid)`, `fetch_measurement` and
`points_with_readings(rid, ...)` all go through it. Its reads run in threads, so
they never stall the notifier on the same event loop.

The manager also writes each run into the database (`pytweezer.database`): a
`runs` row on start and finish (git commit/dirty included), and a `points` row per point, holding its times,
scanned values and 0-d numeric results. Those rows sit next to the monitor
`readings`, so `points_with_readings(rid, "ni_adc")`
(`pytweezer.database.analysis`) gives one row per point with the readings during
it. `pytweezer-db-backfill` loads files the database missed. See
`docs/notes/database.md`.

## The manager and its workers

Code: `pytweezer/servers/experiment_manager.py` (single-threaded REP loop),
`pytweezer/experiment/{queue,worker,catalogue,introspect,client}.py`.

- **Published state:** the manager serves sipyco notifier `"experiment"` on
  `sync_port` (layout in the module): the queue snapshot (`running`, a list,
  `queue`, `history`, `simulated`, `alive`, and `catalogue_`/`recipes_`/
  `saved_parameters_version` counters; waiting tasks carry `waiting_for`) and
  `points`, keyed by rid, every point each running task (and each pipeline's
  last finished one) has measured, with its scanned values and 0-d numeric
  results. Only changes are sent; a GUI connecting late gets it all.
  `GUI/experiments/feed.py`'s `ExperimentFeed` is the Qt side (one per GUI,
  shared by its experiment tabs): `queue_changed`, `point_received`,
  `connection_changed`, `points(rid)` and `held_points()`. The Live Plot applet
  (`GUI/viewers/live_plot.py`) has its own feed and plots chosen scalar results
  accumulated across tasks (context menu → *experiment results*). The Results tab
  reads a running file once for metadata and the planned `/points` table
  (`archive.planned_points`), plots from the feed, and reloads the file when the
  task finishes; HDF5 cannot safely be re-read while it is written. Without a
  connected feed it falls back to polling the archive.

- **Pipelines and device reservations** (`reservations.py`, `queue.py`).
  A pipeline is a named lane (`TaskRequest.pipeline`, default `"main"`, created
  by submitting a new name; `Recipe`s keep theirs). Within one it is strictly
  one task at a time, in order: priority (high first), then rid; a `due_time`
  or hold passes a task over. Each worker is its own process, so pipelines run
  in parallel, but a task starts only if no running or paused task holds a
  device it reserves. A blocked head keeps its pipeline waiting while heads in
  other pipelines may start, even at lower priority; the remedy for a stuck
  task is `set_priority` (queue-view +/− buttons). Reservations are computed at
  submit and stored as `Task.devices`: a plain device reserves itself, a
  composite (CONFIG `"devices"`) itself and its sub-devices, a sub-device only
  itself, and `MotMaster(linked=True)` also its CONFIG `linked_partner` (`"CaF
  MotMaster"` names `"Rb MotMaster"`). `Device.reserves()` is the hook for other
  hidden dependencies. Names match as `get_device()` does. Only queued tasks are subject to reservations: `Rig`, `run_local` and GUI
  panels can drive a device another task holds, though `run_local` still
  refuses devices the class doesn't declare.
- The queue view has a Pipeline column and filter chips (a dot if one is
  running, a count of waiting tasks). A task blocked by a device shows amber
  `waiting: <device> (RID n)` (`all devices` while a running task's
  reservation is unknown); a pipeline's next task whose reservation awaits the
  catalogue's (re-)read of its module shows amber `waiting: reading
  experiments`; one only behind its own pipeline shows `queued`.
  The submit form's Pipeline box defaults to `main`; type a new name to create
  one. `"points"` is per RID, and each pipeline shows its latest task's.
- **Terminate** stops at the next point boundary (finish() runs). **Abort**
  kills that task's worker immediately: a device call in flight (`Go()`, a
  frame grab) still completes on the device. Devices are released the moment a
  task ends however it ended (a `TODO` in `ExperimentQueue.finish`), so a task
  in another pipeline can start on a sequencer that is not yet idle.
- Pause keeps the task's devices open and reserved; hold only applies to waiting tasks.
- State is saved on every change, because on Windows the GUI stops the manager
  with `TerminateProcess`. Workers outlive the manager: on restart each running
  worker is adopted again (pid + create time), otherwise its task is settled
  from its h5 status (`interrupted`/`crashed`). A worker that can't reach the
  manager for `orphan_timeout` s stops and marks itself `interrupted`.
- **Simulation** (`"simulate": SIMULATING` on the manager's entry, so any
  `pytweezer-server` off the server PC): `Device(...)` gives the experiment its
  device's simulated backend, built in the worker by the same `build_spec` as a
  device server, so no device servers or lab network are needed. Data goes to
  `<data_root>/simulated/`; files carry `simulated=True`; the GUI shows a
  banner. `run_local(..., simulate=True)` does the same in a notebook.
- Worker output goes to `logs/experiments/<rid>.log`. Workers bind no ports,
  so `pytweezer-kill-stale` ignores them.

Tests: `tests/test_experiment_*.py`, `tests/test_motmaster_experiment.py`,
`tests/test_results_{browser,archive,analyses}.py`, `tests/test_file_copier.py`,
`tests/test_saved_parameters.py`; `test_experiment_e2e.py` runs a real manager
and worker subprocesses on localhost. Tweezer measurements:
`tests/test_tweezer_array.py`, `tests/test_slm_array.py`,
`tests/test_tweezer_histogram.py`.
