---
name: add-experiment
description: Write, run, queue and read back pytweezer experiments — an Experiment subclass in pytweezer/experiments/ with declared arguments and devices, run per scan point by the Experiment Manager (or in-process with run_local), saving one h5 file per measurement. Use whenever the user wants to write, add or change an experiment or measurement sequence (a scan, a MOTMaster sequence plus camera readout, a calibration run), submit or queue one from a notebook (submit/wait), scan an argument, record data from a run, load a measurement file, or work on the Experiment Manager, its worker processes, the queue, or the Experiments/Results GUI tabs — even if they only describe the physics ("scan the tweezer depth and count atoms") without saying "experiment".
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
  Defaults are validated at import. Submitted values are checked by the GUI
  form and by `submit(ExperimentClass, ...)`; anything else (a
  `"module:Class"` string) is only checked when the task starts.
- **No hardware at module import or class definition.** The manager imports
  every module in a subprocess to list its arguments for the GUI.
- `Device("<CONFIG name>")` gives a lazily created RPC client, closed when the
  task ends. For `run_parallel`, take extra clients with
  `self.device(name, fresh=True)` — one sipyco client per thread.
- `record(name, value, unit="")`: in `run_point` it stores into
  `/results/<name>` row `point.index`; in `prepare`/`finish` into
  `/constants/<name>`. Each name's shape and dtype are fixed by its first
  record (later values may only safely widen, e.g. int→float); a mismatch
  raises and fails the task. Strings and dicts (as JSON) are allowed. A point
  that doesn't record a name gets NaN (floats) and `/recorded/<name>` False.
- An exception in any hook fails the task (status `failed`, traceback in the
  file and the queue). Don't catch-and-print.
- A class only appears in the GUI if it is defined in that module and
  overrides `run_point`, so intermediate base classes stay hidden.

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
shot. `prepare()` puts followers in trigger mode; `run_sequences()` starts each
follower in a thread, then the master `follower_arm_delay` later
(`MotMasterExperiment` class attribute, default 0.5 s). A follower still running
`follower_timeout` (a `MotMaster(...)` keyword, default 60 s) after the master
finishes fails the task, and its device server stays blocked in the triggered
`Go()` (Abort does not free it), so restart that device server or trigger it
before the next task. Camera APIs differ
between drivers — check the driver, not old notebooks.

## Running

From the GUI: **Experiments** tab → pick the class → edit, toggle "Scan" per
argument → Submit; MOTMaster parameters beyond the declared ones are added from
the search box. **Results** tab lists files as they appear (no refresh button)
and quick-plots a scalar against a scan axis; "Resubmit" reloads its
arguments. A running measurement plots live from the manager's published points,
not by re-reading its file; see "The manager and its workers".

From a notebook or script:

```python
from pytweezer.experiment import LinearAxis, ListAxis, Scan, load_measurement, run_local
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait
from pytweezer.experiment.storage import data_root

scan = Scan(
    axes=[LinearAxis(argument="load_time", start=0, stop=2e-3, n=21)],
    repetitions=3,
    order="shuffle",
)  # nested | snake | shuffle
rid = submit(LoadingCurve, scan, shots=2, label="after realignment")
task = wait(rid)  # Task: status, error, h5_path
m = load_measurement(data_root() / task.h5_path)  # results=[...] to load a subset
m.results["atom_number"], m.points["load_time"], m.arguments, m.constants

m = run_local(LoadingCurve, scan)  # in-process, no queue, in memory
```

`submit` needs a class importable by the manager — **not one defined in a
notebook**; use `run_local` for those (pass `path=` to keep a file).
`ExperimentManagerClient` has `pause/resume/terminate/abort/hold/release/
delete/set_priority/snapshot/history/catalogue`.

## Files

`{data_root}/YYYY/MM/DD/{rid:06d}_{Class}.h5`. Root attrs: rid, experiment,
class_name, label, submitter, host, status, error, times, n_points, n_done, git
commit/dirty. Groups: `/arguments` (effective values; `__schema__` attr),
`/scan` (spec JSON), `/points/{index,repetition,<axis>,t_start,t_end}`,
`/results`, `/recorded`, `/constants`, `/source` (the module's code). Rows are
planned points in execution order; only the first `n_done` ran. Files are
never overwritten (mode `"x"`) and readable mid-run (`locking=False`).

`data_root` = `$PYTWEEZER_DATA_DIR`; else, on a PC other than the manager's,
`CONFIG["Servers"]["Experiment Manager"]["client_data_root"]` (the manager's
data share as mounted there) if set; else that entry's `data_root`; else
`<repo>/data`. The queue state lives in `{data_root}/queue_state.json`.

The manager also writes each run into the database (`pytweezer.database`): a
`runs` row on start and finish, and a `points` row per point, holding its times,
scanned values and 0-d numeric results. Those rows sit next to the monitor
`readings`, so `points_with_readings(rid, "ni_adc")`
(`pytweezer.database.analysis`) gives one row per point with the readings during
it. `pytweezer-db-backfill` loads files the database missed. See
`docs/notes/database.md`.

## The manager and its workers

Code: `pytweezer/servers/experiment_manager.py` (single-threaded REP loop),
`pytweezer/experiment/{queue,worker,catalogue,introspect,client}.py`.

- **Published state:** the manager serves sipyco notifier `"experiment"` on
  `sync_port` (layout in the module): the queue snapshot (`running`, `queue`,
  `history`, `catalogue_version`, `simulated`, `alive`) and `points`, every
  point the current (or last) task has measured, with its scanned values and
  0-d numeric results. Only changes are sent; a GUI connecting late gets it all.
  `GUI/experiments/feed.py`'s `ExperimentFeed` is the Qt side (one per GUI,
  shared by the Experiments and Results tabs): `queue_changed`,
  `point_received`, `connection_changed` and `points(rid)`. The Results tab
  reads a running file once for metadata and the planned `/points` table
  (`read_planned_points`), plots from the feed, and reloads the file when the
  task finishes; HDF5 cannot safely be re-read while it is written. Without a
  connected feed it falls back to polling the files.

- One task runs at a time; order is priority (high first), then rid; a
  `due_time` holds a task back.
- **Terminate** stops at the next point boundary (finish() runs). **Abort**
  kills the worker immediately: a device call in flight (`Go()`, a frame grab)
  still completes on the device, so the next task may wait behind it.
- Pause keeps the task's devices open; hold only applies to waiting tasks.
- State is saved on every change, because on Windows the GUI stops the manager
  with `TerminateProcess`. Workers outlive the manager: on restart a running
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
`tests/test_results_browser.py`; `test_experiment_e2e.py` runs a real manager
and worker subprocesses on localhost.
