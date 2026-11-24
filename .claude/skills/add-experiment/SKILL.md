---
name: add-experiment
description: Write, run, queue and read back pytweezer experiments — an Experiment subclass in pytweezer/experiments/ with declared arguments and devices, run per scan point by the Experiment Manager (or in-process with run_local), saving one h5 file per measurement — and the saved parameters they read. Use whenever the user wants to write, add or change an experiment or measurement sequence (a scan, a MOTMaster sequence plus camera readout, a calibration), submit or queue one from a notebook (submit/wait), scan an argument, record data from a run, load a measurement file, read or set saved parameters, give an experiment a Results-tab analysis, or work on the Experiment Manager, its workers, pipelines, device reservations or queue, or the Experiments/Results/Saved parameters tabs — even if they only describe the physics ("scan the load time and count atoms") without saying "experiment". For the Rb tweezer-array experiments themselves also use tweezer-array.
---

# Experiments

An experiment is a class in a module under `pytweezer/experiments/`. The
Experiment Manager (a `CONFIG["Servers"]` process on the server PC) queues
tasks and runs each in a **fresh worker subprocess** that imports the module,
makes one instance and writes **one h5 file per measurement**, flushed after
every point. Optimisers are add-optimiser; the Rb tweezer-array experiments
(`SlmTweezerArray`, `LoadTweezers`, trap sites) are tweezer-array.

## Writing one

```python
# pytweezer/experiments/loading.py
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
        mark = self.camera.start_shot({"rid": self.rid})
        frames = self.camera.wait_frames(mark, self.shots)
        self.record("image", frames[0])
        self.record("atom_number", float(frames.sum()), unit="counts")

    def finish(self): ...  # always runs, also after a failure or terminate
```

- **Methods go in the order `prepare`, `run_point`, `finish`, then everything
  else** (helpers, `measure`, classmethods, overrides), so the class reads top
  down in the order it runs. This applies to every class under `experiments/`
  and `optimisers/`, including a subclass that overrides only some of the three.
- **Arguments are class attributes** (`Number`, `Integer`, `Bool`, `Choice`,
  `Text`; `experiment/arguments.py`). Values are SI; `unit`/`scale` only set
  the display (`Number(5e6, unit="MHz", scale=1e6)` shows "5 MHz"); `group=`
  and `tooltip=` lay out the form. `scannable=False` marks an argument that
  must be one value for the whole task (something `prepare()` depends on);
  scanning it is refused. Defaults are validated at import; submitted values
  by the form and `submit(Class, ...)`, but a `"module:Class"` string only
  when the task starts.
- **No hardware at module import or class definition.** The manager imports
  every module in a subprocess to list its arguments.
- **Declare every device you use.** `Device("<CONFIG name>")` is a lazily
  created RPC client, closed when the task ends; under `run_parallel` take one
  per thread with `self.device(name, fresh=True)`. A name outside the task's
  reserved set raises `UndeclaredDeviceError`, and a queued worker also
  refuses a bare `get_device()` of it. The declarations are what the manager
  reserves.
- **A variant that uses other devices is another experiment** (usually a
  subclass), not an option or a device used only when it happens to be
  available. **Never design around a device's `run_at_startup` flag**: it is
  lab state anyone switches. Declare what you need and fail plainly if it
  isn't running.
- `record(name, value, unit="")`: in `run_point` into `/results/<name>` row
  `point.index`; in `prepare`/`finish` into `/constants/<name>`. The first
  record fixes shape and dtype (later values may only widen, e.g. int→float; a
  mismatch fails the task). Strings and dicts (as JSON) are allowed. A point
  that doesn't record a name gets NaN and `/recorded/<name>` False.
- An exception in any hook fails the task (status `failed`, traceback in the
  file and the queue). Don't catch-and-print.
- A class appears in the GUI only if it is defined in that module and
  overrides `run_point`; a base that inherits a concrete `run_point` hides
  with `run_point = Experiment.run_point`.
- `self.rid` is the task's run id (-1 under `run_local`). An experiment that
  chooses its own points overrides `planned_points(scan)`, raising
  `ValueError` for a scan it can't honour.
- Camera settings experiments (`experiments/camera_settings.py`, base
  `CameraSettings`) build their form from the driver's `CONTROLS`, send only
  the ticked settings and save them as `<camera>_<control>` saved parameters.

## MOTMaster sequences

```python
from pytweezer.experiment import Device
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment, MotMasterInteger


class Tof(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    tof = MotMasterInteger(100, parameter="tDelay1")  # always shown
    camera = Device("Rb ThorCam")

    def prepare(self):
        super().prepare()  # sets the sequencer up; must come first
        self.camera.stop_live()  # DCAM refuses trigger settings while capturing
        self.camera.set_trigger_source("ext")
        self.camera.start_live()

    def run_point(self):
        mark = self.camera.start_shot({"rid": self.rid})
        self.run_motmaster_sequence()  # blocks until Go() returns
        self.record("image", self.camera.wait_frames(mark, 1)[0])

    def finish(self):
        try:
            self.camera.end_shots()
        finally:
            super().finish()
```

- **One `MotMaster` per experiment** (a second is a `TypeError`; drive two
  machines with linked mode). `prepare()` sets linked mode, script,
  iterations, save toggle and untriggered mode on every task. Repeats come
  from `Scan(repetitions=...)`, one point per shot;
  `run_motmaster_sequence(**overrides)` overrides parameters for one run.
- `MotMasterInteger`/`MotMasterNumber` must match the script parameter's .NET
  type (`Int32`/`Double`).
- **Any undeclared identifier is a script parameter**: `submit(Tof,
  tPulse=3e-5)`, `ListAxis(argument="tPulse", ...)`, or the form's search box.
  A misspelt one is accepted at submit and fails in `prepare()` (or before the
  first `Go` if only scanned) with a "did you mean" hint. They live in
  `self.extra_values`; a declared argument wins over a script parameter of the
  same name. Unselected ones keep the script's defaults and aren't recorded.
- `script_from="script"` instead of `script=` takes the script from a
  `scannable=False` `Choice`; `self.script_name()` is the one the task runs.
  `RbMotMasterRunner` offers every script in the Rb folder and records each
  ticked camera's frames.
- `saved=(...)` binds script parameters to saved parameters: a bare name for
  the key of that name, `{parameter: key}` for another. They are read once in
  `prepare()` (unreachable manager: `RuntimeError`; missing key: `KeyError`),
  sent with every run and recorded as the constant `saved_sequence`.
  Precedence: `run_motmaster_sequence` override > form or scan > saved >
  script default. Such an experiment gets a `use_motmaster_parameter_overrides`
  argument; `False` sends none (`saved_bindings()`).

**Linked mode** (one CaF script driving PH-BONESAW and RbTweezer together,
fired by TTL): declare only the primary, `MotMaster("CaF MotMaster",
script=..., linked=True)`; every parameter of both machines, including
`SecondaryTriggerDelay`, is a parameter of that script. The link lasts one
task: `prepare()` sets it and `close_devices()` releases it, even after a
failure. While linked the Rb MOTMaster refuses its own `SetScriptPath`/`Go`;
an Rb-only task takes back a hand-ticked link, failing only if a linked run is
in progress. Minimal example: `experiments/linked_scope_test.py`; MOTMaster
side: EDMSuite's `motmaster-linked-mode` skill.

## Cameras and cloud fits

There is no camera base class: declare the camera as a `Device`, set it up in
`prepare()` (after `super().prepare()`) and read it in your shot.
`LoadTweezers.set_up_hamcam` and `LoadTweezersMOTImage`
(`experiments/load_tweezers.py`) are the examples to copy. Camera APIs differ
between drivers — check the driver, not old notebooks.

- **Cameras stay live**, during experiments too. `prepare()` stops live, sets
  binning, ROI (keep the origin `set_roi` returns: the camera may round it),
  trigger, gain, then starts live. `finish()` calls `end_shots()` in a
  `try`/`finally` around `super().finish()`.
- A shot is `mark = camera.start_shot({"rid": self.rid})`,
  `run_motmaster_sequence(...)`, `camera.wait_frames(mark, n)`. The live
  thread broadcasts the frames tagged with `rid` and `index`; don't broadcast
  them yourself.
- Camera settings are arguments named for the camera (`hamcam_gain`,
  `hamcam_roi_x0`...). A scannable one is re-sent before a shot when its value
  changed; one that changes the frame shape or that `prepare()` relies on is
  `scannable=False`.
- A subclass adding a camera wraps the parent's shot (start its camera, call
  `super().run_point()`, read it) rather than adding hooks to the parent.

To fit a cloud, copy `LoadTweezersMOTImage.record_cloud_fit` under your
camera's prefix: it records `<camera>_<field>`, `_error` and
`<camera>_mean_image` (metres if the camera's CONFIG has `object_pixel_m`;
NaN, never an error, on a failed fit) plus the constant `cloud_mass_amu`.

## Results-tab analyses

`analyses = (...)` (a class attribute, inherited) lists what the Results tab's
Analysis dropdown offers after "Plot against". The file stores the names (root
attr `analyses`), so the tab never imports the experiment. Each analysis has a
Qt-free computation in `pytweezer/analyses/default/` and a view plus its
`ANALYSES` entry (`{name: AnalysisEntry}`) in the same-named module of
`pytweezer/GUI/default_analyses/`, merged in that package's `__init__`. Views
are usually `ComputedAnalysisView` (`GUI/experiments/analyses.py`), which
computes off the GUI thread through `needed_results`/`settings`/`compute`/
`show_result`/`unavailable_reason`; `point_result` reads one point's large
result.

A view reads only what the file recorded, so record what it needs:
`cloud_fit` reads the cloud fit above; `cloud_temperature` fits those sizes against a scanned time argument (native:
seconds; MOTMaster: 10 µs ticks) and needs `cloud_mass_amu`; the tweezer
histograms and `trap_frequency` are in tweezer-array. Analyses are disabled until the measurement
finishes; an unknown name is listed disabled.

## Running

GUI: **Experiments** tab → pick the class, edit, tick "Scan" per argument,
Submit. **Results** lists files as they appear, quick-plots a scalar against a
scan axis (live from the manager's published points while running), and
"Resubmit" reloads a run's arguments.

```python
from pytweezer.experiment import LinearAxis, ListAxis, Scan, run_local
from pytweezer.experiment.archive import fetch_measurement
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait

scan = Scan(
    axes=[LinearAxis(argument="load_time", start=0, stop=2e-3, n=21)],
    repetitions=3,
    order="shuffle",  # nested | snake | shuffle; repeat="scan" repeats the grid
)
rid = submit(LoadingCurve, scan, shots=2, label="after realignment", pipeline="main")
task = wait(rid)  # Task: status, error, h5_path
m = fetch_measurement(rid)  # any PC; results=[...] to load a subset
m.results["atom_number"], m.points["load_time"], m.arguments, m.constants

m = run_local(LoadingCurve, scan)  # in-process, no queue, in memory
m.status, m.attrs["error"]  # a failure is recorded, not raised
```

- `submit` needs a class the manager can import — **not one defined in a
  notebook**; use `run_local` for those (`path=` keeps a file). `submit` is
  refused at once if the class isn't in the manager's catalogue (one for
  `pytweezer.experiments`, one for `pytweezer.optimisers`), its module fails
  to import, or its devices can't be reserved (`ReservationError`; `run_local`
  raises the same). While the catalogue is being read the task queues and
  resolves later.
- `ExperimentManagerClient` (`experiment/client.py`) has the queue controls
  (pause, resume, terminate, abort, hold, release, delete, set_priority), the
  catalogues, recipes and saved parameters.

For analysis, `Dataset` reads a file by rid (or path) on any PC:

```python
from pytweezer.experiment import Dataset

ds = Dataset(rid)  # root=... for another data dir, results=[...] for a subset
ds.summarise()  # same text as the Results tab's metadata pane
df = ds.get_data()  # "dataframe" (default) | "dict" | "array"
ax = ds.plot_scan("atom_number")  # mean ± SEM over repetitions, stamped with the git commit
```

Rows are the points run, in run order: `index`, `repetition`, scanned
arguments, `t_start`, `t_end`, then results (SI). The DataFrame drops array
results; `get_dict` stacks them.

## Recipes and saved parameters

A **recipe** is a named, saved form (arguments, scan, priority, label,
pipeline) in `{data_root}/recipes.json`, shared by every PC: the Experiments
tab's **Recipes** / **Save as recipe…**, or `save_recipe(Class, name, scan,
**args, overwrite=False)`, `submit_recipe(Class, name, **overrides)`,
`recipes(Class)`, `delete_recipe(Class, name)` from `experiment.client`. It
stores every argument except those its scan sweeps (an override may not fix a
scanned one); one naming an argument the experiment no longer declares is
refused until re-saved from the form.

**Saved parameters** are named calibrated values (float, int, bool, string or
numeric array) in `{data_root}/saved_parameters.json`, every change logged to
`saved_parameters_log.jsonl` (`experiment/saved_parameters.py`):

```python
from pytweezer.experiment.client import get_saved_parameter, get_saved_parameters, saved_parameter_history, set_saved_parameters

get_saved_parameter("tImgCoolVCO")  # KeyError names close matches; get_saved_parameters() for all
set_saved_parameters({"level": 0.3}, "realigned the probe")  # all or none; reason and user logged
set_saved_parameters({"new_key": 1.0}, "first calibration", create=True, group="MOT", unit="V")
saved_parameter_history("level", limit=10)  # newest first; rid= for one run's
```

- Unknown keys are refused unless `create=True`; a key keeps its type (int
  accepted for float), an array its dimensions and every axis past the first.
- **Never read one into an argument default**
  (`Number(get_saved_parameter(...))`): that is read at import, and recipes and
  prefilled forms then pin it, silently overriding later recalibrations. Read
  it in `prepare()`/`run_point()`, or declare a **saved-backed argument**:
  `blaze = Number(saved="slm_blaze_x", unit="um", scale=1e-6)` is `None` until
  the task starts, then takes the saved value (a missing key fails the task).
  A value from the form, request, recipe or scan wins; the file's
  `saved_arguments` lists the ones that took saved values.
- The **Saved parameters** tab edits values with a reason and shows history.
  Keys are created only by `create=True`, an accepted `Fit`, or
  `import_parameters_json(path)`. A simulating manager keeps its own set.

## Files and the manager

Each measurement is `{data_root}/YYYY/MM/DD/{rid:06d}_{Class}.h5`, with groups
`/arguments`, `/scan`, `/points`, `/results`, `/recorded`, `/constants` and
`/source`. Read it with `Dataset` or `fetch_measurement`, never by path from
another PC. **Read [manager.md](manager.md) before working on** the file
layout, `data_root`, the copier, the results archive, the database rows, or
the manager itself: its published state, pipelines, device reservations,
terminate/abort, worker adoption and simulation.

Tests: `tests/test_experiment_*.py`, `test_motmaster_experiment.py`,
`test_saved_parameters.py`; more in [manager.md](manager.md).
