---
name: run-experiments
description: Run, queue and read back pytweezer experiments that already exist — the Experiments and Results tabs, submit/wait/run_local from a notebook, scans, the queue controls, recipes, reading measurement files (Dataset, fetch_measurement) and getting or setting saved parameters. Use whenever the user wants to run, queue, submit, scan, resubmit, pause or terminate an experiment, save or use a recipe, load or plot a measurement or a rid, or read, set or look up the history of a saved parameter — even if they only say "scan the load time" or "what did rid 1015 give". To write or change an experiment class use add-experiment; for optimisers use add-optimiser.
---

# Running experiments

The Experiment Manager on the server PC queues tasks and runs each in a fresh
worker subprocess, which writes one h5 file per measurement. Writing an
experiment class is add-experiment.

## Running

GUI: **Experiments** tab → pick the class, edit, tick "Scan" per argument,
Submit. **Results** lists files as they appear, quick-plots a scalar against a
scan axis (live from the manager's published points while running), and
"Resubmit" reloads a run's arguments.

```python
from pytweezer.experiment import LinearAxis, ListAxis, Scan, run_local
from pytweezer.experiment.archive import fetch_measurement
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait
from pytweezer.experiments.loading import LoadingCurve  # any experiment class

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

Each measurement is `{data_root}/YYYY/MM/DD/{rid:06d}_{Class}.h5`. Read it
with `Dataset` or `fetch_measurement`, never by path from another PC.

## Recipes

A **recipe** is a named, saved form (arguments, scan, priority, label,
pipeline) in `{data_root}/recipes.json`, shared by every PC: the Experiments
tab's **Recipes** / **Save as recipe…**, or `save_recipe(Class, name, scan,
**args, overwrite=False)`, `submit_recipe(Class, name, **overrides)`,
`recipes(Class)`, `delete_recipe(Class, name)` from `experiment.client`. It
stores every argument except those its scan sweeps (an override may not fix a
scanned one); one naming an argument the experiment no longer declares is
refused until re-saved from the form.

## Saved parameters

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
- The **Saved parameters** tab edits values with a reason and shows history.
  Keys are created only by `create=True`, an accepted `Fit`, or
  `import_parameters_json(path)`. A simulating manager keeps its own set.
