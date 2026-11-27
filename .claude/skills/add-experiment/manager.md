# The Experiment Manager: files, storage and internals

Reference for add-experiment: read when working on the manager, its workers,
queue, pipelines, reservations, file layout, copier, archive or database rows.

## Files

`{data_root}/YYYY/MM/DD/{rid:06d}_{Class}.h5`. Root attrs include rid,
experiment, class_name, label, submitter, status, error, times, `n_points`,
`n_done`, git commit/dirty and `analyses`. Groups: `/arguments`, `/scan`,
`/points/{index,repetition,<axis>,t_start,t_end}`, `/results`, `/recorded`,
`/constants`, `/source` (the module's code) and `/motmaster` (every MOTMaster
run: `runs/{point,sequence,t}`, plus each distinct sequence record and script
once, by sha256; read lazily by `Dataset.motmaster_sequence`). Rows are planned points in run
order; only the first `n_done` ran. Never overwritten (mode `"x"`), readable
mid-run (`locking=False`).

- `data_root` = `$PYTWEEZER_DATA_DIR`, else
  `CONFIG["Servers"]["Experiment Manager"]["data_root"]`, else `<repo>/data`.
  The queue state is `{data_root}/queue_state.json`.
- A finished file is copied to the same path under `copy_root` (the router's
  USB drive) by a background thread (`experiment/copier.py`); runs never write
  to the network, failed copies retry every minute and survive restarts
  (`pending_copies.json`). Only a manager that binds its sockets copies.
- Other PCs never read the manager's disk: it serves files as sipyco target
  `"results"` on `results_port` (`experiment/archive.py`). `ResultsArchive`
  reads directly, `RemoteArchive` is the same interface over RPC, and
  `default_archive()` picks one; the Results tab, `Dataset` and
  `fetch_measurement` all go through it.
- The manager also writes a `runs` row and a `points` row per point (times,
  scanned values, 0-d numeric results) to the database, beside the monitor
  `readings`: `points_with_readings(rid, "ni_adc")`
  (`pytweezer.database.analysis`). `pytweezer-db-backfill` loads files it
  missed. See `docs/notes/database.md`.

## The manager and its workers

Code: `pytweezer/servers/experiment_manager.py` (single-threaded REP loop) and
`pytweezer/experiment/{queue,worker,catalogue,introspect,client,reservations}.py`.

- **Published state**: sipyco notifier `"experiment"` on `sync_port` holds the
  queue snapshot (`running`, `queue`, `history`, catalogue/recipe/saved
  parameter version counters; waiting tasks carry `waiting_for`) and `points`
  per rid (scanned values and 0-d results of every point measured). The GUI
  side is `GUI/experiments/feed.py`'s `ExperimentFeed`, one per GUI. HDF5
  can't be safely re-read while being written, so the Results tab reads a
  running file once and plots from the feed.
- **Pipelines**: a named lane (`pipeline=`, default `"main"`, created by
  submitting a new name). Within one, one task at a time, by priority then
  rid; a `due_time` or hold passes a task over. Pipelines run in parallel,
  but a task starts only if no running or paused task holds a device it
  reserves. A blocked head waits while other pipelines' heads may start; the
  remedy for a stuck task is `set_priority`.
- **Reservations** are computed at submit (`Task.devices`): a device reserves
  itself; a composite itself and its sub-devices; `MotMaster(linked=True)`
  also its CONFIG `linked_partner`. `Device.reserves()` is the hook for other
  hidden dependencies. Only queued tasks are subject to them: `run_local` and
  GUI panels can drive a held device.
- **Terminate** stops at the next point boundary (`finish()` runs). **Abort**
  kills the worker at once; a device call in flight still completes on the
  device, and devices are released the moment a task ends (a `TODO` in
  `ExperimentQueue.finish`). Pause keeps devices reserved; hold applies only
  to waiting tasks.
- State is saved on every change (on Windows the GUI kills the manager with
  `TerminateProcess`). Workers outlive the manager: on restart a running
  worker is adopted again (pid + create time), else its task is settled from
  its h5 status. A worker that can't reach the manager for `orphan_timeout` s
  marks itself `interrupted`.
- **Simulation** (`"simulate": SIMULATING`, i.e. any `pytweezer-server` off
  the server PC): each `Device(...)` is its simulated backend built in the
  worker, so no device servers are needed; data goes to
  `<data_root>/simulated/` and files carry `simulated=True`.
  `run_local(..., simulate=True)` does the same in a notebook.
- Worker output goes to `logs/experiments/<rid>.log`. Workers bind no ports.

Tests: `tests/test_experiment_*.py`, `test_motmaster_experiment.py`,
`test_results_{browser,archive,analyses}.py`, `test_file_copier.py`,
`test_saved_parameters.py`; `test_experiment_e2e.py` runs a real manager and
workers on localhost.
