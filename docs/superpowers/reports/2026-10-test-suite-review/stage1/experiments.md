# Stage 1: Experiments

## pytweezer/experiment/arguments.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/arguments#1 | `coerce_arguments` raises, naming the bad names, when given an argument the class does not declare and `extra_argument` does not resolve. | 1 | unit | quiet: a misspelt argument (`load_tme=...`) is dropped and the default runs, and the data looks fine | — |
| experiments/arguments#2 | `coerce_arguments` returns every declared argument, with the given values applied over the defaults. | 1 | unit | quiet: the task runs with defaults instead of the submitted values | — |
| experiments/arguments#3 | `Number.coerce` refuses values below `min`, above `max`, non-finite values, bools and strings, and returns a Python float for ints and numpy floats (parametrised). | 1 | unit | quiet: an out-of-range detuning or power reaches the hardware | — |
| experiments/arguments#4 | `Integer.coerce` turns whole floats such as `3.0` into `3` and refuses `2.5` instead of truncating it. | 1 | unit | quiet: 2.5 shots or a 2.5 tick delay silently becomes 2 | — |
| experiments/arguments#5 | `Bool.coerce` accepts bools, numpy bools and 0/1 integers, and refuses anything else, e.g. `2` or `"yes"` (parametrised). | 1 | unit | quiet if `2` or `"no"` were taken as True; loud for numpy bools, which are refused today (see bugs) | — |
| experiments/arguments#6 | `Choice.coerce` refuses a value that is not one of the options, and a `Choice` with no default takes its first option. | 1 | unit | quiet: a mode the experiment does not handle reaches `run_point` | — |
| experiments/arguments#7 | `collect` gives base-class declarations first; a subclass redeclaring a name replaces it, and a subclass setting a plain value under that name removes the argument. | 1 | unit | quiet: a subclass runs with the base class's default or limits, or the form shows an argument that does nothing | — |
| experiments/arguments#8 | `to_schema` carries `unit`, `scale`, `min` and `max` through JSON unchanged, so the form shows and limits values in display units. | 4 | unit | quiet: a value typed in MHz is taken as Hz, or the form allows values the experiment will refuse | — |
| experiments/arguments#9 | A default outside the declared limits is rejected when the class is defined. *Added from stage 2; covered by `tests/test_experiment_arguments.py::test_bad_default_is_rejected_at_declaration`.* | 4 | unit | quiet: an out-of-range default nobody chose is submitted | — |

## pytweezer/experiment/catalogue.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/catalogue#1 | After `refresh()` and `poll()`, a new experiment module in the package directory appears in `entries()` with its classes' schemas (this goes through the real introspect subprocess). | 4 | integration | quiet: a new experiment never shows up in the Experiments tab | — |
| experiments/catalogue#2 | A file whose mtime or size changes is described again, and an unchanged file does not start a new subprocess. | 4 | integration | quiet: the form keeps offering arguments that the edited experiment no longer has, or lacks the new ones | — |
| experiments/catalogue#3 | A module whose file is deleted is dropped from `entries()`, and `refresh()` returns True. | 4 | unit | quiet: the GUI lists an experiment that fails to import when submitted | — |
| experiments/catalogue#4 | A module that fails to import gets an entry with its traceback in `error` and no classes, and the other modules are still listed. | 4 | integration | loud: one broken file empties or stalls the whole catalogue | — |

## pytweezer/experiment/client.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/client#1 | `submit(cls, scan, **args)` sends a request with the class's module and qualname, the args, and the scan including its shuffle seed. | 1 | unit | quiet: the manager runs another class, other values or an unreproducible order | — |
| experiments/client#2 | `submit` with an invalid argument value or an invalid scan axis raises before anything is sent. | 1 | unit | loud but late: the mistake surfaces only when the task reaches the front of the queue, perhaps overnight | — |
| experiments/client#3 | `save_recipe(cls, ...)` stores every argument value, defaults included, except those the scan sweeps. | 1 | unit | quiet: a later change of a default changes what an old recipe runs | — |
| experiments/client#4 | `ReqClient.request` raises `ManagerUnavailable` when no reply arrives within the retries, and the next request on the same client succeeds once the server answers. | 4 | integration | loud: a notebook or GUI hangs or is wedged after one missed reply | — |
| experiments/client#5 | `wait` returns the task once its status is final, and raises `TimeoutError` after `timeout_s` otherwise. | 4 | unit | loud: a notebook blocks for ever or returns before the data exists | — |
| experiments/client#6 | A command that changes state is sent once, even when the manager replies after the timeout; only read-only commands are resent. *Added with the fix in PR #13; covered by `tests/test_experiment_manager.py::test_a_slow_manager_receives_each_submit_once`.* | 1 | integration | quiet: a slow manager queues the same task twice | — |

## pytweezer/experiment/dataset.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/dataset#1 | `get_dataframe` gives one row per point that ran, in run order, with columns `index`, `repetition`, the scanned arguments, `t_start`, `t_end` and then the scalar results; array results are left out. This holds for a shuffled scan that was terminated early. | 2 | integration | quiet: results are paired with the wrong scan values, or rows that never ran appear as data | each row's scanned value equals what the recording experiment saw at that point |
| experiments/dataset#2 | `get_dict` stacks an array result as `(n_done, *shape)`, row `i` being the image recorded at point `i`. | 2 | integration | quiet: images are misaligned with their points | — |
| experiments/dataset#3 | `get_array` holds only numeric scalar results, with fields named after them. | 2 | integration | loud (a cast error), or quiet if a string result were coerced | — |
| experiments/dataset#4 | `plot_scan` plots, at each x in display units (the SI value divided by `scale`), the mean over repetitions of `y` with its standard error as the error bar, one curve per value of `series`. | 3 | integration | quiet: a plot in the lab notebook shows the wrong x scale or the wrong averages | hand-computed means/SEMs of a small known data set; x of 5e6 with scale 1e6 is plotted at 5 |
| experiments/dataset#5 | `git_label` is `git <first 7 chars>`, with `-dirty` appended when the run had uncommitted changes, and `git unknown` when there is no commit. | 4 | unit | quiet: a plot names the wrong code, or hides that the code was dirty | — |

## pytweezer/experiment/experiment_parameter_manager.py

No entries: a thin wrapper over `json` whose lookup is exercised through `TweezerBaseline`'s entry on the saved parameters it sends.

## pytweezer/experiment/experiment.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/experiment#1 | `argument_values()` returns the effective value of every declared argument plus every dotted extra set at construction or by `set_argument`. | 2 | unit | quiet: `/arguments` in the file omits or misstates what ran | — |
| experiments/experiment#2 | With `simulated_devices` set, `device(name)` and `Device` attributes return the simulated backend and never create an RPC client. | 4 | unit | quiet: a simulated run talks to the real device server and drives lab hardware | — |
| experiments/experiment#3 | `device(name)` returns the same client on repeated calls, and `fresh=True` returns a separate one. | 4 | unit | quiet: two threads share one sipyco client and their replies get crossed | — |
| experiments/experiment#4 | `close_devices()` closes every client handed out, fresh ones included, even if one close raises, and clears the cached `Device` attributes. | 4 | unit | quiet: device-server connections leak from task to task | — |

## pytweezer/experiment/__init__.py

No entries: re-exports only.

## pytweezer/experiment/introspect.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/introspect#1 | `runnable_experiments` lists only the `Experiment` subclasses defined in the module that override `run_point`; imported classes and intermediate base classes are left out. | 4 | unit | quiet: a real experiment is hidden from the GUI, or a base class is offered and fails when run | — |
| experiments/introspect#2 | `describe` adds a warning for each `Device` whose name is not in CONFIG, and still lists the class. | 4 | unit | quiet: a misspelt device name is found only when the task starts | — |
| experiments/introspect#3 | `describe` puts an import error's traceback in `error` and returns normally, instead of raising. | 4 | unit | loud: introspection produces no JSON and the catalogue shows an unhelpful error | — |

## pytweezer/experiment/motmaster.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/motmaster#1 | `prepare()` sets, on every sequencer, its script, `run_until_stopped=False`, its iterations and its save toggle, and sets trigger mode on the followers only. A lone `MotMaster` without `master=True` counts as the master (parametrised: lone and multiple). | 1 | unit | quiet: a sequencer runs the previous task's script or iteration count, or the master waits for a trigger that never comes | — |
| experiments/motmaster#2 | `run_sequences()` sends each declared argument under its `parameter=` script name and each dotted extra under its parameter name, to the right sequencer, and nothing for parameters that were not set. | 1 | unit | quiet: a scanned delay never reaches the script, so every point is the same shot | — |
| experiments/motmaster#3 | Sent values take the script's type: an `Int32` parameter gets an `int` (`3.0` -> `3`), a `Double` gets a `float`, and a non-whole value for an `Int32` raises (parametrised). | 1 | unit | quiet (a truncated or mistyped value), or loud only on the .NET side | — |
| experiments/motmaster#4 | `prepare()` raises, naming close matches, when a requested parameter is not in the script, before any `Go()` is sent. | 1 | unit | quiet: a misspelt parameter is ignored and the script's default runs | — |
| experiments/motmaster#5 | A dotted name that is only scanned (absent at `prepare`) is checked against the script at the first point, before that point's sequence starts. | 1 | unit | quiet: a scan of a misspelt parameter records N identical shots | — |
| experiments/motmaster#6 | `extra_argument` refuses a dotted name that a declared argument already targets (e.g. `rb.tDelay1` when `tof` sets `tDelay1`), so the declared limits cannot be bypassed. | 1 | unit | quiet: an out-of-range value reaches the sequencer past the declared `min`/`max` | — |
| experiments/motmaster#7 | `run_sequences()` starts every follower before the master and starts the master `follower_arm_delay` after them. | 1 | unit | quiet: a follower misses the master's trigger, and that species' sequence does not run in step | — |
| experiments/motmaster#8 | When several sequencers fail, the master's error is raised and the followers' errors are logged. | 4 | unit | quiet: the root cause is hidden behind a follower timeout | — |
| experiments/motmaster#9 | A follower still running `follower_timeout` after the master finishes raises `TimeoutError`, and the next `run_sequences()` uses a new client for it. | 4 | unit | loud: the task hangs, or the next call is queued behind the blocked `Go()` | — |
| experiments/motmaster#10 | Class definition refuses several `MotMaster`s without exactly one master, two on one device, two arguments setting one parameter, and a `device=` that is not a `MotMaster` (parametrised). | 4 | unit | loud at import, but if broken the experiment deadlocks on the shared device server | — |
| experiments/motmaster#11 | The failure of any one sequencer, master or a single follower, fails the task. *Added from stage 2; covered by `tests/test_motmaster_experiment.py::test_master_or_follower_failure_fails_the_task`.* | 1 | unit | quiet: the task is marked completed although a follower never ran | — |
| experiments/motmaster#12 | No follower is armed when another follower can't be reached. *Added from stage 2; covered by `tests/test_motmaster_experiment.py::test_failure_opening_a_follower_client_starts_no_follower`.* | 4 | unit | quiet: an armed follower blocks its device server until it times out | — |

## pytweezer/experiment/queue.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/queue#1 | `next_due` picks the waiting task with the highest priority, and the lowest rid among equal priorities. | 1 | unit | quiet: tasks run in the wrong order, e.g. a calibration after the scan that depends on it | — |
| experiments/queue#2 | A held task is skipped by `next_due` until it is released. | 1 | unit | quiet: a held task runs anyway | — |
| experiments/queue#3 | A task whose `due_time` is in the future is not started until that time; a naive `due_time` is read as local time. | 1 | unit | quiet: an overnight run starts at once, or hours late | — |
| experiments/queue#4 | `next_due` returns None while a task is running or paused. | 1 | unit | quiet: two tasks drive the same hardware at once | — |
| experiments/queue#5 | `terminate` on the running task makes `action_for(rid)` return `TERMINATE`, and leaves the task active until the worker reports. | 1 | unit | quiet: Terminate does nothing and the scan carries on | — |
| experiments/queue#6 | `terminate` on a waiting task removes it from the queue. | 1 | unit | quiet: a cancelled task runs later | — |
| experiments/queue#7 | After `pause`, `action_for` is `PAUSE` and `mark_paused` makes the status paused; after `resume` it is `CONTINUE` and running. | 1 | unit | quiet: Pause is ignored, or a paused task never resumes | — |
| experiments/queue#8 | `action_for` a rid that is not the running task returns `TERMINATE`. | 1 | unit | quiet: a stale worker left over from a crash keeps running alongside the new task | — |
| experiments/queue#9 | `QueueStore.save` then `load` gives back an identical state: active tasks, history, `next_rid`, scan seeds, due times and the worker record. | 2 | unit | quiet: after a manager restart, tasks are lost or rids are reused | — |
| experiments/queue#10 | An unparseable state file is moved aside and an empty state returned, so the manager still starts. | 4 | unit | loud: the manager will not start after a power cut that left the file half written | — |
| experiments/queue#11 | A state file that can't be read (as opposed to parsed) is retried briefly; if it still can't be read the error is raised and the file left in place. *Added with the fix in PR #13; covered by `tests/test_experiment_queue.py::test_a_transient_read_error_still_loads_the_saved_queue` and `::test_a_persistent_read_error_raises_and_leaves_the_file`.* | 2 | unit | quiet: a transient read error loses the queue or every recipe | — |

## pytweezer/experiment/recipes.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/recipes#1 | `replay(args)` gives a request with the recipe's arguments overridden by `args`, the recipe's scan, and the recipe's priority and label unless they are given. | 1 | unit | quiet: "submit recipe" runs other values than the recipe or the override | — |
| experiments/recipes#2 | `replay` refuses an override of an argument the recipe scans. | 1 | unit | quiet: the override is silently overwritten by the scan, or vice versa | — |
| experiments/recipes#3 | `unknown_arguments` lists argument and scan-axis names the schema does not declare, but not `attribute.parameter` names on a declared MOTMaster. | 1 | unit | quiet: a recipe using a removed argument is submitted and that argument is ignored | — |
| experiments/recipes#4 | `save` refuses a name that already exists for that experiment unless `overwrite`, comparing names with surrounding whitespace stripped. | 4 | unit | quiet: another user's recipe is replaced | — |
| experiments/recipes#5 | `save` with `overwrite` replaces the recipe in place, leaving exactly one under that name. | 4 | unit | quiet: duplicates appear and an arbitrary one is run | — |
| experiments/recipes#6 | `save` clears `due_time`. | 4 | unit | quiet: every later submission of the recipe is held back to a time long past, or one in the future | — |
| experiments/recipes#7 | Saved recipes, scans included, read back identical through `RecipeStore`. *Added from stage 2; covered by `tests/test_experiment_recipes.py::test_store_round_trip`.* | 2 | unit | quiet: a recipe runs different settings from those saved | — |
| experiments/recipes#8 | Recipes are listed per experiment, sorted, and one name may be used by two experiments. *Added from stage 2; covered by `tests/test_experiment_recipes.py::test_find_filters_and_sorts`.* | 4 | unit | quiet: another experiment's recipe is offered, or two recipes clash | — |

## pytweezer/experiment/runner.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/runner#1 | `prepare` runs once, then `run_point` once per point in the scan's order with that point's values as the instance's attributes, then `finish` once. | 1 | unit | quiet: points run with stale values, or in a different order from `/points` | — |
| experiments/runner#2 | When `control` returns `TERMINATE` or `INTERRUPT` after point k, the run stops there with status terminated or interrupted, `n_done = k+1` and `finish` still runs (parametrised). | 1 | unit | quiet: Terminate runs the rest of the scan, or the file claims points that never ran | — |
| experiments/runner#3 | An exception in `run_point` gives status failed with the traceback as `error`, keeps the points already done, and still runs `finish`. | 1 | unit | quiet: completed points are lost, or the hardware is left mid-sequence because `finish` was skipped | — |
| experiments/runner#4 | An exception in `prepare` runs no points, still runs `finish`, and gives status failed. | 1 | unit | quiet: points run on hardware that was never set up | — |
| experiments/runner#5 | `run_local(..., path=p)` writes a file whose `load_measurement` matches the returned `Measurement`: arguments, points and results. | 2 | integration | quiet: the in-memory and saved data differ | — |
| experiments/runner#6 | `/arguments` in the file holds every argument's effective value (defaults included, plus dotted extras passed as arguments) and the schema includes dotted names that are only scanned. | 2 | integration | quiet: the file does not say what ran; Resubmit reproduces something else | — |
| experiments/runner#7 | After each point, `control` receives a `Progress` carrying that point, its 0-d numeric results and its start and end times; after `prepare` it receives `done=0`. | 2 | unit | quiet: the live plot and the database `points` rows show wrong or missing values | — |
| experiments/runner#8 | An exception in `finish` after a good run makes it failed, and one after a failed `run_point` keeps the first traceback. | 4 | unit | quiet: a broken `finish` (hardware left on) is reported as completed, or the real error is hidden | — |
| experiments/runner#9 | A `KeyboardInterrupt` in `run_point` gives status terminated, with the file finalised and `finish` run. | 4 | unit | quiet: Ctrl-C in a notebook leaves a file marked running and the hardware not reset | — |
| experiments/runner#10 | A simulated `run_local` records `simulated` truthfully in its file. *Added from stage 2; covered by `tests/test_experiment_simulation.py::test_run_local_simulated_experiment`.* | 2 | unit | quiet: simulated data passes for real data | — |
| experiments/runner#11 | The runner closes its device clients on every exit path, including after a failing point. *Added from stage 2; covered by `tests/test_experiment_runner.py::test_devices_closed_even_on_failure`.* | 4 | unit | quiet: a failed run leaves device clients open | — |

## pytweezer/experiment/scan.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/scan#1 | A `nested` scan yields the grid with the first axis outermost, `Point.index` counting 0, 1, 2, ... and each point's values taken from its axis indices. | 1 | unit | quiet: points run in an order other than intended, or values are paired with the wrong row | hand-written 2x3 grid |
| experiments/scan#2 | `repeat="point"` runs a point's repetitions back to back; `repeat="scan"` repeats the whole sweep (parametrised). | 1 | unit | quiet: drifts are attributed to the scanned variable | hand-written sequence for 2 values x 2 repetitions |
| experiments/scan#3 | A `snake` scan over three axes visits every grid point once, and consecutive points differ in exactly one axis by one step. | 1 | unit | quiet: large hardware jumps between points the experimenter thought were adjacent | the adjacency property itself, plus the count equal to the product of the axis lengths |
| experiments/scan#4 | A `shuffle` scan visits every grid point exactly once per pass. | 1 | unit | quiet: some values are never measured, others twice | set of visited index tuples equals the full grid |
| experiments/scan#5 | A shuffled scan built without a seed is given one, and expanding it again after a JSON round trip gives the same order. | 1 | unit | quiet: the order that ran cannot be reproduced from the stored scan | — |
| experiments/scan#6 | A `LinearAxis` on an `Integer` argument gives whole values (e.g. 0..10 in 6 steps gives 0, 2, 4, 6, 8, 10). | 1 | unit | quiet: fractional values reach an integer parameter, or are truncated | hand-written list |
| experiments/scan#7 | A `LinearAxis` on an `Integer` argument whose rounded values repeat raises. | 1 | unit | quiet: the same value is measured twice under different rows | — |
| experiments/scan#8 | Axis values are coerced by the argument: a value outside its `min`/`max`, or a scan of an undeclared argument, raises from `points()`. | 1 | unit | quiet: an out-of-range value reaches the hardware mid-scan | — |
| experiments/scan#9 | A scan with no axes gives `repetitions` points with empty values. | 1 | unit | quiet: a plain submit runs zero times or once regardless of repetitions | — |

## pytweezer/experiments/demo.py

No entries: a device-free demo whose output nobody relies on; it is better used as the fixture experiment in the runner and queue tests.

## pytweezer/experiment/simulation.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/simulation#1 | `SimulatedDevices.get` builds a composite entry with `simulate=True` on the entry and on every one of its sub-devices. | 4 | unit | quiet: a simulated run opens the real hardware of one sub-device | — |
| experiments/simulation#2 | Two sub-devices of one composite, and repeated `get` calls, share one simulated build. | 4 | unit | quiet: simulated devices that should share a rig disagree, so simulated runs mislead | — |
| experiments/simulation#3 | `close()` tears down every build, and one failing teardown does not stop the rest. | 4 | unit | quiet: simulated resources (threads, publishers) leak between `run_local` calls | — |

## pytweezer/experiments/__init__.py

No entries: docstring only.

## pytweezer/experiments/motmaster_arguments_test.py

No entries: a manual check of the MOTMaster form whose mechanics are covered by the `motmaster.py` entries.

## pytweezer/experiments/tweezer_baseline.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/tweezer_baseline#1 | The saved settings in `parameters_file` reach the `rb` MOTMaster under their script names; both imaging pulses get the single saved imaging value (`tImgCoolVCO1` and `tImgCoolVCO2` from `tImgCoolVCO`). | 1 | integration | quiet: shots run with the script's defaults instead of the calibrated cooling and shim values | values written into a temporary parameters.json |
| experiments/tweezer_baseline#2 | A declared argument or a dotted override such as `rb.tMolCoolVCO1` takes precedence over the saved value. | 1 | integration | quiet: a scan of a saved parameter sends the saved value at every point | — |
| experiments/tweezer_baseline#3 | `prepare` sets the camera ROI from `roi_x0`, `roi_width`, `roi_y0` and `roi_height` in the driver's argument order, and sets the sensitivity from `sensitivity`. | 1 | integration | quiet: the images are of the wrong region with the wrong gain | — |
| experiments/tweezer_baseline#4 | Each point records both images and `signal_counts` as the sum of the first image minus the sum of the second, without unsigned-integer wrap-around. | 3 | integration | quiet: a negative difference wraps to about 4e9 counts | images of known constant value, where the second is brighter than the first |
| experiments/tweezer_baseline#5 | The `motmaster_parameters` constant equals the parameters actually sent to the script. | 2 | integration | quiet: the file misreports the calibration a run used | — |

## pytweezer/experiment/summary.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/summary#1 | `aggregate` returns, for each distinct x, the mean and the standard error (sample standard deviation over the square root of n) of the y values at that x. | 3 | unit | quiet: the error bars in the Results tab and notebooks are wrong | y = [1, 2, 3] at one x gives mean 2, SEM 1/sqrt(3) |
| experiments/summary#2 | `aggregate` ignores non-finite y, so points that did not record a value do not count. | 3 | unit | quiet: one unrecorded point turns a whole mean into NaN, or inflates n | [1, NaN, 3] gives mean 2, count 2 |
| experiments/summary#3 | `aggregate` gives one curve per distinct series value, each made only of its own samples. | 3 | unit | quiet: curves for different detunings are mixed | two series with constant, distinct y |
| experiments/summary#4 | A single sample at an x gives a NaN standard error, not zero. | 3 | unit | quiet: a point measured once is plotted as if it had no uncertainty | — |
| experiments/summary#5 | `display_value` divides a number by its `scale` and appends its unit (5e6 with MHz/1e6 gives "5 MHz"), and gives other kinds as `str(value)`. | 4 | unit | quiet: the metadata pane shows "5e+06 MHz" | 5e6 / 1e6 = 5 |

## pytweezer/experiment/task.py

No entries: pydantic models and an enum whose one rule (`finished`) is exercised by the queue entries and by `wait()`.

## pytweezer/experiment/worker.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| experiments/worker#1 | While the manager's reply says pause, `ManagerLink.control` keeps sending heartbeats, and it returns the first non-pause action. | 1 | unit | quiet: Pause stops nothing, or a paused task never resumes | — |
| experiments/worker#2 | When the manager is unreachable for longer than `orphan_timeout`, `control` returns `INTERRUPT`. | 1 | unit | quiet: an orphaned worker keeps firing sequences with nobody watching | — |
| experiments/worker#3 | An outage shorter than `orphan_timeout` is retried, and the run continues. | 4 | unit | quiet: a network blip kills an overnight scan | — |
| experiments/worker#4 | A reply with no `action` (e.g. the manager rejected the token) gives `TERMINATE`. | 1 | unit | quiet: a worker the manager has disowned runs on | — |
| experiments/worker#5 | `run()` reports `finished` with status failed and the traceback when the class cannot be imported or its arguments are invalid. | 4 | unit | loud: the task stays running in the queue for ever | — |
| experiments/worker#6 | `worker.run` passes the manager's `simulate` flag to `open_writer`, so a simulating manager's tasks get simulated devices. *Added from stage 2; covered by `tests/test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices`.* | 1 | integration | quiet: a simulating manager's worker drives real hardware | — |

## Suspected bugs

- pytweezer/experiment/arguments.py: `Bool.coerce` refuses numpy bools, although its comment says they are accepted (`np.bool_` is not a `numbers.Integral`). A notebook passing `flag=arr.any()` or `ListAxis(values=list(mask))` gets an error. Smallest input: `Bool(False).coerce(np.True_)` raises `ValueError`.
- pytweezer/experiment/client.py: `ReqClient` resends a request that got no reply within `timeout_ms`, and `ExperimentManagerClient` defaults to `retries=1`. A request that is not idempotent is therefore carried out twice when the manager is just slow. A slow `submit` queues the task twice, and a slow `save_recipe`/`delete_recipe`/`delete` reports a spurious "already exists"/"no recipe"/"cannot delete" error for an action that did succeed. Smallest input: a REP server that sleeps 2.5 s before replying; `ExperimentManagerClient(endpoint).submit(request)` makes it receive two `submit` commands.
- pytweezer/experiment/catalogue.py: introspection output is the child's whole stdout parsed as one JSON document. So an experiment module, or anything it imports, that prints at import time vanishes from the catalogue, and the misleading error is "introspection exited with code 0". Smallest input: an experiment module whose first line is `print("hi")`; its entry has `classes == []`.
- pytweezer/experiment/queue.py: `ModelStore.load` treats any `OSError` on reading as corruption. A transient read failure therefore makes the live `queue_state.json` or `recipes.json` get moved aside and an empty state returned, losing the queue or every recipe. On Windows that means a `PermissionError` while a backup tool or editor holds the file, which `save` already anticipates. Smallest input: `QueueStore(p).load()` while reading `p` raises `PermissionError`; the file is renamed to `*.unreadable-*.json`, or the rename itself raises, stopping the manager.
- pytweezer/experiment/motmaster.py (low confidence): dotted overrides are always declared as `_MotMasterOverride`, a `Number`, so a script parameter whose default is a bool cannot be set by name, although `_typed` has a branch for bool defaults. Smallest input: a script with `{"flag": False}`; `submit(Exp, **{"rb.flag": True})` is refused ("is not a number"), and `1.0` is passed to .NET unchanged as a float.
