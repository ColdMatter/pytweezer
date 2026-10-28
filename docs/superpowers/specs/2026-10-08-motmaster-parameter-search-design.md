# MOTMaster parameters chosen by search

## Goal

An experiment form shows no MOTMaster script parameters until the user adds
them by name from a search box. Parameters that are not added keep the script's
own defaults. Experiments may drive several MOTMasters at once (one master, the
rest triggered).

## Decisions

- Several MOTMasters per experiment: one is the master (untriggered), the others
  are followers in trigger mode, each `Go()` in its own thread.
- Each MOTMaster's script is fixed by the experiment class.
- The parameter list comes from the device over RPC, so `Int32`/`Double` types
  are exact.
- Searched parameters are scannable and can be submitted from code.
- `MotMasterNumber`/`MotMasterInteger` remain for parameters that should always
  be shown, with units.
- The script defaults are not recorded in the measurement file.
- The old single-sequencer form (`sequencer = Device(...)`, `motmaster_script`)
  is replaced, not kept; only tests use it.

## 1. Declaring MOTMasters

```python
class TofImaging(MotMasterExperiment):
    rb  = MotMaster("Rb MotMaster",  script="RbTweezerBasic", master=True)
    caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad")
    camera = Device("Rb ThorCam")

    tof = MotMasterInteger(100, device="rb", parameter="tDelay1", unit="ms/100")

    def run_point(self):
        self.camera.start_acquisition()
        self.run_sequences()
        self.record("image", self.camera.acquire_n_frames(1)[0])
```

- `MotMaster` is a `Device` with `script`, `master=False`, `iterations=1`,
  `save=False`, and `follower_timeout=60` (seconds). `to_schema()` adds a
  `motmaster` block (device name, script, master flag).
- Checked at import, with no hardware: with two or more MOTMasters exactly one
  has `master=True`; a lone MOTMaster is the master implicitly.
- `MotMasterInteger`/`MotMasterNumber` take `device=` (optional when there is
  one MOTMaster).
- `prepare()`, per MOTMaster: set script, iterations and save toggle; read the
  script's parameters and check every override against them (unknown name fails
  the task, with the nearest names suggested); set the master untriggered and
  the followers triggered. The defaults are used for checking only and are not
  stored.
- `run_sequences(**overrides)` starts each follower's `Go()` in its own thread
  with a fresh client, then the master's, and returns once all have finished.
  Exceptions from any thread are collected and the first is re-raised.

## 2. Discovery and the search box

**Driver.** `MotMasterInterface.script_parameters(script) -> {name: default}`
sets the script path, calls `GetParameters()` and restores the previous path.
Values are plain Python (`int` = `Int32`, `float` = `Double`), matching
`python_to_cs_dict`. It must not run mid-sequence; the RPC server serialises
calls.

**Schema.** `Experiment.schema()` gains the `motmaster` block above.

**GUI.** The Experiments form gets one "MOTMaster: rb (script)" box per
sequencer, initially empty.
- On selecting the experiment a background thread calls `script_parameters`;
  results are cached per (device, script) for the session. The box shows
  "Loading...", then the search field; if the device is unreachable it shows the
  error and a Retry button (no free-typed names, which would bypass the type
  check).
- The search is a completer over the script's names, each shown with type and
  default (`tDelay1 - int - 5`). Choosing one adds an ordinary row named
  `rb.tDelay1` (integer field for `Int32`, float field for `Double`) starting at
  the script default, with the Scan toggle and a remove button. Removing a row
  returns the parameter to the script's own value.
- The request carries only added rows, as dotted names, plus the declared
  arguments as now.
- Resubmit from the Results tab re-adds its rows; a name no longer in the script
  keeps its row, flagged, and the task fails in `prepare()`.

**Simulator.** `SimulatedMotMasterInterface`, decorated with
`@simulate(MotMasterInterface)`, set as `"sim_class"` on both MotMaster entries
in `CONFIG["Devices"]`. Hand-written methods: `script_parameters` (canned dict
per script name, mixed ints and floats, the same dict for any unlisted script),
`set_motmaster_experiment` (remembers the script), `get_params` (that script's
defaults) and `start_motmaster_experiment` (validates overrides against the
table, so a bad name fails in simulation as on hardware). Every other public
method is auto-stubbed. No `MagicMock`: it returns unserialisable objects and
accepts any method name.

## 3. Argument plumbing

- `Experiment.extra_argument(name) -> Argument | None`, a classmethod, default
  `None`.
- `coerce_arguments`, `Scan.axis_values` and the `submit()` check look in the
  declared arguments first, then the hook; a name neither knows still fails at
  `submit()`/scan validation.
- `MotMasterExperiment.extra_argument("rb.tDelay1")` returns a placeholder
  `Number` accepting any finite value, only when `rb` is one of its `MotMaster`
  attributes (so an unknown attribute fails at submit).
- Only submitted extras exist on the instance; `argument_values()` includes
  them, so they are stored in `/arguments` and `/points`.
- The script's type is applied in `run_sequences`: an `Int32` parameter takes an
  integral float and sends an int; a non-integral value fails that point with a
  message naming the parameter and value. A `LinearAxis` over an `Int32`
  parameter from code therefore fails at the first non-whole point; the GUI row
  is an integer field and avoids this.

## Errors

- Unknown parameter: fails `prepare()`.
- Unknown MOTMaster attribute in a dotted name: fails at submit.
- Follower failure: collected, re-raised after the master finishes.
- Master fails before triggering: followers are joined with `follower_timeout`;
  the task fails and the stuck follower is logged. Abort kills the worker but
  not the `Go()` call already in flight on the follower's device server, which
  stays blocked until it is triggered; restart that device server (or trigger
  it) before the next task, whose `prepare()` would otherwise hang on it.

## Tests

- Declaration checks (zero/two masters rejected; lone master implicit).
- Hook: coerce and scan with dotted names.
- Type conversion: int/float handling and the error cases.
- Run order: modes set, followers started before the master, failures propagate.
- Driver: `script_parameters` restores the previous script path.
- Simulator: rejects bad names.
- GUI (pattern of `tests/test_experiment_gui.py`, offscreen): the search box adds
  a row; the request carries only added rows.
- End to end: `run_local(..., simulate=True)`.
- `pytweezer/experiments/motmaster_arguments_test.py` is rewritten on the new API.
