# MOTMaster Parameter Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiments declare one or more MOTMasters; their script parameters are added by name from a search box in the Experiments form (default: none shown), are scannable, and are sent typed to the right sequencer.

**Architecture:** A driver RPC `script_parameters(script)` (plus a hand-written simulated backend) supplies the searchable list. `Experiment` gains an `extra_argument(name)` hook so dotted names like `rb.tDelay1` coerce, scan and store like declared arguments. `MotMasterExperiment` is rewritten around `MotMaster` device attributes (one master, followers in trigger mode, each `Go()` in a thread). The argument editor gets a search box per MOTMaster that adds ordinary `ArgumentRow`s.

**Tech Stack:** Python 3.13, PyQt6 (offscreen in tests), sipyco RPC, h5py storage, pytest, ruff, Poetry.

**Spec:** `docs/superpowers/specs/2026-10-08-motmaster-parameter-search-design.md`

## Global Constraints

- Run tests with `poetry run pytest tests/ -q`; GUI tests need `QT_QPA_PLATFORM=offscreen`.
- UK English in docstrings and text; Google-style docstrings; docstrings describe current behaviour only (no change history).
- Comment only what is genuinely surprising; no explanatory comment blocks.
- Use `get_logger(name)` from `pytweezer/logging_utils.py`, never bare `logging.getLogger`.
- No hardware at module import or class definition (the manager imports every experiment module to list arguments).
- Values are SI; `unit`/`scale` are display only.
- No compatibility fallback for the old `sequencer = Device(...)` / `motmaster_script` form; it is removed.
- The script's defaults are used to check and type overrides but are **not** recorded in the measurement file.
- Ruff must pass (`poetry run ruff check . && poetry run ruff format .`) before each commit; stage only files you changed (leave `.vscode/settings.json` alone).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- The branch is `exp-handler` (already carries the spec commit); the main branch must stay deployable, so the whole suite must pass at every commit.

## Review Focus

Failure modes the spec implies but the feature tests would not otherwise pin; each has a test in the named task.

1. A dotted name whose prefix is not a declared MOTMaster (`xx.tDelay1`) must be rejected at submit, not when the task starts (Task 3).
2. A non-whole value sent to an `Int32` script parameter (e.g. a `LinearAxis` of 0 to 100 in 7 steps) must fail that point with a message naming the parameter, not send a float MOTMaster rejects (Task 3).
3. A script containing non-numeric parameters (bool/string): the search box offers only numeric ones and sends values of other types unconverted (Tasks 3 and 4).
4. The MOTMaster device is unreachable when an experiment is selected: the box shows the error and a Retry button, and declared arguments still submit (Task 4).
5. Resubmitting a request whose searched parameter no longer exists in the script keeps the row, flagged, instead of silently dropping it (Task 4).
6. A follower still waiting on a trigger after the master failed must not hang the worker: the task fails after `follower_timeout` (Task 3).

Not testable without hardware, so check by hand on the lab PC after merge: `follower_arm_delay` (0.5 s) is long enough for each follower's `Go()` to be armed before the master triggers.

---

### Task 1: Driver `script_parameters` and the simulated sequencer

**Files:**
- Modify: `pytweezer/drivers/motmaster.py` (add method after `get_params_csdict`, ~line 277; add class at end of the sequencer section, before `plot_auto_mot_results` ~line 765)
- Modify: `pytweezer/configuration/config.py:118-135` (two `"sim_class"` entries)
- Test: `tests/test_motmaster_experiment.py` (append)

**Interfaces:**
- Produces: `MotMasterInterface.script_parameters(script: str) -> dict[str, Any]` (name to default; `int` = Int32, `float` = Double). `SimulatedMotMasterInterface` with the same public surface, importable from `pytweezer.drivers.motmaster`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_motmaster_experiment.py`; add `SimulatedMotMasterInterface` to the existing `from pytweezer.drivers.motmaster import ...` line)

```python
class _FakeDotNet:
    def __init__(self):
        self.paths = []

    def SetScriptPath(self, path):
        self.paths.append(path)

    def GetParameters(self):
        return {"tDelay1": 5, "tPulse": 20e-6}


def test_script_parameters_reads_a_script_and_restores_the_previous_one():
    driver = _driver()
    driver.motmaster = _FakeDotNet()
    driver.set_motmaster_experiment("A")
    assert driver.script_parameters("B") == {"tDelay1": 5, "tPulse": 20e-6}
    assert driver.motmaster.paths == ["/scripts/A.cs", "/scripts/B.cs", "/scripts/A.cs"]
    assert driver.script == "A"


def test_script_parameters_leaves_no_script_selected_if_none_was():
    driver = _driver()
    driver.motmaster = _FakeDotNet()
    driver.script_parameters("B")
    assert driver.script is None and driver.script_path is None


def test_simulated_sequencer_serves_and_validates_parameters():
    simulated = SimulatedMotMasterInterface()
    parameters = simulated.script_parameters("RbTweezerBasic")
    assert type(parameters["tDelay1"]) is int and type(parameters["tPulse"]) is float
    assert simulated.script_parameters("NotInTheTable") == simulated.script_parameters(
        "AlsoNotInTheTable"
    )
    simulated.set_motmaster_experiment("RbTweezerBasic")
    assert simulated.get_params() == parameters
    simulated.start_motmaster_experiment({"tDelay1": 7})
    with pytest.raises(KeyError, match="nope"):
        simulated.start_motmaster_experiment({"nope": 1})
    with pytest.raises(TypeError, match="tDelay1"):
        simulated.start_motmaster_experiment({"tDelay1": 7.5})


def test_config_wires_the_simulated_sequencer():
    from pytweezer.experiment.simulation import SimulatedDevices

    devices = SimulatedDevices()
    try:
        backend = devices.get("Rb MotMaster")
        assert isinstance(backend, SimulatedMotMasterInterface)
    finally:
        devices.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/test_motmaster_experiment.py -q`
Expected: ImportError for `SimulatedMotMasterInterface`.

- [ ] **Step 3: Implement**

In `pytweezer/drivers/motmaster.py`, after `get_params_csdict`:

```python
    def script_parameters(self, script: str) -> dict[str, Any]:
        """Return ``{name: default}`` for ``script``, leaving the selected script unchanged.

        Must not run during a sequence: it briefly changes MOTMaster's script path.
        """
        previous = self.script
        self.set_motmaster_experiment(script)
        try:
            return self.get_params()
        finally:
            if previous is not None:
                self.set_motmaster_experiment(previous)
            else:
                self.script = self.script_path = None
```

At module end of the driver class section (before `plot_auto_mot_results`), plus `from pytweezer.servers.simulated_device import simulate` in the imports:

```python
SIMULATED_SCRIPT_PARAMETERS: dict[str, dict[str, Any]] = {
    "RbTweezerBasic": {"tDelay1": 5, "tPulse": 20e-6, "coil_current": 1.5, "nShots": 1},
    "CaFTweezerLoad": {"tLoad": 100, "bTop": 0.5, "tHold": 10e-3},
}
_SIMULATED_FALLBACK = {"tDelay1": 5, "tPulse": 20e-6, "coil_current": 1.5}


@simulate(MotMasterInterface)
class SimulatedMotMasterInterface:
    """MOTMaster stand-in with canned script parameters that rejects what the real one would.

    Scripts missing from :data:`SIMULATED_SCRIPT_PARAMETERS` get a fallback parameter set.
    """

    def __init__(self, *args, **kwargs):
        self.script = None

    def script_parameters(self, script: str) -> dict[str, Any]:
        return dict(SIMULATED_SCRIPT_PARAMETERS.get(script, _SIMULATED_FALLBACK))

    def set_motmaster_experiment(self, script: str):
        self.script = script

    def get_params(self) -> dict[str, Any]:
        if self.script is None:
            raise ValueError("MotMaster script not set")
        return self.script_parameters(self.script)

    def start_motmaster_experiment(self, parameters: dict | None = None):
        known = self.get_params()
        for name, value in (parameters or {}).items():
            if name not in known:
                raise KeyError(f"script {self.script!r} has no parameter {name!r}")
            if isinstance(known[name], int) and type(value) is not int:
                raise TypeError(
                    f"{name!r} is an Int32 in script {self.script!r}; got {value!r}"
                )
```

In `pytweezer/configuration/config.py`, add to both `"Rb MotMaster"` and `"CaF MotMaster"` entries (next to `"class"`):

```python
            "sim_class": "pytweezer.drivers.motmaster:SimulatedMotMasterInterface",
```

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/test_motmaster_experiment.py tests/test_experiment_simulation.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
poetry run ruff check pytweezer tests && poetry run ruff format pytweezer tests
git add pytweezer/drivers/motmaster.py pytweezer/configuration/config.py tests/test_motmaster_experiment.py
git commit -m "Add script_parameters to the MOTMaster driver and a simulated sequencer

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The `extra_argument` hook

**Files:**
- Modify: `pytweezer/experiment/arguments.py` (`coerce_arguments`, end of file)
- Modify: `pytweezer/experiment/experiment.py` (`__init__`, `argument_values`, new `extra_argument`, `set_argument`)
- Modify: `pytweezer/experiment/scan.py:76-85` (`Scan.axis_values`)
- Modify: `pytweezer/experiment/runner.py` (`setattr(experiment, name, value)` loop ~line 112; `argument_schema=` ~line 69)
- Test: `tests/test_experiment_arguments.py` (append)

**Interfaces:**
- Produces: `Experiment.extra_argument(name: str) -> Argument | None` (classmethod, default `None`); `Experiment.set_argument(name, value)` (sets the attribute and registers extras); `Experiment._extra_names: list[str]` (submitted or scanned extras, in order); `Experiment.argument_values()` includes extras.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_experiment_arguments.py`; add imports as needed: `Experiment, Number, Scan, ListAxis, run_local` from `pytweezer.experiment`, `coerce_arguments` from `pytweezer.experiment.arguments`)

```python
class Dotted(Experiment):
    base = Number(1.0)

    @classmethod
    def extra_argument(cls, name):
        if name.startswith("x."):
            argument = Number(0.0)
            argument.name = name
            return argument
        return None

    def run_point(self):
        self.record("seen", self.__dict__["x.a"])


def test_hook_names_coerce_and_unknown_names_still_fail():
    assert coerce_arguments(Dotted, {"x.a": 3})["x.a"] == 3.0
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Dotted, {"y.a": 3})


def test_hook_names_can_be_scanned_and_are_stored():
    scan = Scan(axes=[ListAxis(argument="x.a", values=[1, 2])])
    measurement = run_local(Dotted, scan, **{"x.b": 5})
    assert measurement.status == "completed", measurement.attrs["error"]
    assert list(measurement.points["x.a"]) == [1.0, 2.0]
    assert list(measurement.results["seen"]) == [1.0, 2.0]
    assert measurement.arguments["x.b"] == 5.0


def test_scanning_an_unknown_name_still_fails():
    scan = Scan(axes=[ListAxis(argument="y.a", values=[1])])
    with pytest.raises(ValueError, match="no argument 'y.a'"):
        scan.points(Dotted)
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_experiment_arguments.py -q`
Expected: the new tests FAIL (`coerce_arguments` rejects `x.a`).

- [ ] **Step 3: Implement**

`arguments.py`, replace `coerce_arguments`:

```python
def coerce_arguments(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Return every argument of ``cls`` with ``values`` applied over the defaults.

    Names ``cls`` does not declare are looked up with ``cls.extra_argument``;
    those that resolve are coerced and included.
    """
    declared = collect(cls, Argument)
    extra_argument = getattr(cls, "extra_argument", lambda name: None)
    extras = {name: extra_argument(name) for name in values if name not in declared}
    unknown = [name for name, argument in extras.items() if argument is None]
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no argument(s) {sorted(unknown)}; "
            f"known: {sorted(declared)}"
        )
    coerced = {
        name: argument.coerce(values[name]) if name in values else argument.default
        for name, argument in declared.items()
    }
    coerced.update(
        {name: argument.coerce(values[name]) for name, argument in extras.items()}
    )
    return coerced
```

`experiment.py`:

```python
    def __init__(self, args: dict[str, Any] | None = None) -> None:
        self._extra_names: list[str] = []
        for name, value in coerce_arguments(type(self), args or {}).items():
            self.set_argument(name, value)
        ...  # rest unchanged

    @classmethod
    def extra_argument(cls, name: str) -> Argument | None:
        """Return the argument for ``name`` if it is not declared but is valid, else ``None``."""
        return None

    def set_argument(self, name: str, value: Any) -> None:
        setattr(self, name, value)
        if name not in self.arguments() and name not in self._extra_names:
            self._extra_names.append(name)

    def argument_values(self) -> dict[str, Any]:
        names = [*self.arguments(), *self._extra_names]
        return {name: getattr(self, name) for name in names}
```

`scan.py`, in `axis_values`:

```python
        declared = collect(experiment_cls, Argument)
        extra_argument = getattr(experiment_cls, "extra_argument", lambda name: None)
        resolved = {}
        for axis in self.axes:
            argument = declared.get(axis.argument) or extra_argument(axis.argument)
            if argument is None:
                raise ValueError(
                    f"{experiment_cls.__name__} has no argument {axis.argument!r} to scan"
                )
            raw = axis.raw_values()
```
(delete the old `if axis.argument not in declared` check and `argument = declared[...]` line).

`runner.py`: in the point loop replace `setattr(experiment, name, value)` with `experiment.set_argument(name, value)`. For the schema, add a helper and use it in `prepare_run`'s `argument_schema=`:

```python
def _argument_schema(experiment_cls: type, experiment: Experiment, scan: Scan) -> dict:
    schema = experiment_cls.schema()["arguments"]
    for name in [*experiment._extra_names, *(axis.argument for axis in scan.axes)]:
        argument = experiment_cls.extra_argument(name) if name not in schema else None
        if argument is not None:
            schema[name] = argument.to_schema()
    return schema
```
```python
        argument_schema=_argument_schema(experiment_cls, experiment, scan),
```

- [ ] **Step 4: Run to verify pass, including the whole experiment suite**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/ -q -k "experiment or motmaster or results or database"`
Expected: PASS. Scanned values reach the database as keys of a JSON `scan_values` dict (`pytweezer/database/backfill.py:57`), so dotted names need no change there; confirm the backfill test still passes.

- [ ] **Step 5: Lint and commit**

```bash
poetry run ruff check pytweezer tests && poetry run ruff format pytweezer tests
git add pytweezer/experiment/arguments.py pytweezer/experiment/experiment.py pytweezer/experiment/scan.py pytweezer/experiment/runner.py tests/test_experiment_arguments.py
git commit -m "Let experiments resolve undeclared argument names through a hook

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `MotMaster` declarations and the rewritten `MotMasterExperiment`

**Files:**
- Rewrite: `pytweezer/experiment/motmaster.py`
- Rewrite: `tests/test_motmaster_experiment.py` (keep the driver and simulator tests added in Task 1; replace the `FakeSequencer`/`Tof` experiment tests)
- Modify: `pytweezer/experiments/motmaster_arguments_test.py` is rewritten in Task 5; until then delete nothing but make sure it is not imported by tests (its test in `tests/test_motmaster_experiment.py` is removed here).

**Interfaces:**
- Consumes: `Experiment.extra_argument`, `set_argument`, `_extra_names` (Task 2); `script_parameters`/`get_params` driver calls (Task 1).
- Produces:
  - `MotMaster(device_name, *, script, master=False, iterations=1, save=False, follower_timeout=60.0, timeout=None)`, a `Device`; `to_schema()` adds `"motmaster": {"script", "master", "iterations", "save", "follower_timeout"}`.
  - `MotMasterNumber`/`MotMasterInteger(default, *, device=None, parameter=None, ...)`; schema adds `"motmaster_device"` and `"motmaster_parameter"`.
  - `MotMasterExperiment.motmasters() -> {attr: MotMaster}`, `.master_name() -> str`, `.motmaster_parameters() -> {attr: {parameter: value}}`, `.run_sequences(**overrides)` (overrides keyed `"attr.parameter"`), class attribute `follower_arm_delay = 0.5`.

- [ ] **Step 1: Write the failing tests** (replace the experiment-level tests in `tests/test_motmaster_experiment.py`; keep imports of `Path`, `pytest`, `MotMasterInterface`, `SimulatedMotMasterInterface`, `_driver`, `_RaisingDotNet`, `_FakeDotNet` and the Task 1 tests)

```python
import time

from pytweezer.experiment import ListAxis, Scan, run_local
from pytweezer.experiment.arguments import coerce_arguments
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterInteger,
    MotMasterNumber,
)


class FakeSequencer:
    def __init__(self, log, name, params, *, fail_go=False, go_delay=0.0):
        self.log, self.name, self.params = log, name, params
        self.fail_go, self.go_delay = fail_go, go_delay

    def _note(self, *event):
        self.log.append((self.name, *event))

    def set_motmaster_experiment(self, script):
        self._note("script", script)

    def set_run_until_stopped(self, value):
        self._note("run_until_stopped", value)

    def set_iterations(self, value):
        self._note("iterations", value)

    def set_save_toggle(self, value):
        self._note("save", value)

    def set_trigger_mode(self, value):
        self._note("triggered", value)

    def get_params(self):
        return dict(self.params)

    def start_motmaster_experiment(self, parameters=None):
        time.sleep(self.go_delay)
        self._note("go", parameters)
        if self.fail_go:
            raise RuntimeError(f"{self.name} refused")

    def close_rpc(self):
        pass


@pytest.fixture
def rig(monkeypatch):
    log = []
    fakes = {
        "Rb MotMaster": FakeSequencer(
            log, "rb", {"tDelay1": 5, "tPulse": 20e-6, "label": "x"}
        ),
        "CaF MotMaster": FakeSequencer(log, "caf", {"tLoad": 100, "bTop": 0.5}),
    }
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: fakes[name]
    )
    monkeypatch.setattr(MotMasterExperiment, "follower_arm_delay", 0.0)
    return log, fakes


def goes(log, name):
    return [event[2] for event in log if event[0] == name and event[1] == "go"]


class Single(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    tof = MotMasterInteger(100, parameter="tDelay1")
    pulse = MotMasterNumber(1e-6, parameter="tPulse")


class Pair(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
    caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad", follower_timeout=0.1)
    tof = MotMasterInteger(100, device="rb", parameter="tDelay1")


def test_declared_and_searched_parameters_are_sent_typed(rig):
    log, _ = rig
    scan = Scan(axes=[ListAxis(argument="rb.tDelay1", values=[1, 2])])
    measurement = run_local(Single, scan, **{"rb.tPulse": 3e-5})
    assert measurement.status == "completed", measurement.attrs["error"]
    runs = goes(log, "rb")
    assert [run["tDelay1"] for run in runs] == [1, 2]
    assert all(type(run["tDelay1"]) is int and type(run["tPulse"]) is float for run in runs)
    assert runs[0]["tPulse"] == 3e-5
    assert list(measurement.points["rb.tDelay1"]) == [1, 2]
    assert measurement.arguments["rb.tPulse"] == 3e-5


def test_parameters_not_chosen_are_not_sent(rig):
    log, _ = rig
    run_local(Pair)
    assert goes(log, "caf") == [{}]
    assert goes(log, "rb") == [{"tDelay1": 100}]


def test_script_defaults_are_not_recorded(rig):
    measurement = run_local(Single)
    assert not any("default" in name for name in measurement.constants)


def test_non_whole_value_for_an_int32_parameter_fails_the_point(rig):
    measurement = run_local(Single, **{"rb.tDelay1": 1.5})
    assert measurement.status == "failed"
    assert "tDelay1" in measurement.attrs["error"]
    assert "whole number" in measurement.attrs["error"]


def test_unknown_parameter_fails_prepare_with_a_suggestion(rig):
    log, _ = rig
    measurement = run_local(Single, **{"rb.tDelayy": 1})
    assert measurement.status == "failed"
    assert "tDelayy" in measurement.attrs["error"]
    assert "tDelay1" in measurement.attrs["error"]
    assert goes(log, "rb") == []


def test_non_numeric_script_parameters_pass_through_unconverted(rig):
    log, _ = rig
    run_local(Single, **{"rb.label": 3})
    assert goes(log, "rb")[0]["label"] == 3


def test_dotted_name_for_an_unknown_motmaster_is_rejected_at_submit():
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Single, {"xx.tDelay1": 1})
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Single, {"rb.": 1})


def test_master_is_untriggered_and_followers_arm_first(rig):
    log, _ = rig
    run_local(Pair, **{"caf.bTop": 0.7})
    assert ("rb", "triggered", False) in log
    assert ("caf", "triggered", True) in log
    order = [event[0] for event in log if event[1] == "go"]
    assert order == ["caf", "rb"] or order == ["rb", "caf"]
    assert goes(log, "caf") == [{"bTop": 0.7}]


def test_master_failure_fails_the_task(rig):
    _, fakes = rig
    fakes["Rb MotMaster"].fail_go = True
    measurement = run_local(Pair)
    assert measurement.status == "failed"
    assert "rb refused" in measurement.attrs["error"]


def test_follower_failure_fails_the_task(rig):
    _, fakes = rig
    fakes["CaF MotMaster"].fail_go = True
    measurement = run_local(Pair)
    assert measurement.status == "failed"
    assert "caf refused" in measurement.attrs["error"]


def test_follower_that_never_finishes_fails_after_its_timeout(rig):
    _, fakes = rig
    fakes["CaF MotMaster"].go_delay = 1.0
    started = time.monotonic()
    measurement = run_local(Pair)
    assert time.monotonic() - started < 0.9
    assert measurement.status == "failed"
    assert "did not finish within 0.1 s" in measurement.attrs["error"]


def test_declaration_rules_are_checked_at_class_definition():
    with pytest.raises(TypeError, match="exactly one"):

        class TwoMasters(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s", master=True)
            b = MotMaster("CaF MotMaster", script="s", master=True)

    with pytest.raises(TypeError, match="exactly one"):

        class NoMaster(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            b = MotMaster("CaF MotMaster", script="s")

    with pytest.raises(TypeError, match="device="):

        class Ambiguous(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s", master=True)
            b = MotMaster("CaF MotMaster", script="s")
            x = MotMasterNumber(1.0)

    with pytest.raises(TypeError, match="'nope'"):

        class UnknownDevice(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            x = MotMasterNumber(1.0, device="nope")


def test_experiment_without_a_motmaster_fails_when_run():
    class Bare(MotMasterExperiment):
        pass

    measurement = run_local(Bare)
    assert measurement.status == "failed"
    assert "MotMaster" in measurement.attrs["error"]


def test_schema_describes_the_sequencers_and_parameters():
    schema = Pair.schema()
    assert schema["devices"]["caf"]["motmaster"]["script"] == "CaFTweezerLoad"
    assert schema["devices"]["rb"]["motmaster"]["master"] is True
    assert schema["arguments"]["tof"]["motmaster_device"] == "rb"
    assert schema["arguments"]["tof"]["motmaster_parameter"] == "tDelay1"
    assert Single.master_name() == "rb"
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_motmaster_experiment.py -q`
Expected: ImportError for `MotMaster`.

- [ ] **Step 3: Implement** (replace `pytweezer/experiment/motmaster.py`)

```python
"""Experiments that run MOTMaster sequences once per point.

::

    class TofImaging(MotMasterExperiment):
        rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
        caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad")

        tof = MotMasterInteger(100, device="rb", parameter="tDelay1", unit="ms/100")

Each :class:`MotMaster` names a device and the script it runs. Parameters
declared with :class:`MotMasterNumber`/:class:`MotMasterInteger` are always
shown; any other script parameter can be set by passing a dotted argument name
(``"rb.tPulse"``), which the Experiments form offers through a search box.
Parameters not set are left at the script's own defaults. One MOTMaster is the
master; the others are armed in trigger mode and started first.
"""

import difflib
import threading
import time
from typing import Any, ClassVar

from pytweezer.experiment.arguments import Argument, Device, Integer, Number, collect
from pytweezer.experiment.experiment import Experiment, logger


class MotMaster(Device):
    """A MOTMaster sequencer device and the script it runs for this experiment."""

    def __init__(
        self,
        device_name: str,
        *,
        script: str,
        master: bool = False,
        iterations: int = 1,
        save: bool = False,
        follower_timeout: float = 60.0,
        timeout: float | None = None,
    ) -> None:
        super().__init__(device_name, timeout=timeout)
        self.script = script
        self.master = master
        self.iterations = iterations
        self.save = save
        self.follower_timeout = follower_timeout

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "motmaster": {
                "script": self.script,
                "master": self.master,
                "iterations": self.iterations,
                "save": self.save,
                "follower_timeout": self.follower_timeout,
            }
        }


class MotMasterParameter:
    """Marks an argument as a MOTMaster script parameter (default group "MOTMaster").

    ``device`` is the name of a :class:`MotMaster` attribute; it may be omitted
    when the experiment has only one.
    """

    def __init__(
        self,
        *args: Any,
        device: str | None = None,
        parameter: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.parameter = parameter
        kwargs.setdefault("group", "MOTMaster")
        super().__init__(*args, **kwargs)

    @property
    def motmaster_name(self) -> str:
        return self.parameter or self.name

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "motmaster_device": self.device,
            "motmaster_parameter": self.motmaster_name,
        }


class MotMasterNumber(MotMasterParameter, Number):
    """A float MOTMaster parameter (sent to .NET as ``Double``)."""


class MotMasterInteger(MotMasterParameter, Integer):
    """An integer MOTMaster parameter (sent to .NET as ``Int32``)."""


class _MotMasterOverride(Number):
    """Placeholder for a script parameter chosen by name; the script gives its real type."""


class MotMasterExperiment(Experiment):
    """Base class for an experiment driven by one or more MOTMaster scripts.

    :meth:`prepare` configures every sequencer completely on each task, since
    their settings persist between tasks. Repeated shots come from the scan's
    ``repetitions``, giving one stored point per shot.
    """

    #: Seconds to wait after arming the followers before the master starts.
    follower_arm_delay: ClassVar[float] = 0.5

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sequencers = cls.motmasters()
        if not sequencers:
            return
        masters = [name for name, sequencer in sequencers.items() if sequencer.master]
        if len(sequencers) > 1 and len(masters) != 1:
            raise TypeError(
                f"{cls.__name__} has {len(sequencers)} MOTMasters; exactly one "
                f"must have master=True (found {len(masters)})"
            )
        for name, argument in cls.motmaster_arguments().items():
            cls._parameter_device(name, argument)

    @classmethod
    def motmasters(cls) -> dict[str, MotMaster]:
        return collect(cls, MotMaster)

    @classmethod
    def master_name(cls) -> str:
        sequencers = cls.motmasters()
        flagged = [name for name, sequencer in sequencers.items() if sequencer.master]
        return flagged[0] if flagged else next(iter(sequencers))

    @classmethod
    def motmaster_arguments(cls) -> dict[str, MotMasterParameter]:
        return {
            name: argument
            for name, argument in cls.arguments().items()
            if isinstance(argument, MotMasterParameter)
        }

    @classmethod
    def _parameter_device(cls, name: str, argument: MotMasterParameter) -> str:
        sequencers = cls.motmasters()
        if argument.device is None:
            if len(sequencers) != 1:
                raise TypeError(
                    f"{cls.__name__}.{name} needs device= to say which of "
                    f"{sorted(sequencers)} it belongs to"
                )
            return next(iter(sequencers))
        if argument.device not in sequencers:
            raise TypeError(
                f"{cls.__name__}.{name}: device={argument.device!r} is not one of "
                f"the MOTMasters {sorted(sequencers)}"
            )
        return argument.device

    @classmethod
    def extra_argument(cls, name: str) -> Argument | None:
        attribute, dot, parameter = name.partition(".")
        if not dot or not parameter or attribute not in cls.motmasters():
            return None
        argument = _MotMasterOverride(0.0, group=f"MOTMaster: {attribute}")
        argument.name = name
        return argument

    def motmaster_parameters(self) -> dict[str, dict[str, Any]]:
        """``{MotMaster attribute: {script parameter: value}}`` for everything set explicitly."""
        requested: dict[str, dict[str, Any]] = {name: {} for name in self.motmasters()}
        for name, argument in self.motmaster_arguments().items():
            device = self._parameter_device(name, argument)
            requested[device][argument.motmaster_name] = getattr(self, name)
        for name in self._extra_names:
            attribute, _, parameter = name.partition(".")
            requested[attribute][parameter] = getattr(self, name)
        return requested

    def prepare(self) -> None:
        sequencers = self.motmasters()
        if not sequencers:
            raise TypeError(
                f"{type(self).__name__} must declare at least one MotMaster(...)"
            )
        master = self.master_name()
        self._script_parameters: dict[str, dict[str, Any]] = {}
        for attribute, motmaster in sequencers.items():
            client = getattr(self, attribute)
            client.set_motmaster_experiment(motmaster.script)
            client.set_run_until_stopped(False)
            client.set_iterations(motmaster.iterations)
            client.set_save_toggle(motmaster.save)
            client.set_trigger_mode(attribute != master)
            self._script_parameters[attribute] = client.get_params()
        for attribute, parameters in self.motmaster_parameters().items():
            for parameter in parameters:
                self._check_parameter(attribute, parameter)

    def _check_parameter(self, attribute: str, parameter: str) -> None:
        known = self._script_parameters[attribute]
        if parameter in known:
            return
        script = self.motmasters()[attribute].script
        close = difflib.get_close_matches(parameter, known, n=3)
        hint = f"; did you mean {close}?" if close else ""
        raise ValueError(
            f"{attribute}: script {script!r} has no parameter {parameter!r}{hint}"
        )

    def _typed(self, attribute: str, parameter: str, value: Any) -> Any:
        default = self._script_parameters[attribute][parameter]
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            return value
        if isinstance(default, float):
            return float(value)
        if not float(value).is_integer():
            script = self.motmasters()[attribute].script
            raise ValueError(
                f"{attribute}.{parameter} is an Int32 in script {script!r}; "
                f"{value!r} is not a whole number"
            )
        return int(value)

    def run_sequences(self, **overrides: Any) -> None:
        """Run every script once (blocking) with the current parameters plus ``overrides``.

        ``overrides`` are keyed ``"attribute.parameter"``. Followers are started
        in threads first, then the master; the first error is raised once all
        have finished or timed out.
        """
        requested = self.motmaster_parameters()
        for key, value in overrides.items():
            attribute, _, parameter = key.partition(".")
            if attribute not in requested:
                raise KeyError(f"no MotMaster attribute {attribute!r} for {key!r}")
            self._check_parameter(attribute, parameter)
            requested[attribute][parameter] = value
        sent = {
            attribute: {
                parameter: self._typed(attribute, parameter, value)
                for parameter, value in parameters.items()
            }
            for attribute, parameters in requested.items()
        }

        master = self.master_name()
        errors: list[tuple[str, BaseException]] = []

        def go(attribute: str, client: Any) -> None:
            try:
                client.start_motmaster_experiment(sent[attribute])
            except BaseException as error:
                errors.append((attribute, error))

        followers: dict[str, threading.Thread] = {}
        for attribute, motmaster in self.motmasters().items():
            if attribute == master:
                continue
            client = self.device(
                motmaster.device_name, fresh=True, timeout=motmaster.timeout
            )
            thread = threading.Thread(
                target=go, args=(attribute, client), name=f"motmaster-{attribute}"
            )
            thread.daemon = True
            thread.start()
            followers[attribute] = thread
        if followers:
            time.sleep(self.follower_arm_delay)
        go(master, getattr(self, master))
        for attribute, thread in followers.items():
            timeout = self.motmasters()[attribute].follower_timeout
            thread.join(timeout)
            if thread.is_alive():
                errors.append(
                    (
                        attribute,
                        TimeoutError(
                            f"{attribute} did not finish within {timeout} s; "
                            "it may still be waiting for its trigger"
                        ),
                    )
                )
        if errors:
            errors.sort(key=lambda entry: entry[0] != master)
            for attribute, error in errors[1:]:
                logger.error("MOTMaster %s also failed: %r", attribute, error)
            raise errors[0][1]

    def run_point(self) -> None:
        self.run_sequences()
```

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/ -q`
Expected: PASS except nothing else imports the old API (`pytweezer/experiments/motmaster_arguments_test.py` is only imported by the test removed in this task; confirm with `grep -rn "motmaster_arguments_test" tests pytweezer`).

- [ ] **Step 5: Lint and commit**

```bash
poetry run ruff check pytweezer tests && poetry run ruff format pytweezer tests
git add pytweezer/experiment/motmaster.py tests/test_motmaster_experiment.py
git commit -m "Declare MOTMasters per experiment; send searched parameters typed to each

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

(`pytweezer/experiments/motmaster_arguments_test.py` still uses the old API and would fail the manager's introspection; if it is untracked/uncommitted, leave it uncommitted until Task 5 rewrites it.)

---

### Task 4: The search box in the Experiments form

**Files:**
- Create: `pytweezer/GUI/experiments/motmaster_params.py`
- Modify: `pytweezer/GUI/experiments/arg_editor.py` (`ArgumentEditor.__init__`, `set_experiment`, `reset_to_defaults`, `load_request`; add row helpers)
- Modify: `pytweezer/GUI/experiments/panel.py` (`ExperimentsPanel.__init__`, `_queue_changed`)
- Test: `tests/test_experiment_gui.py` (append)

**Interfaces:**
- Consumes: schema `devices[attr]["motmaster"]` and `["device"]` (Task 3); `ArgumentRow`, `ValueField` (existing).
- Produces:
  - `ParameterFetcher(source, *, threaded=True)` with `.request(device, script)`, signals `fetched(str, str, object)` and `failed(str, str, str)`, attribute `.source`, `.threaded`, `.cache`.
  - `DeviceParameterSource(simulated=lambda: False)`, callable `(device, script) -> dict`.
  - `MotMasterBox(attribute, device, script)` with `.defaults: dict[str, int | float]` (numeric only), `.choose(text)`, `.retry`, `.status`, `.parameter_chosen` signal `(str)`.
  - `ArgumentEditor.set_parameter_source(source)`, `.motmaster_boxes: dict[str, MotMasterBox]`, `.add_motmaster_row(attribute, parameter, value=None) -> ArgumentRow`, `.remove_motmaster_row(name)`. Searched rows live in `ArgumentEditor.rows` under their dotted name, so `request()`, `scan()` and the point count need no change.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_experiment_gui.py`; add imports `from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment, MotMasterNumber` and `from pytweezer.GUI.experiments.motmaster_params import DeviceParameterSource`)

```python
class Sequenced(MotMasterExperiment):
    """A MOTMaster experiment."""

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    pulse = MotMasterNumber(1e-6, parameter="tPulse")


SEQUENCED = Sequenced.schema()
PARAMETERS = {"tDelay1": 5, "tPulse": 20e-6, "label": "x", "enabled": True}


class Source:
    def __init__(self, parameters=None, error=None):
        self.parameters, self.error, self.calls = parameters, error, []

    def __call__(self, device, script):
        self.calls.append((device, script))
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.parameters)


def sequenced_editor(qapp, source):
    editor = ArgumentEditor()
    editor.fetcher.threaded = False
    editor.set_parameter_source(source)
    editor.set_experiment(SEQUENCED)
    return editor


def test_motmaster_parameters_are_hidden_until_chosen(qapp):
    editor = sequenced_editor(qapp, Source(PARAMETERS))
    box = editor.motmaster_boxes["rb"]
    assert box.defaults == {"tDelay1": 5, "tPulse": 20e-6}
    assert set(editor.rows) == {"pulse"}
    assert "rb.tDelay1" not in editor.request().args


def test_choosing_a_parameter_adds_a_typed_row_and_only_that_row_is_sent(qapp):
    editor = sequenced_editor(qapp, Source(PARAMETERS))
    editor.motmaster_boxes["rb"].choose("tDelay1 · int · 5")
    row = editor.rows["rb.tDelay1"]
    assert row.schema["kind"] == "integer" and row.value.value() == 5
    assert editor.request().args["rb.tDelay1"] == 5
    assert "rb.tPulse" not in editor.request().args


def test_a_searched_parameter_can_be_scanned_and_removed(qapp):
    editor = sequenced_editor(qapp, Source(PARAMETERS))
    editor.motmaster_boxes["rb"].choose("tPulse")
    editor.rows["rb.tPulse"].scan_button.setChecked(True)
    assert [axis.argument for axis in editor.request().scan.axes] == ["rb.tPulse"]
    editor.remove_motmaster_row("rb.tPulse")
    assert "rb.tPulse" not in editor.rows
    assert editor.request().scan.axes == []


def test_resetting_the_form_removes_searched_rows(qapp):
    editor = sequenced_editor(qapp, Source(PARAMETERS))
    editor.motmaster_boxes["rb"].choose("tDelay1")
    editor.reset_to_defaults()
    assert "rb.tDelay1" not in editor.rows


def test_load_request_restores_searched_rows_and_flags_missing_ones(qapp):
    editor = sequenced_editor(qapp, Source(PARAMETERS))
    request = TaskRequest(
        experiment=SEQUENCED["module"],
        class_name="Sequenced",
        args={"rb.tDelay1": 9, "rb.gone": 1.5},
        scan=Scan(axes=[LinearAxis(argument="rb.tPulse", start=0, stop=1e-5, n=3)]),
    )
    editor.load_request(request)
    assert editor.rows["rb.tDelay1"].value.value() == 9
    assert editor.rows["rb.tPulse"].scanning
    assert "not in script" in editor.rows["rb.gone"].label.toolTip()
    assert "not in script" not in editor.rows["rb.tDelay1"].label.toolTip()


def test_an_unreachable_device_shows_the_error_and_a_retry(qapp):
    source = Source(error="no route to host")
    editor = sequenced_editor(qapp, source)
    box = editor.motmaster_boxes["rb"]
    assert "no route to host" in box.status.text()
    assert not box.retry.isHidden()
    source.error, source.parameters = None, PARAMETERS
    box.retry.click()
    assert box.retry.isHidden() and "tDelay1" in box.defaults
    assert editor.request().args == {"pulse": 1e-6}


def test_parameters_are_fetched_once_per_script(qapp):
    source = Source(PARAMETERS)
    editor = sequenced_editor(qapp, source)
    editor.set_experiment(SEQUENCED)
    assert source.calls == [("Rb MotMaster", "RbTweezerBasic")]


def test_the_device_source_can_read_the_simulated_sequencer():
    source = DeviceParameterSource(simulated=lambda: True)
    parameters = source("Rb MotMaster", "RbTweezerBasic")
    assert "tDelay1" in parameters
```

- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/test_experiment_gui.py -q`
Expected: ImportError for `motmaster_params`.

- [ ] **Step 3: Implement**

`pytweezer/GUI/experiments/motmaster_params.py`:

```python
"""Search-and-add boxes for MOTMaster script parameters in the argument editor."""

import threading
from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCompleter,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

_SEPARATOR = " · "


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class DeviceParameterSource:
    """Reads a script's parameters from its MOTMaster device (or the simulated one)."""

    def __init__(self, simulated: Callable[[], bool] = lambda: False) -> None:
        self._simulated = simulated
        self._simulated_devices = None

    def __call__(self, device: str, script: str) -> dict[str, Any]:
        if self._simulated():
            if self._simulated_devices is None:
                from pytweezer.experiment.simulation import SimulatedDevices

                self._simulated_devices = SimulatedDevices()
            return self._simulated_devices.get(device).script_parameters(script)
        from pytweezer.servers.device_client import get_device

        client = get_device(device)
        try:
            return client.script_parameters(script)
        finally:
            client.close_rpc()


class ParameterFetcher(QtCore.QObject):
    """Fetches script parameters off the GUI thread and caches them per (device, script)."""

    fetched = QtCore.pyqtSignal(str, str, object)
    failed = QtCore.pyqtSignal(str, str, str)
    _finished = QtCore.pyqtSignal(object, object, object)

    def __init__(self, source: Callable[[str, str], dict], *, threaded: bool = True):
        super().__init__()
        self.source = source
        self.threaded = threaded
        self.cache: dict[tuple[str, str], dict] = {}
        self._pending: set[tuple[str, str]] = set()
        self._finished.connect(self._on_finished)

    def request(self, device: str, script: str) -> None:
        key = (device, script)
        if key in self.cache:
            self.fetched.emit(device, script, self.cache[key])
        elif key not in self._pending:
            self._pending.add(key)
            if self.threaded:
                threading.Thread(target=self._run, args=(key,), daemon=True).start()
            else:
                self._run(key)

    def _run(self, key: tuple[str, str]) -> None:
        try:
            self._finished.emit(key, self.source(*key), None)
        except Exception as error:
            self._finished.emit(key, None, str(error) or type(error).__name__)

    def _on_finished(self, key, parameters, error) -> None:
        self._pending.discard(key)
        if error is None:
            self.cache[key] = parameters
            self.fetched.emit(*key, parameters)
        else:
            self.failed.emit(*key, error)


class MotMasterBox(QGroupBox):
    """One sequencer's group: a search field and the rows chosen from it."""

    parameter_chosen = QtCore.pyqtSignal(str)

    def __init__(self, attribute: str, device: str, script: str, parent=None):
        super().__init__(f"MOTMaster: {attribute} ({script})", parent)
        self.setObjectName("EditorGroup")
        self.attribute, self.device, self.script = attribute, device, script
        self.defaults: dict[str, int | float] = {}
        self.status = QLabel("Loading parameters…")
        self.status.setProperty("role", "regionHint")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search for a script parameter to add")
        self.search.setEnabled(False)
        self.search.returnPressed.connect(lambda: self.choose(self.search.text()))
        self.retry = QPushButton("Retry")
        self.retry.setVisible(False)
        self.completer = QCompleter([], self.search)
        self.completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.completer.activated.connect(self.choose)
        self.search.setCompleter(self.completer)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setColumnStretch(4, 1)
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.retry)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addLayout(top)
        layout.addLayout(self.grid)

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        self.defaults = {name: v for name, v in parameters.items() if is_number(v)}
        labels = [
            f"{name}{_SEPARATOR}{'int' if isinstance(v, int) else 'float'}{_SEPARATOR}{v}"
            for name, v in self.defaults.items()
        ]
        self.completer.setModel(QtCore.QStringListModel(labels, self.completer))
        self.search.setEnabled(True)
        self.retry.setVisible(False)
        self.status.setText(f"{len(self.defaults)} parameters in the script")

    def set_error(self, error: str) -> None:
        self.search.setEnabled(False)
        self.retry.setVisible(True)
        self.status.setText(f"Could not read the script's parameters: {error}")

    def choose(self, text: str) -> None:
        name = text.split(_SEPARATOR)[0].strip()
        if name in self.defaults:
            self.search.clear()
            self.parameter_chosen.emit(name)
```

`arg_editor.py` changes (imports: `from pytweezer.GUI.experiments.motmaster_params import DeviceParameterSource, MotMasterBox, ParameterFetcher, is_number`; `QToolButton` is already imported):

In `ArgumentEditor.__init__`, before building the layout:

```python
        self.fetcher = ParameterFetcher(DeviceParameterSource())
        self.fetcher.fetched.connect(self._parameters_fetched)
        self.fetcher.failed.connect(self._parameters_failed)
        self.motmaster_boxes = {}
        self._searched = {}
```

New methods:

```python
    def set_parameter_source(self, source):
        self.fetcher.source = source

    def _add_motmaster_boxes(self, schema):
        self.motmaster_boxes = {}
        self._searched = {}
        for attribute, device in schema.get("devices", {}).items():
            motmaster = device.get("motmaster")
            if motmaster is None:
                continue
            box = MotMasterBox(attribute, device["device"], motmaster["script"])
            box.parameter_chosen.connect(
                lambda name, attribute=attribute: self.add_motmaster_row(attribute, name)
            )
            box.retry.clicked.connect(
                lambda _=False, box=box: self.fetcher.request(box.device, box.script)
            )
            self.motmaster_boxes[attribute] = box
            self.arguments_layout.addWidget(box)
            self.fetcher.request(box.device, box.script)

    def add_motmaster_row(self, attribute, parameter, value=None):
        """Add (or return) the row for script parameter ``parameter`` of ``attribute``."""
        name = f"{attribute}.{parameter}"
        if name in self.rows:
            return self.rows[name]
        box = self.motmaster_boxes[attribute]
        default = box.defaults.get(parameter, value)
        schema = {
            "kind": "integer" if isinstance(default, int) else "number",
            "default": default,
            "tooltip": f"{parameter} in script {box.script}",
            "group": box.title(),
        }
        row = ArgumentRow(name, schema, box)
        grid_row = box.grid.rowCount()
        row.add_to(box.grid, grid_row)
        remove = QToolButton()
        remove.setText("×")
        remove.setToolTip("Return this parameter to the script's own value")
        remove.clicked.connect(lambda: self.remove_motmaster_row(name))
        box.grid.addWidget(remove, grid_row, 3)
        row.remove_button = remove
        row.changed.connect(self._update_count)
        self.rows[name] = row
        self._searched[name] = (attribute, parameter)
        if parameter not in box.defaults and box.defaults:
            self._flag_missing(row)
        self._update_count()
        return row

    def remove_motmaster_row(self, name):
        row = self.rows.pop(name)
        attribute, _ = self._searched.pop(name)
        grid = self.motmaster_boxes[attribute].grid
        for widget in (row.label, row.stack, row.scan_button, row.remove_button):
            grid.removeWidget(widget)
            widget.deleteLater()
        row.deleteLater()
        self._update_count()

    def _flag_missing(self, row):
        row.label.setToolTip(f"{row.name}: not in script")
        set_state(row.label, "crashed")

    def _parameters_fetched(self, device, script, parameters):
        for attribute, box in self.motmaster_boxes.items():
            if (box.device, box.script) != (device, script):
                continue
            box.set_parameters(parameters)
            for name, (owner, parameter) in self._searched.items():
                if owner == attribute and parameter not in box.defaults:
                    self._flag_missing(self.rows[name])

    def _parameters_failed(self, device, script, error):
        for box in self.motmaster_boxes.values():
            if (box.device, box.script) == (device, script):
                box.set_error(error)
```

In `set_experiment`, after the `for group, arguments in groups.items():` loop and before `if not self.rows:` add `self._add_motmaster_boxes(schema)`; also reset `self.rows = {}` stays before. Change `if not self.rows:` to `if not self.rows and not self.motmaster_boxes:`.

In `reset_to_defaults`, first line: `for name in list(self._searched): self.remove_motmaster_row(name)`.

In `load_request`, after `self.reset_to_defaults()` and before filling:

```python
        searched = [*request.args, *(axis.argument for axis in request.scan.axes)]
        for name in searched:
            attribute, _, parameter = name.partition(".")
            if parameter and attribute in self.motmaster_boxes and name not in self.rows:
                value = request.args.get(name, 0)
                self.add_motmaster_row(attribute, parameter, value)
```
For a scanned name not in `args`, the row default of `0` makes it an integer row; correct it by using the axis's first value: replace `request.args.get(name, 0)` with
`request.args.get(name, next((a.raw_values()[0] for a in request.scan.axes if a.argument == name), 0))`. (An int-typed first value gives an integer row, as the script default would once fetched; `add_motmaster_row` prefers `box.defaults` when available.)

`panel.py`: import `DeviceParameterSource`; in `__init__` before creating the editor set `self._simulated = False`; after `self.editor = ArgumentEditor()` add `self.editor.set_parameter_source(DeviceParameterSource(lambda: self._simulated))`; in `_queue_changed` first line add `self._simulated = bool(snapshot.get("simulated"))`.

- [ ] **Step 4: Run to verify pass**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/test_experiment_gui.py tests/test_results_browser.py -q`
Expected: PASS. If a `test_load_request` expectation fails because the stack width or row count changed, fix the code, not the old test.

- [ ] **Step 5: Look at it, then commit**

Use the `run-pytweezer` skill to launch `pytweezer-server` simulated, open the Experiments tab, select a `MotMasterExperiment`, and check: the MOTMaster box lists no rows, the search offers `name · type · default`, choosing adds a row, scanning it shows the scan editor, × removes it. Fix any visual problem against the `pytweezer-gui-design` skill before committing.

```bash
poetry run ruff check pytweezer tests && poetry run ruff format pytweezer tests
git add pytweezer/GUI/experiments/motmaster_params.py pytweezer/GUI/experiments/arg_editor.py pytweezer/GUI/experiments/panel.py tests/test_experiment_gui.py
git commit -m "Add a MOTMaster parameter search box to the Experiments form

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The test experiment, an end-to-end run, and the skill docs

**Files:**
- Rewrite: `pytweezer/experiments/motmaster_arguments_test.py`
- Modify: `.claude/skills/add-experiment/SKILL.md` (the "MOTMaster sequences" section and its mention in "Running")
- Test: `tests/test_motmaster_experiment.py` (append one end-to-end test)

**Interfaces:**
- Consumes: everything from Tasks 1 to 4.

- [ ] **Step 1: Write the failing end-to-end test** (append)

```python
def test_simulated_end_to_end_with_searched_parameters():
    from pytweezer.experiments.motmaster_arguments_test import MotMasterArgumentsTest

    scan = Scan(axes=[ListAxis(argument="rb.tDelay1", values=[10, 20])])
    measurement = run_local(
        MotMasterArgumentsTest, scan, simulate=True, **{"rb.tPulse": 3e-5}
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    assert list(measurement.points["rb.tDelay1"]) == [10, 20]
    assert list(measurement.results["sent_tDelay1"]) == [10, 20]
    assert measurement.arguments["rb.tPulse"] == 3e-5
```
- [ ] **Step 2: Run to verify failure**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/test_motmaster_experiment.py -q`
Expected: FAIL (the experiment file still uses the removed API).

- [ ] **Step 3: Rewrite the experiment**

```python
"""Exercise the MOTMaster argument interface without a camera."""

from pytweezer.experiment import Bool, Choice, Text
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment, MotMasterNumber


class MotMasterArgumentsTest(MotMasterExperiment):
    """Run a script with parameters chosen from the form's search box.

    ``pulse_time`` is always shown; any other script parameter is added by name.
    ``dry_run``, ``mode`` and ``note`` are ordinary arguments and are not sent.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")

    pulse_time = MotMasterNumber(
        20e-6, parameter="tPulse", unit="us", scale=1e-6, min=0
    )

    dry_run = Bool(False, tooltip="Record the parameters without running the script")
    mode = Choice(["fast", "slow"])
    note = Text("")

    def run_point(self):
        sent = self.motmaster_parameters()["rb"]
        for name, value in sent.items():
            self.record(f"sent_{name}", value)
        if not self.dry_run:
            self.run_sequences()
```

- [ ] **Step 4: Update the skill**

In `.claude/skills/add-experiment/SKILL.md`, replace the "MOTMaster sequences" code block and the two paragraphs after it with the new form (`MotMaster(...)` attributes, `master=True`, optional `MotMasterNumber(..., device=, parameter=)`, dotted names `"rb.tPulse"` for submit/scan, `run_sequences`, `follower_timeout`, script defaults unrecorded, `MotMasterInteger` vs `MotMasterNumber` types still matching the script), and add one sentence to "Running" saying MOTMaster parameters other than declared ones are added from the search box in the Experiments form. Keep it to the same length as the section it replaces.

- [ ] **Step 5: Full verification and commit**

Run: `QT_QPA_PLATFORM=offscreen poetry run pytest tests/ -q` and `poetry run ruff check . && poetry run ruff format --check .`
Expected: everything passes. Then:

```bash
git add pytweezer/experiments/motmaster_arguments_test.py tests/test_motmaster_experiment.py .claude/skills/add-experiment/SKILL.md
git commit -m "Rewrite the MOTMaster arguments test experiment on the new API; update the skill

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** Section 1 (declaration, `prepare`, `run_sequences`, no recorded defaults) is Task 3. Section 2 (driver RPC, schema, search box, fetching and caching, Retry, resubmit, simulator) is Tasks 1 and 4; the spec's separate `motmaster` schema block is delivered as `devices[attr]["motmaster"]` because `MotMaster` is a `Device` and `schema()` already lists devices. Section 3 (hook, scan, typing, errors, tests) is Tasks 2 and 3. The old form's removal is Task 3 and Task 5.
- **Behaviour changes to confirm at review:** the old `motmaster_triggered` class variable and the recorded `motmaster_script` constant are gone (spec: master is untriggered; defaults not recorded). The GUI offers numeric script parameters only.
- **Names used across tasks:** `extra_argument`, `set_argument`, `_extra_names`, `motmasters()`, `master_name()`, `motmaster_parameters()`, `run_sequences`, `follower_arm_delay`, `ParameterFetcher.request`, `MotMasterBox.choose`, `add_motmaster_row`/`remove_motmaster_row` are used identically wherever they appear.
