import time
from pathlib import Path

import pytest

from pytweezer.drivers.motmaster import MotMasterInterface, SimulatedMotMasterInterface
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
    assert all(
        type(run["tDelay1"]) is int and type(run["tPulse"]) is float for run in runs
    )
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


def test_unknown_parameter_that_is_only_scanned_fails_before_any_sequence_runs(rig):
    log, _ = rig
    scan = Scan(axes=[ListAxis(argument="rb.tDelayy", values=[1, 2])])
    measurement = run_local(Single, scan)
    assert measurement.status == "failed"
    assert "has no parameter 'tDelayy'" in measurement.attrs["error"]
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


def test_follower_client_is_opened_once_per_task(rig, monkeypatch):
    _, fakes = rig
    opened = []

    def get_device(name, timeout):
        opened.append(name)
        return fakes[name]

    monkeypatch.setattr("pytweezer.experiment.experiment._get_device", get_device)
    scan = Scan(axes=[ListAxis(argument="tof", values=[1, 2, 3])])
    measurement = run_local(Pair, scan)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert opened.count("CaF MotMaster") == 2


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


class _RaisingDotNet:
    def SetScriptPath(self, path):
        raise RuntimeError("no such script")

    def Go(self, *args):
        raise RuntimeError("Go failed")


def _driver():
    driver = MotMasterInterface.__new__(MotMasterInterface)
    driver.motmaster = _RaisingDotNet()
    driver.script_root = Path("/scripts")
    driver.script = None
    driver.interval = 0
    return driver


def test_driver_raises_instead_of_printing():
    driver = _driver()
    with pytest.raises(RuntimeError, match="no such script"):
        driver.set_motmaster_experiment("missing")
    assert driver.script is None
    driver.script = "x"
    with pytest.raises(RuntimeError, match="Go failed"):
        driver.start_motmaster_experiment()


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
