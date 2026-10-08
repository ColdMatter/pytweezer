from pathlib import Path

import pytest

from pytweezer.drivers.motmaster import MotMasterInterface, SimulatedMotMasterInterface
from pytweezer.experiment import Device, ListAxis, Scan, run_local
from pytweezer.experiment.motmaster import (
    MotMasterExperiment,
    MotMasterInteger,
    MotMasterNumber,
)


class FakeSequencer:
    def __init__(self):
        self.calls = []
        self.fail_go = False

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            if name == "start_motmaster_experiment" and self.fail_go:
                raise RuntimeError("MOTMaster refused")
            if name == "get_params":
                return {"tDelay1": 5, "other": 1.0}
            return None

        return call

    def close_rpc(self):
        self.calls.append(("close_rpc",))


@pytest.fixture
def sequencer(monkeypatch):
    fake = FakeSequencer()
    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: fake
    )
    return fake


class Tof(MotMasterExperiment):
    sequencer = Device("Rb MotMaster")
    motmaster_script = "RbTweezerBasic"
    motmaster_triggered = True

    tof = MotMasterInteger(100, parameter="tDelay1")
    current = MotMasterNumber(1.5, unit="A")


def test_parameters_are_forwarded_with_their_declared_types(sequencer):
    measurement = run_local(
        Tof, Scan(axes=[ListAxis(argument="tof", values=[10, 20])]), current=2
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    names = [call[0] for call in sequencer.calls]
    assert names[:5] == [
        "set_motmaster_experiment",
        "set_run_until_stopped",
        "set_iterations",
        "set_save_toggle",
        "set_trigger_mode",
    ]
    runs = [
        call[1] for call in sequencer.calls if call[0] == "start_motmaster_experiment"
    ]
    assert runs == [{"tDelay1": 10, "current": 2.0}, {"tDelay1": 20, "current": 2.0}]
    assert all(
        type(run["tDelay1"]) is int and type(run["current"]) is float for run in runs
    )
    assert measurement.constants["motmaster_script"] == "RbTweezerBasic"
    assert measurement.constants["motmaster_script_defaults"] == {
        "tDelay1": 5,
        "other": 1.0,
    }
    assert names[-1] == "close_rpc"


def test_schema_marks_motmaster_parameters():
    arguments = Tof.schema()["arguments"]
    assert arguments["tof"]["group"] == "MOTMaster"
    assert arguments["tof"]["motmaster_parameter"] == "tDelay1"
    assert arguments["current"]["motmaster_parameter"] == "current"


def test_sequence_failure_fails_the_task(sequencer):
    sequencer.fail_go = True
    measurement = run_local(Tof)
    assert measurement.status == "failed"
    assert "MOTMaster refused" in measurement.attrs["error"]


def test_missing_declarations_are_reported(sequencer):
    class NoScript(MotMasterExperiment):
        sequencer = Device("Rb MotMaster")

    class NoDevice(MotMasterExperiment):
        motmaster_script = "x"

    assert "motmaster_script" in run_local(NoScript).attrs["error"]
    assert "sequencer = Device" in run_local(NoDevice).attrs["error"]


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
