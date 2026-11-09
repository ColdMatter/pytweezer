import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pytweezer.drivers import motmaster as motmaster_driver
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
    def __init__(self, log, name, params):
        self.log, self.name, self.params = log, name, params
        self.fail_go, self.go_delay = False, 0.0

    def close_rpc(self):
        pass

    def set_run_until_stopped(self, value):
        self.log.append((self.name, "run_until_stopped", value))

    def set_iterations(self, value):
        self.log.append((self.name, "iterations", value))

    def set_save_toggle(self, value):
        self.log.append((self.name, "save", value))

    def set_linked(self, value):
        self.log.append((self.name, "linked", value))

    def set_motmaster_experiment(self, script):
        self.log.append((self.name, "script", script))

    def set_trigger_mode(self, value):
        self.log.append((self.name, "triggered", value))

    def get_params(self):
        return dict(self.params)

    def start_motmaster_experiment(self, parameters=None):
        self.log.append((self.name, "start"))
        time.sleep(self.go_delay)
        self.log.append((self.name, "go", parameters))
        if self.fail_go:
            raise RuntimeError(f"{self.name} refused")


@pytest.fixture
def rig(monkeypatch):
    log, opened = [], []
    rb_params = {
        "tDelay1": 5,
        "tPulse": 20e-6,
        "nShots": 1,
        "bField": 0.5,
        "tag": "x",
        "threshold": 1.0,
    }
    fakes = {
        "Rb MotMaster": FakeSequencer(log, "rb", rb_params),
        "CaF MotMaster": FakeSequencer(log, "caf", {"tLoad": 100, "bTop": 0.5}),
    }

    def get_device(name, timeout):
        opened.append(name)
        return fakes[name]

    monkeypatch.setattr("pytweezer.experiment.experiment._get_device", get_device)
    return SimpleNamespace(log=log, fakes=fakes, opened=opened)


def goes(rig, name):
    return [event[2] for event in rig.log if event[0] == name and event[1] == "go"]


class Single(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    tof = MotMasterInteger(100, parameter="tDelay1")
    pulse = MotMasterNumber(1e-6, parameter="tPulse")


class Linked(MotMasterExperiment):
    caf = MotMaster("CaF MotMaster", script="LinkedTweezerLoad", linked=True)
    load = MotMasterInteger(100, parameter="tLoad")


def test_declared_and_searched_parameters_are_sent_typed(rig):
    scan = Scan(axes=[ListAxis(argument="nShots", values=[1, 2])])
    measurement = run_local(Single, scan, bField=3, tag=3)
    assert measurement.status == "completed", measurement.attrs["error"]
    runs = goes(rig, "rb")
    assert [run["nShots"] for run in runs] == [1, 2]
    # Typed by the script's default: Int32 as int, Double as float, anything else as given.
    for run in runs:
        types = [type(run[p]) for p in ("nShots", "tDelay1", "bField", "tPulse")]
        assert types == [int, int, float, float]
    assert runs[0] == {
        "tDelay1": 100,
        "tPulse": 1e-6,
        "nShots": 1,
        "bField": 3.0,
        "tag": 3.0,
    }
    assert list(measurement.points["nShots"]) == [1, 2]
    assert measurement.arguments["bField"] == 3.0


@pytest.mark.parametrize(
    "fixed, scanned, expected",
    [
        ({"nShots": 1.5}, None, ["nShots", "whole number"]),
        ({"tDelayy": 1}, None, ["tDelayy", "tDelay1"]),
        ({"pulsee": 1}, None, ["pulsee", "pulse"]),
        ({}, "tDelayy", ["tDelayy"]),
    ],
    ids=[
        "fraction-for-int32",
        "unknown-with-suggestion",
        "suggests-a-declared-argument",
        "unknown-only-scanned",
    ],
)
def test_bad_script_parameter_fails_before_any_sequence_runs(
    rig, fixed, scanned, expected
):
    scan = Scan(axes=[ListAxis(argument=scanned, values=[1, 2])] if scanned else [])
    measurement = run_local(Single, scan, **fixed)
    assert measurement.status == "failed"
    assert all(text in measurement.attrs["error"] for text in expected)
    assert goes(rig, "rb") == []


@pytest.mark.parametrize("name", ["rb.tDelay1", "hamcam.gain", "rb.", "1x", "tDelay1"])
def test_names_that_cannot_be_script_parameters_are_refused(name):
    # tDelay1 is already set by the declared `tof`.
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Single, {name: 1})
    with pytest.raises(ValueError, match="no argument"):
        Scan(axes=[ListAxis(argument=name, values=[1])]).points(Single)


def test_a_script_parameter_named_like_an_attribute_leaves_the_attribute_alone(rig):
    class WithThreshold(Single):
        threshold = 0.5

        def run_point(self):
            self.record("attribute", self.threshold)
            super().run_point()

    measurement = run_local(WithThreshold, threshold=3)
    assert goes(rig, "rb")[0]["threshold"] == 3.0
    assert list(measurement.results["attribute"]) == [0.5]


def test_overrides_beat_the_form(rig):
    class Override(Single):
        def run_point(self):
            self.run_sequences(tDelay1=7, bField=2)

    run_local(Override, bField=1)
    sent = goes(rig, "rb")[0]
    assert (sent["tDelay1"], sent["bField"]) == (7, 2.0)


class Configured(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", iterations=3, save=True)


def test_the_sequencer_is_set_up_as_declared(rig):
    run_local(Configured)
    settings = {
        event[1]: event[2]
        for event in rig.log
        if event[1]
        in ("script", "run_until_stopped", "iterations", "save", "triggered")
    }
    assert settings == {
        "script": "RbTweezerBasic",
        "run_until_stopped": False,
        "iterations": 3,
        "save": True,
        "triggered": False,
    }


def test_linked_sequencer_is_linked_untriggered_and_run_alone(rig):
    run_local(Linked, bTop=0.7)
    assert ("caf", "linked", True) in rig.log
    assert ("caf", "triggered", False) in rig.log
    assert goes(rig, "caf") == [{"tLoad": 100, "bTop": 0.7}]
    assert rig.opened == ["CaF MotMaster"]


@pytest.mark.parametrize("fail_go", [False, True], ids=["completed", "failed"])
def test_link_is_released_when_the_task_ends(rig, fail_go):
    # Left on, it would lock the secondary against every other task.
    rig.fakes["CaF MotMaster"].fail_go = fail_go
    run_local(Linked)
    assert rig.log[-1] == ("caf", "linked", False)


def test_an_unlinked_sequencer_is_unlinked_before_its_script_is_set(rig):
    # A MOTMaster still claimed by an earlier linked task refuses SetScriptPath.
    run_local(Configured)
    events = [event[1:] for event in rig.log if event[1] in ("linked", "script")]
    assert events[:2] == [("linked", False), ("script", "RbTweezerBasic")]


def test_declaration_rules_are_checked_at_class_definition():
    with pytest.raises(TypeError, match="linked mode"):

        class Two(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            b = MotMaster("CaF MotMaster", script="s")

    with pytest.raises(TypeError, match="Twice.tPulse and Twice.pulse"):

        class Twice(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            tPulse = MotMasterNumber(1.0)
            pulse = MotMasterNumber(1.0, parameter="tPulse")


class FakeDotNet:
    def __init__(self, fail=False):
        self.fail, self.paths = fail, []

    def SetScriptPath(self, path):
        if self.fail:
            raise RuntimeError("no such script")
        self.paths.append(path)

    def GetParameters(self):
        return {"tDelay1": 5, "tPulse": 20e-6}

    def Go(self, *args):
        raise RuntimeError("Go failed")


class FakeLinkableDotNet(FakeDotNet):
    def __init__(self):
        super().__init__()
        self.IsLinked, self.set_linked_calls, self.go_calls = False, [], 0
        self.LockedBy, self.secondary_run_in_progress = None, False

    @property
    def IsLockedByPrimary(self):
        return self.LockedBy is not None

    def SetLinked(self, value):
        self.set_linked_calls.append(value)
        self.IsLinked = value

    def TakeBackControl(self):
        if self.secondary_run_in_progress:
            return False
        self.LockedBy = None
        return True

    def Go(self, *args):
        self.go_calls += 1


def _driver(fail=False, motmaster=None):
    driver = MotMasterInterface.__new__(MotMasterInterface)
    driver.motmaster = motmaster or FakeDotNet(fail)
    driver.script_root = Path("/scripts")
    driver.script = driver.script_path = None
    driver.interval = 0
    driver.linked = False
    return driver


def test_linked_script_is_not_run_once_the_secondary_takes_back_control(dotnet_types):
    driver = _driver(motmaster=FakeLinkableDotNet())
    driver.set_motmaster_experiment("LinkedTweezerLoad")
    driver.set_linked(True)
    driver.start_motmaster_experiment()
    driver.motmaster.IsLinked = False  # what MOTMaster does on LinkLostException
    with pytest.raises(RuntimeError, match="no longer linked"):
        driver.start_motmaster_experiment()
    assert driver.motmaster.go_calls == 1


def test_linking_an_already_linked_motmaster_keeps_its_claim():
    driver = _driver(motmaster=FakeLinkableDotNet())
    driver.set_linked(True)
    driver.set_linked(True)
    assert driver.motmaster.set_linked_calls == [True]


def test_unlinking_takes_back_a_motmaster_left_claimed_as_a_secondary():
    driver = _driver(motmaster=FakeLinkableDotNet())
    driver.motmaster.LockedBy = "PH-BONESAW"
    driver.set_linked(False)
    assert not driver.motmaster.IsLockedByPrimary


def test_unlinking_refuses_a_secondary_mid_linked_run():
    driver = _driver(motmaster=FakeLinkableDotNet())
    driver.motmaster.LockedBy = "PH-BONESAW"
    driver.motmaster.secondary_run_in_progress = True
    with pytest.raises(RuntimeError, match="PH-BONESAW"):
        driver.set_linked(False)


def test_unlinking_a_motmaster_built_before_linked_mode_does_nothing():
    driver = _driver()
    driver.set_linked(False)
    assert driver.linked is False


def test_driver_raises_instead_of_printing(dotnet_types):
    driver = _driver(fail=True)
    with pytest.raises(RuntimeError, match="no such script"):
        driver.set_motmaster_experiment("missing")
    assert driver.script is None
    driver.script = "x"
    with pytest.raises(RuntimeError, match="Go failed"):
        driver.start_motmaster_experiment()


def test_script_parameters_leaves_the_selected_script_unchanged():
    driver = _driver()
    driver.script_parameters("B")
    assert driver.script is None and driver.script_path is None
    driver.set_motmaster_experiment("A")
    assert driver.script_parameters("B") == {"tDelay1": 5, "tPulse": 20e-6}
    assert driver.motmaster.paths[1:] == [
        str(Path("/scripts") / f"{script}.cs") for script in "ABA"
    ]
    assert driver.script == "A"


class FakePicomotor:
    """The Newport stage at the USB edge; records what it is asked to do."""

    def __init__(self):
        self.moves, self.velocity = [], None

    def setup_velocity(self, speed, accel):
        self.velocity = (speed, accel)

    def move_by(self, axis, steps):
        self.moves.append((axis, steps))

    def wait_move(self):
        pass


def _picomotor_driver(monkeypatch, **defaults):
    """A driver connected to a :class:`FakePicomotor` with these picomotor defaults."""
    stage = FakePicomotor()
    newport = SimpleNamespace(
        get_usb_devices_number_picomotor=lambda: 1, Picomotor8742=lambda: stage
    )
    monkeypatch.setitem(
        sys.modules, "pylablib.devices", SimpleNamespace(Newport=newport)
    )
    driver = MotMasterInterface.__new__(MotMasterInterface)
    driver.hardware_controller = None
    driver.config = {
        "dll_paths": {},
        "picomotor": {
            "connect": True,
            "motor_to_axis": {"m1": 2},
            "defaults": defaults,
        },
    }
    driver.connect()
    return driver, stage


def test_yag_spot_moves_use_the_configured_velocity(monkeypatch):
    driver, stage = _picomotor_driver(monkeypatch, speed=100, acceleration=1000)
    driver._move_picomotor_with_default_settings()
    assert stage.velocity == (100, 1000)


def test_yag_spot_moves_step_the_configured_motor(monkeypatch):
    driver, stage = _picomotor_driver(monkeypatch, motor="m1", steps=10, max_steps=100)
    driver._move_picomotor_with_default_settings()
    assert stage.moves == [(2, 10)]


def test_yag_spot_moves_travel_max_steps_then_back(monkeypatch):
    driver, stage = _picomotor_driver(monkeypatch, motor="m1", steps=10, max_steps=20)
    for _ in range(6):
        driver._move_picomotor_with_default_settings()
    steps = [step for _axis, step in stage.moves]
    assert steps == [10, 10, -10, -10, 10, 10]
    assert all(type(step) is int for step in steps)


class Int32(int):
    """Stands in for .NET's System.Int32, which only exists under pythonnet."""


class Dictionary(dict):
    def __class_getitem__(cls, types):
        return cls


@pytest.fixture
def dotnet_types(monkeypatch):
    for name, value in {
        "Int32": Int32,
        "Dictionary": Dictionary,
        "String": str,
        "Object": object,
    }.items():
        monkeypatch.setattr(motmaster_driver, name, value, raising=False)
    monkeypatch.setattr(motmaster_driver.time, "sleep", lambda _s: None)


class RecordingDotNet(FakeDotNet):
    """MOTMaster over .NET; records the parameters of every ``Go``."""

    def __init__(self):
        super().__init__()
        self.gos = []

    def Go(self, parameters=None):
        self.gos.append(dict(parameters or {}))


def test_python_values_reach_motmaster_as_int32_or_unchanged(dotnet_types):
    sent = _driver().python_to_cs_dict(
        {"a": 3, "b": np.int64(4), "c": 2.5, "d": np.float64(0.5), "e": "x"}
    )
    assert {name: type(value) for name, value in sent.items()} == {
        "a": Int32,
        "b": Int32,
        "c": float,
        "d": np.float64,
        "e": str,
    }
    assert sent == {"a": 3, "b": 4, "c": 2.5, "d": 0.5, "e": "x"}


def test_a_parameter_scan_runs_one_shot_per_value_in_order(dotnet_types):
    driver = _driver(motmaster=RecordingDotNet())
    results = driver.scan_motmaster_parameter(
        "S", "tDelay1", [30, 10, 20], callback=lambda value: 2 * value
    )
    assert driver.motmaster.gos == [{"tDelay1": 30}, {"tDelay1": 10}, {"tDelay1": 20}]
    assert results == [60, 20, 40]


def test_simulated_sequencer_serves_and_validates_parameters():
    simulated = SimulatedMotMasterInterface()
    parameters = simulated.script_parameters("RbTweezerBasic")
    assert type(parameters["tDelay1"]) is int and type(parameters["tPulse"]) is float
    simulated.set_motmaster_experiment("RbTweezerBasic")
    assert simulated.get_params() == parameters
    simulated.start_motmaster_experiment({"tDelay1": 7})
    with pytest.raises(KeyError, match="nope"):
        simulated.start_motmaster_experiment({"nope": 1})
    with pytest.raises(TypeError, match="tDelay1"):
        simulated.start_motmaster_experiment({"tDelay1": 7.5})


def test_simulated_end_to_end_with_searched_parameters():
    from pytweezer.experiments.motmaster_arguments_test import MotMasterArgumentsTest

    scan = Scan(axes=[ListAxis(argument="tDelay1", values=[10, 20])])
    measurement = run_local(
        MotMasterArgumentsTest, scan, simulate=True, coil_current=2.0
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    assert list(measurement.points["tDelay1"]) == [10, 20]
    assert list(measurement.results["sent_tDelay1"]) == [10, 20]
    assert list(measurement.results["sent_coil_current"]) == [2.0, 2.0]
    assert list(measurement.results["sent_tPulse"]) == [20e-6, 20e-6]
    assert measurement.arguments["coil_current"] == 2.0


def test_a_simulated_run_lasts_long_enough_for_simulated_cameras(monkeypatch):
    slept = []
    monkeypatch.setattr("pytweezer.drivers.motmaster.time.sleep", slept.append)
    simulated = SimulatedMotMasterInterface()
    simulated.set_motmaster_experiment("RbTweezerBasic")
    simulated.start_motmaster_experiment()
    assert sum(slept) >= SimulatedMotMasterInterface.sim_run_s > 0


def test_simulated_laser_ramp_reaches_its_target(monkeypatch):
    monkeypatch.setattr("pytweezer.drivers.motmaster.time.sleep", lambda _s: None)
    simulated = SimulatedMotMasterInterface()
    laser = next(iter(simulated.laser_set_points_wml()))
    simulated.ramp_laser_set_point_wml(laser, 400.0)
    assert simulated.laser_set_points_wml()[laser] == 400.0


def test_ramp_steps_towards_the_target_then_sets_it_exactly(monkeypatch):
    monkeypatch.setattr("pytweezer.drivers.motmaster.time.sleep", lambda _s: None)
    set_points = {"BX": 1.0}
    sent = []

    def set_slave_frequency(laser, value):
        sent.append(value)
        set_points[laser] = value

    driver = MotMasterInterface.__new__(MotMasterInterface)
    driver.wavemeter_lock = SimpleNamespace(
        getSlaveFrequency=lambda laser: set_points[laser],
        setSlaveFrequency=set_slave_frequency,
    )
    driver.ramp_laser_set_point_wml("BX", 1.0 + 3.5e-5)
    assert sent[:3] == pytest.approx([1.00001, 1.00002, 1.00003])
    assert sent[-1] == 1.0 + 3.5e-5
    assert len(sent) == 4
