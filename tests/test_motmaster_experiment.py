import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    def __init__(self, log, name, params):
        self.log, self.name, self.params = log, name, params
        self.fail_go, self.go_delay = False, 0.0

    def _ignore(self, *args):
        pass

    set_run_until_stopped = set_iterations = set_save_toggle = close_rpc = _ignore

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
    log, opened, unreachable = [], [], set()
    rb_params = {
        "tDelay1": 5,
        "tPulse": 20e-6,
        "nShots": 1,
        "bField": 0.5,
        "label": "x",
    }
    fakes = {
        "Rb MotMaster": FakeSequencer(log, "rb", rb_params),
        "CaF MotMaster": FakeSequencer(log, "caf", {"tLoad": 100, "bTop": 0.5}),
        "K MotMaster": FakeSequencer(log, "k", {}),
    }

    def get_device(name, timeout):
        opened.append(name)
        # The first open is the task's own client; later ones are fresh follower clients.
        if name in unreachable and opened.count(name) > 1:
            raise ConnectionError(f"{name} unreachable")
        return fakes[name]

    monkeypatch.setattr("pytweezer.experiment.experiment._get_device", get_device)
    monkeypatch.setattr(MotMasterExperiment, "follower_arm_delay", 0.0)
    return SimpleNamespace(log=log, fakes=fakes, opened=opened, unreachable=unreachable)


def goes(rig, name):
    return [event[2] for event in rig.log if event[0] == name and event[1] == "go"]


class Single(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")
    tof = MotMasterInteger(100, parameter="tDelay1")
    pulse = MotMasterNumber(1e-6, parameter="tPulse")


class Pair(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
    caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad", follower_timeout=0.1)
    tof = MotMasterInteger(100, device="rb", parameter="tDelay1")


class Linked(MotMasterExperiment):
    caf = MotMaster("CaF MotMaster", script="LinkedTweezerLoad", linked=True)
    load = MotMasterInteger(100, parameter="tLoad")


class Trio(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
    caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad")
    k = MotMaster("K MotMaster", script="KTweezer")


def test_declared_and_searched_parameters_are_sent_typed(rig):
    scan = Scan(axes=[ListAxis(argument="rb.nShots", values=[1, 2])])
    measurement = run_local(Single, scan, **{"rb.bField": 3, "rb.label": 3})
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
        "label": 3.0,
    }
    assert list(measurement.points["rb.nShots"]) == [1, 2]
    assert measurement.arguments["rb.bField"] == 3.0


@pytest.mark.parametrize(
    "fixed, scanned, expected",
    [
        ({"rb.nShots": 1.5}, None, ["nShots", "whole number"]),
        ({"rb.tDelayy": 1}, None, ["tDelayy", "tDelay1"]),
        ({}, "rb.tDelayy", ["tDelayy"]),
    ],
    ids=["fraction-for-int32", "unknown-with-suggestion", "unknown-only-scanned"],
)
def test_bad_script_parameter_fails_before_any_sequence_runs(
    rig, fixed, scanned, expected
):
    scan = Scan(axes=[ListAxis(argument=scanned, values=[1, 2])] if scanned else [])
    measurement = run_local(Single, scan, **fixed)
    assert measurement.status == "failed"
    assert all(text in measurement.attrs["error"] for text in expected)
    assert goes(rig, "rb") == []


@pytest.mark.parametrize("name", ["xx.tDelay1", "rb.", "rb.tDelay1"])
def test_dotted_names_that_are_not_searchable_are_rejected_at_submit(name):
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Single, {name: 1})
    with pytest.raises(ValueError, match="no argument"):
        Scan(axes=[ListAxis(argument=name, values=[1])]).points(Single)


def test_master_is_untriggered_and_followers_start_first(rig, monkeypatch):
    monkeypatch.setattr(MotMasterExperiment, "follower_arm_delay", 0.05)
    run_local(Pair, **{"caf.bTop": 0.7})
    assert ("rb", "triggered", False) in rig.log
    assert ("caf", "triggered", True) in rig.log
    assert [event[0] for event in rig.log if event[1] == "start"] == ["caf", "rb"]
    # Only parameters that were chosen are sent; the rest keep the script's defaults.
    assert goes(rig, "caf") == [{"bTop": 0.7}]
    assert goes(rig, "rb") == [{"tDelay1": 100}]


def test_linked_sequencer_is_linked_untriggered_and_run_alone(rig):
    run_local(Linked, **{"caf.bTop": 0.7})
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


def test_every_sequencer_is_unlinked_before_any_script_is_set(rig):
    # A secondary still claimed by an earlier linked task refuses SetScriptPath.
    run_local(Pair)
    events = [event[:2] for event in rig.log if event[1] in ("linked", "script")]
    assert events[:2] == [("rb", "linked"), ("caf", "linked")]
    assert ("rb", "linked", False) in rig.log and ("caf", "linked", False) in rig.log


def test_follower_client_is_opened_once_per_task(rig):
    scan = Scan(axes=[ListAxis(argument="tof", values=[1, 2, 3])])
    measurement = run_local(Pair, scan)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert rig.opened.count("CaF MotMaster") == 2


def test_failure_opening_a_follower_client_starts_no_follower(rig):
    rig.unreachable.add("K MotMaster")
    measurement = run_local(Trio)
    assert measurement.status == "failed"
    assert "K MotMaster unreachable" in measurement.attrs["error"]
    assert goes(rig, "caf") == []


@pytest.mark.parametrize("device", ["Rb MotMaster", "CaF MotMaster"])
def test_master_or_follower_failure_fails_the_task(rig, device):
    rig.fakes[device].fail_go = True
    measurement = run_local(Pair)
    assert measurement.status == "failed"
    assert f"{rig.fakes[device].name} refused" in measurement.attrs["error"]


def test_follower_that_never_finishes_fails_after_its_timeout(rig):
    rig.fakes["CaF MotMaster"].go_delay = 1.0
    started = time.monotonic()
    measurement = run_local(Pair)
    assert time.monotonic() - started < 0.9
    assert measurement.status == "failed"
    error = measurement.attrs["error"]
    assert "did not finish within 0.1 s" in error
    assert "Restart the 'CaF MotMaster' device server" in error


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

    with pytest.raises(TypeError, match="Twice.tPulse and Twice.pulse"):

        class Twice(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            tPulse = MotMasterNumber(1.0)
            pulse = MotMasterNumber(1.0, parameter="tPulse")

    with pytest.raises(TypeError, match="a and b both use 'Rb MotMaster'"):

        class SharedDevice(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s", master=True)
            b = MotMaster("Rb MotMaster", script="t")

    with pytest.raises(TypeError, match="declare no other MotMaster"):

        class LinkedPlusFollower(MotMasterExperiment):
            a = MotMaster("CaF MotMaster", script="s", master=True, linked=True)
            b = MotMaster("Rb MotMaster", script="t")

    with pytest.raises(TypeError, match="'nope'"):

        class UnknownDevice(MotMasterExperiment):
            a = MotMaster("Rb MotMaster", script="s")
            x = MotMasterNumber(1.0, device="nope")


def test_schema_describes_the_sequencers_and_parameters():
    schema = Pair.schema()
    assert schema["devices"]["caf"]["motmaster"]["script"] == "CaFTweezerLoad"
    assert schema["devices"]["rb"]["motmaster"]["master"] is True
    assert schema["arguments"]["tof"]["motmaster_device"] == "rb"
    assert schema["arguments"]["tof"]["motmaster_parameter"] == "tDelay1"
    assert Single.master_name() == "rb"


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

    def SetLinked(self, value):
        self.set_linked_calls.append(value)
        self.IsLinked = value

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


def test_linked_script_is_not_run_once_the_secondary_takes_back_control():
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


def test_unlinking_a_motmaster_built_before_linked_mode_does_nothing():
    driver = _driver()
    driver.set_linked(False)
    assert driver.linked is False


def test_driver_raises_instead_of_printing():
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
        "/scripts/A.cs",
        "/scripts/B.cs",
        "/scripts/A.cs",
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


class FakeCamera:
    """Hands out one-pixel frames with the given counts, in order of frame number."""

    def __init__(self, counts):
        self.counts = counts

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def acquire_n_frames(self, n, start_frame=0):
        return [[[count]] for count in self.counts[start_frame : start_frame + n]]


@pytest.mark.parametrize(
    "shot_counts, error", [([2.0], float("nan")), ([0.0, 2.0], 1.0)], ids=["1", "2"]
)
def test_auto_mot_error_bars_are_standard_errors_of_the_shots(
    monkeypatch, shot_counts, error
):
    def run_each(*args):
        values, callback = args[2], args[4]
        return [callback(value) for value in values]

    # The two scan loops drive MOTMaster over .NET; each calls back once per point.
    driver = MotMasterInterface.__new__(MotMasterInterface)
    driver.config = {"lasers": {"v00": "cavity"}}
    driver.wavemeter_lock = SimpleNamespace(getSlaveFrequency=lambda laser: 1.0)
    driver.scan_motmaster_parameter = run_each
    driver.scan_wm_laser_set_points = run_each
    monkeypatch.setattr("pytweezer.drivers.motmaster.time.sleep", lambda _s: None)

    n = len(shot_counts)
    # The background shot, the one scan point, then parking on the best point.
    camera = FakeCamera([0.0] * n + shot_counts + [0.0] * n)
    results = driver.auto_mot({"v00": [1.0]}, camera, shots_per_point=n)
    assert results["v00"]["errors"] == [pytest.approx(error, nan_ok=True)]


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

    scan = Scan(axes=[ListAxis(argument="rb.tDelay1", values=[10, 20])])
    measurement = run_local(
        MotMasterArgumentsTest, scan, simulate=True, **{"rb.coil_current": 2.0}
    )
    assert measurement.status == "completed", measurement.attrs["error"]
    assert list(measurement.points["rb.tDelay1"]) == [10, 20]
    assert list(measurement.results["sent_tDelay1"]) == [10, 20]
    assert list(measurement.results["sent_coil_current"]) == [2.0, 2.0]
    assert list(measurement.results["sent_tPulse"]) == [20e-6, 20e-6]
    assert measurement.arguments["rb.coil_current"] == 2.0


def test_a_simulated_run_lasts_long_enough_for_simulated_cameras(monkeypatch):
    slept = []
    monkeypatch.setattr("pytweezer.drivers.motmaster.time.sleep", slept.append)
    simulated = SimulatedMotMasterInterface()
    simulated.set_motmaster_experiment("RbTweezerBasic")
    simulated.start_motmaster_experiment()
    assert sum(slept) >= SimulatedMotMasterInterface.sim_run_s > 0
