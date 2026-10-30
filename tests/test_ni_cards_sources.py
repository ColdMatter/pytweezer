import json
import pathlib
import sys
import time
import types

import pytest

from pytweezer.drivers import motmaster, ni_cards_sources
from pytweezer.drivers.ni_cards_sources import (
    LastRun,
    ManagerQueue,
    MotMasterNotAnswering,
    MotMasterTooOld,
    RemotingMotMasterStatus,
    last_run_from_report,
)

FULL_REPORT = {
    "RunNumber": 7,
    "Script": "LoadTweezer",
    "Finished": 1_761_746_520,
    "OutputsKnown": True,
    "Values": {"tweezerAmplitude": 1, "shutter": True},
}


class _Reflected:
    """The reflection path to a .NET property: ``GetType().GetProperty(name).GetValue``."""

    def __init__(self, read):
        self._read = read

    def GetType(self):
        return self

    def GetProperty(self, name):
        assert name == "IsRunning"
        return self

    def GetValue(self, proxy, index):
        try:
            return self._read()
        except Exception as error:
            wrapper = type("TargetInvocationException", (Exception,), {})()
            wrapper.InnerException = error
            raise wrapper from error


class FakeController:
    """A remoting proxy whose ``IsRunning`` may only be read by reflection."""

    def __init__(self, is_running=lambda: False, report=None):
        self._reflected = _Reflected(is_running)
        self._report = report

    def GetType(self):
        return self._reflected.GetType()

    @property
    def IsRunning(self):
        raise AssertionError("a property read holds the GIL under pythonnet")

    def GetLastRunOutputs(self):
        return self._report


class NoLastRunController:
    def GetType(self):
        return _Reflected(lambda: True)


def status_with(monkeypatch, controller=None, error=None, timeout_s=2.0):
    def factory(config_file):
        if error is not None:
            raise error
        return controller

    monkeypatch.setattr(ni_cards_sources, "motmaster_controller", factory)
    return RemotingMotMasterStatus("CaF MotMaster", timeout_s=timeout_s)


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (None, None),
        (
            FULL_REPORT,
            LastRun(
                7,
                "LoadTweezer",
                1_761_746_520.0,
                True,
                {"tweezerAmplitude": 1.0, "shutter": True},
            ),
        ),
        (
            {**FULL_REPORT, "OutputsKnown": False, "Values": {}},
            LastRun(7, "LoadTweezer", 1_761_746_520.0, False, {}),
        ),
    ],
)
def test_last_run_from_report_converts_the_contract(report, expected):
    result = last_run_from_report(report)
    assert result == expected
    if result is not None:
        assert {k: type(v) for k, v in result.values.items()} == {
            k: type(v) for k, v in expected.values.items()
        }


def test_status_call_that_never_returns_raises_not_answering(monkeypatch):
    status = status_with(
        monkeypatch, FakeController(is_running=lambda: time.sleep(5)), timeout_s=0.1
    )
    start = time.monotonic()
    with pytest.raises(MotMasterNotAnswering):
        status.is_running()
    assert time.monotonic() - start < 1


def test_build_without_last_run_method_raises_too_old(monkeypatch):
    status = status_with(monkeypatch, NoLastRunController())
    with pytest.raises(MotMasterTooOld):
        status.last_run()
    assert status.is_running() is True


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError(), NameError("clr"), KeyError("motmaster"), ValueError()],
)
def test_unreachable_motmaster_reads_as_none_and_warns_once(monkeypatch, error):
    warnings = []
    monkeypatch.setattr(
        ni_cards_sources.logger, "warning", lambda *args: warnings.append(args)
    )
    status = status_with(monkeypatch, error=error)
    assert status.is_running() is None
    assert status.last_run() is None
    assert len(warnings) == 1
    assert status.setup_error


@pytest.mark.parametrize(
    "name", ["SocketException", "RemotingException", "IOException"]
)
def test_dotnet_connection_failure_drops_the_proxy_and_reconnects(monkeypatch, name):
    error_class = type(name, (Exception,), {})
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise error_class("connection broke")
        return True

    status = status_with(monkeypatch, FakeController(is_running=flaky))
    assert status.is_running() is None
    assert not status.setup_error
    assert status.is_running() is True


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "running": [
                    {"rid": 42, "devices": ["CaF MotMaster", "Rb MotMaster"]},
                    {"rid": 43, "devices": ["CaF ThorCam"]},
                    {"rid": 44, "devices": None},
                ]
            },
            {"CaF MotMaster": 42, "Rb MotMaster": 42, "CaF ThorCam": 43},
        ),
        ({"running": {"rid": 5, "devices": ["CaF MotMaster"]}}, {"CaF MotMaster": 5}),
        ({"running": None}, {}),
        ({}, {}),
    ],
    ids=["list", "older-manager-dict", "older-manager-none", "key-missing"],
)
def test_running_devices_maps_reserved_devices_to_rid(monkeypatch, state, expected):
    class StubMirror:
        connected = True

        def __init__(self, *args, **kwargs):
            pass

        def read(self, fn):
            return fn(state)

        def close(self):
            pass

    monkeypatch.setattr(ni_cards_sources, "SyncMirror", StubMirror)
    queue = ManagerQueue()
    assert queue.running_devices() == expected
    StubMirror.connected = False
    assert queue.running_devices() is None
    queue.close()


def test_motmaster_controller_adds_references_once_per_process(monkeypatch, tmp_path):
    (tmp_path / "mm.json").write_text(
        json.dumps(
            {
                "dll_paths": {"a": "C:/mm/A.dll"},
                "motmaster": {
                    "exe_path": "C:/mm/MOTMaster.exe",
                    "remote_path": "tcp://x",
                },
            }
        )
    )
    added = []
    fake_clr = types.SimpleNamespace(AddReference=added.append)
    fake_activator = types.SimpleNamespace(GetObject=lambda cls, url: (cls, url))
    monkeypatch.setattr(motmaster, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(motmaster, "_referenced", set())
    monkeypatch.setattr(motmaster, "clr", fake_clr, raising=False)
    monkeypatch.setattr(motmaster, "Activator", fake_activator, raising=False)
    monkeypatch.setitem(
        sys.modules, "MOTMaster", types.SimpleNamespace(Controller="Controller")
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert motmaster.motmaster_controller("mm.json") == ("Controller", "tcp://x")
    path_length = len(sys.path)
    motmaster.motmaster_controller("mm.json")

    assert added == ["C:/mm/A.dll", "C:/mm/MOTMaster.exe"]
    assert len(sys.path) == path_length


def test_missing_dll_does_not_grow_sys_path_on_every_attempt(monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(motmaster, "_referenced", set())
    monkeypatch.setattr(
        motmaster, "clr", types.SimpleNamespace(AddReference=missing), raising=False
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    for _ in range(3):
        with pytest.raises(FileNotFoundError):
            motmaster._add_reference("C:/mm/Missing.dll")

    assert sys.path.count(str(pathlib.Path("C:/mm"))) == 1
