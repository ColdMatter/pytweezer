import time

import pytest

from pytweezer.drivers import ni_cards_sources
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


class FakeController:
    def __init__(self, is_running=lambda: False, report=None):
        self._is_running = is_running
        self._report = report

    @property
    def IsRunning(self):
        return self._is_running()

    def GetLastRunOutputs(self):
        return self._report


class NoLastRunController:
    IsRunning = True


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


def test_unreachable_motmaster_reads_as_none(monkeypatch):
    status = status_with(monkeypatch, error=ConnectionRefusedError())
    assert status.is_running() is None
    assert status.last_run() is None


def test_dotnet_connection_failure_drops_the_proxy_and_reconnects(monkeypatch):
    class SocketException(Exception):
        pass

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise SocketException("No connection could be made")
        return True

    status = status_with(monkeypatch, FakeController(is_running=flaky))
    assert status.is_running() is None
    assert status.is_running() is True


def test_running_devices_maps_reserved_devices_to_rid(monkeypatch):
    class StubMirror:
        connected = True

        def __init__(self, *args, **kwargs):
            pass

        def snapshot(self):
            return {
                "running": [
                    {"rid": 42, "devices": ["CaF MotMaster", "Rb MotMaster"]},
                    {"rid": 43, "devices": ["CaF ThorCam"]},
                    {"rid": 44, "devices": None},
                ]
            }

        def close(self):
            pass

    monkeypatch.setattr(ni_cards_sources, "SyncMirror", StubMirror)
    queue = ManagerQueue()
    assert queue.running_devices() == {
        "CaF MotMaster": 42,
        "Rb MotMaster": 42,
        "CaF ThorCam": 43,
    }
    StubMirror.connected = False
    assert queue.running_devices() is None
    queue.close()
