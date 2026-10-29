"""The worker and its link to the Experiment Manager, with the manager's socket faked."""

from datetime import datetime

import pytest

from pytweezer.experiment import worker
from pytweezer.experiment.client import ManagerUnavailable
from pytweezer.experiment.experiment import Experiment
from pytweezer.experiment.runner import Progress
from pytweezer.experiment.task import Action, Task
from pytweezer.servers import device_client


class FakeManager:
    """Answers every request with ``reply``, or is unreachable when it is None."""

    def __init__(self, reply):
        self.reply = reply

    def request(self, payload):
        if self.reply is None:
            raise ManagerUnavailable("no reply")
        return self.reply


def control(reply, monkeypatch):
    monkeypatch.setattr(worker, "RETRY_INTERVAL_S", 0.01)
    link = worker.ManagerLink(FakeManager(reply), rid=1, token="t", orphan_timeout=0.05)
    return link.control(Progress(1, 3))


def test_a_manager_unreachable_past_the_orphan_timeout_interrupts_the_run(
    monkeypatch,
):
    assert control(None, monkeypatch) == Action.INTERRUPT


def test_an_error_reply_terminates_the_run(monkeypatch):
    reply = {"ok": False, "error": "ValueError: unknown worker event"}
    assert control(reply, monkeypatch) == Action.TERMINATE


class OpensADevice(Experiment):
    """Reaches a device directly, bypassing its declarations."""

    def run_point(self):
        device_client.get_device("Rb MotMaster")


class StartedManager:
    """Hands out one task, then records every event the worker sends."""

    def __init__(self, task, h5_path):
        self.task, self.h5_path = task, h5_path
        self.sent = []

    def request(self, payload):
        self.sent.append(payload)
        if payload["event"] == "started":
            return {
                "action": "continue",
                "task": self.task.model_dump(mode="json"),
                "h5_path": str(self.h5_path),
            }
        return {"action": "continue"}


@pytest.mark.parametrize(
    "devices, error",
    [([], "UndeclaredDeviceError"), (None, "task has no device reservation")],
)
def test_worker_restricts_devices_to_the_reservation(
    tmp_path, monkeypatch, devices, error
):
    monkeypatch.setattr(device_client, "_allowed_devices", None)
    task = Task(
        experiment=OpensADevice.__module__,
        class_name=OpensADevice.__qualname__,
        rid=7,
        t_submit=datetime.now().astimezone(),
        devices=devices,
    )
    manager = StartedManager(task, tmp_path / "7.h5")
    worker.run(worker.ManagerLink(manager, rid=7, token="t"))
    finished = manager.sent[-1]
    assert (finished["event"], finished["status"]) == ("finished", "failed")
    assert error in finished["error"]
