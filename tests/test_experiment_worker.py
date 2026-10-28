"""The worker's link to the Experiment Manager, with the manager's socket faked."""

from pytweezer.experiment import worker
from pytweezer.experiment.client import ManagerUnavailable
from pytweezer.experiment.runner import Progress
from pytweezer.experiment.task import Action


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
