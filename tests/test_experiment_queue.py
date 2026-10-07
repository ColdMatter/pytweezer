from datetime import timedelta

import pytest

from pytweezer.experiment.queue import (
    HISTORY_LIMIT,
    ExperimentQueue,
    QueueError,
    QueueStore,
    WorkerRecord,
    now,
)
from pytweezer.experiment.task import Action, TaskRequest, TaskStatus


def request(**kwargs):
    return TaskRequest(
        experiment="pytweezer.experiments.demo", class_name="RabiDemo", **kwargs
    )


def worker(rid):
    return WorkerRecord(rid=rid, token="t", pid=123, create_time=1.0)


def start(queue, rid):
    return queue.mark_started(rid, worker(rid), f"{rid}.h5")


def test_rids_are_sequential_and_order_is_priority_then_rid():
    queue = ExperimentQueue()
    a = queue.submit(request())
    b = queue.submit(request(priority=5))
    c = queue.submit(request())
    assert [a.rid, b.rid, c.rid] == [1, 2, 3]
    assert [t.rid for t in queue.ordered()] == [2, 1, 3]
    assert queue.next_due().rid == 2


def test_only_one_task_runs_at_a_time():
    queue = ExperimentQueue()
    queue.submit(request())
    queue.submit(request())
    start(queue, 1)
    assert queue.next_due() is None
    with pytest.raises(QueueError, match="already running"):
        start(queue, 2)
    queue.finish(1, TaskStatus.COMPLETED)
    assert queue.next_due().rid == 2
    assert queue.state.worker is None


def test_due_time_holds_a_task_back_and_naive_times_are_local():
    queue = ExperimentQueue()
    later = now() + timedelta(hours=1)
    task = queue.submit(request(due_time=later.replace(tzinfo=None)))
    assert task.due_time.tzinfo is not None
    assert queue.next_due() is None
    assert queue.next_due(later + timedelta(seconds=1)).rid == task.rid


def test_hold_release_delete():
    queue = ExperimentQueue()
    queue.submit(request())
    queue.hold(1)
    assert queue.next_due() is None
    queue.release(1)
    assert queue.next_due().rid == 1
    queue.delete(1)
    assert queue.ordered() == []
    with pytest.raises(QueueError, match="no task"):
        queue.get(1)


def test_running_task_cannot_be_held_or_deleted():
    queue = ExperimentQueue()
    queue.submit(request())
    start(queue, 1)
    for verb in (queue.hold, queue.delete):
        with pytest.raises(QueueError, match="running"):
            verb(1)


def test_pause_resume_terminate_reach_the_worker_as_actions():
    queue = ExperimentQueue()
    queue.submit(request())
    start(queue, 1)
    assert queue.action_for(1) == Action.CONTINUE
    queue.pause(1)
    assert queue.action_for(1) == Action.PAUSE
    assert queue.running.status == TaskStatus.RUNNING
    queue.mark_paused(1)
    assert queue.running.status == TaskStatus.PAUSED
    queue.resume(1)
    assert queue.action_for(1) == Action.CONTINUE
    assert queue.running.status == TaskStatus.RUNNING
    queue.terminate(1)
    queue.pause(1)  # a pending terminate is not downgraded
    assert queue.action_for(1) == Action.TERMINATE
    assert queue.action_for(99) == Action.TERMINATE


def test_terminate_drops_a_waiting_task():
    queue = ExperimentQueue()
    queue.submit(request())
    queue.terminate(1)
    assert queue.ordered() == []


def test_finish_moves_to_history_newest_first():
    queue = ExperimentQueue()
    for _ in range(3):
        queue.submit(request())
    for rid in (1, 2):
        start(queue, rid)
        queue.mark_progress(rid, 3, 4)
        queue.finish(rid, TaskStatus.FAILED if rid == 2 else TaskStatus.COMPLETED, "e")
    snapshot = queue.snapshot()
    assert snapshot["running"] is None
    assert [t["rid"] for t in snapshot["queue"]] == [3]
    assert [t["rid"] for t in snapshot["history"]] == [2, 1]
    assert snapshot["history"][0]["status"] == "failed"
    assert queue.get(1).points_done == 3
    assert [t.rid for t in queue.history(before_rid=2)] == [1]
    with pytest.raises(QueueError, match="not a final status"):
        queue.finish(3, TaskStatus.RUNNING)


def test_history_is_capped():
    queue = ExperimentQueue()
    for _ in range(HISTORY_LIMIT + 5):
        task = queue.submit(request())
        queue.finish(task.rid, TaskStatus.TERMINATED)
    assert len(queue.state.history) == HISTORY_LIMIT
    assert queue.state.history[0].rid == 6


def test_last_request_is_remembered_per_experiment():
    queue = ExperimentQueue()
    queue.submit(request(args={"atoms": 5}))
    queue.submit(request(args={"atoms": 7}))
    assert queue.state.last_requests["pytweezer.experiments.demo:RabiDemo"].args == {
        "atoms": 7
    }


def test_store_round_trip(tmp_path):
    store = QueueStore(tmp_path / "state.json")
    assert store.load().next_rid == 1
    queue = ExperimentQueue()
    queue.submit(request(due_time=now()))
    queue.submit(request())
    start(queue, 2)
    store.save(queue.state)
    again = ExperimentQueue(store.load())
    assert again.state == queue.state
    assert again.running.rid == 2
    assert again.state.worker.pid == 123


def test_unreadable_store_is_set_aside(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    assert QueueStore(path).load().next_rid == 1
    assert not path.exists()
    assert len(list(tmp_path.glob("state.unreadable-*.json"))) == 1
