import json
from datetime import timedelta
from pathlib import Path

import pytest

from pytweezer.experiment import queue as queue_module
from pytweezer.experiment.queue import (
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


def submit(queue, devices=(), **kwargs):
    return queue.submit(request(**kwargs), devices=list(devices))


def rids(tasks):
    return [task.rid for task in tasks]


def worker(rid):
    return WorkerRecord(rid=rid, token="t", pid=123, create_time=1.0)


def start(queue, rid):
    return queue.mark_started(rid, worker(rid), f"{rid}.h5")


def test_rids_are_sequential_and_order_is_priority_then_rid():
    queue = ExperimentQueue()
    a = submit(queue)
    b = submit(queue, priority=5)
    c = submit(queue)
    assert [a.rid, b.rid, c.rid] == [1, 2, 3]
    assert rids(queue.ordered()) == [2, 1, 3]
    assert rids(queue.startable()) == [2]


def test_a_pipeline_runs_one_task_at_a_time():
    queue = ExperimentQueue()
    submit(queue)
    submit(queue)
    start(queue, 1)
    assert queue.startable() == []
    with pytest.raises(QueueError, match="already running task 1"):
        start(queue, 2)
    queue.finish(1, TaskStatus.COMPLETED)
    assert rids(queue.startable()) == [2]
    assert queue.state.workers == []


def test_due_time_holds_a_task_back_and_naive_times_are_local():
    queue = ExperimentQueue()
    later = now() + timedelta(hours=1)
    task = submit(queue, due_time=later.replace(tzinfo=None))
    assert task.due_time.tzinfo is not None
    assert queue.startable() == []
    assert rids(queue.startable(later + timedelta(seconds=1))) == [task.rid]


def test_only_waiting_tasks_can_be_held_deleted_or_dropped():
    queue = ExperimentQueue()
    for _ in range(3):
        submit(queue)
    queue.hold(1)
    assert rids(queue.startable()) == [2]
    queue.release(1)
    assert rids(queue.startable()) == [1]
    queue.delete(1)
    queue.terminate(2)
    assert [t.rid for t in queue.ordered()] == [3]
    with pytest.raises(QueueError, match="no task"):
        queue.get(1)
    start(queue, 3)
    for verb in (queue.hold, queue.delete):
        with pytest.raises(QueueError, match="running"):
            verb(3)


def test_pause_resume_terminate_reach_the_worker_as_actions():
    queue = ExperimentQueue()
    submit(queue)
    start(queue, 1)
    assert queue.action_for(1) == Action.CONTINUE
    queue.pause(1)
    assert queue.action_for(1) == Action.PAUSE
    assert [t.status for t in queue.running] == [TaskStatus.RUNNING]
    queue.mark_paused(1)
    assert [t.status for t in queue.running] == [TaskStatus.PAUSED]
    queue.resume(1)
    assert queue.action_for(1) == Action.CONTINUE
    assert [t.status for t in queue.running] == [TaskStatus.RUNNING]
    queue.terminate(1)
    queue.pause(1)  # a pending terminate is not downgraded
    assert queue.action_for(1) == Action.TERMINATE
    assert queue.action_for(99) == Action.TERMINATE


def test_finish_moves_to_history_newest_first():
    queue = ExperimentQueue()
    for _ in range(3):
        submit(queue)
    for rid in (1, 2):
        start(queue, rid)
        queue.mark_progress(rid, 3, 4)
        queue.finish(rid, TaskStatus.FAILED if rid == 2 else TaskStatus.COMPLETED, "e")
    snapshot = queue.snapshot()
    assert snapshot["running"] == []
    assert [t["rid"] for t in snapshot["queue"]] == [3]
    assert [t["rid"] for t in snapshot["history"]] == [2, 1]
    assert snapshot["history"][0]["status"] == "failed"
    assert queue.get(1).points_done == 3
    assert [t.rid for t in queue.history(before_rid=2)] == [1]
    with pytest.raises(QueueError, match="not a final status"):
        queue.finish(3, TaskStatus.RUNNING)


def test_store_round_trip_and_unreadable_file(tmp_path):
    path = tmp_path / "state.json"
    store = QueueStore(path)
    assert store.load().next_rid == 1
    queue = ExperimentQueue()
    submit(queue, due_time=now())
    submit(queue, ["X"], pipeline="CaF")
    start(queue, 2)
    store.save(queue.state)
    again = ExperimentQueue(store.load())
    assert again.state == queue.state
    assert rids(again.running) == [2]

    path.write_text("{not json")
    assert store.load().next_rid == 1
    assert not path.exists()
    assert len(list(tmp_path.glob("state.unreadable-*.json"))) == 1


def test_pipelines_run_side_by_side_but_each_runs_one_task():
    q = ExperimentQueue()
    q.submit(request(pipeline="CaF"), devices=["CaF MotMaster"])
    q.submit(request(pipeline="CaF"), devices=["Rb HamCam"])
    q.submit(request(pipeline="Rb"), devices=["Rb MotMaster"])
    assert rids(q.startable()) == [1, 3]


def test_a_shared_device_blocks_across_pipelines_and_says_why():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    blocked = submit(q, ["X", "Y"], pipeline="B")
    start(q, 1)
    assert q.startable() == []
    assert q.waiting_for(blocked) == {"X": 1}
    assert q.snapshot()["queue"][0]["waiting_for"] == {"X": 1}


def test_a_task_waiting_behind_its_own_pipeline_has_no_waiting_for():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    behind = submit(q, ["X"], pipeline="A")
    start(q, 1)
    assert q.waiting_for(behind) == {}
    assert q.snapshot()["queue"][0]["waiting_for"] == {}


def test_lower_priority_in_another_pipeline_overtakes_a_blocked_task():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    start(q, 1)
    submit(q, ["X"], pipeline="B", priority=5)
    submit(q, ["Y"], pipeline="C", priority=0)
    assert rids(q.startable()) == [3]


def test_a_blocked_head_is_not_overtaken_within_its_pipeline():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    start(q, 1)
    submit(q, ["X"], pipeline="B", priority=5)
    submit(q, ["Y"], pipeline="B")
    assert q.startable() == []


def test_one_tick_never_double_books_a_device():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A", priority=0)
    submit(q, ["X"], pipeline="B", priority=3)
    assert rids(q.startable()) == [2]


def test_a_paused_task_keeps_its_devices():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    submit(q, ["X"], pipeline="B")
    start(q, 1)
    q.pause(1)
    q.mark_paused(1)
    assert q.get(1).status == TaskStatus.PAUSED
    assert q.startable() == []


def test_raising_priority_changes_who_gets_a_freed_device():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    start(q, 1)
    submit(q, ["X"], pipeline="B")
    submit(q, ["X"], pipeline="C")
    q.set_priority(3, 9)
    q.finish(1, TaskStatus.COMPLETED)
    assert rids(q.startable()) == [3]


def test_unresolved_devices_never_start_until_set():
    q = ExperimentQueue()
    task = q.submit(request())
    assert task.devices is None
    assert q.startable() == []
    q.set_devices(task.rid, ["X"])
    assert rids(q.startable()) == [task.rid]
    start(q, task.rid)
    with pytest.raises(QueueError, match="running"):
        q.set_devices(task.rid, ["Y"])


def test_a_running_task_with_unknown_devices_blocks_everything():
    q = ExperimentQueue()
    q.submit(request(pipeline="A")).status = TaskStatus.RUNNING
    head = submit(q, ["Y"], pipeline="B")
    assert q.startable() == []
    assert q.waiting_for(head) == {"(unknown)": 1}
    with pytest.raises(QueueError, match="held"):
        start(q, head.rid)


def test_mark_started_refuses_a_clash_or_a_busy_pipeline():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    submit(q, ["Y"], pipeline="A")
    submit(q, ["X", "Z"], pipeline="B")
    q.submit(request(pipeline="C"))
    start(q, 1)
    with pytest.raises(QueueError, match="pipeline 'A' is already running task 1"):
        start(q, 2)
    with pytest.raises(QueueError, match="'X' held by task 1"):
        start(q, 3)
    with pytest.raises(QueueError, match="not resolved"):
        start(q, 4)
    assert rids(q.running) == [1]
    assert rids(q.state.workers) == [1]


def test_pipeline_name_is_stripped_and_defaults_to_main():
    assert request().pipeline == "main"
    assert request(pipeline="  ").pipeline == "main"
    assert request(pipeline=" CaF ").pipeline == "CaF"


@pytest.mark.parametrize("had_worker", [True, False])
def test_old_state_file_migrates_worker_and_pipeline(tmp_path, had_worker):
    record = {"rid": 1, "token": "t", "pid": 123, "create_time": 1.0}
    task = {
        "schema_version": 1,
        "experiment": "pytweezer.experiments.demo",
        "class_name": "RabiDemo",
        "rid": 1,
        "status": "running",
        "t_submit": "2026-10-01T12:00:00+01:00",
    }
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "next_rid": 2,
                "active": [task],
                "worker": record if had_worker else None,
            }
        )
    )
    state = QueueStore(path).load()
    assert state.schema_version == 2
    assert state.workers == ([WorkerRecord(**record)] if had_worker else [])
    [loaded] = state.active
    assert (loaded.pipeline, loaded.devices) == ("main", None)


def test_finish_releases_only_that_tasks_worker():
    q = ExperimentQueue()
    submit(q, ["X"], pipeline="A")
    submit(q, ["Y"], pipeline="B")
    start(q, 1)
    start(q, 2)
    q.pause(1)
    assert (q.action_for(1), q.action_for(2)) == (Action.PAUSE, Action.CONTINUE)
    q.finish(1, TaskStatus.ABORTED)
    assert rids(q.running) == [2]
    assert (q.worker(1), q.worker(2)) == (None, worker(2))
    assert q.action_for(1) == Action.TERMINATE
    assert [t["rid"] for t in q.snapshot()["running"]] == [2]


def _saved_queue_failing_reads(tmp_path, monkeypatch, failures):
    """A store holding a saved queue whose next ``failures`` reads raise PermissionError."""
    path = tmp_path / "state.json"
    store = QueueStore(path)
    queue = ExperimentQueue()
    queue.submit(request())
    store.save(queue.state)
    read_text = Path.read_text
    remaining = [failures]

    def flaky_read_text(self, *args, **kwargs):
        if self == path and remaining[0] > 0:
            remaining[0] -= 1
            raise PermissionError("file held by a backup tool")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    return store, queue.state


def test_a_transient_read_error_still_loads_the_saved_queue(tmp_path, monkeypatch):
    store, saved = _saved_queue_failing_reads(tmp_path, monkeypatch, failures=1)
    assert store.load() == saved


def test_a_persistent_read_error_raises_and_leaves_the_file(tmp_path, monkeypatch):
    store, _ = _saved_queue_failing_reads(tmp_path, monkeypatch, failures=10**6)
    monkeypatch.setattr(queue_module.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        store.load()
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
