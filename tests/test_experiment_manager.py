"""Experiment Manager request handling and restart reconciliation, mostly without sockets."""

import json
import threading
import time

import h5py
import psutil
import pytest
import zmq

from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    delete_recipe,
    recipes,
    save_recipe,
    submit,
    submit_recipe,
)
from pytweezer.experiment.queue import WorkerRecord
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest
from pytweezer.experiments.demo import RabiDemo
from pytweezer.servers import experiment_manager as em

DEMO = "pytweezer.experiments.demo"
DEAD_PID = 2**22 + 12345
PULSE_SCAN = Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=1e-5, n=3)])
BROKEN = {
    "module": "pytweezer.experiments.broken",
    "classes": [],
    "warnings": [],
    "error": "Traceback (most recent call last):\n  ...\nSyntaxError: invalid syntax\n",
}


class StaticCatalogue:
    """A catalogue whose entries are given, never read from files."""

    def __init__(self, entries):
        self._entries = entries
        self.busy = False

    def refresh(self):
        return False

    def poll(self):
        return False

    def entries(self):
        return self._entries

    def close(self):
        pass


@pytest.fixture
def make_manager(tmp_path, recording_db):
    demo = {
        "module": DEMO,
        "classes": [RabiDemo.schema()],
        "warnings": [],
        "error": None,
    }

    def make():
        return em.ExperimentManager(
            root=tmp_path,
            catalogue=StaticCatalogue([demo, BROKEN]),
            bind=False,
            db=recording_db,
        )

    return make


def request(**kwargs):
    return TaskRequest(experiment=DEMO, class_name="RabiDemo", **kwargs).model_dump(
        mode="json"
    )


def test_submit_and_query(make_manager):
    manager = make_manager()
    reply = manager.handle({"command": "submit", "request": request(label="a")})
    assert reply == {"ok": True, "rid": 1}
    task = manager.handle({"command": "get", "rid": 1})["task"]
    assert task["label"] == "a" and task["status"] == "queued"
    last = manager.handle(
        {"command": "last_request", "experiment": DEMO, "class_name": "RabiDemo"}
    )
    assert last["request"]["label"] == "a"
    snapshot = manager.handle({"command": "snapshot"})["snapshot"]
    assert [t["rid"] for t in snapshot["queue"]] == [1]


def test_errors_come_back_as_replies(make_manager):
    manager = make_manager()
    reply = manager.handle({"command": "nope"})
    assert not reply["ok"] and "unknown command" in reply["error"]
    reply = manager.handle({"command": "pause", "rid": 5})
    assert not reply["ok"] and "not running" in reply["error"]
    reply = manager.handle({"command": "submit", "request": {"experiment": "x"}})
    assert not reply["ok"] and "class_name" in reply["error"]


def test_state_survives_a_restart_and_rids_never_repeat(make_manager, tmp_path):
    make_manager().handle({"command": "submit", "request": request()})
    (tmp_path / "2026" / "01" / "01").mkdir(parents=True)
    (tmp_path / "2026" / "01" / "01" / "000040_RabiDemo.h5").touch()
    again = make_manager()
    assert [t.rid for t in again.queue.ordered()] == [1]
    assert again.handle({"command": "submit", "request": request()})["rid"] == 41


def _running_task(manager, tmp_path, pid=0, create_time=0.0, status="running"):
    manager.handle({"command": "submit", "request": request()})
    relpath = "2026/01/01/000001_RabiDemo.h5"
    record = WorkerRecord(rid=1, token="tok", pid=pid, create_time=create_time)
    manager.queue.mark_started(1, record, relpath)
    path = tmp_path / relpath
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(status=status, error="from file" if status == "failed" else "")
    manager._save()
    return path


def test_worker_events_drive_the_task_and_reach_the_database(
    make_manager, tmp_path, recording_db
):
    manager = make_manager()
    path = _running_task(manager, tmp_path)
    with h5py.File(path, "r+") as f:
        f.create_group("arguments").attrs.update(frequency=2.0, __schema__="{}")
    manager._record_run(1)
    base = {"command": "worker", "rid": 1, "token": "tok"}

    stale = manager.handle(base | {"token": "old", "event": "heartbeat"})
    assert stale["action"] == "terminate"
    assert manager.handle(base | {"event": "started"})["h5_path"] == str(path)
    manager.handle(base | {"event": "point", "done": 0, "total": 3, "index": None})

    manager.handle({"command": "pause", "rid": 1})
    point = {
        "event": "point",
        "done": 1,
        "total": 3,
        "index": 0,
        "values": {"frequency": 2.0},
        "scalars": {"counts": 5.0},
        "t_start": 100.0,
        "t_end": 101.0,
    }
    assert manager.handle(base | point)["action"] == "pause"
    manager.handle(base | {"event": "heartbeat", "paused": True})
    assert manager.queue.get(1).status == "paused"
    manager.handle({"command": "resume", "rid": 1})
    assert manager.handle(base | {"event": "heartbeat"})["action"] == "continue"

    manager.handle(base | {"event": "finished", "status": "completed"})
    task = manager.queue.get(1)
    assert task.status == "completed" and task.points_done == 1
    assert manager.queue.state.workers == []

    assert recording_db.points == [
        {
            "rid": 1,
            "point_index": 0,
            "t_start": 100.0,
            "t_end": 101.0,
            "scan_values": {"frequency": 2.0},
            "scalars": {"counts": 5.0},
        }
    ]
    started, finished = recording_db.runs
    assert started["status"] == "running" and started["t_end"] is None
    assert started["h5_path"] == "2026/01/01/000001_RabiDemo.h5"
    assert finished["status"] == "completed" and finished["t_end"] is not None
    # The effective arguments come from the file, not the request.
    assert finished["arguments"] == {"frequency": 2.0}
    assert finished["simulated"] is manager.simulate


@pytest.mark.parametrize(
    "in_file, settled, error",
    [("running", "interrupted", None), ("failed", "failed", "from file")],
)
def test_restart_settles_a_dead_workers_task(
    make_manager, tmp_path, recording_db, in_file, settled, error
):
    path = _running_task(make_manager(), tmp_path, pid=DEAD_PID, status=in_file)
    task = make_manager().queue.get(1)
    assert (task.status, task.error) == (settled, error)
    assert recording_db.runs[-1]["status"] == settled
    with h5py.File(path, "r") as f:
        assert f.attrs["status"] == settled


@pytest.mark.parametrize("age_offset, adopted", [(0, True), (-100, False)])
def test_restart_adopts_a_live_worker_but_not_a_reused_pid(
    make_manager, tmp_path, age_offset, adopted
):
    me = psutil.Process()
    create_time = me.create_time() + age_offset
    _running_task(make_manager(), tmp_path, pid=me.pid, create_time=create_time)
    again = make_manager()
    assert (again.process is not None) is adopted
    assert again.queue.get(1).status == ("running" if adopted else "interrupted")


def recipe(**kwargs):
    fields = {"experiment": DEMO, "class_name": "RabiDemo", "name": "check", **kwargs}
    return Recipe(**fields).model_dump(mode="json")


def save(manager, **kwargs):
    reply = manager.handle({"command": "save_recipe", "recipe": recipe(**kwargs)})
    assert reply == {"ok": True}


def test_recipes_are_filled_in_published_and_survive_a_restart(make_manager):
    manager = make_manager()
    version = manager._snapshot()["recipes_version"]
    save(manager, args={"atoms": 50}, scan=PULSE_SCAN)
    save(manager, class_name="Gone", args={"x": 1})
    assert manager._snapshot()["recipes_version"] == version + 2

    listed = make_manager().handle({"command": "recipes"})["recipes"]
    saved = {r["class_name"]: r["args"] for r in listed}
    declared = RabiDemo.schema()["arguments"]
    assert saved["RabiDemo"] == {
        "atoms": 50,
        "rabi_frequency": declared["rabi_frequency"]["default"],
        "point_delay": declared["point_delay"]["default"],
    }
    assert saved["Gone"] == {"x": 1}, "not in the catalogue: kept as given"

    delete = {"experiment": DEMO, "class_name": "Gone", "name": "check"}
    assert manager.handle({"command": "delete_recipe", **delete}) == {"ok": True}
    assert manager._snapshot()["recipes_version"] == version + 3


GONE = {"experiment": "pytweezer.experiments.gone", "class_name": "Gone"}


@pytest.mark.parametrize(
    "saved, submitted, busy, expected",
    [
        (
            {"args": {"atoms": 5, "old_knob": 1}},
            {},
            False,
            ["no longer has", "old_knob"],
        ),
        (
            {"scan": Scan(axes=[ListAxis(argument="old", values=[1])])},
            {},
            False,
            ["no longer has", "'old'"],
        ),
        ({}, {"args": {"atom": 3}}, False, ["has no argument", "'atom'"]),
        (GONE, {}, False, ["not in the catalogue"]),
        (GONE, {}, True, ["try again"]),
        (
            {"experiment": BROKEN["module"], "class_name": "Anything"},
            {},
            False,
            ["fails to import", "SyntaxError: invalid syntax"],
        ),
    ],
)
def test_a_recipe_that_no_longer_fits_its_experiment_is_refused(
    make_manager, saved, submitted, busy, expected
):
    manager = make_manager()
    save(manager, **saved)
    manager.catalogue.busy = busy
    key = {"experiment": DEMO, "class_name": "RabiDemo", "name": "check"}
    key |= {k: v for k, v in saved.items() if k in key}
    reply = manager.handle({"command": "submit_recipe", **key, **submitted})
    assert not reply["ok"]
    for text in expected:
        assert text in reply["error"]
    assert manager.queue.ordered() == []


class InProcess:
    """Routes client requests straight into a manager, through JSON as on the wire."""

    def __init__(self, manager):
        self.manager = manager

    def request(self, payload, *, retry=True):
        return self.manager.handle(json.loads(json.dumps(payload)))

    def close(self):
        pass


def test_a_slow_manager_receives_each_submit_once():
    server = zmq.Context.instance().socket(zmq.REP)
    server.bind("tcp://127.0.0.1:*")
    received = []

    def serve_slowly():
        while server.poll(300):
            received.append(server.recv_json()["command"])
            time.sleep(0.2)
            server.send_json({"ok": True, "rid": 1})

    thread = threading.Thread(target=serve_slowly)
    thread.start()
    client = ExperimentManagerClient(
        server.getsockopt_string(zmq.LAST_ENDPOINT), timeout_ms=50
    )
    with pytest.raises(ManagerUnavailable):
        client.submit(TaskRequest(experiment=DEMO, class_name="RabiDemo"))
    thread.join()
    client.close()
    server.close(linger=0)
    assert received == ["submit"]


def test_notebook_recipes_round_trip(make_manager):
    manager = make_manager()
    client = ExperimentManagerClient(endpoint="tcp://127.0.0.1:1")
    client._req = InProcess(manager)
    save_recipe(RabiDemo, "check", PULSE_SCAN, label="nightly", client=client, atoms=50)
    [saved] = recipes(RabiDemo, client=client)
    assert saved.args == {"rabi_frequency": 50e3, "atoms": 50, "point_delay": 0.2}
    assert saved.label == "nightly" and "@" in saved.submitter

    rid = submit_recipe(
        RabiDemo, "check", priority=2, label="override", client=client, atoms=60
    )
    task = manager.queue.get(rid)
    assert task.args == {"rabi_frequency": 50e3, "atoms": 60, "point_delay": 0.2}
    assert [axis.argument for axis in task.scan.axes] == ["pulse_time"]
    assert (task.priority, task.label) == (2, "override") and "@" in task.submitter

    with pytest.raises(ManagerError, match="already exists"):
        save_recipe(RabiDemo, "check", client=client)
    save_recipe(RabiDemo, "check", client=client, overwrite=True, atoms=1)
    assert recipes(client=client)[0].args["atoms"] == 1

    save_recipe(f"{DEMO}:RabiDemo", "string", client=client, atoms=5)
    string_form = next(r for r in recipes(client=client) if r.name == "string")
    assert string_form.args == {
        "pulse_time": 10e-6,
        "rabi_frequency": 50e3,
        "atoms": 5,
        "point_delay": 0.2,
    }

    delete_recipe(f"{DEMO}:RabiDemo", "check", client=client)
    delete_recipe(f"{DEMO}:RabiDemo", "string", client=client)
    assert recipes(client=client) == []


def test_notebook_calls_check_arguments_before_sending():
    class Unreachable:
        def __getattr__(self, name):
            raise AssertionError("nothing should be sent")

    client = Unreachable()
    with pytest.raises(ValueError, match="no argument"):
        submit(RabiDemo, client=client, atom=3)
    with pytest.raises(ValueError, match="no argument"):
        save_recipe(RabiDemo, "x", client=client, atom=1)
    with pytest.raises(ValueError, match="no argument"):
        submit_recipe(RabiDemo, "x", client=client, atom=1)
    with pytest.raises(ValueError, match="below the minimum"):
        submit(
            RabiDemo, Scan(axes=[ListAxis(argument="atoms", values=[0])]), client=client
        )
    with pytest.raises(ValueError, match="run_local"):
        submit(type("Local", (), {"__module__": "__main__"}), client=client)
