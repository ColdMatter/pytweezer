"""Experiment Manager request handling and restart reconciliation, mostly without sockets."""

import sys
import threading
import time

import h5py
import psutil
import pytest
import zmq

from pytweezer.experiment import Experiment
from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    delete_recipe,
    get_saved_parameter,
    recipes,
    save_recipe,
    set_saved_parameters,
    submit,
    submit_recipe,
)
from pytweezer.experiment.queue import WorkerRecord
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest
from pytweezer.experiments.demo import RabiDemo
from pytweezer.servers import experiment_manager as em
from tests.helpers import BROKEN, DEMO, demo_schema, in_process_client

DEAD_PID = 2**22 + 12345
PULSE_SCAN = Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=1e-5, n=3)])


def request(class_name="RabiDemo", **kwargs):
    return TaskRequest(experiment=DEMO, class_name=class_name, **kwargs).model_dump(
        mode="json"
    )


class FakeWorker:
    """Stands in for a worker process: alive until killed."""

    def __init__(self, pid):
        self.pid = pid
        self.killed = False

    def create_time(self):
        return 1.0

    def is_running(self):
        return not self.killed

    def status(self):
        return psutil.STATUS_RUNNING

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


@pytest.fixture
def workers(monkeypatch):
    """Every worker the manager spawns, by rid, faked."""
    spawned = {}

    def spawn(manager, rid, token):
        spawned[rid] = FakeWorker(pid=1000 + rid)
        return spawned[rid]

    monkeypatch.setattr(em.ExperimentManager, "_spawn", spawn)
    return spawned


def submit_to(manager, class_name="RabiDemo", **kwargs):
    reply = manager.handle(
        {"command": "submit", "request": request(class_name=class_name, **kwargs)}
    )
    assert reply["ok"], reply
    return reply["rid"]


def finish(manager, rid, status="completed"):
    token = manager.queue.worker(rid).token
    manager.handle(
        {
            "command": "worker",
            "rid": rid,
            "token": token,
            "event": "finished",
            "status": status,
        }
    )


def running(manager):
    return [task.rid for task in manager.queue.running]


def test_two_pipelines_start_two_workers(make_manager, workers):
    manager = make_manager()
    submit_to(manager, pipeline="Rb")
    submit_to(manager, "CaFDemo", pipeline="CaF")
    manager.tick()
    assert sorted(manager.processes) == [1, 2]
    assert running(manager) == [1, 2]


def test_clashing_tasks_start_one_after_the_other(make_manager, workers):
    manager = make_manager()
    submit_to(manager, pipeline="A")
    submit_to(manager, pipeline="B")
    manager.tick()
    assert running(manager) == [1]
    assert manager.queue.get(2).devices == ["Rb MotMaster"]
    finish(manager, 1)
    manager.tick()
    assert running(manager) == [2]


@pytest.mark.parametrize("command", ["submit", "submit_recipe"])
def test_submit_refuses_a_bad_declaration(make_manager, command):
    manager = make_manager()
    error = "'CaF MotMaster' is linked but its CONFIG entry has no 'linked_partner'"
    classes = manager.catalogue.entries()[0]["classes"]
    classes.append(demo_schema("Linked", reserves=None, error=error))
    if command == "submit":
        reply = manager.handle(
            {"command": "submit", "request": request(class_name="Linked")}
        )
    else:
        save(manager, class_name="Linked")
        reply = manager.handle(
            {
                "command": "submit_recipe",
                "experiment": DEMO,
                "class_name": "Linked",
                "name": "check",
            }
        )
    assert not reply["ok"]
    assert "linked_partner" in reply["error"]
    assert manager.queue.ordered() == []


def test_reservation_resolved_later_when_catalogue_was_busy(make_manager, workers):
    manager = make_manager()
    manager.catalogue.busy = True
    rid = submit_to(manager, "Later")
    assert manager.queue.get(rid).devices is None
    manager.tick()
    assert manager.processes == {}

    manager.catalogue.entries()[0]["classes"].append(
        demo_schema("Later", ["Rb HamCam"])
    )
    manager.catalogue.busy = False
    manager.tick()
    assert running(manager) == [rid]
    assert manager.queue.get(rid).devices == ["Rb HamCam"]


def test_unresolvable_reservation_fails_the_task(make_manager, workers):
    manager = make_manager()
    manager.catalogue.busy = True
    rid = submit_to(manager, "Gone")
    manager.catalogue.busy = False
    manager.tick()
    task = manager.queue.get(rid)
    assert task.status == "failed"
    assert "not in the catalogue" in task.error
    assert workers == {}


def test_a_module_being_reread_is_not_reserved_from_its_old_description(
    make_manager, workers
):
    manager = make_manager()
    save(manager)
    manager.catalogue.rereading = {DEMO}
    rid = submit_to(manager)
    assert manager.queue.get(rid).devices is None
    reply = manager.handle(
        {
            "command": "submit_recipe",
            "experiment": DEMO,
            "class_name": "RabiDemo",
            "name": "check",
        }
    )
    assert not reply["ok"] and "try again shortly" in reply["error"]

    manager.catalogue.entries()[0]["classes"][0]["reserves"] = ["Rb HamCam"]
    manager.catalogue.rereading = set()
    manager.tick()
    assert manager.queue.get(rid).devices == ["Rb HamCam"]


def test_abort_kills_only_that_worker(make_manager, workers):
    manager = make_manager()
    submit_to(manager, pipeline="Rb")
    submit_to(manager, "CaFDemo", pipeline="CaF")
    manager.tick()
    assert manager.handle({"command": "abort", "rid": 1}) == {"ok": True}
    assert (workers[1].killed, workers[2].killed) == (True, False)
    assert manager.queue.get(1).status == "aborted"
    assert running(manager) == [2]
    assert list(manager.processes) == [2]


def test_points_are_kept_per_task(make_manager, workers):
    manager = make_manager()
    submit_to(manager, pipeline="Rb")
    submit_to(manager, "CaFDemo", pipeline="CaF")
    submit_to(manager, "CaFDemo", pipeline="Rb")
    manager.tick()
    for rid in (1, 2):
        manager.handle(
            {
                "command": "worker",
                "rid": rid,
                "token": manager.queue.worker(rid).token,
                "event": "point",
                "done": 1,
                "total": 1,
                "index": 0,
                "scalars": {"rid": float(rid)},
            }
        )
    points = manager.notifier.raw_view["points"]
    assert {key: [row["scalars"] for row in rows] for key, rows in points.items()} == {
        "1": [{"rid": 1.0}],
        "2": [{"rid": 2.0}],
    }

    finish(manager, 1)
    finish(manager, 2)
    manager.tick()
    assert running(manager) == [3]
    points = manager.notifier.raw_view["points"]
    assert set(points) == {"2", "3"}, "the next task in a pipeline replaces its points"
    assert points["3"] == []


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


def test_restart_adopts_every_live_worker(make_manager, tmp_path):
    manager = make_manager()
    for pipeline in ("A", "B", "C"):
        manager.queue.submit(
            TaskRequest(experiment=DEMO, class_name="RabiDemo", pipeline=pipeline),
            devices=[pipeline],
        )
    child = psutil.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    me = psutil.Process()
    try:
        live = {1: child, 2: me}
        for rid, process in live.items():
            record = WorkerRecord(
                rid=rid, token="t", pid=process.pid, create_time=process.create_time()
            )
            manager.queue.mark_started(rid, record, f"{rid}.h5")
        reused_pid = WorkerRecord(
            rid=3, token="t", pid=me.pid, create_time=me.create_time() - 100
        )
        manager.queue.mark_started(3, reused_pid, "3.h5")
        manager._save()

        again = make_manager()
        assert sorted(again.processes) == [1, 2]
        assert running(again) == [1, 2]
        assert again.queue.get(3).status == "interrupted"
    finally:
        child.kill()
        child.wait()


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
    client = in_process_client(manager)
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
        submit(type("Local", (Experiment,), {"__module__": "__main__"}), client=client)


def test_a_written_value_is_read_back_after_a_restart_and_bumps_the_version(
    make_manager, tmp_path
):
    manager = make_manager()
    version = manager._snapshot()["saved_parameters_version"]
    reply = manager.handle(
        {
            "command": "set_saved_parameters",
            "values": {"offset": 131.5},
            "reason": "r",
            "who": "me",
            "create": True,
        }
    )
    assert reply["ok"], reply
    assert (tmp_path / "saved_parameters_log.jsonl").exists()
    [parameter] = manager.handle({"command": "saved_parameters"})["parameters"]
    assert (parameter["key"], parameter["value"]) == ("offset", 131.5)
    assert manager._snapshot()["saved_parameters_version"] == version + 1
    restarted = make_manager()
    assert restarted.handle({"command": "saved_parameters"})["parameters"] == [
        parameter
    ]
    [entry] = restarted.handle({"command": "saved_parameter_history", "key": "offset"})[
        "entries"
    ]
    assert entry["new"] == 131.5


@pytest.mark.parametrize("failing", ["save", "append"])
def test_a_saved_parameter_write_that_fails_to_persist_changes_nothing(
    make_manager, tmp_path, monkeypatch, failing
):
    manager = make_manager()

    def set_value(value, **extra):
        return manager.handle(
            {
                "command": "set_saved_parameters",
                "values": {"offset": value},
                "reason": "r",
                "who": "me",
                **extra,
            }
        )

    assert set_value(1.0, create=True)["ok"]
    version = manager._snapshot()["saved_parameters_version"]
    target = (
        manager.saved_parameter_store
        if failing == "save"
        else manager.saved_parameter_log
    )

    def refuse(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(target, failing, refuse)
    reply = set_value(2.0)
    assert not reply["ok"] and "locked" in reply["error"]
    [parameter] = manager.handle({"command": "saved_parameters"})["parameters"]
    assert parameter["value"] == 1.0
    assert manager._snapshot()["saved_parameters_version"] == version
    monkeypatch.undo()
    [stored] = make_manager().handle({"command": "saved_parameters"})["parameters"]
    assert stored["value"] == 1.0
    assert set_value(3.0)["ok"]


def test_a_misspelt_key_names_the_closest_one(make_manager):
    client = in_process_client(make_manager())
    set_saved_parameters({"cool_resonance": 4.05}, "r", create=True, client=client)
    assert get_saved_parameter("cool_resonance", client=client) == 4.05
    with pytest.raises(KeyError, match="cool_resonance"):
        get_saved_parameter("cool_resonanse", client=client)
