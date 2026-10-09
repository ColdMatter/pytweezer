"""Experiment Manager request handling and restart reconciliation, without sockets."""

import os

import h5py
import psutil
import pytest

from pytweezer.configuration.config import CONFIG
from pytweezer.experiment.catalogue import Catalogue
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.experiment.queue import WorkerRecord
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest
from pytweezer.experiments.demo import RabiDemo
from pytweezer.servers import experiment_manager as em


def request(**kwargs):
    return TaskRequest(
        experiment="pytweezer.experiments.demo", class_name="RabiDemo", **kwargs
    ).model_dump(mode="json")


@pytest.fixture
def make_manager(tmp_path, recording_db):
    def make():
        return em.ExperimentManager(
            root=tmp_path,
            catalogue=Catalogue(package="none", directory=tmp_path / "none"),
            bind=False,
            db=recording_db,
        )

    return make


def test_config_entry_is_allocated_after_every_other_port():
    manager_ports = {
        CONFIG["Servers"]["Experiment Manager"][key] for key in ("port", "sync_port")
    }
    other_ports = {
        entry[key]
        for category in ("Servers", "Devices")
        for name, entry in CONFIG[category].items()
        if name != "Experiment Manager"
        for key in ("port", "pub_port", "sub_port", "rpc_port")
        if key in entry
    }
    assert min(manager_ports) > max(other_ports)


def test_submit_and_query(make_manager):
    manager = make_manager()
    reply = manager.handle({"command": "submit", "request": request(label="a")})
    assert reply == {"ok": True, "rid": 1}
    task = manager.handle({"command": "get", "rid": 1})["task"]
    assert task["label"] == "a" and task["status"] == "queued"
    last = manager.handle(
        {
            "command": "last_request",
            "experiment": "pytweezer.experiments.demo",
            "class_name": "RabiDemo",
        }
    )
    assert last["request"]["label"] == "a"
    snapshot = manager.handle({"command": "snapshot"})["snapshot"]
    assert [t["rid"] for t in snapshot["queue"]] == [1]


def test_errors_come_back_as_replies(make_manager):
    manager = make_manager()
    assert manager.handle({"command": "nope"}) == {
        "ok": False,
        "error": "unknown command 'nope'",
    }
    reply = manager.handle({"command": "pause", "rid": 5})
    assert not reply["ok"] and "not running" in reply["error"]
    reply = manager.handle({"command": "submit", "request": {"experiment": "x"}})
    assert not reply["ok"] and "class_name" in reply["error"]


def test_state_survives_a_restart_and_rids_never_repeat(make_manager, tmp_path):
    manager = make_manager()
    manager.handle({"command": "submit", "request": request()})
    (tmp_path / "2026" / "01" / "01").mkdir(parents=True)
    (tmp_path / "2026" / "01" / "01" / "000040_RabiDemo.h5").touch()
    again = make_manager()
    assert [t.rid for t in again.queue.ordered()] == [1]
    assert again.handle({"command": "submit", "request": request()})["rid"] == 41


def test_unknown_worker_is_told_to_terminate(make_manager):
    manager = make_manager()
    reply = manager.handle(
        {"command": "worker", "rid": 1, "token": "x", "event": "point"}
    )
    assert reply == {"ok": True, "action": "terminate"}


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


def test_worker_events_drive_the_task(make_manager, tmp_path):
    manager = make_manager()
    _running_task(manager, tmp_path)
    base = {"command": "worker", "rid": 1, "token": "tok"}

    started = manager.handle(base | {"event": "started"})
    assert started["task"]["rid"] == 1
    assert started["h5_path"].endswith("000001_RabiDemo.h5")

    manager.handle({"command": "pause", "rid": 1})
    reply = manager.handle(base | {"event": "point", "done": 1, "total": 3, "index": 0})
    assert reply["action"] == "pause"
    manager.handle(base | {"event": "heartbeat", "paused": True})
    assert manager.queue.get(1).status == "paused"
    manager.handle({"command": "resume", "rid": 1})
    assert manager.handle(base | {"event": "heartbeat"})["action"] == "continue"

    manager.handle(base | {"event": "finished", "status": "completed"})
    task = manager.queue.get(1)
    assert task.status == "completed" and task.points_done == 1
    assert manager.queue.state.worker is None


def test_runs_and_points_reach_the_database(make_manager, tmp_path, recording_db):
    manager = make_manager()
    path = _running_task(manager, tmp_path)
    with h5py.File(path, "r+") as f:
        f.create_group("arguments").attrs.update(frequency=2.0, __schema__="{}")
    manager._record_run(1)
    base = {"command": "worker", "rid": 1, "token": "tok"}

    manager.handle(
        base
        | {
            "event": "point",
            "done": 1,
            "total": 3,
            "index": 0,
            "values": {"frequency": 2.0},
            "scalars": {"counts": 5.0},
            "t_start": 100.0,
            "t_end": 101.0,
        }
    )
    manager.handle(base | {"event": "point", "done": 0, "total": 3, "index": None})
    manager.handle(base | {"event": "finished", "status": "completed"})

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
    assert finished["arguments"] == {"frequency": 2.0}
    assert finished["simulated"] is manager.simulate


def test_a_run_settled_on_restart_is_recorded(make_manager, tmp_path, recording_db):
    manager = make_manager()
    _running_task(manager, tmp_path, pid=2**22 + 12345)
    make_manager()
    assert recording_db.runs[-1]["status"] == "interrupted"


def test_restart_with_dead_worker_marks_task_interrupted(make_manager, tmp_path):
    manager = make_manager()
    path = _running_task(manager, tmp_path, pid=2**22 + 12345)
    again = make_manager()
    assert again.queue.get(1).status == "interrupted"
    with h5py.File(path, "r") as f:
        assert f.attrs["status"] == "interrupted"


def test_restart_trusts_a_final_status_already_in_the_file(make_manager, tmp_path):
    manager = make_manager()
    _running_task(manager, tmp_path, pid=2**22 + 12345, status="failed")
    task = make_manager().queue.get(1)
    assert task.status == "failed" and task.error == "from file"


def test_restart_adopts_a_live_worker(make_manager, tmp_path):
    me = psutil.Process(os.getpid())
    manager = make_manager()
    _running_task(manager, tmp_path, pid=me.pid, create_time=me.create_time())
    again = make_manager()
    assert again.queue.running.rid == 1
    assert again.process.pid == me.pid


def test_restart_does_not_adopt_a_reused_pid(make_manager, tmp_path):
    me = psutil.Process(os.getpid())
    manager = make_manager()
    _running_task(manager, tmp_path, pid=me.pid, create_time=me.create_time() - 100)
    assert make_manager().queue.get(1).status == "interrupted"


def test_published_state_follows_the_queue_and_points(make_manager, tmp_path):
    manager = make_manager()
    _running_task(manager, tmp_path)
    base = {"command": "worker", "rid": 1, "token": "tok"}
    state = manager.notifier.raw_view

    manager.handle(
        base
        | {
            "event": "point",
            "done": 1,
            "total": 2,
            "index": 0,
            "values": {"frequency": 2.0},
            "scalars": {"counts": 5.0},
        }
    )
    manager.tick()
    assert state["running"]["rid"] == 1 and state["running"]["points_done"] == 1
    assert state["points"]["rid"] == 1
    assert [row["scalars"] for row in state["points"]["rows"]] == [{"counts": 5.0}]

    manager.handle(base | {"event": "finished", "status": "completed"})
    manager.tick()
    assert state["running"] is None
    assert state["history"][0]["rid"] == 1
    assert state["points"]["rid"] == 1, "rows outlive the task until the next starts"


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


class Sequenced(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")

    def run_point(self):
        pass


def entry(cls):
    return {
        "module": cls.__module__,
        "classes": [cls.schema()],
        "warnings": [],
        "error": None,
    }


BROKEN = {
    "module": "pytweezer.experiments.broken",
    "classes": [],
    "warnings": [],
    "error": "Traceback (most recent call last):\n  ...\nSyntaxError: invalid syntax\n",
}
DEMO = "pytweezer.experiments.demo"


@pytest.fixture
def recipe_manager(tmp_path, recording_db):
    def make():
        return em.ExperimentManager(
            root=tmp_path,
            catalogue=StaticCatalogue([entry(RabiDemo), entry(Sequenced), BROKEN]),
            bind=False,
            db=recording_db,
        )

    return make


def recipe(**kwargs):
    fields = {"experiment": DEMO, "class_name": "RabiDemo", "name": "check", **kwargs}
    return Recipe(**fields).model_dump(mode="json")


def save(manager, **kwargs):
    reply = manager.handle({"command": "save_recipe", "recipe": recipe(**kwargs)})
    assert reply == {"ok": True}


def submit_recipe(manager, **fields):
    return manager.handle(
        {
            "command": "submit_recipe",
            "experiment": DEMO,
            "class_name": "RabiDemo",
            "name": "check",
            **fields,
        }
    )


def test_recipes_are_saved_published_and_survive_a_restart(recipe_manager, tmp_path):
    manager = recipe_manager()
    version = manager._snapshot()["recipes_version"]
    save(manager, args={"atoms": 50})
    assert manager._snapshot()["recipes_version"] == version + 1
    assert (tmp_path / "recipes.json").exists()
    [saved] = recipe_manager().handle({"command": "recipes"})["recipes"]
    assert saved["name"] == "check" and saved["args"] == {"atoms": 50}


def test_saving_over_a_recipe_needs_overwrite(recipe_manager):
    manager = recipe_manager()
    save(manager)
    reply = manager.handle({"command": "save_recipe", "recipe": recipe(label="new")})
    assert not reply["ok"] and "already exists" in reply["error"]
    reply = manager.handle(
        {"command": "save_recipe", "recipe": recipe(label="new"), "overwrite": True}
    )
    assert reply["ok"]
    [saved] = manager.handle({"command": "recipes"})["recipes"]
    assert saved["label"] == "new"


def test_recipes_are_listed_per_experiment_and_deleted(recipe_manager):
    manager = recipe_manager()
    save(manager)
    save(manager, experiment=Sequenced.__module__, class_name="Sequenced", name="seq")
    listed = manager.handle(
        {"command": "recipes", "experiment": DEMO, "class_name": "RabiDemo"}
    )["recipes"]
    assert [r["name"] for r in listed] == ["check"]
    version = manager._snapshot()["recipes_version"]
    delete = {
        "command": "delete_recipe",
        "experiment": DEMO,
        "class_name": "RabiDemo",
        "name": "check",
    }
    assert manager.handle(delete) == {"ok": True}
    assert manager._snapshot()["recipes_version"] == version + 1
    assert [r["name"] for r in manager.handle({"command": "recipes"})["recipes"]] == [
        "seq"
    ]
    reply = manager.handle(delete)
    assert not reply["ok"] and "no recipe 'check'" in reply["error"]


def test_a_recipe_is_queued_with_its_settings_and_overrides(recipe_manager):
    manager = recipe_manager()
    scan = Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=1e-5, n=3)])
    save(manager, args={"atoms": 50, "rabi_frequency": 1e3}, scan=scan, priority=2)
    reply = submit_recipe(
        manager, args={"atoms": 7}, label="override", submitter="me@pc"
    )
    assert reply == {"ok": True, "rid": 1}
    task = manager.queue.get(1)
    assert task.args == {"atoms": 7, "rabi_frequency": 1e3}
    assert [axis.argument for axis in task.scan.axes] == ["pulse_time"]
    assert (task.priority, task.label, task.submitter) == (2, "override", "me@pc")
    assert task.due_time is None


def test_a_recipe_using_a_removed_argument_is_refused(recipe_manager):
    manager = recipe_manager()
    save(manager, args={"atoms": 5, "old_knob": 1})
    reply = submit_recipe(manager)
    assert not reply["ok"]
    assert "no longer has" in reply["error"] and "old_knob" in reply["error"]
    save(
        manager, name="scanned", scan=Scan(axes=[ListAxis(argument="old", values=[1])])
    )
    reply = submit_recipe(manager, name="scanned")
    assert not reply["ok"] and "'old'" in reply["error"]
    assert manager.queue.ordered() == []


def test_an_override_must_name_an_argument_and_not_a_scanned_one(recipe_manager):
    manager = recipe_manager()
    save(manager, scan=Scan(axes=[ListAxis(argument="atoms", values=[1, 2])]))
    reply = submit_recipe(manager, args={"atom": 3})
    assert not reply["ok"]
    assert "has no argument" in reply["error"] and "'atom'" in reply["error"]
    assert "no longer" not in reply["error"]
    reply = submit_recipe(manager, args={"atoms": 3})
    assert not reply["ok"] and "scans" in reply["error"]
    assert manager.queue.ordered() == []


def test_motmaster_script_parameters_are_left_to_the_run(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": Sequenced.__module__, "class_name": "Sequenced"}
    save(manager, name="seq", args={"rb.tPulse": 2e-6}, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "seq", **fields})
    assert reply["ok"]
    save(manager, name="bad", args={"cs.tPulse": 1}, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "bad", **fields})
    assert not reply["ok"] and "cs.tPulse" in reply["error"]


def test_a_recipe_for_a_missing_experiment_is_refused(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": "pytweezer.experiments.gone", "class_name": "Gone"}
    save(manager, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"] and "not in the catalogue" in reply["error"]
    manager.catalogue.busy = True
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"] and "try again" in reply["error"]


def test_a_recipe_for_a_module_that_fails_to_import_says_why(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": BROKEN["module"], "class_name": "Anything"}
    save(manager, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"]
    assert "fails to import" in reply["error"]
    assert "SyntaxError: invalid syntax" in reply["error"]
