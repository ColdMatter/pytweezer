"""Experiment Manager + real worker subprocesses over localhost sockets."""

import socket
import textwrap
import threading
import time

import pytest

from pytweezer.experiment import ListAxis, Scan, load_measurement
from pytweezer.experiment.catalogue import Catalogue
from pytweezer.experiment.client import ExperimentManagerClient, submit, wait
from pytweezer.servers import experiment_manager as em

EXPERIMENT = textwrap.dedent(
    """
    import os
    import time

    from pytweezer.experiment import Device, Experiment, Integer, Number


    class Steps(Experiment):
        delay = Number(0.0)
        crash_at = Integer(-1)
        step = Integer(0)

        def run_point(self):
            if self.point.index == self.crash_at:
                os._exit(3)
            time.sleep(self.delay)
            self.record("index", self.point.index)


    class NotRunnable(Experiment):
        pass


    class UsesSequencer(Experiment):
        sequencer = Device("Rb MotMaster")

        def run_point(self):
            self.sequencer.set_iterations(1)
            self.record("backend", type(self.sequencer).__name__)
    """
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Harness:
    def __init__(self, tmp_path, monkeypatch, db):
        self.db = db
        package = tmp_path / "labexp"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "steps.py").write_text(EXPERIMENT)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.setenv("PYTWEEZER_LOG_DIR", str(tmp_path / "logs"))
        self.conf = {
            "host": "127.0.0.1",
            "port": free_port(),
            "pub_port": free_port(),
            "orphan_timeout": 5.0,
        }
        monkeypatch.setattr(
            em, "get_config", lambda: {"Servers": {"Experiment Manager": self.conf}}
        )
        self.root = tmp_path / "data"
        self.package_dir = package
        self.thread = None
        self.manager = None
        self.client = ExperimentManagerClient(
            f"tcp://127.0.0.1:{self.conf['port']}", timeout_ms=3000
        )

    def start(self):
        ready = threading.Event()

        def serve():
            self.manager = em.ExperimentManager(
                root=self.root,
                catalogue=Catalogue(package="labexp", directory=self.package_dir),
                db=self.db,
            )
            ready.set()
            self.manager.serve_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        assert ready.wait(10)

    def stop(self):
        self.manager.stop()
        self.thread.join(10)

    def submit(self, scan=None, **args):
        return submit("labexp.steps:Steps", scan, client=self.client, **args)

    def wait_until(self, rid, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.client.get(rid)
            if predicate(task):
                return task
            time.sleep(0.05)
        raise TimeoutError(f"task {rid}: {self.client.get(rid)}")

    def measurement(self, task):
        return load_measurement(self.root / task.h5_path)


@pytest.fixture
def harness(tmp_path, monkeypatch, recording_db):
    harness = Harness(tmp_path, monkeypatch, recording_db)
    harness.start()
    yield harness
    running = harness.manager.queue.running
    if running is not None:
        harness.client.abort(running.rid)
    harness.stop()
    harness.client.close()


def five_points():
    return Scan(axes=[ListAxis(argument="step", values=[1, 2, 3, 4, 5])])


def test_tasks_run_in_order_and_write_measurements(harness):
    first = harness.submit(Scan(repetitions=3))
    second = harness.submit(Scan(repetitions=2))
    task = wait(second, client=harness.client, poll_s=0.1, timeout_s=30)
    assert task.status == "completed"
    assert harness.client.get(first).t_end <= task.t_start
    measurement = harness.measurement(task)
    assert measurement.rid == second
    assert list(measurement.results["index"]) == [0, 1]
    history = harness.client.snapshot()["history"]
    assert [t["rid"] for t in history] == [second, first]

    points = [p for p in harness.db.points if p["rid"] == second]
    assert [p["point_index"] for p in points] == [0, 1]
    assert list(measurement.points["t_end"]) == [p["t_end"] for p in points]
    assert harness.db.runs[-1]["rid"] == second
    assert harness.db.runs[-1]["status"] == "completed"


def test_catalogue_lists_runnable_experiments(harness):
    deadline = time.monotonic() + 30
    modules = []
    while time.monotonic() < deadline:
        modules = harness.client.catalogue()
        if modules:
            break
        time.sleep(0.2)
    [entry] = modules
    assert entry["module"] == "labexp.steps"
    assert entry["error"] is None
    assert [c["class_name"] for c in entry["classes"]] == ["Steps", "UsesSequencer"]


def test_terminate_stops_at_a_point_boundary(harness):
    rid = harness.submit(five_points(), delay=0.3)
    harness.wait_until(rid, lambda t: t.points_done >= 1)
    harness.client.terminate(rid)
    task = harness.wait_until(rid, lambda t: t.status.finished)
    assert task.status == "terminated"
    measurement = harness.measurement(task)
    assert measurement.status == "terminated"
    assert 1 <= measurement.n_done < 5


def test_pause_and_resume(harness):
    rid = harness.submit(five_points(), delay=0.1)
    harness.wait_until(rid, lambda t: t.points_done >= 1)
    harness.client.pause(rid)
    paused = harness.wait_until(rid, lambda t: t.status == "paused")
    time.sleep(0.5)
    assert harness.client.get(rid).points_done == paused.points_done
    harness.client.resume(rid)
    task = harness.wait_until(rid, lambda t: t.status.finished)
    assert task.status == "completed"
    assert task.points_done == 5


def test_crashed_worker_is_recorded(harness):
    rid = harness.submit(five_points(), crash_at=2)
    task = harness.wait_until(rid, lambda t: t.status.finished)
    assert task.status == "crashed"
    assert "code 3" in task.error
    assert harness.measurement(task).status == "crashed"
    # the queue keeps going
    assert (
        harness.wait_until(harness.submit(), lambda t: t.status.finished).status
        == "completed"
    )


def test_abort_kills_the_worker(harness):
    rid = harness.submit(five_points(), delay=5.0)
    harness.wait_until(rid, lambda t: t.status == "running" and t.h5_path)
    time.sleep(1.5)  # let the worker create its file
    harness.client.abort(rid)
    task = harness.wait_until(rid, lambda t: t.status.finished, timeout=5)
    assert task.status == "aborted"


def test_worker_survives_a_manager_restart(harness):
    rid = harness.submit(five_points(), delay=0.4)
    harness.wait_until(rid, lambda t: t.points_done >= 1)
    harness.stop()
    time.sleep(0.5)
    harness.start()
    assert harness.manager.process is not None  # adopted
    task = harness.wait_until(rid, lambda t: t.status.finished)
    assert task.status == "completed"
    assert harness.measurement(task).n_done == 5


def test_submit_checks_arguments_before_queueing():
    from pytweezer.experiments.demo import RabiDemo

    class Unreachable:
        def submit(self, request):
            raise AssertionError("should not be sent")

    with pytest.raises(ValueError, match="no argument"):
        submit(RabiDemo, client=Unreachable(), atom=3)
    with pytest.raises(ValueError, match="below the minimum"):
        submit(
            RabiDemo,
            Scan(axes=[ListAxis(argument="atoms", values=[0])]),
            client=Unreachable(),
        )
    with pytest.raises(ValueError, match="run_local"):
        submit(type("Local", (), {"__module__": "__main__"}), client=Unreachable())


def test_simulating_manager_gives_workers_simulated_devices(harness):
    harness.manager.simulate = True  # as "simulate": SIMULATING in CONFIG
    rid = submit("labexp.steps:UsesSequencer", client=harness.client)
    task = harness.wait_until(rid, lambda t: t.status.finished)
    assert task.status == "completed", task.error
    measurement = harness.measurement(task)
    assert measurement.attrs["simulated"] is True
    assert list(measurement.results["backend"]) == ["SimulatedMotMasterInterface"]
