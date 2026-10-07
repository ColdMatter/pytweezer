import pytest

from pytweezer.experiment import (
    Device,
    Experiment,
    Integer,
    ListAxis,
    Number,
    Scan,
    TaskStatus,
    run_local,
)
from pytweezer.experiment.runner import Progress, open_writer, run_points
from pytweezer.experiment.task import Action


class Recorder(Experiment):
    gain = Number(2.0)
    x = Number(0.0)
    fail_at = Integer(-1)
    fail_in_finish = Integer(0)

    def __init__(self, args=None):
        super().__init__(args)
        self.calls = []

    def prepare(self):
        self.calls.append("prepare")
        self.record("setup", 1)

    def run_point(self):
        self.calls.append(("point", self.point.index, self.x))
        if self.point.index == self.fail_at:
            raise RuntimeError("boom")
        self.record("y", self.gain * self.x)

    def finish(self):
        self.calls.append("finish")
        if self.fail_in_finish:
            raise RuntimeError("finish failed")


SCAN = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])])


def run(args=None, control=None):
    experiment, points, writer = open_writer(
        None, Recorder, args or {}, SCAN, {"rid": 1}
    )
    status, error = run_points(experiment, points, writer, control)
    return experiment, status, error, writer


def test_hook_order_and_scanned_values():
    experiment, status, error, writer = run()
    assert status == TaskStatus.COMPLETED and error is None
    assert experiment.calls == [
        "prepare",
        ("point", 0, 1.0),
        ("point", 1, 2.0),
        ("point", 2, 3.0),
        "finish",
    ]
    writer.close()


def test_run_local_stores_effective_arguments_and_results():
    measurement = run_local(Recorder, SCAN, gain=3)
    assert measurement.status == "completed"
    assert measurement.arguments["gain"] == 3.0
    assert list(measurement.results["y"]) == [3.0, 6.0, 9.0]
    assert measurement.constants["setup"] == 1


def test_run_local_rejects_bad_arguments_up_front():
    with pytest.raises(ValueError, match="no argument"):
        run_local(Recorder, SCAN, gian=3)


def test_failure_runs_finish_and_keeps_completed_points():
    experiment, status, error, writer = run({"fail_at": 1})
    assert status == TaskStatus.FAILED
    assert "boom" in error
    assert experiment.calls[-1] == "finish"
    assert writer.file.attrs["n_done"] == 1
    assert writer.file.attrs["status"] == "failed"
    assert "boom" in writer.file.attrs["error"]
    writer.close()


def test_failure_in_finish_fails_the_task():
    _, status, error, writer = run({"fail_in_finish": 1})
    assert status == TaskStatus.FAILED
    assert "finish failed" in error
    writer.close()


def test_terminate_at_point_boundary():
    seen = []

    def control(progress: Progress):
        seen.append(progress.done)
        return Action.TERMINATE if progress.done == 2 else Action.CONTINUE

    experiment, status, _, writer = run(control=control)
    assert status == TaskStatus.TERMINATED
    assert seen == [0, 1, 2]
    assert experiment.calls[-1] == "finish"
    assert writer.file.attrs["n_done"] == 2
    writer.close()


def test_control_sees_point_scalars_and_interrupt_status():
    scalars = []

    def control(progress: Progress):
        scalars.append(progress.scalars)
        return Action.INTERRUPT if progress.done == 1 else Action.CONTINUE

    _, status, _, writer = run(control=control)
    assert status == TaskStatus.INTERRUPTED
    assert scalars == [None, {"y": 2.0}]
    writer.close()


def test_control_sees_when_each_point_ran():
    times = []

    def control(progress: Progress):
        times.append((progress.t_start, progress.t_end))
        return Action.CONTINUE

    *_, writer = run(control=control)
    assert times[0] == (None, None)
    for (t_start, t_end), stored in zip(times[1:], writer.file["points/t_end"]):
        assert t_start <= t_end == stored
    writer.close()


def test_terminate_before_first_point():
    experiment, status, _, writer = run(control=lambda progress: Action.TERMINATE)
    assert status == TaskStatus.TERMINATED
    assert experiment.calls == ["prepare", "finish"]
    writer.close()


def test_devices_closed_even_on_failure(monkeypatch):
    closed = []

    class Client:
        def close_rpc(self):
            closed.append(True)

    monkeypatch.setattr(
        "pytweezer.experiment.experiment._get_device", lambda name, timeout: Client()
    )

    class UsesDevice(Experiment):
        cam = Device("Rb ThorCam")

        def run_point(self):
            self.cam
            raise RuntimeError("after opening")

    measurement = run_local(UsesDevice)
    assert measurement.status == "failed"
    assert closed == [True]
