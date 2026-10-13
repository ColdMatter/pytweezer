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
from pytweezer.experiment.runner import open_writer, run_points
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


def run(control=None, **args):
    """Returns ``(experiment, status, error, file attrs, stored point end times)``."""
    experiment, points, writer = open_writer(None, Recorder, args, SCAN, {"rid": 1})
    with writer.file:
        status, error = run_points(experiment, points, writer, control)
        t_end = list(writer.file["points/t_end"])
        return experiment, status, error, dict(writer.file.attrs), t_end


def test_hook_order_and_scanned_values():
    experiment, status, error, *_ = run()
    assert status == TaskStatus.COMPLETED and error is None
    assert experiment.calls == [
        "prepare",
        ("point", 0, 1.0),
        ("point", 1, 2.0),
        ("point", 2, 3.0),
        "finish",
    ]


def test_run_local_stores_effective_arguments_and_results():
    measurement = run_local(Recorder, SCAN, gain=3)
    assert measurement.status == "completed"
    assert measurement.arguments["gain"] == 3.0
    assert list(measurement.results["y"]) == [3.0, 6.0, 9.0]
    assert measurement.constants["setup"] == 1
    with pytest.raises(ValueError, match="no argument"):
        run_local(Recorder, SCAN, gian=3)


@pytest.mark.parametrize(
    "args, message, n_done",
    [({"fail_at": 1}, "boom", 1), ({"fail_in_finish": 1}, "finish failed", 3)],
)
def test_failure_runs_finish_and_keeps_completed_points(args, message, n_done):
    experiment, status, error, attrs, _ = run(**args)
    assert status == TaskStatus.FAILED and message in error
    assert experiment.calls[-1] == "finish"
    assert attrs["n_done"] == n_done
    assert attrs["status"] == "failed" and message in attrs["error"]


@pytest.mark.parametrize(
    "action, stop_after, status",
    [
        (Action.TERMINATE, 0, TaskStatus.TERMINATED),
        (Action.TERMINATE, 2, TaskStatus.TERMINATED),
        (Action.INTERRUPT, 1, TaskStatus.INTERRUPTED),
    ],
)
def test_control_stops_at_a_point_boundary(action, stop_after, status):
    def control(progress):
        return action if progress.done == stop_after else Action.CONTINUE

    experiment, got, _, attrs, _ = run(control)
    assert got == status
    assert experiment.calls[-1] == "finish"
    assert len(experiment.calls) == stop_after + 2
    assert attrs["n_done"] == stop_after


def test_control_sees_each_points_scalars_and_times():
    seen = []

    def control(progress):
        seen.append(progress)
        return Action.CONTINUE

    *_, stored_t_end = run(control)
    assert seen[0].scalars is None and seen[0].t_start is None
    assert [p.scalars for p in seen[1:]] == [{"y": 2.0}, {"y": 4.0}, {"y": 6.0}]
    for progress, stored in zip(seen[1:], stored_t_end, strict=True):
        assert progress.t_start <= progress.t_end == stored


def test_devices_closed_even_on_failure(fake_devices):
    class UsesDevice(Experiment):
        cam = Device("Rb ThorCam")

        def run_point(self):
            self.cam
            raise RuntimeError("after opening")

    assert run_local(UsesDevice).status == "failed"
    assert [client.closed for client in fake_devices] == [True]
