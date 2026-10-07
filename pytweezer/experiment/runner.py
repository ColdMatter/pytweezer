"""Run an experiment over its scan points and write the measurement file.

:func:`run_points` is the one execution loop, shared by the manager's worker
process and by :func:`run_local` for interactive use.
"""

import inspect
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pytweezer.experiment.experiment import Experiment
from pytweezer.experiment.scan import Point, Scan
from pytweezer.experiment.simulation import SimulatedDevices
from pytweezer.experiment.storage import (
    Measurement,
    MeasurementWriter,
    git_provenance,
    load_measurement,
)
from pytweezer.experiment.task import Action, TaskStatus


@dataclass
class Progress:
    done: int
    total: int
    point: Point | None = None
    scalars: dict[str, float] | None = None


#: Called after prepare() and after every point. Returns what to do next; it
#: may block (e.g. while paused) before answering.
Control = Callable[[Progress], Action]


def open_writer(
    path: Path | str | None,
    experiment_cls: type[Experiment],
    args: dict[str, Any],
    scan: Scan,
    header: dict[str, Any],
    *,
    simulate: bool = False,
) -> tuple[Experiment, list[Point], MeasurementWriter]:
    """Instantiate the experiment, expand the scan and create its measurement file.

    With ``simulate`` the experiment's devices are simulated in-process.
    """
    experiment = experiment_cls(args)
    if simulate:
        experiment.simulated_devices = SimulatedDevices()
    points = scan.points(experiment_cls)
    writer = MeasurementWriter(
        path,
        header={
            "experiment": experiment_cls.__module__,
            "class_name": experiment_cls.__name__,
            **git_provenance(),
            "simulated": simulate,
            **header,
        },
        argument_values=experiment.argument_values(),
        argument_schema=experiment_cls.schema()["arguments"],
        scan=scan,
        points=points,
        sources=_sources(experiment_cls),
    )
    return experiment, points, writer


def _sources(experiment_cls: type) -> dict[str, str]:
    module = inspect.getmodule(experiment_cls)
    try:
        return {experiment_cls.__module__: inspect.getsource(module)}
    except (OSError, TypeError):
        # Classes defined interactively have no source file.
        return {}


def run_points(
    experiment: Experiment,
    points: list[Point],
    writer: MeasurementWriter,
    control: Control | None = None,
) -> tuple[TaskStatus, str | None]:
    """Run ``prepare``, every point, and ``finish``; finalise the file. Returns ``(status, error)``."""
    control = control or (lambda progress: Action.CONTINUE)
    total = len(points)
    status, error = TaskStatus.COMPLETED, None

    def stop_requested(action: Action) -> bool:
        nonlocal status
        if action == Action.TERMINATE:
            status = TaskStatus.TERMINATED
        elif action == Action.INTERRUPT:
            status = TaskStatus.INTERRUPTED
        return status != TaskStatus.COMPLETED

    experiment._recorder = writer.record
    try:
        experiment.prepare()
        if not stop_requested(control(Progress(0, total))):
            for point in points:
                experiment.point = point
                for name, value in point.values.items():
                    setattr(experiment, name, value)
                writer.begin_point(point)
                experiment.run_point()
                writer.end_point()
                experiment.point = None
                progress = Progress(point.index + 1, total, point, writer.point_scalars)
                if stop_requested(control(progress)):
                    break
    except KeyboardInterrupt:
        status = TaskStatus.TERMINATED
    except Exception:
        status, error = TaskStatus.FAILED, traceback.format_exc()
    finally:
        experiment.point = None
        writer.abandon_point()
        try:
            experiment.finish()
        except Exception:
            if status != TaskStatus.FAILED:
                status, error = TaskStatus.FAILED, traceback.format_exc()
        experiment.close_devices()
        experiment._recorder = None
        writer.finalise(status, error)
    return status, error


def run_local(
    experiment_cls: type[Experiment],
    scan: Scan | None = None,
    *,
    path: Path | str | None = None,
    label: str = "",
    simulate: bool = False,
    **args: Any,
) -> Measurement:
    """Run ``experiment_cls`` in this process, bypassing the queue.

    For notebooks and debugging. The measurement is kept in memory unless
    ``path`` is given. ``simulate=True`` uses in-process simulated devices.
    A failure doesn't raise: check ``.status`` and ``.attrs["error"]`` on the
    returned measurement.
    """
    experiment, points, writer = open_writer(
        path,
        experiment_cls,
        args,
        scan or Scan(),
        {"rid": -1, "label": label},
        simulate=simulate,
    )
    try:
        run_points(experiment, points, writer)
        measurement = load_measurement(writer.file)
        measurement.path = writer.path
        return measurement
    finally:
        writer.close()
