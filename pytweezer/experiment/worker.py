"""Entry point of the process that runs one task for the Experiment Manager.

``python -m pytweezer.experiment.worker --endpoint tcp://host:port --rid N --token T``

The worker asks the manager for its task, runs it, and reports after every
point; the reply says whether to continue, pause or terminate. It may open
only the devices the task reserves, and fails the task before opening any if
the class now reserves a device that was not reserved when it was queued. If
the manager stays unreachable for ``orphan_timeout`` seconds the worker stops
at the next point and marks the measurement ``interrupted``.

Workers bind no ports, so ``pytweezer-kill-stale`` doesn't look for them; the
manager finds and reconciles its workers itself.
"""

import argparse
import functools
import importlib
import time
import traceback
from typing import Any

from pytweezer.experiment.client import ManagerUnavailable, ReqClient
from pytweezer.experiment.reservations import ReservationError
from pytweezer.experiment.runner import Progress, open_writer, run_points
from pytweezer.experiment.task import Action, Task, TaskStatus
from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_client import restrict_devices

logger = get_logger("pytweezer.experiment.worker")

PAUSE_HEARTBEAT_S = 1.0
RETRY_INTERVAL_S = 1.0


class ManagerLink:
    """The worker's side of the conversation with the manager."""

    def __init__(
        self,
        client: ReqClient,
        rid: int,
        token: str,
        orphan_timeout: float = 30.0,
    ) -> None:
        self.client = client
        self.rid = rid
        self.token = token
        self.orphan_timeout = orphan_timeout

    def send(self, event: str, **fields: Any) -> dict[str, Any] | None:
        """Send one event, retrying until ``orphan_timeout``; ``None`` if the manager is gone."""
        payload = {
            "command": "worker",
            "rid": self.rid,
            "token": self.token,
            "event": event,
            **fields,
        }
        give_up = time.monotonic() + self.orphan_timeout
        while True:
            try:
                return self.client.request(payload)
            except ManagerUnavailable:
                if time.monotonic() > give_up:
                    logger.error(
                        "Experiment Manager unreachable for %.0f s", self.orphan_timeout
                    )
                    return None
                time.sleep(RETRY_INTERVAL_S)

    def _action(self, reply: dict[str, Any] | None) -> Action:
        if reply is None:
            return Action.INTERRUPT
        return Action(reply.get("action", Action.TERMINATE))

    def control(self, progress: Progress) -> Action:
        point = progress.point
        action = self._action(
            self.send(
                "point",
                done=progress.done,
                total=progress.total,
                index=point.index if point else None,
                values=point.values if point else {},
                scalars=progress.scalars or {},
                t_start=progress.t_start,
                t_end=progress.t_end,
            )
        )
        while action == Action.PAUSE:
            time.sleep(PAUSE_HEARTBEAT_S)
            action = self._action(self.send("heartbeat", paused=True))
        return action


def resolve_class(module_name: str, qualname: str) -> type:
    module = importlib.import_module(module_name)
    return functools.reduce(getattr, qualname.split("."), module)


def run(link: ManagerLink) -> int:
    reply = link.send("started")
    if reply is None or "task" not in reply:
        logger.error("Could not get task %s from the manager", link.rid)
        return 1
    link.orphan_timeout = float(reply.get("orphan_timeout", link.orphan_timeout))
    task = Task.model_validate(reply["task"])
    if task.devices is None:
        logger.error("Task %s has no device reservation", task.rid)
        link.send(
            "finished", status=TaskStatus.FAILED, error="task has no device reservation"
        )
        return 1
    restrict_devices(task.devices)

    def fail(error: str) -> int:
        logger.error("Task %s could not start:\n%s", task.rid, error)
        link.send("finished", status=TaskStatus.FAILED, error=error)
        return 1

    try:
        experiment_cls = resolve_class(task.experiment, task.class_name)
    except Exception:
        return fail(traceback.format_exc())
    try:
        unreserved = set(experiment_cls.reserved_devices()) - set(task.devices)
    except ReservationError as error:
        return fail(str(error))
    if unreserved:
        names = ", ".join(repr(name) for name in sorted(unreserved))
        verb = "was" if len(unreserved) == 1 else "were"
        return fail(
            f"{experiment_cls.__name__} now reserves {names}, which {verb} not "
            "reserved when it was queued; resubmit it"
        )

    try:
        experiment, points, writer = open_writer(
            reply["h5_path"],
            experiment_cls,
            task.args,
            task.scan,
            {
                "rid": task.rid,
                "label": task.label,
                "submitter": task.submitter,
                "t_submit": task.t_submit.isoformat(),
            },
            simulate=bool(reply.get("simulate", False)),
            allowed_devices=task.devices,
        )
    except Exception:
        return fail(traceback.format_exc())

    try:
        status, error = run_points(experiment, points, writer, link.control)
    finally:
        writer.close()
    if error:
        logger.error("Task %s failed:\n%s", task.rid, error)
    if status != TaskStatus.INTERRUPTED:
        # Interrupted means the manager is already gone; when it returns it
        # reads the final status from the file.
        link.send("finished", status=status, error=error)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rid", type=int, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args(argv)
    client = ReqClient(args.endpoint, timeout_ms=2000, retries=0)
    try:
        return run(ManagerLink(client, args.rid, args.token))
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
