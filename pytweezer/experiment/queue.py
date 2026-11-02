"""The experiment queue as plain data, and its on-disk store.

:class:`ExperimentQueue` holds every rule about task states, ordering and
which tasks may start together. It has no sockets, threads or processes; the
Experiment Manager wraps it and saves its :class:`QueueState` after every
change.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from pytweezer.experiment.task import Action, Task, TaskRequest, TaskStatus
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.experiment.queue")

HISTORY_LIMIT = 1000
#: Stands for every device while a running task's reservations are unresolved.
UNKNOWN_DEVICES = "(unknown)"
_LIVE = (TaskStatus.RUNNING, TaskStatus.PAUSED)


class QueueError(ValueError):
    """A request that the queue's current state doesn't allow."""


class WorkerRecord(BaseModel):
    """Enough to recognise a running worker process again after a manager restart."""

    rid: int
    token: str
    pid: int
    create_time: float


class QueueState(BaseModel):
    schema_version: int = 2
    next_rid: int = 1
    active: list[Task] = Field(default_factory=list)
    history: list[Task] = Field(default_factory=list)
    last_requests: dict[str, TaskRequest] = Field(default_factory=dict)
    workers: list[WorkerRecord] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _from_version_1(cls, data: Any) -> Any:
        """Version 1 kept a single ``worker`` record, or ``None``."""
        if isinstance(data, dict) and "worker" in data:
            data = dict(data)
            worker = data.pop("worker")
            data["workers"] = [] if worker is None else [worker]
            data["schema_version"] = 2
        return data


def now() -> datetime:
    return datetime.now().astimezone()


class ExperimentQueue:
    def __init__(self, state: QueueState | None = None) -> None:
        self.state = state or QueueState()

    # -- queries -----------------------------------------------------------

    def get(self, rid: int) -> Task:
        for task in self.state.active:
            if task.rid == rid:
                return task
        for task in reversed(self.state.history):
            if task.rid == rid:
                return task
        raise QueueError(f"no task with rid {rid}")

    @property
    def running(self) -> list[Task]:
        """Running and paused tasks, by rid."""
        live = [task for task in self.state.active if task.status in _LIVE]
        return sorted(live, key=lambda task: task.rid)

    def worker(self, rid: int) -> WorkerRecord | None:
        for record in self.state.workers:
            if record.rid == rid:
                return record
        return None

    def ordered(self) -> list[Task]:
        """Waiting tasks in the order they would start (ignoring due times)."""
        waiting = [
            t
            for t in self.state.active
            if t.status in (TaskStatus.QUEUED, TaskStatus.HELD)
        ]
        return sorted(waiting, key=lambda t: (-t.priority, t.rid))

    def heads(self, at: datetime | None = None) -> list[Task]:
        """Each idle pipeline's next task, if one is due, by priority then rid.

        A pipeline is idle when none of its tasks is running or paused. Held
        tasks and tasks not yet due are passed over for the next in line.
        """
        at = at or now()
        busy = {task.pipeline for task in self.running}
        heads: dict[str, Task] = {}
        for task in self.ordered():
            if task.pipeline in busy or task.pipeline in heads:
                continue
            if task.status == TaskStatus.QUEUED and (
                task.due_time is None or task.due_time <= at
            ):
                heads[task.pipeline] = task
        return list(heads.values())

    def startable(self, at: datetime | None = None) -> list[Task]:
        """The heads to start now, none of them sharing a device with another task.

        Heads are taken by priority; each starts if none of its devices is held
        by a running task or by a head taken before it. A head that can't start
        keeps its pipeline waiting, but heads of other pipelines may go ahead
        of it. A head whose devices are unresolved never starts.
        """
        held = self._holders()
        if UNKNOWN_DEVICES in held:
            return []
        claimed = set(held)
        started = []
        for head in self.heads(at):
            if head.devices is None:
                continue
            if claimed.isdisjoint(head.devices):
                started.append(head)
                claimed |= set(head.devices)
        return started

    def waiting_for(self, task: Task) -> dict[str, int]:
        """The devices keeping head ``task`` from starting, each with the rid holding it.

        Empty unless ``task`` heads its pipeline and a running task holds one
        of its devices.
        """
        if task.status != TaskStatus.QUEUED:
            return {}
        if task.rid not in {head.rid for head in self.heads()}:
            return {}
        return self._clashes(task)

    def _holders(self) -> dict[str, int]:
        """Each device held by a running task, mapped to that task's rid."""
        held = {}
        for task in self.running:
            if task.devices is None:
                return {UNKNOWN_DEVICES: task.rid}
            held.update(dict.fromkeys(task.devices, task.rid))
        return held

    def _clashes(self, task: Task) -> dict[str, int]:
        held = self._holders()
        if UNKNOWN_DEVICES in held:
            return held
        return {device: held[device] for device in task.devices or () if device in held}

    def action_for(self, rid: int) -> Action:
        try:
            task = self._running(rid, "act on")
        except QueueError:
            return Action.TERMINATE
        return task.requested or Action.CONTINUE

    def snapshot(self, history: int = 50) -> dict[str, Any]:
        return {
            "running": [task.model_dump(mode="json") for task in self.running],
            "queue": [
                {**task.model_dump(mode="json"), "waiting_for": self.waiting_for(task)}
                for task in self.ordered()
            ],
            "history": [
                task.model_dump(mode="json")
                for task in reversed(self.state.history[-history:])
            ],
        }

    def history(self, before_rid: int | None = None, limit: int = 50) -> list[Task]:
        older = [
            task
            for task in reversed(self.state.history)
            if before_rid is None or task.rid < before_rid
        ]
        return older[:limit]

    # -- submitter requests ------------------------------------------------

    def submit(
        self,
        request: TaskRequest,
        at: datetime | None = None,
        devices: list[str] | None = None,
    ) -> Task:
        """Queue ``request``; it can't start until its ``devices`` are known."""
        due = request.due_time
        if due is not None and due.tzinfo is None:
            due = due.astimezone()
        task = Task(
            **request.model_dump(exclude={"due_time"}),
            due_time=due,
            rid=self.state.next_rid,
            t_submit=at or now(),
            devices=devices,
        )
        self.state.next_rid += 1
        self.state.active.append(task)
        self.state.last_requests[request.key] = request
        return task

    def hold(self, rid: int) -> None:
        task = self._waiting(rid, "hold")
        task.status = TaskStatus.HELD

    def release(self, rid: int) -> None:
        task = self._waiting(rid, "release")
        task.status = TaskStatus.QUEUED

    def set_devices(self, rid: int, devices: list[str]) -> None:
        task = self._waiting(rid, "set the devices of")
        task.devices = list(devices)

    def delete(self, rid: int) -> None:
        task = self._waiting(rid, "delete")
        self.state.active.remove(task)

    def set_priority(self, rid: int, priority: int) -> None:
        task = self.get(rid)
        if task.status.finished:
            raise QueueError(f"task {rid} has already finished")
        task.priority = int(priority)

    def pause(self, rid: int) -> None:
        task = self._running(rid, "pause")
        if task.requested != Action.TERMINATE:
            task.requested = Action.PAUSE

    def resume(self, rid: int) -> None:
        task = self._running(rid, "resume")
        if task.requested == Action.PAUSE:
            task.requested = None
        task.status = TaskStatus.RUNNING

    def terminate(self, rid: int) -> None:
        """Stop a running task at its next point boundary, or drop a waiting one."""
        task = self.get(rid)
        if task.status in _LIVE:
            task.requested = Action.TERMINATE
        else:
            self.delete(rid)

    def _waiting(self, rid: int, verb: str) -> Task:
        task = self.get(rid)
        if task.status not in (TaskStatus.QUEUED, TaskStatus.HELD):
            raise QueueError(f"cannot {verb} task {rid}: it is {task.status}")
        return task

    def _running(self, rid: int, verb: str) -> Task:
        for task in self.running:
            if task.rid == rid:
                return task
        raise QueueError(f"cannot {verb} task {rid}: it is not running")

    # -- worker lifecycle --------------------------------------------------

    def mark_started(
        self,
        rid: int,
        worker: WorkerRecord,
        h5_path: str,
        at: datetime | None = None,
    ) -> Task:
        task = self.get(rid)
        if task.devices is None:
            raise QueueError(f"cannot start task {rid}: its devices are not resolved")
        for other in self.running:
            if other.pipeline == task.pipeline:
                raise QueueError(
                    f"cannot start task {rid}: pipeline {task.pipeline!r} is "
                    f"already running task {other.rid}"
                )
        if clashes := self._clashes(task):
            held = ", ".join(
                f"{device!r} held by task {holder}"
                for device, holder in clashes.items()
            )
            raise QueueError(f"cannot start task {rid}: {held}")
        task.status = TaskStatus.RUNNING
        task.t_start = at or now()
        task.h5_path = h5_path
        self.state.workers.append(worker)
        return task

    def mark_progress(self, rid: int, done: int, total: int) -> None:
        task = self._running(rid, "update")
        task.points_done = done
        task.points_total = total

    def mark_paused(self, rid: int) -> None:
        task = self._running(rid, "pause")
        if task.requested == Action.PAUSE:
            task.status = TaskStatus.PAUSED

    def finish(
        self,
        rid: int,
        status: TaskStatus,
        error: str | None = None,
        at: datetime | None = None,
    ) -> Task:
        task = self.get(rid)
        if not status.finished:
            raise QueueError(f"{status} is not a final status")
        if task in self.state.active:
            self.state.active.remove(task)
            self.state.history.append(task)
            del self.state.history[:-HISTORY_LIMIT]
        task.status = status
        task.error = error
        task.requested = None
        task.t_end = at or now()
        # TODO: devices are released however the task ended. After an abort a device
        # call in flight still completes, a killed linked primary can leave its
        # secondary claimed, and a timed-out follower waits for its trigger, so another
        # pipeline may start on a sequencer that is not yet idle.
        self.state.workers = [w for w in self.state.workers if w.rid != rid]
        return task


class ModelStore:
    """Persists one pydantic model as JSON, replacing the file atomically.

    Subclasses set :attr:`model`. A corrupt file is moved aside and an empty
    model returned, so it never stops the manager starting. A file that can't
    be read is retried briefly, then the error is raised and the file left alone.
    """

    model: type[BaseModel]

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> BaseModel:
        try:
            text = self._read_text()
        except FileNotFoundError:
            return self.model()
        try:
            return self.model.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError):
            aside = self.path.with_suffix(f".unreadable-{int(time.time())}.json")
            logger.exception("%s is unreadable; moved to %s", self.path, aside)
            self.path.replace(aside)
            return self.model()

    def _read_text(self) -> str:
        for attempt in range(20):
            try:
                return self.path.read_text()
            except PermissionError:
                # Windows refuses a read while another process holds the file
                # exclusively (a backup tool); it is usually released quickly.
                if attempt == 19:
                    raise
                time.sleep(0.05)

    def _dump(self, state: BaseModel) -> str:
        return state.model_dump_json(indent=1)

    def save(self, state: BaseModel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self._dump(state))
        for attempt in range(20):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                # Windows refuses to replace a file another process has open
                # (an editor, a backup tool); it is usually released quickly.
                if attempt == 19:
                    raise
                time.sleep(0.05)


class QueueStore(ModelStore):
    """Persists a :class:`QueueState`."""

    model = QueueState
