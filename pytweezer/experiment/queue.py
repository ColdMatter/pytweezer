"""The experiment queue as plain data, and its on-disk store.

:class:`ExperimentQueue` holds every rule about task states and ordering. It
has no sockets, threads or processes; the Experiment Manager wraps it and
saves its :class:`QueueState` after every change.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from pytweezer.experiment.task import Action, Task, TaskRequest, TaskStatus
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.experiment.queue")

HISTORY_LIMIT = 1000


class QueueError(ValueError):
    """A request that the queue's current state doesn't allow."""


class WorkerRecord(BaseModel):
    """Enough to recognise a running worker process again after a manager restart."""

    rid: int
    token: str
    pid: int
    create_time: float


class QueueState(BaseModel):
    schema_version: int = 1
    next_rid: int = 1
    active: list[Task] = Field(default_factory=list)
    history: list[Task] = Field(default_factory=list)
    last_requests: dict[str, TaskRequest] = Field(default_factory=dict)
    worker: WorkerRecord | None = None


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
    def running(self) -> Task | None:
        for task in self.state.active:
            if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
                return task
        return None

    def ordered(self) -> list[Task]:
        """Waiting tasks in the order they would start (ignoring due times)."""
        waiting = [
            t
            for t in self.state.active
            if t.status in (TaskStatus.QUEUED, TaskStatus.HELD)
        ]
        return sorted(waiting, key=lambda t: (-t.priority, t.rid))

    def next_due(self, at: datetime | None = None) -> Task | None:
        """The task to start now, or ``None``. Only one task runs at a time."""
        if self.running is not None:
            return None
        at = at or now()
        for task in self.ordered():
            if task.status == TaskStatus.QUEUED and (
                task.due_time is None or task.due_time <= at
            ):
                return task
        return None

    def action_for(self, rid: int) -> Action:
        task = self.running
        if task is None or task.rid != rid:
            return Action.TERMINATE
        return task.requested or Action.CONTINUE

    def snapshot(self, history: int = 50) -> dict[str, Any]:
        running = self.running
        return {
            "running": running.model_dump(mode="json") if running else None,
            "queue": [task.model_dump(mode="json") for task in self.ordered()],
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

    def submit(self, request: TaskRequest, at: datetime | None = None) -> Task:
        due = request.due_time
        if due is not None and due.tzinfo is None:
            due = due.astimezone()
        task = Task(
            **request.model_dump(exclude={"due_time"}),
            due_time=due,
            rid=self.state.next_rid,
            t_submit=at or now(),
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
        if task is self.running:
            task.requested = Action.TERMINATE
        else:
            self.delete(rid)

    def _waiting(self, rid: int, verb: str) -> Task:
        task = self.get(rid)
        if task.status not in (TaskStatus.QUEUED, TaskStatus.HELD):
            raise QueueError(f"cannot {verb} task {rid}: it is {task.status}")
        return task

    def _running(self, rid: int, verb: str) -> Task:
        task = self.running
        if task is None or task.rid != rid:
            raise QueueError(f"cannot {verb} task {rid}: it is not running")
        return task

    # -- worker lifecycle --------------------------------------------------

    def mark_started(
        self,
        rid: int,
        worker: WorkerRecord,
        h5_path: str,
        at: datetime | None = None,
    ) -> Task:
        task = self.get(rid)
        if self.running is not None:
            raise QueueError(f"task {self.running.rid} is already running")
        task.status = TaskStatus.RUNNING
        task.t_start = at or now()
        task.h5_path = h5_path
        self.state.worker = worker
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
        if self.state.worker is not None and self.state.worker.rid == rid:
            self.state.worker = None
        return task


class QueueStore:
    """Persists a :class:`QueueState` as JSON, replacing the file atomically."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> QueueState:
        try:
            return QueueState.model_validate_json(self.path.read_text())
        except FileNotFoundError:
            return QueueState()
        except (ValidationError, json.JSONDecodeError, OSError):
            aside = self.path.with_suffix(f".unreadable-{int(time.time())}.json")
            logger.exception(
                "Queue state %s is unreadable; moved to %s", self.path, aside
            )
            self.path.replace(aside)
            return QueueState()

    def save(self, state: QueueState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=1))
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
