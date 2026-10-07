"""Tasks: a request to run one experiment with given arguments and scan.

These models are what the manager persists and what crosses the wire between
manager, workers, GUIs and notebooks.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from pytweezer.experiment.scan import Scan

SCHEMA_VERSION = 1


class TaskStatus(StrEnum):
    QUEUED = "queued"
    HELD = "held"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"
    ABORTED = "aborted"
    CRASHED = "crashed"
    INTERRUPTED = "interrupted"

    @property
    def finished(self) -> bool:
        return self not in _ACTIVE


_ACTIVE = {TaskStatus.QUEUED, TaskStatus.HELD, TaskStatus.RUNNING, TaskStatus.PAUSED}


class Action(StrEnum):
    """What a running task should do at the next point boundary."""

    CONTINUE = "continue"
    PAUSE = "pause"
    TERMINATE = "terminate"
    # The manager is unreachable: stop, and record that nobody asked us to.
    INTERRUPT = "interrupt"


class TaskRequest(BaseModel):
    """What a submitter asks for."""

    experiment: str = Field(
        description="dotted module name, e.g. pytweezer.experiments.demo"
    )
    class_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    scan: Scan = Field(default_factory=Scan)
    priority: int = 0
    due_time: datetime | None = None
    label: str = ""
    submitter: str = ""

    @property
    def key(self) -> str:
        return f"{self.experiment}:{self.class_name}"


class Task(TaskRequest):
    """A request once the manager has accepted it."""

    schema_version: int = SCHEMA_VERSION
    rid: int
    status: TaskStatus = TaskStatus.QUEUED
    requested: Action | None = None
    t_submit: datetime
    t_start: datetime | None = None
    t_end: datetime | None = None
    points_done: int = 0
    points_total: int | None = None
    h5_path: str | None = Field(None, description="relative to the data root")
    error: str | None = None
