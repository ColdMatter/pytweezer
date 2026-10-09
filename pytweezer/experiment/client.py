"""Talking to the Experiment Manager: a Qt-free REQ client plus notebook helpers.

From a notebook::

    from pytweezer.experiment import Scan, LinearAxis
    from pytweezer.experiment.client import submit, submit_recipe, wait
    from pytweezer.experiments.demo import RabiDemo

    rid = submit(RabiDemo, Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=2e-5, n=21)]))
    task = wait(rid)
    rid = submit_recipe(RabiDemo, "nightly check", atoms=300)
"""

import getpass
import socket
import time
from datetime import datetime
from typing import Any

import zmq

from pytweezer.configuration.config import get_config
from pytweezer.experiment.arguments import coerce_arguments
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.scan import Scan
from pytweezer.experiment.task import Task, TaskRequest

SERVER_NAME = "Experiment Manager"


class ManagerUnavailable(ConnectionError):
    """The manager didn't answer in time."""


class ManagerError(RuntimeError):
    """The manager answered, refusing the request."""


def manager_conf(server_name: str = SERVER_NAME) -> dict[str, Any]:
    try:
        return get_config()["Servers"][server_name]
    except KeyError:
        raise KeyError(f'no CONFIG["Servers"][{server_name!r}] entry') from None


def connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("*", "0.0.0.0") else host


def rep_endpoint(server_name: str = SERVER_NAME) -> str:
    conf = manager_conf(server_name)
    return f"tcp://{connect_host(conf['host'])}:{conf['port']}"


#: The manager's published state; layout in :mod:`~pytweezer.servers.experiment_manager`.
NOTIFIER_NAME = "experiment"


def sync_address(server_name: str = SERVER_NAME) -> tuple[str, int]:
    """``(host, port)`` where the manager publishes its state (sipyco sync_struct)."""
    conf = manager_conf(server_name)
    return connect_host(conf["host"]), conf["sync_port"]


class ReqClient:
    """REQ socket that recovers from a lost reply by recreating itself (the
    ZMQ "lazy pirate" pattern), so a dead server never wedges the caller.
    """

    def __init__(self, endpoint: str, timeout_ms: int = 2000, retries: int = 1) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        for _attempt in range(self.retries + 1):
            if self._socket is None:
                self._socket = self._context.socket(zmq.REQ)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.connect(self.endpoint)
            self._socket.send_json(payload)
            if self._socket.poll(self.timeout_ms, zmq.POLLIN):
                return self._socket.recv_json()
            self.close()
        raise ManagerUnavailable(
            f"no reply from {self.endpoint} within {self.timeout_ms} ms"
        )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class ExperimentManagerClient:
    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout_ms: int = 2000,
        retries: int = 1,
    ) -> None:
        self._req = ReqClient(endpoint or rep_endpoint(), timeout_ms, retries)

    def call(self, command: str, **fields: Any) -> dict[str, Any]:
        reply = self._req.request({"command": command, **fields})
        if not reply.get("ok"):
            raise ManagerError(reply.get("error", "unknown error"))
        return reply

    def close(self) -> None:
        self._req.close()

    def ping(self) -> dict[str, Any]:
        return self.call("ping")

    def snapshot(self) -> dict[str, Any]:
        return self.call("snapshot")["snapshot"]

    def catalogue(self) -> list[dict[str, Any]]:
        return self.call("catalogue")["modules"]

    def submit(self, request: TaskRequest) -> int:
        return self.call("submit", request=request.model_dump(mode="json"))["rid"]

    def get(self, rid: int) -> Task:
        return Task.model_validate(self.call("get", rid=rid)["task"])

    def history(self, before_rid: int | None = None, limit: int = 50) -> list[Task]:
        tasks = self.call("history", before_rid=before_rid, limit=limit)["tasks"]
        return [Task.model_validate(task) for task in tasks]

    def last_request(self, experiment: str, class_name: str) -> TaskRequest | None:
        reply = self.call("last_request", experiment=experiment, class_name=class_name)
        request = reply["request"]
        return TaskRequest.model_validate(request) if request else None

    def recipes(
        self, experiment: str | None = None, class_name: str | None = None
    ) -> list[Recipe]:
        reply = self.call("recipes", experiment=experiment, class_name=class_name)
        return [Recipe.model_validate(recipe) for recipe in reply["recipes"]]

    def save_recipe(self, recipe: Recipe, overwrite: bool = False) -> None:
        self.call(
            "save_recipe", recipe=recipe.model_dump(mode="json"), overwrite=overwrite
        )

    def delete_recipe(self, experiment: str, class_name: str, name: str) -> None:
        self.call(
            "delete_recipe", experiment=experiment, class_name=class_name, name=name
        )

    def submit_recipe(
        self,
        experiment: str,
        class_name: str,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        priority: int | None = None,
        label: str | None = None,
        submitter: str = "",
    ) -> int:
        return self.call(
            "submit_recipe",
            experiment=experiment,
            class_name=class_name,
            name=name,
            args=args or {},
            priority=priority,
            label=label,
            submitter=submitter,
        )["rid"]

    def set_priority(self, rid: int, priority: int) -> None:
        self.call("set_priority", rid=rid, priority=priority)

    def pause(self, rid: int) -> None:
        self.call("pause", rid=rid)

    def resume(self, rid: int) -> None:
        self.call("resume", rid=rid)

    def terminate(self, rid: int) -> None:
        self.call("terminate", rid=rid)

    def abort(self, rid: int) -> None:
        self.call("abort", rid=rid)

    def hold(self, rid: int) -> None:
        self.call("hold", rid=rid)

    def release(self, rid: int) -> None:
        self.call("release", rid=rid)

    def delete(self, rid: int) -> None:
        self.call("delete", rid=rid)


def submitter_name() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


def _resolve(
    experiment: type | str,
    scan: Scan | None = None,
    args: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """``(module, class_name)`` for ``experiment``, checked against the arguments."""
    if isinstance(experiment, str):
        module, _, class_name = experiment.partition(":")
    else:
        module, class_name = experiment.__module__, experiment.__qualname__
        # Fail here rather than when the task reaches the front of the queue.
        coerce_arguments(experiment, args or {})
        if scan is not None:
            scan.axis_values(experiment)
    if module == "__main__" or "<locals>" in class_name or not class_name:
        raise ValueError(
            f"{experiment!r} can't be imported by the manager; define it in a module "
            "under pytweezer/experiments/, or run it here with run_local()"
        )
    return module, class_name


def submit(
    experiment: type | str,
    scan: Scan | None = None,
    *,
    priority: int = 0,
    label: str = "",
    due_time: datetime | None = None,
    client: ExperimentManagerClient | None = None,
    **args: Any,
) -> int:
    """Queue an experiment and return its rid.

    ``experiment`` is an Experiment subclass importable by the manager (so not
    one defined in a notebook: use :func:`~pytweezer.experiment.run_local` for
    those) or a ``"module:ClassName"`` string.
    """
    scan = scan or Scan()
    module, class_name = _resolve(experiment, scan, args)
    request = TaskRequest(
        experiment=module,
        class_name=class_name,
        args=args,
        scan=scan,
        priority=priority,
        label=label,
        due_time=due_time,
        submitter=submitter_name(),
    )
    client = client or ExperimentManagerClient()
    return client.submit(request)


def save_recipe(
    experiment: type | str,
    name: str,
    scan: Scan | None = None,
    /,
    *,
    priority: int = 0,
    label: str = "",
    overwrite: bool = False,
    client: ExperimentManagerClient | None = None,
    **args: Any,
) -> None:
    """Save these settings as recipe ``name``, shared with every PC.

    With a class, every argument is stored (defaults included), so a later
    change of default does not change what the recipe runs.
    """
    scan = scan or Scan()
    module, class_name = _resolve(experiment, scan, args)
    if not isinstance(experiment, str):
        scanned = {axis.argument for axis in scan.axes}
        args = {
            key: value
            for key, value in coerce_arguments(experiment, args).items()
            if key not in scanned
        }
    recipe = Recipe(
        experiment=module,
        class_name=class_name,
        name=name,
        args=args,
        scan=scan,
        priority=priority,
        label=label,
        submitter=submitter_name(),
    )
    (client or ExperimentManagerClient()).save_recipe(recipe, overwrite)


def submit_recipe(
    experiment: type | str,
    name: str,
    /,
    *,
    priority: int | None = None,
    label: str | None = None,
    client: ExperimentManagerClient | None = None,
    **args: Any,
) -> int:
    """Queue recipe ``name`` with ``args`` overriding it; return the rid."""
    module, class_name = _resolve(experiment, args=args)
    return (client or ExperimentManagerClient()).submit_recipe(
        module,
        class_name,
        name,
        args=args,
        priority=priority,
        label=label,
        submitter=submitter_name(),
    )


def recipes(
    experiment: type | str | None = None,
    /,
    *,
    client: ExperimentManagerClient | None = None,
) -> list[Recipe]:
    """Saved recipes, for one experiment or for all."""
    module = class_name = None
    if experiment is not None:
        module, class_name = _resolve(experiment)
    return (client or ExperimentManagerClient()).recipes(module, class_name)


def delete_recipe(
    experiment: type | str,
    name: str,
    /,
    *,
    client: ExperimentManagerClient | None = None,
) -> None:
    module, class_name = _resolve(experiment)
    (client or ExperimentManagerClient()).delete_recipe(module, class_name, name)


def wait(
    rid: int,
    *,
    poll_s: float = 1.0,
    timeout_s: float | None = None,
    client: ExperimentManagerClient | None = None,
) -> Task:
    """Block until task ``rid`` has finished, and return it."""
    client = client or ExperimentManagerClient()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        task = client.get(rid)
        if task.status.finished:
            return task
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"task {rid} still {task.status} after {timeout_s} s")
        time.sleep(poll_s)
