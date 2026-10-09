"""The Experiment Manager: owns the experiment queue and runs one task at a time.

A single-threaded loop serves REQ/REP commands (from GUIs, notebooks and the
worker) and starts each due task in a fresh worker subprocess
(:mod:`pytweezer.experiment.worker`). Its state is published as the sipyco
notifier ``"experiment"`` on ``sync_port`` (see :data:`NOTIFIER_NAME` for the
layout), so every GUI holds a live copy of the queue and of the points the
current task has measured so far.

The queue state is saved after every change: on Windows the GUI stops this
process with ``TerminateProcess``, so nothing can rely on a clean shutdown. A
worker survives its manager; on restart the manager adopts it again if it is
still running, or records how its task ended.
"""

import argparse
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import psutil
import zmq
from sipyco.sync_struct import Notifier

from pytweezer.configuration.config import get_config
from pytweezer.configuration.paths import tweezerpath
from pytweezer.database.writer import DBWriter
from pytweezer.experiment.catalogue import Catalogue
from pytweezer.experiment.client import NOTIFIER_NAME, connect_host
from pytweezer.experiment.queue import (
    ExperimentQueue,
    QueueError,
    QueueStore,
    WorkerRecord,
    now,
)
from pytweezer.experiment.recipes import (
    Recipe,
    RecipeBook,
    RecipeError,
    RecipeStore,
    unknown_arguments,
)
from pytweezer.experiment.storage import (
    data_root,
    highest_rid,
    measurement_relpath,
    read_arguments,
    read_header,
)
from pytweezer.experiment.task import Action, TaskRequest, TaskStatus
from pytweezer.logging_utils import get_daily_log_path, get_logger
from pytweezer.servers.sync import SyncServer

logger = get_logger("pytweezer.servers.experiment_manager")

#: The published state (:data:`~pytweezer.experiment.client.NOTIFIER_NAME`)
#: is laid out as::
#:
#:     {"started", "catalogue_version", "recipes_version", "simulated",
#:      "alive",                         # epoch s, at least every PUBLISH_INTERVAL_S
#:      "running", "queue", "history",   # as ExperimentQueue.snapshot()
#:      "points": {"rid": int | None,    # the task now (or last) running
#:                 "rows": [{"index", "values", "scalars", "t_start", "t_end"}]}}
PUBLISH_INTERVAL_S = 2.0
CATALOGUE_RESCAN_S = 10.0
_WORKER_COMMAND = [sys.executable, "-m", "pytweezer.experiment.worker"]


class ExperimentManager:
    def __init__(
        self,
        server_name: str = "Experiment Manager",
        *,
        root: Path | str | None = None,
        catalogue: Catalogue | None = None,
        bind: bool = True,
        db: DBWriter | None = None,
    ) -> None:
        self.conf = get_config()["Servers"][server_name]
        self.root = Path(root) if root is not None else data_root()
        self.orphan_timeout = float(self.conf.get("orphan_timeout", 30.0))
        #: Workers use in-process simulated devices instead of device servers.
        self.simulate = bool(self.conf.get("simulate", False))
        host, port = self.conf["host"], self.conf["port"]
        self.rep_address = f"tcp://{host}:{port}"
        self.worker_endpoint = f"tcp://{connect_host(host)}:{port}"
        self.log_dir = get_daily_log_path().parent / "experiments"

        self.store = QueueStore(self.root / "queue_state.json")
        self.queue = ExperimentQueue(self.store.load())
        self.recipe_store = RecipeStore(self.root / "recipes.json")
        self.recipes = RecipeBook(self.recipe_store.load())
        state = self.queue.state
        state.next_rid = max(state.next_rid, highest_rid(self.root) + 1)
        self.catalogue = catalogue or Catalogue()
        #: Receives a ``runs`` row per start/finish and a ``points`` row per point.
        self.db = db if db is not None else DBWriter()

        self.process: psutil.Process | None = None
        self._abort_requested: int | None = None
        self._running = True
        # Bumped whenever the catalogue changes, so GUIs know to fetch it again.
        self._catalogue_version = 0
        self._recipes_version = 0
        self._dirty = True
        self._last_publish = 0.0
        self._last_rescan = 0.0
        self.started = now().isoformat()

        #: Without ``bind`` nothing is published and mutations apply directly.
        self.notifier = Notifier(
            {**self._snapshot(), "points": {"rid": None, "rows": []}}
        )
        self.rep = self.sync = None
        if bind:
            context = zmq.Context.instance()
            self.rep = context.socket(zmq.REP)
            self.rep.setsockopt(zmq.LINGER, 0)
            self.rep.bind(self.rep_address)
            self.sync = SyncServer(
                {NOTIFIER_NAME: self.notifier}, host, self.conf["sync_port"]
            )

        self._reconcile_worker()
        self._save()

    # -- main loop -----------------------------------------------------------

    def serve_forever(self) -> None:
        logger.info("Experiment Manager serving on %s", self.rep_address)
        if self.simulate:
            logger.warning(
                "Experiment Manager in SIMULATION MODE: devices are simulated, "
                "data goes to %s",
                self.root,
            )
        try:
            while self._running:
                if self.rep.poll(100, zmq.POLLIN):
                    request = self.rep.recv_json()
                    self.rep.send_json(self.handle(request))
                self.tick()
        finally:
            # Deliberately leave a running worker alone: the next manager adopts it.
            self.catalogue.close()
            self.db.close()
            self.rep.close(linger=0)
            self.sync.close()

    def stop(self) -> None:
        self._running = False

    def tick(self) -> None:
        self._reap_worker()
        self._start_next()
        monotonic = time.monotonic()
        if monotonic - self._last_rescan > CATALOGUE_RESCAN_S:
            self._last_rescan = monotonic
            self._catalogue_changed(self.catalogue.refresh())
        self._catalogue_changed(self.catalogue.poll())
        if self._dirty or monotonic - self._last_publish > PUBLISH_INTERVAL_S:
            self._publish_queue()

    # -- requests ------------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {command!r}"}
        try:
            return {"ok": True, **(handler(request) or {})}
        except (QueueError, KeyError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        except Exception as error:
            logger.exception("Error handling %r", command)
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def _cmd_ping(self, request):
        return {"started": self.started, "pid": os.getpid(), "simulated": self.simulate}

    def _cmd_snapshot(self, request):
        return {"snapshot": self._snapshot()}

    def _cmd_catalogue(self, request):
        self._catalogue_changed(self.catalogue.refresh())
        return {
            "modules": self.catalogue.entries(),
            "busy": self.catalogue.busy,
            "version": self._catalogue_version,
        }

    def _cmd_get(self, request):
        return {"task": self.queue.get(int(request["rid"])).model_dump(mode="json")}

    def _cmd_history(self, request):
        tasks = self.queue.history(
            request.get("before_rid"), int(request.get("limit", 50))
        )
        return {"tasks": [task.model_dump(mode="json") for task in tasks]}

    def _cmd_last_request(self, request):
        key = f"{request['experiment']}:{request['class_name']}"
        last = self.queue.state.last_requests.get(key)
        return {"request": last.model_dump(mode="json") if last else None}

    def _cmd_submit(self, request):
        task = self.queue.submit(TaskRequest.model_validate(request["request"]))
        logger.info("Queued task %s: %s.%s", task.rid, task.experiment, task.class_name)
        self._changed()
        return {"rid": task.rid}

    def _cmd_recipes(self, request):
        recipes = self.recipes.find(
            request.get("experiment"), request.get("class_name")
        )
        return {
            "recipes": [recipe.model_dump(mode="json") for recipe in recipes],
            "version": self._recipes_version,
        }

    def _cmd_save_recipe(self, request):
        recipe = self.recipes.save(
            self._with_default_arguments(Recipe.model_validate(request["recipe"])),
            overwrite=bool(request.get("overwrite")),
        )
        logger.info(
            "Saved recipe %r for %s.%s",
            recipe.name,
            recipe.experiment,
            recipe.class_name,
        )
        self._recipes_changed()

    def _with_default_arguments(self, recipe: Recipe) -> Recipe:
        """``recipe`` with every unscanned declared argument given, defaults filled in.

        A recipe whose experiment the catalogue cannot describe is kept as given.
        """
        try:
            schema = self._schema(recipe.experiment, recipe.class_name)
        except RecipeError:
            return recipe
        scanned = {axis.argument for axis in recipe.scan.axes}
        defaults = {
            name: argument["default"]
            for name, argument in schema["arguments"].items()
            if name not in scanned
        }
        return recipe.model_copy(update={"args": {**defaults, **recipe.args}})

    def _cmd_delete_recipe(self, request):
        self.recipes.delete(
            request["experiment"], request["class_name"], request["name"]
        )
        self._recipes_changed()

    def _cmd_submit_recipe(self, request):
        recipe = self.recipes.get(
            request["experiment"], request["class_name"], request["name"]
        )
        overrides = request.get("args") or {}
        task_request = recipe.replay(
            overrides,
            priority=request.get("priority"),
            label=request.get("label"),
            submitter=request.get("submitter", ""),
        )
        unknown = unknown_arguments(
            task_request, self._schema(recipe.experiment, recipe.class_name)
        )
        if bad_overrides := [name for name in unknown if name in overrides]:
            raise RecipeError(f"{recipe.class_name} has no argument(s) {bad_overrides}")
        if unknown:
            raise RecipeError(
                f"{recipe.class_name} no longer has argument(s) {unknown}; load "
                f"recipe {recipe.name!r} into the form, check it and save it again"
            )
        task = self.queue.submit(task_request)
        logger.info("Queued task %s from recipe %r", task.rid, recipe.name)
        self._changed()
        return {"rid": task.rid}

    def _schema(self, module: str, class_name: str) -> dict[str, Any]:
        for entry in self.catalogue.entries():
            if entry["module"] != module:
                continue
            if entry.get("error"):
                reason = entry["error"].strip().splitlines()[-1]
                raise RecipeError(f"{module} fails to import: {reason}")
            for schema in entry["classes"]:
                if schema["class_name"] == class_name:
                    return schema
        if self.catalogue.busy:
            raise RecipeError(
                f"{module}.{class_name} is not in the catalogue yet; the manager "
                "is still reading the experiments, so try again shortly"
            )
        raise RecipeError(f"{module}.{class_name} is not in the catalogue")

    def _cmd_set_priority(self, request):
        self.queue.set_priority(int(request["rid"]), int(request["priority"]))
        self._changed()

    def _cmd_hold(self, request):
        self.queue.hold(int(request["rid"]))
        self._changed()

    def _cmd_release(self, request):
        self.queue.release(int(request["rid"]))
        self._changed()

    def _cmd_delete(self, request):
        self.queue.delete(int(request["rid"]))
        self._changed()

    def _cmd_pause(self, request):
        self.queue.pause(int(request["rid"]))
        self._changed()

    def _cmd_resume(self, request):
        self.queue.resume(int(request["rid"]))
        self._changed()

    def _cmd_terminate(self, request):
        self.queue.terminate(int(request["rid"]))
        self._changed()

    def _cmd_abort(self, request):
        """Kill the worker now. A device call already in flight still completes on the device."""
        rid = int(request["rid"])
        task = self.queue.running
        if task is None or task.rid != rid:
            raise QueueError(f"cannot abort task {rid}: it is not running")
        if self.process is None:
            self._settle_unreported(rid, TaskStatus.ABORTED, None)
            return
        self._abort_requested = rid
        try:
            self.process.kill()
            self.process.wait(timeout=5)
        except psutil.Error:
            pass
        self._reap_worker()

    def _cmd_worker(self, request):
        rid, token = int(request["rid"]), request.get("token")
        record = self.queue.state.worker
        if record is None or record.rid != rid or record.token != token:
            # A worker this manager no longer recognises: stop it.
            return {"action": Action.TERMINATE}
        event = request.get("event")
        task = self.queue.get(rid)
        if event == "started":
            return {
                "action": self.queue.action_for(rid),
                "task": task.model_dump(mode="json"),
                "h5_path": str(self.root / task.h5_path),
                "orphan_timeout": self.orphan_timeout,
                "simulate": self.simulate,
            }
        if event == "point":
            self.queue.mark_progress(rid, int(request["done"]), int(request["total"]))
            if request.get("index") is not None:
                self.db.record_point(
                    rid,
                    request["index"],
                    request.get("t_start"),
                    request.get("t_end"),
                    request.get("values", {}),
                    request.get("scalars", {}),
                )
                self._add_point(
                    rid,
                    {
                        "index": request["index"],
                        "values": request.get("values", {}),
                        "scalars": request.get("scalars", {}),
                        "t_start": request.get("t_start"),
                        "t_end": request.get("t_end"),
                    },
                )
            self._dirty = True
        elif event == "heartbeat":
            if request.get("paused") and task.status != TaskStatus.PAUSED:
                self.queue.mark_paused(rid)
                self._changed()
        elif event == "finished":
            self._finish(rid, TaskStatus(request["status"]), request.get("error"))
            return {"action": Action.TERMINATE}
        else:
            raise ValueError(f"unknown worker event {event!r}")
        return {"action": self.queue.action_for(rid)}

    # -- workers -------------------------------------------------------------

    def _start_next(self) -> None:
        if self.process is not None:
            return  # the previous worker is still exiting
        task = self.queue.next_due()
        if task is None:
            return
        token = secrets.token_hex(8)
        relpath = measurement_relpath(task.rid, task.class_name, now())
        try:
            self.process = self._spawn(task.rid, token)
        except OSError as error:
            self._finish(
                task.rid, TaskStatus.FAILED, f"could not start worker: {error}"
            )
            return
        record = WorkerRecord(
            rid=task.rid,
            token=token,
            pid=self.process.pid,
            create_time=self.process.create_time(),
        )
        self.queue.mark_started(task.rid, record, relpath.as_posix())
        self._mutate(self.notifier.__setitem__, "points", {"rid": task.rid, "rows": []})
        self._record_run(task.rid)
        logger.info("Started task %s (worker pid %s)", task.rid, self.process.pid)
        self._changed()

    def _spawn(self, rid: int, token: str) -> psutil.Popen:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with open(self.log_dir / f"{rid:06d}.log", "ab") as log:
            # Output goes to a file, not a pipe, so the worker outlives this process.
            return psutil.Popen(
                [
                    *_WORKER_COMMAND,
                    "--endpoint",
                    self.worker_endpoint,
                    "--rid",
                    str(rid),
                    "--token",
                    token,
                ],
                cwd=tweezerpath,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )

    def _reap_worker(self) -> None:
        if self.process is None or _alive(self.process):
            return
        returncode = getattr(self.process, "returncode", None)
        self.process = None
        record = self.queue.state.worker
        if record is None:
            return  # the worker reported its own end
        aborted = self._abort_requested == record.rid
        self._abort_requested = None
        status = TaskStatus.ABORTED if aborted else TaskStatus.CRASHED
        error = (
            None
            if aborted
            else f"worker exited with code {returncode} without reporting"
        )
        self._settle_unreported(record.rid, status, error)

    def _reconcile_worker(self) -> None:
        """After a restart: adopt a still-running worker, or settle its task."""
        record = self.queue.state.worker
        if record is not None:
            try:
                process = psutil.Process(record.pid)
                if abs(process.create_time() - record.create_time) < 0.01 and _alive(
                    process
                ):
                    self.process = process
                    logger.info("Adopted running worker for task %s", record.rid)
                    return
            except psutil.Error:
                pass
            self._settle_unreported(record.rid, TaskStatus.INTERRUPTED, None)
        for task in list(self.queue.state.active):
            if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
                self._settle_unreported(task.rid, TaskStatus.INTERRUPTED, None)

    def _settle_unreported(
        self, rid: int, fallback: TaskStatus, error: str | None
    ) -> None:
        """Finish a task whose worker ended without telling us, trusting its file if it got that far."""
        task = self.queue.get(rid)
        path = self.root / task.h5_path if task.h5_path else None
        status = fallback
        if path is not None and path.exists():
            try:
                header = read_header(path)
                recorded = TaskStatus(header.get("status", ""))
                if recorded.finished:
                    status, error = recorded, header.get("error") or None
                else:
                    with h5py.File(path, "r+", locking=False) as f:
                        f.attrs.update(status=str(fallback), error=error or "")
            except (OSError, ValueError):
                logger.warning("Could not read or update %s", path, exc_info=True)
        self._finish(rid, status, error)

    def _finish(self, rid: int, status: TaskStatus, error: str | None) -> None:
        self.queue.finish(rid, status, error)
        self._record_run(rid)
        logger.info("Task %s %s", rid, status)
        if error:
            logger.warning("Task %s error: %s", rid, error.strip().splitlines()[-1])
        self._changed()

    def _record_run(self, rid: int) -> None:
        task = self.queue.get(rid)
        arguments = task.args
        path = self.root / task.h5_path if task.h5_path else None
        if task.status.finished and path is not None and path.exists():
            # The file holds the effective values, defaults included.
            try:
                arguments = read_arguments(path)
            except (OSError, KeyError):
                pass
        self.db.record_run(
            {
                "rid": task.rid,
                "experiment": task.experiment,
                "class_name": task.class_name,
                "label": task.label,
                "submitter": task.submitter,
                "arguments": arguments,
                "scan": task.scan.model_dump(mode="json"),
                "status": str(task.status),
                "error": task.error,
                "t_submit": task.t_submit,
                "t_start": task.t_start,
                "t_end": task.t_end,
                "points_done": task.points_done,
                "points_total": task.points_total,
                "h5_path": task.h5_path,
                "simulated": self.simulate,
            }
        )

    # -- state and publishing ------------------------------------------------

    def _catalogue_changed(self, changed: bool) -> None:
        if changed:
            self._catalogue_version += 1
            self._dirty = True

    def _changed(self) -> None:
        self._save()
        self._dirty = True

    def _recipes_changed(self) -> None:
        self.recipe_store.save(self.recipes.state)
        self._recipes_version += 1
        self._dirty = True

    def _save(self) -> None:
        self.store.save(self.queue.state)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "catalogue_version": self._catalogue_version,
            "recipes_version": self._recipes_version,
            "simulated": self.simulate,
            "alive": time.time(),
            **self.queue.snapshot(),
        }

    def _publish_queue(self) -> None:
        self._dirty = False
        self._last_publish = time.monotonic()
        self._mutate(self._apply_snapshot, self._snapshot())

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        current = self.notifier.raw_view
        for key, value in snapshot.items():
            if current.get(key) != value:
                self.notifier[key] = value

    def _add_point(self, rid: int, row: dict[str, Any]) -> None:
        def add():
            if self.notifier.raw_view["points"]["rid"] != rid:
                # A worker adopted after a manager restart.
                self.notifier["points"] = {"rid": rid, "rows": []}
            self.notifier["points"]["rows"].append(row)

        self._mutate(add)

    def _mutate(self, fn, *args) -> None:
        """Run a notifier mutation where it is safe: on the publisher's loop."""
        if self.sync is None:
            fn(*args)
            return
        self.sync.call(fn, *args).add_done_callback(_log_failure)


def _log_failure(future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.error(
            "Publishing the experiment state failed", exc_info=future.exception()
        )


def _alive(process: psutil.Process) -> bool:
    try:
        if isinstance(process, psutil.Popen):
            return process.poll() is None
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Experiment Manager")
    parser.add_argument(
        "name",
        nargs="?",
        default="Experiment Manager",
        help='CONFIG["Servers"] key, optionally prefixed with "Servers/"',
    )
    args, _unknown = parser.parse_known_args()
    name = args.name.removeprefix("Servers/")

    manager = ExperimentManager(name)

    def _shutdown(_signo, _frame):
        manager.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    manager.serve_forever()


if __name__ == "__main__":
    main()
