import copy
import time
from typing import Any, Optional

import numpy as np

from pytweezer.analysis.print_messages import print_error
from pytweezer.experiment.experiment import get_experiment
from pytweezer.experiment.motmaster_client import MotMasterClient
from pytweezer.servers.properties import Properties


def _normalize_scan_values(scan_values: Optional[list[list[Any]]], n_runs: int) -> list[list[Any]]:
    if scan_values is None:
        return []
    arr = np.asarray(scan_values)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if int(arr.shape[0]) != int(n_runs):
        raise ValueError(
            f"scan_values length ({arr.shape[0]}) must match n_runs ({n_runs})"
        )
    return arr.tolist()


def _normalize_scan_sequence(
    scan_sequence: Optional[list[int]], n_runs: int, randomize_scan: bool
) -> list[int]:
    if scan_sequence is None or len(scan_sequence) == 0:
        sequence = np.arange(int(n_runs), dtype=int)
    else:
        sequence = np.asarray(scan_sequence, dtype=int)
        if len(sequence) != int(n_runs):
            raise ValueError(
                f"scan_sequence length ({len(sequence)}) must match n_runs ({n_runs})"
            )

    if randomize_scan:
        np.random.shuffle(sequence)

    return sequence.tolist()


def build_task_dict(
    *,
    filepath: str,
    exp_name: str,
    args: Optional[dict[str, Any]] = None,
    n_runs: int = 1,
    n_reps: int = 0,
    scanpars: Optional[list[str]] = None,
    scanvals: Optional[list[list[Any]]] = None,
    scansequence: Optional[list[int]] = None,
    label: str = "notebook",
    task: int = 0,
    priority: int = 0,
    randomize_scan: bool = False,
) -> dict[str, Any]:
    """Build a task dict with the same schema used by ExperimentManager."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if n_reps < 0:
        raise ValueError("n_reps must be >= 0")

    scanpars = list(scanpars or [])
    scanvals = _normalize_scan_values(scanvals, n_runs=n_runs)
    scansequence = _normalize_scan_sequence(
        scansequence, n_runs=n_runs, randomize_scan=randomize_scan
    )

    if scanvals and scanpars and len(scanpars) != len(scanvals[0]):
        raise ValueError(
            "scanpars length must match number of columns in scanvals"
        )

    task_dict = {
        "task": int(task),
        "label": str(label),
        "expName": str(exp_name),
        "run": 0,
        "priority": int(priority),
        "args": dict(args or {}),
        "nRuns": int(n_runs),
        "scanpars": scanpars,
        "scanvals": scanvals,
        "nReps": int(n_reps),
        "scansequence": scansequence,
        "repetition": 0,
        "terminated": False,
        "filepath": str(filepath),
        "status": "Queued",
        "dueDateTime": "",
    }
    return task_dict


class NotebookExperimentRunner:
    """Headless experiment runner for notebooks.

    This mirrors the core run/scan/repetition logic of ExperimentManager,
    but it does not require Qt models, browser widgets, or dashboard services.
    """

    def __init__(self, motmaster_client: Optional[MotMasterClient] = None):
        self._props = Properties("Manager")
        self.motmaster_client = motmaster_client or MotMasterClient()

        self.running = False
        self.paused = False
        self.terminated = False

        self.taskNr = 0
        self.run_nr = 0
        self.rep = 0

        self.experiment = None
        self._current_task: Optional[dict[str, Any]] = None

    def run_task(self, task_dict: dict[str, Any]) -> dict[str, Any]:
        """Run one task dict and return the final task state."""
        task = copy.deepcopy(task_dict)
        self._validate_task(task)

        self.taskNr = int(task["task"])
        self._current_task = task

        experiment_cls = get_experiment(task["filepath"], task["expName"])
        self.experiment = experiment_cls(self._props, motmaster_client=self.motmaster_client)
        self.experiment.expName = task["expName"]

        self.run_nr = int(task["run"])
        self.rep = int(task["repetition"])
        self.terminated = bool(task["terminated"])

        self._props.set("ExperimentQ/current_scan", task["scanpars"])
        try:
            self._start_measurement(task)

            while self.rep <= int(task["nReps"]) and self.running:
                self._set_rep_nr(task, self.rep)
                self._runloop(task)
                self.rep += 1
                if self.terminated:
                    self._terminate_task(task)
                    break

            if self.running and task["status"] != "Failed":
                self._end_measurement(task)
        except Exception as error:
            task["status"] = "Failed"
            self.running = False
            if self.experiment is not None:
                try:
                    self.experiment.cleanup()
                except Exception:
                    pass
            print_error(
                f"NotebookRunner: task {self.taskNr} failed with error: {error}",
                "error",
            )
            raise

        return task

    def run(
        self,
        *,
        filepath: str,
        exp_name: str,
        args: Optional[dict[str, Any]] = None,
        n_runs: int = 1,
        n_reps: int = 0,
        scanpars: Optional[list[str]] = None,
        scanvals: Optional[list[list[Any]]] = None,
        scansequence: Optional[list[int]] = None,
        label: str = "notebook",
        task: int = 0,
        priority: int = 0,
        randomize_scan: bool = False,
    ) -> dict[str, Any]:
        """Convenience wrapper: build task dict and run it immediately."""
        task_dict = build_task_dict(
            filepath=filepath,
            exp_name=exp_name,
            args=args,
            n_runs=n_runs,
            n_reps=n_reps,
            scanpars=scanpars,
            scanvals=scanvals,
            scansequence=scansequence,
            label=label,
            task=task,
            priority=priority,
            randomize_scan=randomize_scan,
        )
        return self.run_task(task_dict)

    def pause(self):
        """Toggle pause state between runs."""
        self.paused = not self.paused

    def terminate(self):
        """Request graceful termination after the current run."""
        self.terminated = True
        if self._current_task is not None:
            self._current_task["terminated"] = True

    @staticmethod
    def _validate_task(task: dict[str, Any]) -> None:
        required = [
            "task",
            "label",
            "expName",
            "run",
            "args",
            "nRuns",
            "scanpars",
            "scanvals",
            "nReps",
            "scansequence",
            "repetition",
            "terminated",
            "filepath",
            "status",
        ]
        missing = [k for k in required if k not in task]
        if missing:
            raise KeyError(f"Task is missing required keys: {missing}")

    def _start_measurement(self, task: dict[str, Any]) -> None:
        self.experiment.build()
        self.experiment._task = self.taskNr
        self.experiment._run = int(self.run_nr)
        self.experiment._repetition = int(self.rep)
        self.experiment._rep = int(self.rep)
        for name, value in task["args"].items():
            self.experiment.set_argument_value(name, value)

        self.running = True
        if self.paused:
            task["status"] = "Paused"
        else:
            task["status"] = "Running"

        print_error(
            f"NotebookRunner: task {self.taskNr} ({task['expName']} {task['label']}) starting",
            "bold",
        )

    def _runloop(self, task: dict[str, Any]) -> None:
        n_runs = int(task["nRuns"])
        scan_pars = list(task["scanpars"])
        scan_values = np.asarray(task["scanvals"])
        scan_sequence = np.asarray(task["scansequence"], dtype=int)

        while self.run_nr < n_runs and self.running:
            self.terminated = bool(task["terminated"]) or self.terminated
            while self.paused and not self.terminated:
                task["status"] = "Paused"
                time.sleep(0.01)

            if self.terminated:
                return

            task["status"] = "Scanning" if n_runs > 1 else "Running"

            for idx, parname in enumerate(scan_pars):
                if parname != "--NONE--" and scan_values.size > 0:
                    self.experiment.set_argument_value(
                        parname,
                        scan_values[scan_sequence[self.run_nr], idx],
                    )

            self._set_run_nr(task, self.run_nr)
            self.experiment._start_run(self.taskNr)
            self.run_nr += 1

        self.run_nr = 0

    def _end_measurement(self, task: dict[str, Any]) -> None:
        self.experiment.cleanup()
        task["status"] = "Done"
        self.running = False
        print_error(
            f"NotebookRunner: task {self.taskNr} ({task['expName']} {task['label']}) completed",
            "bold",
        )

    def _terminate_task(self, task: dict[str, Any]) -> None:
        task["status"] = "Terminating"
        self.running = False
        self.experiment.cleanup()
        print_error(
            f"NotebookRunner: task {self.taskNr} ({task['expName']} {task['label']}) terminated",
            "warning",
        )

    def _set_run_nr(self, task: dict[str, Any], value: int) -> None:
        self.experiment._run = int(value)
        task["run"] = int(value)

    def _set_rep_nr(self, task: dict[str, Any], value: int) -> None:
        self.experiment._repetition = int(value)
        self.experiment._rep = int(value)
        task["repetition"] = int(value)


def run_experiment_notebook(
    *,
    filepath: str,
    exp_name: str,
    args: Optional[dict[str, Any]] = None,
    n_runs: int = 1,
    n_reps: int = 0,
    scanpars: Optional[list[str]] = None,
    scanvals: Optional[list[list[Any]]] = None,
    scansequence: Optional[list[int]] = None,
    label: str = "notebook",
    task: int = 0,
    priority: int = 0,
    randomize_scan: bool = False,
    motmaster_client: Optional[MotMasterClient] = None,
) -> dict[str, Any]:
    """One-call notebook convenience function for running an experiment."""
    runner = NotebookExperimentRunner(motmaster_client=motmaster_client)
    return runner.run(
        filepath=filepath,
        exp_name=exp_name,
        args=args,
        n_runs=n_runs,
        n_reps=n_reps,
        scanpars=scanpars,
        scanvals=scanvals,
        scansequence=scansequence,
        label=label,
        task=task,
        priority=priority,
        randomize_scan=randomize_scan,
    )
