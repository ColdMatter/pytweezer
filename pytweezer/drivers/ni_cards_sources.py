"""Where the NI Cards driver learns that its pattern boards are in use.

`RemotingMotMasterStatus` asks MOTMaster itself whether it is running a
sequence, and what its last finished run left on the outputs.
`ManagerQueue` reads which devices the Experiment Manager's running tasks have
reserved. Both report ``None`` for "cannot tell" so that the driver can say so
instead of guessing.
"""

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pytweezer.configuration.config import get_config
from pytweezer.drivers.motmaster import motmaster_controller
from pytweezer.experiment.client import NOTIFIER_NAME, sync_address
from pytweezer.logging_utils import get_logger
from pytweezer.servers.sync import SyncMirror

logger = get_logger("pytweezer.drivers.ni_cards_sources")

# pythonnet may not be installed, so the .NET exceptions are matched by name.
_UNREACHABLE_DOTNET_ERRORS = {"SocketException", "RemotingException", "IOException"}


@dataclass(frozen=True)
class LastRun:
    """What MOTMaster's last finished run left behind."""

    number: int
    script: str
    finished: float
    outputs_known: bool
    values: dict[str, float | bool]


def _as_dict(mapping: Any) -> dict:
    # A .NET dictionary is iterated as key-value pairs, so go through Keys.
    if hasattr(mapping, "Keys"):
        return {key: mapping[key] for key in mapping.Keys}
    return dict(mapping)


def last_run_from_report(report: Mapping[str, Any] | None) -> LastRun | None:
    """Convert MOTMaster's ``GetLastRunOutputs()`` report.

    Args:
        report: The report dictionary, or ``None`` before any run has finished.

    Returns:
        The run, or ``None`` if ``report`` is ``None``.
    """
    if report is None:
        return None
    report = _as_dict(report)
    outputs_known = bool(report["OutputsKnown"])
    values = {}
    if outputs_known:
        for channel, value in _as_dict(report["Values"]).items():
            values[str(channel)] = value if isinstance(value, bool) else float(value)
    return LastRun(
        number=int(report["RunNumber"]),
        script=str(report["Script"]),
        finished=float(report["Finished"]),
        outputs_known=outputs_known,
        values=values,
    )


class MotMasterNotAnswering(Exception):
    """MOTMaster did not answer within the timeout."""


class MotMasterTooOld(Exception):
    """The MOTMaster build has no ``GetLastRunOutputs``."""


class RemotingMotMasterStatus:
    """Read-only view of a MOTMaster over .NET remoting.

    The proxy is built on the first call; any failure to build it counts as
    MOTMaster being unreachable, and is logged as a warning the first time so a
    configuration error does not pass for a closed MOTMaster. Each call runs on a worker thread and
    is abandoned after ``timeout_s``; while an abandoned call is still stuck,
    further calls raise `MotMasterNotAnswering` at once rather than pile up
    threads.

    Args:
        device_name: The MOTMaster's key in ``CONFIG["Devices"]``.
        timeout_s: How long to wait for MOTMaster to answer.
    """

    def __init__(self, device_name: str, timeout_s: float = 2.0):
        self._config_file = get_config()["Devices"][device_name]["config_file"]
        self._timeout_s = timeout_s
        self._proxy = None
        self._lock = threading.Lock()
        self._stuck: threading.Thread | None = None
        self._warned = False

    def is_running(self) -> bool | None:
        """Whether MOTMaster is running a sequence; ``None`` if unreachable.

        Raises:
            MotMasterNotAnswering: MOTMaster did not answer in time.
        """
        return self._call(self._read_is_running)

    @staticmethod
    def _read_is_running(proxy) -> bool:
        # pythonnet holds the GIL during a .NET property read but not during a
        # method call, so a blocked read must go through reflection for the
        # timeout to be able to fire.
        return bool(proxy.GetType().GetProperty("IsRunning").GetValue(proxy, None))

    def last_run(self) -> LastRun | None:
        """MOTMaster's last finished run.

        ``None`` means MOTMaster is unreachable or has not finished a run yet.

        Raises:
            MotMasterNotAnswering: MOTMaster did not answer in time.
            MotMasterTooOld: The MOTMaster build cannot report its last run.
        """

        def read(proxy):
            try:
                report = proxy.GetLastRunOutputs()
            except AttributeError as error:
                raise MotMasterTooOld(str(error)) from error
            except Exception as error:
                if type(error).__name__ == "MissingMethodException":
                    raise MotMasterTooOld(str(error)) from error
                raise
            return last_run_from_report(report)

        return self._call(read)

    def _call(self, fn: Callable[[Any], Any]) -> Any:
        with self._lock:
            if self._stuck is not None and self._stuck.is_alive():
                raise MotMasterNotAnswering("an earlier call is still waiting")
            outcome: dict[str, Any] = {}

            def work():
                try:
                    if self._proxy is None:
                        try:
                            self._proxy = motmaster_controller(self._config_file)
                        except Exception as error:
                            outcome["unbuildable"] = error
                            return
                    outcome["result"] = fn(self._proxy)
                except BaseException as error:
                    outcome["error"] = error

            worker = threading.Thread(target=work, name="motmaster-status", daemon=True)
            worker.start()
            worker.join(self._timeout_s)
            if worker.is_alive():
                self._stuck = worker
                raise MotMasterNotAnswering(f"no answer in {self._timeout_s} s")
            if "unbuildable" in outcome:
                self._unreachable(outcome["unbuildable"], warn=True)
                return None
            error = outcome.get("error")
            if error is None:
                self._warned = False
                return outcome["result"]
            if isinstance(error, OSError) or (
                type(error).__name__ in _UNREACHABLE_DOTNET_ERRORS
            ):
                self._unreachable(error, warn=False)
                return None
            raise error

    def _unreachable(self, error: BaseException, warn: bool) -> None:
        self._proxy = None
        if warn and not self._warned:
            logger.warning(
                "Cannot reach MOTMaster (%s): %s: %s",
                self._config_file,
                type(error).__name__,
                error,
            )
        else:
            logger.debug("MOTMaster unreachable: %r", error)
        self._warned = self._warned or warn


class SimulatedMotMasterStatus:
    """Stands in for a MOTMaster that never runs and has no last run."""

    def is_running(self) -> bool | None:
        return False

    def last_run(self) -> LastRun | None:
        return None


def _devices_by_rid(state: Any) -> dict[str, int]:
    # An older manager sends "running" as one task or None, not a list.
    running = (state or {}).get("running")
    if isinstance(running, dict):
        running = [running]
    held = {}
    for task in running or ():
        for device in task.get("devices") or ():
            held[device] = task["rid"]
    return held


class ManagerQueue:
    """Which devices the Experiment Manager's running tasks have reserved."""

    def __init__(self):
        self._mirror = SyncMirror(*sync_address(), NOTIFIER_NAME)

    def running_devices(self) -> dict[str, int] | None:
        """Device name to the rid of the running task holding it.

        Paused tasks count as running. ``None`` if the manager is unreachable.
        """
        if not self._mirror.connected:
            return None
        return self._mirror.read(_devices_by_rid)

    def close(self) -> None:
        self._mirror.close()
