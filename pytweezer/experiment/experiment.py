"""The :class:`Experiment` base class."""

import inspect
from typing import Any

from pytweezer.experiment.arguments import (
    Argument,
    Device,
    coerce_arguments,
    collect,
)
from pytweezer.experiment.scan import Point
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.experiment")


def _get_device(name: str, timeout: float | None) -> Any:
    from pytweezer.servers.device_client import get_device

    return get_device(name, timeout=timeout)


class Experiment:
    """Base class for an experiment.

    Subclasses declare arguments and devices as class attributes (see
    :mod:`pytweezer.experiment.arguments`) and implement :meth:`run_point`.
    A fresh instance is created for every task, with each argument attribute
    holding its effective value for that task.

    Per task the runner calls :meth:`prepare` once, :meth:`run_point` once per
    scan point (with :attr:`point` set), and :meth:`finish` once, even if an
    earlier hook raised.

    Device state persists between tasks, so :meth:`prepare` should set
    everything the experiment relies on rather than assume what a previous
    task left behind.
    """

    point: Point | None = None

    def __init__(self, args: dict[str, Any] | None = None) -> None:
        for name, value in coerce_arguments(type(self), args or {}).items():
            setattr(self, name, value)
        self._clients: list[Any] = []
        self._shared_clients: dict[str, Any] = {}
        self._recorder: Any = None
        #: A :class:`~pytweezer.experiment.simulation.SimulatedDevices` when
        #: running in simulation; devices then come from it instead of RPC.
        self.simulated_devices: Any = None

    @classmethod
    def arguments(cls) -> dict[str, Argument]:
        return collect(cls, Argument)

    @classmethod
    def devices(cls) -> dict[str, Device]:
        return collect(cls, Device)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """JSON-serialisable description of the class, for argument editors."""
        return {
            "class_name": cls.__name__,
            "module": cls.__module__,
            "doc": inspect.getdoc(cls) or "",
            "arguments": {
                name: argument.to_schema() for name, argument in cls.arguments().items()
            },
            "devices": {name: dev.to_schema() for name, dev in cls.devices().items()},
        }

    def argument_values(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.arguments()}

    def prepare(self) -> None:
        """Called once before the first point."""

    def run_point(self) -> None:
        """Called once per scan point; :attr:`point` describes which."""
        raise NotImplementedError

    def finish(self) -> None:
        """Called once after the last point, also after a failure or termination."""

    def record(self, name: str, value: Any, unit: str = "") -> None:
        """Store ``value`` in the measurement file.

        From :meth:`run_point` the value belongs to the current point and is
        stored under ``/results/<name>``; every point's value for one name
        must have the same shape. From :meth:`prepare` or :meth:`finish` it is
        stored once under ``/constants/<name>``.
        """
        if self._recorder is None:
            raise RuntimeError(
                "record() only works while the experiment is run by run_local() "
                "or the experiment manager"
            )
        self._recorder(name, value, unit)

    def device(self, name: str, *, fresh: bool = False, timeout: float | None = None):
        """Return an RPC client for the device ``name``, closed when the task ends.

        Clients are not thread-safe: pass ``fresh=True`` for an extra client
        to use from another thread, e.g. with
        :func:`pytweezer.parallel.run_parallel`. In simulation every call
        returns the same in-process simulated backend.
        """
        if self.simulated_devices is not None:
            return self.simulated_devices.get(name)
        if not fresh and name in self._shared_clients:
            return self._shared_clients[name]
        client = _get_device(name, timeout)
        self._clients.append(client)
        if not fresh:
            self._shared_clients[name] = client
        return client

    def close_devices(self) -> None:
        while self._clients:
            client = self._clients.pop()
            try:
                client.close_rpc()
            except Exception:
                logger.warning("Error closing a device client", exc_info=True)
        self._shared_clients.clear()
        if self.simulated_devices is not None:
            self.simulated_devices.close()
        for name in self.devices():
            self.__dict__.pop(name, None)
