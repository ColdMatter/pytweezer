"""The :class:`Experiment` base class."""

import inspect
from collections.abc import Iterable
from typing import Any, ClassVar

from pytweezer.experiment.arguments import (
    Argument,
    Device,
    coerce_arguments,
    collect,
)
from pytweezer.experiment.reservations import canonical_device
from pytweezer.experiment.scan import Point, Scan
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
    #: Names of the analyses the Results tab offers for this experiment's
    #: files, beside its universal plot; see
    #: :mod:`pytweezer.GUI.experiments.analyses`.
    analyses: ClassVar[tuple[str, ...]] = ()

    def __init__(self, args: dict[str, Any] | None = None) -> None:
        #: Undeclared arguments that :meth:`extra_argument` accepted, by name.
        #: They are kept here rather than as attributes, so they can share a
        #: name with an attribute of the experiment.
        self.extra_values: dict[str, Any] = {}
        for name, value in coerce_arguments(type(self), args or {}).items():
            self.set_argument(name, value)
        self._clients: list[Any] = []
        self._shared_clients: dict[str, Any] = {}
        self._recorder: Any = None
        #: A :class:`~pytweezer.experiment.simulation.SimulatedDevices` when
        #: running in simulation; devices then come from it instead of RPC.
        self.simulated_devices: Any = None
        #: Canonical names of the devices :meth:`device` may open; ``None``
        #: means unrestricted.
        self.allowed_devices: frozenset[str] | None = None
        #: The task's run id; -1 for :func:`~pytweezer.experiment.run_local`.
        self.rid: int | None = None

    @classmethod
    def arguments(cls) -> dict[str, Argument]:
        return collect(cls, Argument)

    @classmethod
    def devices(cls) -> dict[str, Device]:
        return collect(cls, Device)

    @classmethod
    def reserved_devices(cls) -> list[str]:
        """Sorted canonical names of every CONFIG device this experiment occupies.

        Raises:
            ReservationError: If a declared device is unknown, inactive or
                mis-configured.
        """
        reserved: set[str] = set()
        for device in cls.devices().values():
            reserved |= device.reserves()
        return sorted(reserved)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """JSON-serialisable description of the class, for argument editors.

        An optimiser that proposes saved parameters also has an ``"optimiser"``
        entry describing it; a shared optimiser base that proposes none does not.
        """
        schema = {
            "class_name": cls.__name__,
            "module": cls.__module__,
            "doc": inspect.getdoc(cls) or "",
            "arguments": {
                name: argument.to_schema() for name, argument in cls.arguments().items()
            },
            "devices": {name: dev.to_schema() for name, dev in cls.devices().items()},
            "analyses": list(cls.analyses),
        }
        if hasattr(cls, "optimiser_schema") and cls.proposes():
            schema["optimiser"] = cls.optimiser_schema()
        return schema

    @classmethod
    def extra_argument(cls, name: str) -> Argument | None:
        """Return the argument for ``name`` if it is not declared but is valid, else ``None``."""
        return None

    def set_argument(self, name: str, value: Any) -> None:
        if name in self.arguments():
            setattr(self, name, value)
        else:
            self.extra_values[name] = value

    def resolve_saved_arguments(self, scanned: Iterable[str] = ()) -> dict[str, str]:
        """Give each unset saved-backed argument not in ``scanned`` its saved value.

        Returns:
            ``{argument: saved key}`` for every argument so resolved.

        Raises:
            KeyError: A saved parameter they need does not exist.
        """
        unset = {
            name: argument
            for name, argument in self.arguments().items()
            if argument.saved and name not in scanned and getattr(self, name) is None
        }
        if not unset:
            return {}
        from pytweezer.experiment.client import get_saved_parameters

        saved = get_saved_parameters()
        missing = sorted({a.saved for a in unset.values()} - set(saved))
        if missing:
            raise KeyError(f"no saved parameter {', '.join(map(repr, missing))}")
        for name, argument in unset.items():
            self.set_argument(name, argument.coerce(saved[argument.saved]))
        return {name: argument.saved for name, argument in unset.items()}

    def argument_values(self) -> dict[str, Any]:
        declared = {name: getattr(self, name) for name in self.arguments()}
        return declared | self.extra_values

    def planned_points(self, scan: Scan) -> list[Point]:
        """The points to run for the submitted ``scan``.

        The default expands ``scan``. An experiment that chooses its own
        number of points, which may depend on its argument values, overrides
        this and raises :class:`ValueError` for a scan it cannot honour.
        """
        return scan.points(type(self))

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
        :func:`pytweezer.helper_scripts.parallel.run_parallel`. In simulation every call
        returns the same in-process simulated backend.

        Raises:
            UndeclaredDeviceError: If :attr:`allowed_devices` is set and ``name``
                is not among the devices this experiment declares.
        """
        if (
            self.allowed_devices is not None
            and canonical_device(name) not in self.allowed_devices
        ):
            from pytweezer.servers.device_client import UndeclaredDeviceError

            raise UndeclaredDeviceError(
                f"{type(self).__name__} uses device {name!r} without declaring it; "
                f"declare it as a class attribute, e.g. camera = Device({name!r})"
            )
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
