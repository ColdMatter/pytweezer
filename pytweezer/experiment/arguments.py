"""Declarative experiment arguments and device handles.

Arguments are declared as class attributes of an
:class:`~pytweezer.experiment.experiment.Experiment` subclass::

    class LoadingRate(Experiment):
        detuning = Number(-12e6, unit="MHz", scale=1e6, min=-50e6, max=0)
        shots = Integer(10, min=1)
        camera = Device("Rb ThorCam")

Everything the GUI needs (names, defaults, limits, units) is read from the
class, so building an argument form never instantiates the experiment or
touches hardware.

Values are always held in SI units. ``unit`` and ``scale`` only control how a
value is displayed: ``Number(5e6, unit="MHz", scale=1e6)`` shows as ``5 MHz``.
"""

import math
import numbers
from typing import Any


class Argument:
    """Base class for a declared experiment argument.

    On an instance, the attribute reads as the effective (coerced) value for
    the current task; on the class it reads as the descriptor itself.
    """

    kind = "argument"

    def __init__(self, default: Any, *, tooltip: str = "", group: str = "") -> None:
        self.name = ""
        self.tooltip = tooltip
        self.group = group
        self.default = self.coerce(default)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: object, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return self.default

    def coerce(self, value: Any) -> Any:
        """Return ``value`` converted to this argument's type, or raise ``ValueError``."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "default": self.default,
            "tooltip": self.tooltip,
            "group": self.group,
        }

    def _fail(self, value: Any, reason: str) -> ValueError:
        return ValueError(f"argument {self.name!r}: {value!r} {reason}")


class Number(Argument):
    kind = "number"

    def __init__(
        self,
        default: float,
        *,
        unit: str = "",
        scale: float = 1.0,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        ndecimals: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.unit = unit
        self.scale = float(scale)
        self.min = min
        self.max = max
        self.step = step
        self.ndecimals = ndecimals
        super().__init__(default, **kwargs)

    def coerce(self, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise self._fail(value, "is not a number")
        value = float(value)
        if not math.isfinite(value):
            raise self._fail(value, "is not finite")
        if self.min is not None and value < self.min:
            raise self._fail(value, f"is below the minimum {self.min!r}")
        if self.max is not None and value > self.max:
            raise self._fail(value, f"is above the maximum {self.max!r}")
        return value

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "unit": self.unit,
            "scale": self.scale,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "ndecimals": self.ndecimals,
        }


class Integer(Argument):
    kind = "integer"

    def __init__(
        self,
        default: int,
        *,
        unit: str = "",
        min: int | None = None,
        max: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.unit = unit
        self.min = min
        self.max = max
        super().__init__(default, **kwargs)

    def coerce(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise self._fail(value, "is not an integer")
        if not float(value).is_integer():
            raise self._fail(value, "is not a whole number")
        value = int(value)
        if self.min is not None and value < self.min:
            raise self._fail(value, f"is below the minimum {self.min!r}")
        if self.max is not None and value > self.max:
            raise self._fail(value, f"is above the maximum {self.max!r}")
        return value

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "unit": self.unit,
            "min": self.min,
            "max": self.max,
        }


class Bool(Argument):
    kind = "bool"

    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        # numpy bools and 0/1 integers arrive from JSON and h5 round trips
        if isinstance(value, numbers.Integral) and value in (0, 1):
            return bool(value)
        raise self._fail(value, "is not a bool")


class Choice(Argument):
    kind = "choice"

    def __init__(self, options: list[str], default: str | None = None, **kwargs):
        if not options:
            raise ValueError("Choice needs at least one option")
        self.options = list(options)
        super().__init__(self.options[0] if default is None else default, **kwargs)

    def coerce(self, value: Any) -> str:
        if value not in self.options:
            raise self._fail(value, f"is not one of {self.options!r}")
        return value

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {"options": self.options}


class Text(Argument):
    kind = "text"

    def __init__(self, default: str = "", **kwargs: Any) -> None:
        super().__init__(default, **kwargs)

    def coerce(self, value: Any) -> str:
        if not isinstance(value, str):
            raise self._fail(value, "is not a string")
        return value


class Device:
    """A device the experiment talks to, resolved by its ``CONFIG["Devices"]`` name.

    The RPC client is created on first attribute access inside a running task
    and closed when the task ends.
    """

    def __init__(self, device_name: str, *, timeout: float | None = None) -> None:
        self.device_name = device_name
        self.timeout = timeout
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        client = obj.device(self.device_name, timeout=self.timeout)
        # Cache on the instance so later accesses skip this descriptor.
        obj.__dict__[self.name] = client
        return client

    def to_schema(self) -> dict[str, Any]:
        return {"device": self.device_name, "timeout": self.timeout}


def collect(cls: type, kind: type) -> dict[str, Any]:
    """Return ``{attribute name: declaration}`` for every ``kind`` declared on ``cls``.

    Base-class declarations come first; a subclass may override one by name.
    """
    found: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, kind):
                found[name] = value
            elif name in found:
                del found[name]
    return found


def coerce_arguments(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Return every argument of ``cls`` with ``values`` applied over the defaults.

    Names ``cls`` does not declare are looked up with ``cls.extra_argument``;
    those that resolve are coerced and included.
    """
    declared = collect(cls, Argument)
    extra_argument = getattr(cls, "extra_argument", lambda name: None)
    extras = {name: extra_argument(name) for name in values if name not in declared}
    unknown = [name for name, argument in extras.items() if argument is None]
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no argument(s) {sorted(unknown)}; "
            f"known: {sorted(declared)}"
        )
    coerced = {
        name: argument.coerce(values[name]) if name in values else argument.default
        for name, argument in declared.items()
    }
    coerced.update(
        {name: argument.coerce(values[name]) for name, argument in extras.items()}
    )
    return coerced
