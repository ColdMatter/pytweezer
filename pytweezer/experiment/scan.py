"""Scans: which argument values a task runs, in which order.

A :class:`Scan` is plain data (it round-trips through JSON and is stored in
every measurement file). :meth:`Scan.points` expands it against an experiment's
declared arguments into the ordered list of :class:`Point`\\ s the runner
executes; ``Point.index`` is the row that point's data lands in.
"""

import itertools
import random
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, Field, model_validator

from pytweezer.experiment.arguments import Argument, Integer, collect


class LinearAxis(BaseModel):
    kind: Literal["linear"] = "linear"
    argument: str
    start: float
    stop: float
    n: int = Field(ge=1)

    def raw_values(self) -> list[float]:
        return np.linspace(self.start, self.stop, self.n).tolist()


class ListAxis(BaseModel):
    kind: Literal["list"] = "list"
    argument: str
    values: list[Any] = Field(min_length=1)

    def raw_values(self) -> list[Any]:
        return list(self.values)


Axis = Annotated[LinearAxis | ListAxis, Field(discriminator="kind")]


class Point(NamedTuple):
    index: int
    repetition: int
    axis_indices: tuple[int, ...]
    values: dict[str, Any]


class Scan(BaseModel):
    """Axes to sweep plus repetitions and ordering.

    ``order`` is how the axis grid is traversed (the first axis is the
    outermost loop); ``snake`` reverses inner loops on alternate passes so
    consecutive points differ in one axis by one step. ``repeat="point"`` runs
    each repetition of a point back to back, ``"scan"`` repeats the whole sweep.
    A shuffled scan with no ``seed`` is given one on creation, so the order
    actually run can always be reproduced from the stored scan.
    """

    axes: list[Axis] = Field(default_factory=list)
    repetitions: int = Field(1, ge=1)
    order: Literal["nested", "snake", "shuffle"] = "nested"
    repeat: Literal["point", "scan"] = "point"
    seed: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "Scan":
        names = [axis.argument for axis in self.axes]
        duplicated = {name for name in names if names.count(name) > 1}
        if duplicated:
            raise ValueError(f"argument(s) scanned twice: {sorted(duplicated)}")
        if self.order == "shuffle" and self.seed is None:
            self.seed = random.SystemRandom().randrange(2**31)
        return self

    def axis_values(self, experiment_cls: type) -> dict[str, list[Any]]:
        """Return ``{argument: coerced values}`` for each axis, validated against ``experiment_cls``."""
        declared = collect(experiment_cls, Argument)
        resolved = {}
        for axis in self.axes:
            if axis.argument not in declared:
                raise ValueError(
                    f"{experiment_cls.__name__} has no argument {axis.argument!r} to scan"
                )
            argument = declared[axis.argument]
            raw = axis.raw_values()
            if isinstance(argument, Integer) and isinstance(axis, LinearAxis):
                raw = [round(value) for value in raw]
                if len(set(raw)) != len(raw):
                    raise ValueError(
                        f"scan of integer argument {axis.argument!r} from "
                        f"{axis.start} to {axis.stop} in {axis.n} steps repeats values "
                        "after rounding; use fewer steps or a list"
                    )
            resolved[axis.argument] = [argument.coerce(value) for value in raw]
        return resolved

    def points(self, experiment_cls: type) -> list[Point]:
        axis_values = self.axis_values(experiment_cls)
        names = list(axis_values)
        shape = [len(values) for values in axis_values.values()]
        rng = random.Random(self.seed)

        def one_pass() -> list[tuple[int, ...]]:
            if self.order == "snake":
                return _snake(shape)
            grid = list(itertools.product(*(range(n) for n in shape)))
            if self.order == "shuffle":
                rng.shuffle(grid)
            return grid

        if self.repeat == "point":
            sequence = [
                (rep, idx) for idx in one_pass() for rep in range(self.repetitions)
            ]
        else:
            sequence = [
                (rep, idx) for rep in range(self.repetitions) for idx in one_pass()
            ]

        return [
            Point(
                index=index,
                repetition=rep,
                axis_indices=idx,
                values={name: axis_values[name][i] for name, i in zip(names, idx)},
            )
            for index, (rep, idx) in enumerate(sequence)
        ]


def _snake(shape: list[int]) -> list[tuple[int, ...]]:
    if not shape:
        return [()]
    inner = _snake(shape[1:])
    return [
        (i, *rest)
        for i in range(shape[0])
        for rest in (inner if i % 2 == 0 else inner[::-1])
    ]
