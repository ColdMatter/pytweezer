"""Optimisers: mixins that run a measurement experiment to propose saved parameters.

An optimiser combines a building block with a measurement experiment, which
provides ``measure(**settings) -> dict[str, float]``::

    class DemoStagedScan(StagedScan, DemoResonance):
        objective = "counts"
        stages: ClassVar[list[Stage]] = [
            Stage("detuning", start=-5.0, stop=5.0, points=21, save_as="demo_resonance"),
        ]

The mixin comes first, so it sets the points and runs each one; the
measurement's own ``prepare`` and ``finish`` still run. The values it proposes
are recorded as constants ``proposed.<saved key>``; nothing is saved until a
person accepts them.
"""

import math
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass
from typing import Any, ClassVar

import numpy as np

from pytweezer.experiment.arguments import Argument, Integer, Number, collect
from pytweezer.experiment.scan import Point, Scan


@dataclass(frozen=True)
class Stage:
    """One setting of a :class:`StagedScan`, swept from ``start`` to ``stop``.

    Its best value is proposed for the saved parameter ``save_as``.
    """

    setting: str
    _: KW_ONLY
    start: float
    stop: float
    points: int
    save_as: str


class Optimiser:
    """Base of the optimiser mixins.

    Subclasses name one result of ``measure()`` as :attr:`objective` and a
    :attr:`goal` of ``"max"`` or ``"min"``. Arguments a subclass generates
    from its declarations (:meth:`generated_arguments`) are added to every
    subclass that does not declare one of the same name itself.
    """

    objective: str
    goal: str = "max"
    kind: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = collect(cls, Argument)
        inherited = getattr(cls, "_generated_names", frozenset())
        generated = set()
        for name, argument in cls.generated_arguments().items():
            if name in declared and (name in vars(cls) or name not in inherited):
                continue
            argument.__set_name__(cls, name)
            setattr(cls, name, argument)
            generated.add(name)
        cls._generated_names = frozenset(generated)

    @classmethod
    def generated_arguments(cls) -> dict[str, Argument]:
        """The arguments this optimiser's declarations give it."""
        return {}

    @classmethod
    def proposes(cls) -> list[str]:
        """The saved-parameter keys this optimiser proposes values for."""
        return []

    @classmethod
    def optimiser_schema(cls) -> dict[str, Any]:
        return {"kind": cls.kind, "proposes": cls.proposes()}

    def objective_of(self, results: dict[str, float]) -> float:
        """The objective of one shot, from ``measure()``'s results."""
        return results[self.objective]

    def best(self, settings: list[float], objectives: list[float]) -> float:
        """The setting whose mean objective is best for :attr:`goal`.

        Repeat shots of a setting are averaged, ignoring non-finite objectives.

        Raises:
            ValueError: If no setting has a finite objective.
        """
        by_setting: dict[float, list[float]] = defaultdict(list)
        for setting, objective in zip(settings, objectives, strict=True):
            if math.isfinite(objective):
                by_setting[setting].append(objective)
        if not by_setting:
            raise ValueError("no setting gave a finite objective")
        means = {
            setting: float(np.mean(shots)) for setting, shots in by_setting.items()
        }
        choose = {"max": max, "min": min}[self.goal]
        return float(choose(means, key=means.__getitem__))


class StagedScan(Optimiser):
    """Scans one setting per stage, each at the best values of the stages before.

    Each :class:`Stage` adds the arguments ``<setting>_start``,
    ``<setting>_stop`` and ``<setting>_points``. Within a stage the sweep runs
    ``repeats`` times. When a stage's points are done, :meth:`best` gives its
    proposal, which every later stage passes to ``measure()``.

    Each point records ``stage`` (its index), ``setting``, ``objective`` and
    every result of ``measure()``. :meth:`finish` records ``proposed.<key>``
    for each finished stage, ``stages_done`` and ``stages_total``.
    """

    kind = "staged"
    stages: ClassVar[list[Stage]]
    repeats = Integer(1, min=1)

    @classmethod
    def generated_arguments(cls) -> dict[str, Argument]:
        arguments: dict[str, Argument] = {}
        for stage in getattr(cls, "stages", []):
            arguments[f"{stage.setting}_start"] = Number(stage.start)
            arguments[f"{stage.setting}_stop"] = Number(stage.stop)
            arguments[f"{stage.setting}_points"] = Integer(stage.points, min=1)
        return arguments

    @classmethod
    def proposes(cls) -> list[str]:
        return [stage.save_as for stage in cls.stages]

    def planned_points(self, scan: Scan) -> list[Point]:
        if scan.axes or scan.repetitions != 1:
            raise ValueError(
                f"{type(self).__name__} sets its own points; submit it without "
                "scan axes or repetitions"
            )
        return [Point(index, 0, (), {}) for index in range(len(self._schedule()))]

    def _schedule(self) -> list[tuple[int, float]]:
        """``(stage index, setting value)`` for every point, in run order."""
        schedule = []
        for index, stage in enumerate(self.stages):
            values = np.linspace(
                getattr(self, f"{stage.setting}_start"),
                getattr(self, f"{stage.setting}_stop"),
                getattr(self, f"{stage.setting}_points"),
            ).tolist()
            for _ in range(self.repeats):
                schedule.extend((index, value) for value in values)
        return schedule

    def prepare(self) -> None:
        self._best_values: list[float] = []
        self._shots: list[tuple[float, float]] = []
        self._points_schedule = self._schedule()
        super().prepare()

    def run_point(self) -> None:
        stage_index, setting = self._points_schedule[self.point.index]
        stage = self.stages[stage_index]
        settings = {
            earlier.setting: value
            for earlier, value in zip(self.stages, self._best_values, strict=False)
        }
        settings[stage.setting] = setting
        results = self.measure(**settings)
        objective = float(self.objective_of(results))
        self.record("stage", stage_index)
        self.record("setting", setting)
        self.record("objective", objective)
        for name, value in results.items():
            self.record(name, value)

        self._shots.append((setting, objective))
        is_last_of_stage = (
            self.point.index + 1 == len(self._points_schedule)
            or self._points_schedule[self.point.index + 1][0] != stage_index
        )
        if is_last_of_stage:
            stage_settings, objectives = zip(*self._shots, strict=True)
            try:
                value = self.best(list(stage_settings), list(objectives))
            except ValueError as error:
                raise ValueError(
                    f"stage {stage_index} ({stage.setting}): {error}"
                ) from error
            self._best_values.append(value)
            self._shots = []

    def finish(self) -> None:
        try:
            for stage, value in zip(self.stages, self._best_values, strict=False):
                self.record(f"proposed.{stage.save_as}", value)
            self.record("stages_done", len(self._best_values))
            self.record("stages_total", len(self.stages))
        finally:
            super().finish()
