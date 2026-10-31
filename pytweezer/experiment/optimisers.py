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
import skopt

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


@dataclass(frozen=True)
class Dimension:
    """One setting searched by a :class:`BayesianOptimisation` between ``low`` and ``high``.

    Its best value is proposed for the saved parameter ``save_as``.
    """

    setting: str
    _: KW_ONLY
    low: float
    high: float
    save_as: str


class Optimiser:
    """Base of the optimiser mixins.

    Subclasses name one result of ``measure()`` as :attr:`objective` and a
    :attr:`goal` of ``"max"`` or ``"min"``. Arguments a subclass generates
    from its declarations (:meth:`generated_arguments`) are added to every
    subclass that does not declare one of the same name itself. A class that
    declares stages or dimensions is checked when it is defined.
    """

    objective: str
    goal: str = "max"
    kind: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.declarations():
            cls.check_declarations()
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
    def declarations(cls) -> list[Any]:
        """The stages or dimensions this optimiser declares, in order."""
        return []

    @classmethod
    def check_declarations(cls) -> None:
        """Refuse declarations that cannot run.

        Raises:
            TypeError: If two declarations share a setting or a ``save_as``,
                or :attr:`goal` is neither ``"max"`` nor ``"min"``.
        """
        for field_name in ("setting", "save_as"):
            names = [getattr(declared, field_name) for declared in cls.declarations()]
            if repeated := sorted({name for name in names if names.count(name) > 1}):
                raise TypeError(
                    f"{cls.__name__}: {field_name} {', '.join(map(repr, repeated))} "
                    "is declared more than once"
                )
        if cls.goal not in ("max", "min"):
            raise TypeError(
                f"{cls.__name__}: goal is {cls.goal!r}; it must be 'max' or 'min'"
            )

    @classmethod
    def generated_arguments(cls) -> dict[str, Argument]:
        """The arguments this optimiser's declarations give it."""
        return {}

    @classmethod
    def proposes(cls) -> list[str]:
        """The saved-parameter keys this optimiser proposes values for."""
        return [declared.save_as for declared in cls.declarations()]

    @classmethod
    def optimiser_schema(cls) -> dict[str, Any]:
        """The optimiser's description, for the catalogue and the Optimisers tab.

        ``settings[i]`` is the setting whose best value is proposed for
        ``proposes[i]``; ``objective`` is ``None`` for a class that names none.
        """
        return {
            "kind": cls.kind,
            "goal": cls.goal,
            "objective": getattr(cls, "objective", None),
            "proposes": cls.proposes(),
            "settings": [declared.setting for declared in cls.declarations()],
        }

    def planned_points(self, scan: Scan) -> list[Point]:
        if scan.axes or scan.repetitions != 1:
            raise ValueError(
                f"{type(self).__name__} sets its own points; submit it without "
                "scan axes or repetitions"
            )
        return [Point(index, 0, (), {}) for index in range(self.number_of_points())]

    def number_of_points(self) -> int:
        """How many points the run has, from the argument values."""
        raise NotImplementedError

    def objective_of(self, results: dict[str, float]) -> float:
        """The objective of one shot, from ``measure()``'s results."""
        return results[self.objective]

    def best(self, settings: list[Any], objectives: list[float]) -> Any:
        """The proposal, from every setting measured and its objective.

        Each building block defines the shapes it passes and expects back.
        """
        raise NotImplementedError


class StagedScan(Optimiser):
    """Scans one setting per stage, each at the best values of the stages before.

    Each :class:`Stage` adds the arguments ``<setting>_start``,
    ``<setting>_stop`` and ``<setting>_points``. Within a stage the sweep runs
    ``repeats`` times. When a stage's points are done, :meth:`best` gives its
    proposal, which every later stage passes to ``measure()``.

    Each point records ``stage`` (its index), ``setting``, ``objective`` and
    every result of ``measure()``, so ``measure()`` must not return a result
    named ``stage``, ``setting`` or ``objective``. :meth:`finish` records
    ``proposed.<key>`` for each finished stage, ``stages_done`` and
    ``stages_total``.

    While a run is in progress, :attr:`stage` is the :class:`Stage` being
    scanned, so an overridden :meth:`best` can tell which stage it chooses for.
    """

    kind = "staged"
    stages: ClassVar[list[Stage]]
    stage: Stage
    repeats = Integer(1, min=1)

    @classmethod
    def declarations(cls) -> list[Stage]:
        return list(getattr(cls, "stages", []))

    @classmethod
    def generated_arguments(cls) -> dict[str, Argument]:
        arguments: dict[str, Argument] = {}
        for stage in cls.declarations():
            arguments[f"{stage.setting}_start"] = Number(stage.start)
            arguments[f"{stage.setting}_stop"] = Number(stage.stop)
            arguments[f"{stage.setting}_points"] = Integer(stage.points, min=1)
        return arguments

    def best(self, settings: list[float], objectives: list[float]) -> float:
        """The setting of this :attr:`stage` whose mean objective is best for :attr:`goal`.

        Args:
            settings: The setting of each shot of the stage, repeats included.
            objectives: The objective of each shot.

        Returns:
            The setting to propose. Repeat shots of a setting are averaged,
            ignoring non-finite objectives.

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

    def number_of_points(self) -> int:
        return len(self._schedule())

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
        stage = self.stage = self.stages[stage_index]
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


class BayesianOptimisation(Optimiser):
    """Searches every :class:`Dimension` at once with a Gaussian-process model.

    Each dimension adds the arguments ``<setting>_low`` and ``<setting>_high``.
    The run makes ``n_calls`` evaluations, the first ``n_initial`` of them at
    random settings. Each point asks a ``skopt.Optimizer`` (expected
    improvement) for settings, calls ``measure()`` and tells it the objective.
    The exploration parameter ``xi`` falls geometrically from ``xi_start`` to
    ``xi_end`` across the calls. A non-finite objective is told as the worst
    finite one so far, or not told if there is none yet.

    Each point records every setting by name, ``objective``, the ``xi`` its
    settings were chosen with, and every result of ``measure()``, so
    ``measure()`` must not return a result named after a setting, ``objective``
    or ``xi``. :meth:`finish` records ``evaluations_done`` and ``evaluations_total`` and,
    after at least one evaluation, ``proposed.<key>`` for each dimension from
    :meth:`best`.
    """

    kind = "bayesian"
    dimensions: ClassVar[list[Dimension]]
    n_calls = Integer(30, min=1)
    n_initial = Integer(5, min=1)
    seed = Integer(0)
    xi_start = Number(0.1)
    xi_end = Number(0.001)

    @classmethod
    def declarations(cls) -> list[Dimension]:
        return list(getattr(cls, "dimensions", []))

    @classmethod
    def check_declarations(cls) -> None:
        """Refuse declarations that cannot run.

        Raises:
            TypeError: As :meth:`Optimiser.check_declarations`, or if a
                dimension's ``low`` is not below its ``high``.
        """
        super().check_declarations()
        for dimension in cls.declarations():
            if not dimension.low < dimension.high:
                raise TypeError(
                    f"{cls.__name__}: dimension {dimension.setting!r} has low "
                    f"{dimension.low} not below high {dimension.high}"
                )

    @classmethod
    def generated_arguments(cls) -> dict[str, Argument]:
        arguments: dict[str, Argument] = {}
        for dimension in cls.declarations():
            arguments[f"{dimension.setting}_low"] = Number(dimension.low)
            arguments[f"{dimension.setting}_high"] = Number(dimension.high)
        return arguments

    def number_of_points(self) -> int:
        return self.n_calls

    def prepare(self) -> None:
        self._evaluations: list[tuple[list[float], float]] = []
        self._told: list[float] = []
        self._optimizer = skopt.Optimizer(
            dimensions=[
                (
                    getattr(self, f"{dimension.setting}_low"),
                    getattr(self, f"{dimension.setting}_high"),
                )
                for dimension in self.dimensions
            ],
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=self.n_initial,
            acq_func_kwargs={"xi": self._xi(0)},
            random_state=self.seed,
        )
        super().prepare()

    def _xi(self, index: int) -> float:
        fraction = index / max(self.n_calls - 1, 1)
        return self.xi_start * (self.xi_end / self.xi_start) ** fraction

    def run_point(self) -> None:
        xi = self._optimizer.acq_func_kwargs["xi"]
        values = [float(value) for value in self._optimizer.ask()]
        results = self.measure(
            **{
                dimension.setting: value
                for dimension, value in zip(self.dimensions, values, strict=True)
            }
        )
        objective = float(self.objective_of(results))
        for dimension, value in zip(self.dimensions, values, strict=True):
            self.record(dimension.setting, value)
        self.record("objective", objective)
        self.record("xi", xi)
        for name, value in results.items():
            self.record(name, value)

        self._evaluations.append((values, objective))
        # skopt chooses the next point inside tell(), so set its xi first.
        self._optimizer.acq_func_kwargs = {"xi": self._xi(self.point.index + 1)}
        if math.isfinite(objective):
            loss = -objective if self.goal == "max" else objective
        elif self._told:
            loss = max(self._told)
        else:
            return
        self._told.append(loss)
        self._optimizer.tell(values, loss)

    def best(self, settings: list[list[float]], objectives: list[float]) -> list[float]:
        """The evaluation whose finite objective is best for :attr:`goal`.

        Args:
            settings: Each evaluation's settings, one per dimension.
            objectives: Each evaluation's objective.

        Returns:
            The best evaluation's settings, one per dimension.

        Raises:
            ValueError: If no evaluation has a finite objective.
        """
        finite = [
            (objective, values)
            for values, objective in zip(settings, objectives, strict=True)
            if math.isfinite(objective)
        ]
        if not finite:
            raise ValueError("no evaluation gave a finite objective")
        choose = {"max": max, "min": min}[self.goal]
        return choose(finite, key=lambda shot: shot[0])[1]

    def finish(self) -> None:
        try:
            self.record("evaluations_done", len(self._evaluations))
            self.record("evaluations_total", self.n_calls)
            if self._evaluations:
                settings, objectives = zip(*self._evaluations, strict=True)
                best_values = self.best(list(settings), list(objectives))
                for dimension, value in zip(self.dimensions, best_values, strict=True):
                    self.record(f"proposed.{dimension.save_as}", value)
        finally:
            super().finish()
