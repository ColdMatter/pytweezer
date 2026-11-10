from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, BayesianOptimisation, Dimension
from pytweezer.experiments.tweezer_array_measurement import TweezerArrayMeasurement

#: ``(setting, script parameter, span)`` for both molasses stages' light.
_MOLASSES_LIGHT = [
    (f"{beam}_{knob.lower()}{stage}", f"tMol{beam.title()}{knob}{stage}", span)
    for stage in (1, 2)
    for beam in ("cool", "rep")
    for knob, span in (("VCO", 0.5), ("VVA", 0.2))
]


class OptimiseMolassesLight(BayesianOptimisation, TweezerArrayMeasurement):
    """Find the cooling and repump VCOs and VVAs of both molasses stages that recapture the most atoms after a drop.

    Scored as :class:`~pytweezer.optimisers.optimise_molasses_shims.OptimiseMolassesShims`:
    below ``loading_floor`` the score is the loading; above it, the floor plus
    the recapture probability after a ``drop_duration`` drop.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension(setting, span=span, save_as=parameter, limits=(0, 10))
        for setting, parameter, span in _MOLASSES_LIGHT
    ]
    n_calls = Integer(80, min=1, group=SEARCH)
    n_initial = Integer(16, min=1, group=SEARCH)
    loading_floor = Number(0.5, min=0, max=1, group=SEARCH)
    drop_duration = Integer(
        3, min=0, group="Analysis", tooltip="the script's tDropDuration"
    )

    def measure(self, **settings: float | None) -> dict[str, float]:
        """Score the molasses light; a missing or ``None`` setting keeps its saved value."""
        light = {
            parameter: settings[setting]
            for setting, parameter, _ in _MOLASSES_LIGHT
            if settings.get(setting) is not None
        }
        return self.score_batch(
            BackgroundDrop=1, tDropDuration=self.drop_duration, **light
        )

    def objective_of(self, results: dict[str, float]) -> float:
        if results["loading"] < self.loading_floor:
            return results["loading"]
        return self.loading_floor + results["survival"]
