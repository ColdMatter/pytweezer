from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Dimension, shortfall, shortness
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.optimisers.tweezer_optimisation import TweezerOptimisation

#: ``(setting, script parameter, span)`` for both molasses stages' light.
_MOLASSES_LIGHT = [
    (f"{beam}_{knob.lower()}{stage}", f"tMol{beam.title()}{knob}{stage}", span)
    for stage in (1, 2)
    for beam in ("cool", "rep")
    for knob, span in (("VCO", 0.5), ("VVA", 0.2))
]


class OptimiseMolassesLight(TweezerOptimisation, LoadTweezers):
    """Find the shortest molasses, with the cooling and repump VCOs and VVAs of both stages, that keeps loading and drop survival above their thresholds."""

    objective = "score"
    drop = "drop_duration"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension(setting, span=span, save_as=parameter, limits=(0, 10))
        for setting, parameter, span in _MOLASSES_LIGHT
    ] + [
        # FIXME: placeholder time limits (1, 20), to check against the script
        Dimension(
            "molasses_time", save_as="tMolassesDuration", low=1, high=20, whole=True
        )
    ]
    n_calls = Integer(80, min=1, group=SEARCH)
    n_initial = Integer(16, min=1, group=SEARCH)
    loading_threshold = Number(0.55, min=0, max=1, group=SEARCH)
    survival_threshold = Number(0.8, min=0, max=1, group=SEARCH)
    shortfall_penalty = Number(10, min=0, group=SEARCH)
    drop_duration = Integer(
        3, min=0, group="Analysis", tooltip="the script's tDropDuration"
    )

    def objective_of(
        self, scores: dict[str, float], settings: dict[str, float]
    ) -> float:
        return (
            shortness(settings["molasses_time"], self.bounds["molasses_time"])
            - self.shortfall_penalty
            * shortfall(scores["loading"], self.loading_threshold)
            - self.shortfall_penalty
            * shortfall(scores["survival"], self.survival_threshold)
        )
