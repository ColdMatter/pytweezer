from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Dimension, shortfall, shortness
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.optimisers.tweezer_optimisation import TweezerOptimisation


class OptimiseMolassesShims(TweezerOptimisation, LoadTweezers):
    """Find the shortest molasses, with the shim currents, that keeps loading and drop survival above their thresholds.

    Each evaluation drops the atoms for ``drop_duration`` between the two images.
    """

    objective = "score"
    drop = "drop_duration"
    # The shims are analog outputs, in volts, of a +-10 V card.
    dimensions: ClassVar[list[Dimension]] = [
        Dimension(
            "shim_x", span=1.0, save_as="tMolassesShimXccValue", limits=(-10, 10)
        ),
        Dimension(
            "shim_y", span=1.0, save_as="tMolassesShimYccValue", limits=(-10, 10)
        ),
        Dimension(
            "shim_z", span=0.25, save_as="tMolassesShimZccValue", limits=(-10, 10)
        ),
        Dimension("molasses_time", save_as="tMolassesDuration", low=1, high=20),
    ]
    n_calls = Integer(50, min=1, group=SEARCH)
    n_initial = Integer(10, min=1, group=SEARCH)
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
