from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Dimension, shortfall, shortness
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.optimisers.tweezer_optimisation import TweezerOptimisation

#: FIXME: placeholder, replace with the script's parameter name
MOT_LOAD_TIME = "tMOTLoadTime"


class OptimiseMOTLight(TweezerOptimisation, LoadTweezers):
    """Find the MOT-loading cooling and repump VCOs and VVAs that load the array best, at the saved load time."""

    objective = "loading"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("cool_vco", span=0.5, save_as="tMOTLoadCoolVCO", limits=(0, 10)),
        Dimension("cool_vva", span=0.2, save_as="tMOTLoadCoolVVA", limits=(0, 10)),
        Dimension("rep_vco", span=0.5, save_as="tMOTLoadRepVCO", limits=(0, 10)),
        Dimension("rep_vva", span=0.2, save_as="tMOTLoadRepVVA", limits=(0, 10)),
    ]
    n_calls = Integer(40, min=1, group=SEARCH)
    n_initial = Integer(10, min=1, group=SEARCH)


class OptimiseMOTLightMinTime(OptimiseMOTLight):
    """Find the shortest MOT load, with the cooling and repump VCOs and VVAs, that still loads the array to ``loading_threshold``."""

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        *OptimiseMOTLight.dimensions,
        # FIXME: placeholder limits, in the script's units, to check with the load time
        Dimension("load_time", save_as=MOT_LOAD_TIME, low=10, high=500, whole=True),
    ]
    loading_threshold = Number(0.55, min=0, max=1, group=SEARCH)
    shortfall_penalty = Number(10, min=0, group=SEARCH)

    def objective_of(
        self, scores: dict[str, float], settings: dict[str, float]
    ) -> float:
        return shortness(
            settings["load_time"], self.bounds["load_time"]
        ) - self.shortfall_penalty * shortfall(
            scores["loading"], self.loading_threshold
        )
