from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer
from pytweezer.experiment.optimisers import SEARCH, BayesianOptimisation, Dimension
from pytweezer.experiments.tweezer_array_measurement import TweezerArrayMeasurement


class OptimiseMOTLight(BayesianOptimisation, TweezerArrayMeasurement):
    """Find the MOT-loading cooling and repump VCOs and VVAs that load the array best.

    Each evaluation images the atoms twice with no drop between, and scores the
    fraction of sites loaded in the first image.
    """

    objective = "loading"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("cool_vco", span=0.5, save_as="tMOTLoadCoolVCO", limits=(0, 10)),
        Dimension("cool_vva", span=0.2, save_as="tMOTLoadCoolVVA", limits=(0, 10)),
        Dimension("rep_vco", span=0.5, save_as="tMOTLoadRepVCO", limits=(0, 10)),
        Dimension("rep_vva", span=0.2, save_as="tMOTLoadRepVVA", limits=(0, 10)),
    ]
    n_calls = Integer(40, min=1, group=SEARCH)
    n_initial = Integer(10, min=1, group=SEARCH)

    def measure(
        self,
        *,
        cool_vco: float | None = None,
        cool_vva: float | None = None,
        rep_vco: float | None = None,
        rep_vva: float | None = None,
    ) -> dict[str, float]:
        """Score the MOT light; ``None`` keeps a saved value."""
        light = {
            parameter: value
            for parameter, value in (
                ("tMOTLoadCoolVCO", cool_vco),
                ("tMOTLoadCoolVVA", cool_vva),
                ("tMOTLoadRepVCO", rep_vco),
                ("tMOTLoadRepVVA", rep_vva),
            )
            if value is not None
        }
        return self.score_batch(BackgroundDrop=0, **light)
