from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Dimension
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.optimisers.tweezer_optimisation import TweezerOptimisation


class OptimiseTweezerImaging(TweezerOptimisation, LoadTweezers):
    """Find the imaging cooling-light VCO and VVA that keep the most atoms through imaging.

    Each evaluation images the atoms twice with no drop between. The score
    weights the fidelity by ``fidelity_weight`` and the survival by the rest.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("imaging_vco", span=0.5, save_as="tImgCoolVCO"),
        Dimension("imaging_vva", span=0.2, save_as="tImgCoolVVA"),
    ]
    n_initial = Integer(10, min=1, group=SEARCH)
    fidelity_weight = Number(0.5, min=0, max=1, group=SEARCH)

    def measure(
        self, *, imaging_vco: float | None = None, imaging_vva: float | None = None
    ) -> dict[str, float]:
        """Score the imaging light; ``None`` keeps a saved value."""
        imaging = {}
        if imaging_vco is not None:
            imaging |= {"tImgCoolVCO1": imaging_vco, "tImgCoolVCO2": imaging_vco}
        if imaging_vva is not None:
            imaging |= {"tImgCoolVVA1": imaging_vva, "tImgCoolVVA2": imaging_vva}
        return self.score_batch(**self.drop_overrides(), **imaging)

    def objective_of(
        self, scores: dict[str, float], settings: dict[str, float]
    ) -> float:
        w = self.fidelity_weight
        return w * scores["fidelity"] + (1 - w) * scores["survival"]
