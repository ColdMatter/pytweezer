from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, BayesianOptimisation, Dimension
from pytweezer.experiments.tweezer_array_measurement import TweezerArrayMeasurement


class OptimiseTweezerImaging(BayesianOptimisation, TweezerArrayMeasurement):
    """Find the imaging cooling-light VCO and VVA that keep the most atoms through imaging.

    Each evaluation images the atoms twice with no drop between. The score is
    the survival while the fidelity exceeds ``fidelity_floor``, and
    0.8 × fidelity below it, so a setting too dim to tell full sites from empty
    ones scores low however many atoms it appears to keep.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("imaging_vco", span=0.5, save_as="tImgCoolVCO"),
        Dimension("imaging_vva", span=0.2, save_as="tImgCoolVVA"),
    ]
    n_initial = Integer(10, min=1, group=SEARCH)
    fidelity_floor = Number(0.99, min=0, max=1, group=SEARCH)

    def measure(
        self, *, imaging_vco: float | None = None, imaging_vva: float | None = None
    ) -> dict[str, float]:
        """Score the imaging light; ``None`` keeps a saved value."""
        imaging = {}
        if imaging_vco is not None:
            imaging |= {"tImgCoolVCO1": imaging_vco, "tImgCoolVCO2": imaging_vco}
        if imaging_vva is not None:
            imaging |= {"tImgCoolVVA1": imaging_vva, "tImgCoolVVA2": imaging_vva}
        return self.score_batch(BackgroundDrop=0, **imaging)

    def objective_of(self, results: dict[str, float]) -> float:
        if results["fidelity"] > self.fidelity_floor:
            return results["survival"]
        return 0.8 * results["fidelity"]
