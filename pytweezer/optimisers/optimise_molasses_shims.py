from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, BayesianOptimisation, Dimension
from pytweezer.experiments.tweezer_array_measurement import TweezerArrayMeasurement


class OptimiseMolassesShims(BayesianOptimisation, TweezerArrayMeasurement):
    """Find the molasses shim currents that recapture the most atoms after a drop.

    Each evaluation drops the atoms for ``drop_duration`` between the two
    images. Below ``loading_floor`` the score is the loading; above it, the
    floor plus the recapture probability, so every setting that loads the
    array outscores every one that does not, and among those the coldest wins.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("shim_x", span=1.0, save_as="tMolassesShimXccValue"),
        Dimension("shim_y", span=1.0, save_as="tMolassesShimYccValue"),
        Dimension("shim_z", span=0.25, save_as="tMolassesShimZccValue"),
    ]
    n_calls = Integer(50, min=1, group=SEARCH)
    n_initial = Integer(10, min=1, group=SEARCH)
    loading_floor = Number(0.5, min=0, max=1, group=SEARCH)
    drop_duration = Integer(
        3, min=0, group="Analysis", tooltip="the script's tDropDuration"
    )

    def measure(
        self,
        *,
        shim_x: float | None = None,
        shim_y: float | None = None,
        shim_z: float | None = None,
    ) -> dict[str, float]:
        """Score the molasses shims; ``None`` keeps a saved value."""
        shims = {
            parameter: value
            for parameter, value in (
                ("tMolassesShimXccValue", shim_x),
                ("tMolassesShimYccValue", shim_y),
                ("tMolassesShimZccValue", shim_z),
            )
            if value is not None
        }
        return self.score_batch(
            BackgroundDrop=1, tDropDuration=self.drop_duration, **shims
        )

    def objective_of(self, results: dict[str, float]) -> float:
        if results["loading"] < self.loading_floor:
            return results["loading"]
        return self.loading_floor + results["survival"]
