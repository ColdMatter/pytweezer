from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.optimisers import SEARCH, Dimension
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
    """Find the cooling and repump VCOs and VVAs of both molasses stages that recapture the most atoms after a drop.

    Scored as :class:`~pytweezer.optimisers.optimise_molasses_shims.OptimiseMolassesShims`:
    below ``loading_floor`` the score is the loading; above it, the floor plus
    the recapture probability after a ``drop_duration`` drop.
    """

    objective = "score"
    drop = "drop_duration"
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

    def objective_of(
        self, results: dict[str, float], settings: dict[str, float]
    ) -> float:
        if results["loading"] < self.loading_floor:
            return results["loading"]
        return self.loading_floor + results["survival"]
