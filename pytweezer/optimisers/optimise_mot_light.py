from __future__ import annotations

from typing import ClassVar

from pytweezer.experiment.arguments import Integer
from pytweezer.experiment.optimisers import SEARCH, Dimension
from pytweezer.experiments.load_tweezers import LoadTweezers
from pytweezer.optimisers.tweezer_optimisation import TweezerOptimisation


class OptimiseMOTLight(TweezerOptimisation, LoadTweezers):
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
