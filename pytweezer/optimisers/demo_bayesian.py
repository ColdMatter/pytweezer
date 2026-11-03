from typing import ClassVar

from pytweezer.experiment.optimisers import BayesianOptimisation, Dimension
from pytweezer.experiments.demo_resonance import DemoResonance


class DemoBayesian(BayesianOptimisation, DemoResonance):
    """Finds the demo resonance and power together by Bayesian optimisation."""

    objective = "counts"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("detuning", low=-5.0, high=5.0, save_as="demo_resonance"),
        Dimension("power", low=0.1, high=1.0, save_as="demo_power"),
    ]
