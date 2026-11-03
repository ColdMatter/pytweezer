from typing import ClassVar

from pytweezer.experiment.optimisers import Stage, StagedScan
from pytweezer.experiments.demo_resonance import DemoResonance


class DemoStagedScan(StagedScan, DemoResonance):
    """Finds the demo resonance, then the power at that detuning."""

    objective = "counts"
    stages: ClassVar[list[Stage]] = [
        Stage("detuning", start=-5.0, stop=5.0, points=21, save_as="demo_resonance"),
        Stage("power", start=0.1, stop=1.0, points=10, save_as="demo_power"),
    ]
