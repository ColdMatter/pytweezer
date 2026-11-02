"""Device-free experiments for trying out the queue, GUI and storage."""

import time
from typing import ClassVar

import numpy as np

from pytweezer.experiment import Experiment, Integer, Number
from pytweezer.experiment.optimisers import (
    BayesianOptimisation,
    Dimension,
    Stage,
    StagedScan,
)


class RabiDemo(Experiment):
    """Simulated Rabi flopping with projection noise and a fake camera image."""

    pulse_time = Number(10e-6, unit="us", scale=1e-6, min=0, ndecimals=2)
    rabi_frequency = Number(50e3, unit="kHz", scale=1e3, min=0)
    atoms = Integer(200, min=1)
    point_delay = Number(0.2, unit="s", min=0, tooltip="wall-clock time per point")

    def prepare(self):
        self.rng = np.random.default_rng()
        y, x = np.mgrid[:32, :32]
        self.cloud = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 50)
        self.record("cloud_shape", self.cloud)

    def run_point(self):
        time.sleep(self.point_delay)
        excited_probability = np.sin(np.pi * self.rabi_frequency * self.pulse_time) ** 2
        excited = self.rng.binomial(self.atoms, excited_probability)
        image = self.rng.poisson(excited * self.cloud / self.cloud.sum() * 50)
        self.record("excited_fraction", excited / self.atoms)
        self.record("image", image.astype(np.uint16))


class DemoResonance(Experiment):
    """Simulated counts on a Lorentzian resonance, scaled by power, with Poisson noise."""

    noise_seed = Integer(0)

    centre = 1.3
    half_width = 0.8
    peak_counts = 1000

    def prepare(self):
        self.rng = np.random.default_rng(self.noise_seed)

    def measure(self, *, detuning: float = 0.0, power: float = 1.0) -> dict[str, float]:
        lineshape = self.half_width**2 / (
            (detuning - self.centre) ** 2 + self.half_width**2
        )
        return {"counts": float(self.rng.poisson(self.peak_counts * power * lineshape))}

    def run_point(self):
        for name, value in self.measure().items():
            self.record(name, value)


class DemoStagedScan(StagedScan, DemoResonance):
    """Finds the demo resonance, then the power at that detuning."""

    objective = "counts"
    stages: ClassVar[list[Stage]] = [
        Stage("detuning", start=-5.0, stop=5.0, points=21, save_as="demo_resonance"),
        Stage("power", start=0.1, stop=1.0, points=10, save_as="demo_power"),
    ]


class DemoBayesian(BayesianOptimisation, DemoResonance):
    """Finds the demo resonance and power together by Bayesian optimisation."""

    objective = "counts"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("detuning", low=-5.0, high=5.0, save_as="demo_resonance"),
        Dimension("power", low=0.1, high=1.0, save_as="demo_power"),
    ]
