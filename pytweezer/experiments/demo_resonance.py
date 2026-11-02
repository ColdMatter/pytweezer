"""A device-free resonance for trying out the optimisers."""

import numpy as np

from pytweezer.experiment import Experiment, Integer


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
