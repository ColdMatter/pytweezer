"""Device-free experiments for trying out the queue, GUI and storage."""

import time

import numpy as np

from pytweezer.experiment import Experiment, Integer, Number


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
