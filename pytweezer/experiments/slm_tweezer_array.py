"""Put a rectangular tweezer array on the Rb SLM through the rearrangement rig, and show it.

The rig computes the mask, writes it to the SLM and remembers the array, so
analyses and later experiments can ask it what is there. Each new mask is
broadcast for the SLM Monitor applet: on the image hub, the simulated
intensity in the trap plane (``slm_simulated``) and viewed top-down, summed
over y with z running along the rows (``slm_top_down``), and the 8-bit mask on
the SLM (``slm_phasemask``); on the data hub, how the weight optimisation
converged (``slm_convergence``: one row per
:data:`~pytweezer.helper_scripts.array_hologram.CONVERGENCE` quantity, one
column per iteration, the names in the header's ``curves``).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from pytweezer.experiment import Bool, Device, Experiment, Integer, Number
from pytweezer.experiment.arguments import Argument, collect
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.helper_scripts import array_hologram
from pytweezer.servers import DataClient, ImageClient

if TYPE_CHECKING:
    # Imports cupy where it is installed, so only for the annotation.
    from pytweezer.coordinators.rearrangement import Rearrangement


#: What the rig's hologram says about the mask's quality, recorded per point.
HOLOGRAM_QUALITY = ("uniformity", "min_max_ratio", "mse")


def array_summary(hologram: dict, blaze_m: tuple[float, float]) -> dict:
    """What a file keeps of a rig hologram: trap positions and weights, and the simulated trap plane.

    Positions are µm in the focal plane relative to the blaze spot, which is at
    ``blaze_um``; ``simulated_trap_plane`` (row index along y, peak 1) has its
    pixel centres at ``grid_x_um``/``grid_y_um``, also relative to the blaze spot.
    """
    return {
        "x_um": np.asarray(hologram["x_um"], float).tolist(),
        "y_um": np.asarray(hologram["y_um"], float).tolist(),
        "weights": np.asarray(hologram["weights"], float).tolist(),
        "blaze_um": [blaze_m[0] * 1e6, blaze_m[1] * 1e6],
        "grid_x_um": np.asarray(hologram["grid_x_um"], float).tolist(),
        "grid_y_um": np.asarray(hologram["grid_y_um"], float).tolist(),
        "simulated_trap_plane": np.round(
            np.asarray(hologram["simulated_trap_plane"], float), 4
        ).tolist(),
    }


class SlmArray:
    """The arguments and steps that put an NxM tweezer array on the Rb SLM, for experiments to mix in.

    See :mod:`pytweezer.helper_scripts.array_hologram` for how the mask is made.
    The optics come from saved parameters: the blaze, Fresnel lens and pupil
    through saved-backed arguments, and the generator, Zernike correction and
    depth envelope (:data:`~pytweezer.helper_scripts.array_hologram.SAVED_KEYS`)
    directly; :meth:`read_slm_settings` records the last two sets as
    ``generator_settings`` and ``envelope``.
    """

    rig: Rearrangement = Device("Rb Rearrangement Rig")

    rows = Integer(15, min=1, group="Array")
    columns = Integer(15, min=1, group="Array")
    spacing = Number(5e-6, unit="um", scale=1e-6, min=0, group="Array")
    angle = Number(math.radians(2.5), unit="deg", scale=math.pi / 180, group="Array")
    compensate_envelope = Bool(
        True,
        group="Array",
        tooltip="weight traps to cancel the measured depth fall-off",
    )

    blaze_x = Number(saved="slm_blaze_x", unit="um", scale=1e-6, group="Optics")
    blaze_y = Number(saved="slm_blaze_y", unit="um", scale=1e-6, group="Optics")
    fresnel_focal_length = Number(
        saved="slm_fresnel_focal_length", unit="mm", scale=1e-3, group="Optics"
    )
    zernike = Bool(
        False, group="Optics", tooltip="add the Zernike aberration correction"
    )

    pupil_radius = Number(
        saved="slm_pupil_radius", unit="mm", scale=1e-3, min=0, group="Pupil"
    )
    pupil_dump_x = Number(
        saved="slm_pupil_dump_x", unit="um", scale=1e-6, group="Pupil"
    )
    pupil_dump_y = Number(
        saved="slm_pupil_dump_y", unit="um", scale=1e-6, group="Pupil"
    )

    iterations = Integer(2000, min=1, group="Optimisation")
    damping = Number(0.5, min=0, max=1, group="Optimisation")
    phase_randomness = Number(
        1.0,
        min=0,
        max=1,
        group="Optimisation",
        tooltip="spread of the initial trap phases",
    )

    def read_slm_settings(self) -> None:
        """Read and record the saved generator settings and envelope; call it in ``prepare``."""
        saved = get_saved_parameters()
        missing = sorted(set(array_hologram.SAVED_KEYS) - set(saved))
        if missing:
            raise KeyError(f"no saved parameter {', '.join(map(repr, missing))}")
        self.generator_settings = array_hologram.generator_settings(saved)
        self.envelope = array_hologram.envelope(saved)
        self.record("generator_settings", self.generator_settings)
        self.record("envelope", self.envelope)
        self.stream = ImageClient("slm")
        self.convergence_stream = DataClient("slm")
        self._slm_arrays = []

    def write_slm_array(self) -> dict:
        """Compute this point's mask, write it to the SLM, broadcast it and return the rig's hologram."""
        blaze_um = (self.blaze_x * 1e6, self.blaze_y * 1e6)
        hologram = self.rig.make_array(
            rows=self.rows,
            columns=self.columns,
            spacing_um=self.spacing * 1e6,
            angle_deg=math.degrees(self.angle),
            generator_settings=self.generator_settings,
            envelope=self.envelope if self.compensate_envelope else None,
            blaze_um=blaze_um,
            fresnel_focal_length_mm=self.fresnel_focal_length * 1e3,
            zernike=self.zernike,
            pupil_radius_um=self.pupil_radius * 1e6,
            pupil_dump_um=(self.pupil_dump_x * 1e6, self.pupil_dump_y * 1e6),
            iterations=self.iterations,
            damping=self.damping,
            phase_randomness=self.phase_randomness,
        )
        self.broadcast(hologram, blaze_um)
        return hologram

    def update_slm_array(self, settle_time: float) -> None:
        """Write the mask if the SLM arguments differ from the last one this task wrote, and record its quality.

        Each write waits ``settle_time`` seconds before returning. Every point
        records ``slm_array``, the index into :meth:`record_slm_arrays`'s list of
        the array on the SLM.
        """
        settings = {name: getattr(self, name) for name in collect(SlmArray, Argument)}
        if settings != getattr(self, "_written_slm_settings", None):
            self._slm_hologram = self.write_slm_array()
            self._written_slm_settings = settings
            self._slm_arrays.append(
                array_summary(self._slm_hologram, (self.blaze_x, self.blaze_y))
            )
            time.sleep(settle_time)
        for name in HOLOGRAM_QUALITY:
            self.record(name, self._slm_hologram[name])
        self.record("slm_array", len(self._slm_arrays) - 1)

    def record_slm_arrays(self) -> None:
        """Record the arrays :meth:`update_slm_array` wrote as the constant ``slm_arrays``; call it in ``finish``.

        ``slm_arrays["arrays"]`` lists, per array, its trap positions and the
        simulated trap plane (see :func:`array_summary`). Nothing is recorded if
        no array was written.
        """
        arrays = getattr(self, "_slm_arrays", [])
        if arrays:
            self.record("slm_arrays", {"arrays": arrays})

    def broadcast(self, hologram: dict, blaze_um: tuple[float, float]) -> None:
        images = array_hologram.monitor_images(
            hologram, blaze_um, self.generator_settings["slm_pitch_um"]
        )
        for channel, (header, image) in images.items():
            self.stream.send(
                {"point": self.point.index, **header}, image, channel=channel
            )
        self.convergence_stream.send(
            {"point": self.point.index, "curves": list(array_hologram.CONVERGENCE)},
            hologram["convergence"],
            channel="_convergence",
        )


class SlmTweezerArray(SlmArray, Experiment):
    """Put an NxM tweezer array on the Rb SLM, one optimised phase mask per point."""

    sleep_time = Number(1, unit="s")

    def prepare(self) -> None:
        self.read_slm_settings()

    def run_point(self) -> None:
        hologram = self.write_slm_array()
        self.record("phasemask", hologram["mask"])
        for name in HOLOGRAM_QUALITY:
            self.record(name, hologram[name])
        time.sleep(self.sleep_time)
