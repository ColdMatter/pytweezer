"""Put a rectangular tweezer array on the Rb SLM, and show it.

The SLM computes the mask, displays it and remembers the array, so
analyses and later experiments can ask it what is there. Each point broadcasts
on the image hub, for the SLM Monitor applet: the requested trap intensities
(``slm_target``), the simulated intensity in the trap plane
(``slm_simulated``) and viewed top-down, summed over y with z running along the
rows (``slm_top_down``), and the 8-bit mask on the SLM (``slm_phasemask``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pytweezer.experiment import Bool, Device, Experiment, Integer, Number
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.helper_scripts import array_hologram
from pytweezer.servers import ImageClient

if TYPE_CHECKING:
    # Imports cupy where it is installed, so only for the annotation.
    from pytweezer.drivers.slm import SLM


def render_target(
    x_um: np.ndarray,
    y_um: np.ndarray,
    weights: np.ndarray,
    spot_sigma_um: float,
    grid_x_um: np.ndarray,
    grid_y_um: np.ndarray,
) -> np.ndarray:
    """Draw each trap as a Gaussian spot of peak ``weight`` on a grid of pixel centres.

    Returns:
        The image, row index along y.
    """
    # A Gaussian spot is separable, so the image is a (ny, N) @ (N, nx) product.
    two_sigma2 = 2 * spot_sigma_um**2
    x_profiles = np.exp(-((grid_x_um[None, :] - x_um[:, None]) ** 2) / two_sigma2)
    y_profiles = np.exp(-((grid_y_um[None, :] - y_um[:, None]) ** 2) / two_sigma2)
    image = (y_profiles * weights[:, None]).T @ x_profiles
    return image.astype(np.float32)


def placement(grid_x_um: np.ndarray, grid_y_um: np.ndarray) -> dict:
    """Image-header keys that place an image whose pixel centres are the grids, in metres."""
    pixel_x_um = grid_x_um[1] - grid_x_um[0]
    pixel_y_um = grid_y_um[1] - grid_y_um[0]
    return {
        "_imgresolution": [pixel_x_um * 1e-6, pixel_y_um * 1e-6],
        "_offset": [grid_x_um[0] / pixel_x_um - 0.5, grid_y_um[0] / pixel_y_um - 0.5],
    }


class SlmTweezerArray(Experiment):
    """Put an NxM tweezer array on the Rb SLM, one optimised phase mask per point.

    See :mod:`pytweezer.helper_scripts.array_hologram` for how the mask is made.
    The optics come from saved parameters: the blaze, Fresnel lens and pupil
    through saved-backed arguments, and the generator, Zernike correction and
    depth envelope (:data:`~pytweezer.helper_scripts.array_hologram.SAVED_KEYS`)
    directly; :meth:`prepare` records the last two sets as
    ``generator_settings`` and ``envelope``.
    """

    slm: SLM = Device("Rb SLM")

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

    def prepare(self) -> None:
        saved = get_saved_parameters()
        missing = sorted(set(array_hologram.SAVED_KEYS) - set(saved))
        if missing:
            raise KeyError(f"no saved parameter {', '.join(map(repr, missing))}")
        self.generator_settings = array_hologram.generator_settings(saved)
        self.envelope = array_hologram.envelope(saved)
        self.record("generator_settings", self.generator_settings)
        self.record("envelope", self.envelope)
        self.stream = ImageClient("slm")

    def run_point(self) -> None:
        blaze_um = (self.blaze_x * 1e6, self.blaze_y * 1e6)
        hologram = self.slm.make_array(
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
        self.record("phasemask", hologram["mask"])
        for name in ("uniformity", "min_max_ratio", "mse"):
            self.record(name, hologram[name])

    def broadcast(self, hologram: dict, blaze_um: tuple[float, float]) -> None:
        grid_x_um = np.asarray(hologram["grid_x_um"]) + blaze_um[0]
        grid_y_um = np.asarray(hologram["grid_y_um"]) + blaze_um[1]
        weights = np.asarray(hologram["weights"])
        target = render_target(
            np.asarray(hologram["x_um"]) + blaze_um[0],
            np.asarray(hologram["y_um"]) + blaze_um[1],
            weights / weights.max(),
            hologram["focal_pixel_um"],
            grid_x_um,
            grid_y_um,
        )
        trap_plane = placement(grid_x_um, grid_y_um)
        top_down = placement(grid_x_um, np.asarray(hologram["grid_z_um"]))
        for channel, image, header in (
            ("_target", target, trap_plane),
            ("_simulated", hologram["simulated_trap_plane"], trap_plane),
            ("_top_down", hologram["simulated_top_down"], top_down),
        ):
            self.stream.send(
                {"point": self.point.index, **header}, image, channel=channel
            )
        mask = hologram["mask"]
        pitch_m = self.generator_settings["slm_pitch_um"] * 1e-6
        ny, nx = mask.shape
        self.stream.send(
            {
                "point": self.point.index,
                "_imgresolution": [pitch_m, pitch_m],
                "_offset": [-nx / 2, -ny / 2],
            },
            mask,
            channel="_phasemask",
        )
