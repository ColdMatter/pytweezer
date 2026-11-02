"""Compute a rectangular tweezer-array phase mask, put it on the Rb SLM and show it.

Each point broadcasts the requested trap intensities (``slm_target``) and the
8-bit mask on the SLM (``slm_phasemask``) on the image hub, for the SLM Monitor
applet. The mask is computed on the GPU, so this runs only where cupy is
installed (PH-BEAST).
"""

import math

import numpy as np

from pytweezer.drivers.slm import SLM
from pytweezer.experiment import Bool, Device, Experiment, Integer, Number
from pytweezer.servers import ImageClient

GENERATOR_SETTINGS = {
    "wavelength_um": 0.852,
    "focal_length_mm": 17.3,
    "slm_pitch_um": 17,
    "slm_res": (1024, 1024),
    "input_beam_waist_mm": 16,
    "zernike_coeff_dict": {
        5: 1.195,
        6: 0.725,
        7: 0.970,
        8: 0.478,
        9: -1.091,
        10: 0.303,
        11: 0.021,
        12: 0.072,
        13: 0.049,
    },
}

#: Measured fall-off of trap depth across the focal plane, as
#: ``offset + amplitude * exp(-r^2 / 2 sigma^2)`` about a centre in focal-plane
#: coordinates (blaze included). Traps are weighted by it so they come out equal.
ENVELOPE = {
    "amplitude": -2.14197604e03,
    "centre_um": (28.8556150, -5.9008040),
    "sigma_um": (2.95002910e03, 2.64281503e03),
    "offset": 2.14239505e03,
}


def envelope_weights(x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
    (x0, y0), (sx, sy) = ENVELOPE["centre_um"], ENVELOPE["sigma_um"]
    gaussian = np.exp(
        -((x_um - x0) ** 2) / (2 * sx**2) - (y_um - y0) ** 2 / (2 * sy**2)
    )
    return ENVELOPE["amplitude"] * gaussian + ENVELOPE["offset"]


def render_target(
    x_um: np.ndarray,
    y_um: np.ndarray,
    weights: np.ndarray,
    spot_sigma_um: float,
    pixel_um: float,
    margin_um: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Draw each trap as a Gaussian spot of peak ``weight``.

    Returns:
        The image, row index along y, and the (x, y) position in µm of its
        first pixel's corner.
    """
    x0 = x_um.min() - margin_um
    y0 = y_um.min() - margin_um
    nx = math.ceil((x_um.max() + margin_um - x0) / pixel_um)
    ny = math.ceil((y_um.max() + margin_um - y0) / pixel_um)
    x_centres = x0 + (np.arange(nx) + 0.5) * pixel_um
    y_centres = y0 + (np.arange(ny) + 0.5) * pixel_um
    # A Gaussian spot is separable, so the image is a (ny, N) @ (N, nx) product.
    two_sigma2 = 2 * spot_sigma_um**2
    x_profiles = np.exp(-((x_centres[None, :] - x_um[:, None]) ** 2) / two_sigma2)
    y_profiles = np.exp(-((y_centres[None, :] - y_um[:, None]) ** 2) / two_sigma2)
    image = (y_profiles * weights[:, None]).T @ x_profiles
    return image.astype(np.float32), (x0, y0)


def apply_pupil(
    generator, phase, radius_um: float, dump_dx_um: float, dump_dy_um: float
):
    """Keep ``phase`` inside a circular pupil and steer the light outside it away."""
    import cupy as cp

    x, y = cp.meshgrid(generator.x_slm, generator.y_slm)
    inside = x**2 + y**2 <= radius_um**2
    dump = generator.generate_blazed_grating_phasemask(dump_dx_um, dump_dy_um)
    return cp.where(inside, phase, dump)


class SlmTweezerArray(Experiment):
    """Put an NxM tweezer array on the Rb SLM, one optimised phase mask per point.

    The traps sit on a grid of ``spacing``, rotated by ``angle``, centred on the
    blaze offset. The mask is the weighted-superposition optimum for that grid,
    plus a Fresnel lens, the blaze and optionally the Zernike correction, inside
    a circular pupil; light outside the pupil is steered off by the dump grating.
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

    blaze_x = Number(48e-6, unit="um", scale=1e-6, group="Optics")
    blaze_y = Number(4e-6, unit="um", scale=1e-6, group="Optics")
    fresnel_focal_length = Number(1.125, unit="mm", scale=1e-3, group="Optics")
    zernike = Bool(
        False, group="Optics", tooltip="add the Zernike aberration correction"
    )

    pupil_radius = Number(8.704e-3, unit="mm", scale=1e-3, min=0, group="Pupil")
    pupil_dump_x = Number(100e-6, unit="um", scale=1e-6, group="Pupil")
    pupil_dump_y = Number(0.0, unit="um", scale=1e-6, group="Pupil")

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
        from pytweezer.helper_scripts.phasemask import (
            OptimisationBasedPhasemaskGeneratorGPU,
        )

        self.generator = OptimisationBasedPhasemaskGeneratorGPU(
            fresnel_f_mm=self.fresnel_focal_length * 1e3,
            blaze_dx_dy_um=(self.blaze_x * 1e6, self.blaze_y * 1e6),
            **GENERATOR_SETTINGS,
        )
        self.stream = ImageClient("slm")
        self.record("generator_settings", GENERATOR_SETTINGS)
        self.record("envelope", ENVELOPE)

    def run_point(self) -> None:
        generator = self.generator
        blaze_um = (self.blaze_x * 1e6, self.blaze_y * 1e6)

        weights, phases, x_um, y_um = generator.generate_weighted_array(
            np.ones((self.rows, self.columns)),
            self.spacing * 1e6,
            init_phase_randomness=self.phase_randomness,
            angle_deg=math.degrees(self.angle),
        )
        if self.compensate_envelope:
            weights = envelope_weights(x_um + blaze_um[0], y_um + blaze_um[1])

        array_phase, _, (uniformity, min_max, mse, _) = (
            generator.superposition_optimization(
                [weights, phases, x_um, y_um],
                max_iter=self.iterations,
                damping=self.damping,
                verbose=False,
            )
        )
        layers = [
            array_phase,
            generator.generate_fresnel_lens_phasemask(self.fresnel_focal_length * 1e3),
            generator.generate_blazed_grating_phasemask(*blaze_um),
        ]
        if self.zernike:
            layers.append(generator.zernike)
        phase = apply_pupil(
            generator,
            generator.superimpose(layers),
            self.pupil_radius * 1e6,
            self.pupil_dump_x * 1e6,
            self.pupil_dump_y * 1e6,
        )
        mask = generator.transform_phase_8bit(phase).get()
        self.slm.update_mask(mask)

        self.broadcast(x_um + blaze_um[0], y_um + blaze_um[1], weights, mask)
        self.record("phasemask", mask)
        self.record("uniformity", float(uniformity[-1]))
        self.record("min_max_ratio", float(min_max[-1]))
        self.record("mse", float(mse[-1]))

    def broadcast(self, x_um, y_um, weights, mask) -> None:
        spot_sigma_um = self.generator.dx_f
        pixel_um = spot_sigma_um / 4
        target, (x0_um, y0_um) = render_target(
            x_um,
            y_um,
            weights / weights.max(),
            spot_sigma_um,
            pixel_um,
            margin_um=2 * self.spacing * 1e6 + 3 * spot_sigma_um,
        )
        self.stream.send(
            {
                "point": self.point.index,
                "_imgresolution": [pixel_um * 1e-6, pixel_um * 1e-6],
                "_offset": [x0_um / pixel_um, y0_um / pixel_um],
            },
            target,
            channel="_target",
        )
        pitch_m = GENERATOR_SETTINGS["slm_pitch_um"] * 1e-6
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
