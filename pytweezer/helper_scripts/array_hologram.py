"""Phase masks for rectangular tweezer arrays on the Rb SLM.

The traps sit on a grid of ``spacing_um``, rotated by ``angle_deg``, centred on
the blaze offset. The mask is the weighted-superposition optimum for that grid,
plus a Fresnel lens, the blaze and optionally the Zernike correction, inside a
circular pupil; light outside the pupil is steered off by the dump grating.

Trap positions are in µm in the focal plane, relative to the blaze spot. The
GPU stack (cupy, and lap through the generator) is imported only when a mask is
computed, so this module imports anywhere.
"""

import numpy as np

GENERATOR_SETTINGS = {
    "wavelength_um": 0.852,
    "focal_length_mm": 17.3,
    "slm_pitch_um": 17,
    "slm_res": (1024, 1024),
    "input_beam_waist_mm": 16,
    "fresnel_f_mm": 1125,
    "blaze_dx_dy_um": (48.0, 4.0),
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


def make_generator():
    from pytweezer.helper_scripts.phasemask import (
        OptimisationBasedPhasemaskGeneratorGPU,
    )

    return OptimisationBasedPhasemaskGeneratorGPU(**GENERATOR_SETTINGS)


def apply_pupil(
    generator, phase, radius_um: float, dump_dx_um: float, dump_dy_um: float
):
    """Keep ``phase`` inside a circular pupil and steer the light outside it away."""
    import cupy as cp

    x, y = cp.meshgrid(generator.x_slm, generator.y_slm)
    inside = x**2 + y**2 <= radius_um**2
    dump = generator.generate_blazed_grating_phasemask(dump_dx_um, dump_dy_um)
    return cp.where(inside, phase, dump)


def array_hologram(
    generator,
    *,
    rows: int,
    columns: int,
    spacing_um: float,
    angle_deg: float,
    compensate_envelope: bool,
    blaze_um: tuple[float, float],
    fresnel_focal_length_mm: float,
    zernike: bool,
    pupil_radius_um: float,
    pupil_dump_um: tuple[float, float],
    iterations: int,
    damping: float,
    phase_randomness: float,
) -> dict:
    """The 8-bit mask for a ``rows`` x ``columns`` array, its traps and how well it converged."""
    weights, phases, x_um, y_um = generator.generate_weighted_array(
        np.ones((rows, columns)),
        spacing_um,
        init_phase_randomness=phase_randomness,
        angle_deg=angle_deg,
    )
    if compensate_envelope:
        weights = envelope_weights(x_um + blaze_um[0], y_um + blaze_um[1])

    array_phase, _, (uniformity, min_max, mse, _) = (
        generator.superposition_optimization(
            [weights, phases, x_um, y_um],
            max_iter=iterations,
            damping=damping,
            verbose=False,
        )
    )
    layers = [
        array_phase,
        generator.generate_fresnel_lens_phasemask(fresnel_focal_length_mm),
        generator.generate_blazed_grating_phasemask(*blaze_um),
    ]
    if zernike:
        layers.append(generator.zernike)
    phase = apply_pupil(
        generator, generator.superimpose(layers), pupil_radius_um, *pupil_dump_um
    )
    return {
        "mask": generator.transform_phase_8bit(phase).get(),
        "x_um": np.asarray(x_um, dtype=float),
        "y_um": np.asarray(y_um, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "focal_pixel_um": float(generator.dx_f),
        "uniformity": float(uniformity[-1]),
        "min_max_ratio": float(min_max[-1]),
        "mse": float(mse[-1]),
    }
