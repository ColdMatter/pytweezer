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

#: Saved parameter (SI) -> generator keyword, and the factor to the generator's unit.
GENERATOR_PARAMETERS = {
    "slm_wavelength": ("wavelength_um", 1e6),
    "slm_focal_length": ("focal_length_mm", 1e3),
    "slm_pixel_pitch": ("slm_pitch_um", 1e6),
    "slm_input_beam_waist": ("input_beam_waist_mm", 1e3),
}
#: The measured fall-off of trap depth across the focal plane, as
#: ``offset + amplitude * exp(-r^2 / 2 sigma^2)`` about a centre in
#: focal-plane coordinates (blaze included). Traps are weighted by it so they
#: come out equal. Centre and sigma are ``[x, y]`` in metres.
ENVELOPE_PARAMETERS = (
    "slm_envelope_amplitude",
    "slm_envelope_offset",
    "slm_envelope_centre",
    "slm_envelope_sigma",
)
#: Saved parameters the array needs besides its arguments: the generator's,
#: the SLM's resolution (``[rows, columns]`` of pixels), the Zernike
#: correction (rows of ``[index, coefficient]``) and the envelope.
SAVED_KEYS = (
    *GENERATOR_PARAMETERS,
    "slm_resolution",
    "slm_zernike_coefficients",
    *ENVELOPE_PARAMETERS,
)


def generator_settings(saved: dict) -> dict:
    """The phase-mask generator's keywords, in its units, from the saved parameters."""
    settings = {
        keyword: float(saved[key]) * factor
        for key, (keyword, factor) in GENERATOR_PARAMETERS.items()
    }
    settings["slm_res"] = tuple(int(n) for n in saved["slm_resolution"])
    settings["zernike_coeff_dict"] = {
        int(index): float(coefficient)
        for index, coefficient in saved["slm_zernike_coefficients"]
    }
    return settings


def envelope(saved: dict) -> dict:
    """The depth envelope from the saved parameters, with its centre and sigma in µm."""
    return {
        "amplitude": float(saved["slm_envelope_amplitude"]),
        "offset": float(saved["slm_envelope_offset"]),
        "centre_um": tuple(float(v) * 1e6 for v in saved["slm_envelope_centre"]),
        "sigma_um": tuple(float(v) * 1e6 for v in saved["slm_envelope_sigma"]),
    }


def envelope_weights(envelope: dict, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
    (x0, y0), (sx, sy) = envelope["centre_um"], envelope["sigma_um"]
    gaussian = np.exp(
        -((x_um - x0) ** 2) / (2 * sx**2) - (y_um - y0) ** 2 / (2 * sy**2)
    )
    return envelope["amplitude"] * gaussian + envelope["offset"]


def make_generator(settings: dict):
    """A generator for ``settings``: :func:`generator_settings` plus the Fresnel lens and blaze."""
    from pytweezer.helper_scripts.phasemask import (
        OptimisationBasedPhasemaskGeneratorGPU,
    )

    return OptimisationBasedPhasemaskGeneratorGPU(**settings)


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
    envelope: dict | None,
    blaze_um: tuple[float, float],
    fresnel_focal_length_mm: float,
    zernike: bool,
    pupil_radius_um: float,
    pupil_dump_um: tuple[float, float],
    iterations: int,
    damping: float,
    phase_randomness: float,
) -> dict:
    """The 8-bit mask for a ``rows`` x ``columns`` array, its traps and how well it converged.

    With an ``envelope`` (see :func:`envelope`) the traps are weighted by it.
    """
    weights, phases, x_um, y_um = generator.generate_weighted_array(
        np.ones((rows, columns)),
        spacing_um,
        init_phase_randomness=phase_randomness,
        angle_deg=angle_deg,
    )
    if envelope is not None:
        weights = envelope_weights(envelope, x_um + blaze_um[0], y_um + blaze_um[1])

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
