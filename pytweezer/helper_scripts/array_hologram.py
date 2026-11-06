"""Phase masks for rectangular tweezer arrays on the Rb SLM.

The traps sit on a grid of ``spacing_um``, rotated by ``angle_deg``, centred on
the blaze offset. The mask is the weighted-superposition optimum for that grid,
plus a Fresnel lens, the blaze and optionally the Zernike correction, inside a
circular pupil; light outside the pupil is steered off by the dump grating.

Trap positions, and the grids the mask's light is simulated on, are in µm in
the focal plane, relative to the blaze spot. The GPU stack (cupy, and lap
through the generator) is imported only when a mask is computed, so this module
imports anywhere.
"""

import math

import numpy as np

#: Simulated-image pixels per focal-plane pixel of the generator, along x and y.
OVERSAMPLING = 4
#: The top-down view spans this many depths of focus either side of the
#: trap plane, in planes a quarter of one apart.
DEPTHS_OF_FOCUS = 6

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


def focal_grid(
    x_um: np.ndarray, y_um: np.ndarray, pixel_um: float, margin_um: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres, in µm, of an image covering the traps with ``margin_um`` around them."""
    x0 = x_um.min() - margin_um
    y0 = y_um.min() - margin_um
    nx = math.ceil((x_um.max() + margin_um - x0) / pixel_um)
    ny = math.ceil((y_um.max() + margin_um - y0) / pixel_um)
    return x0 + (np.arange(nx) + 0.5) * pixel_um, y0 + (np.arange(ny) + 0.5) * pixel_um


def depth_of_focus_um(generator, pupil_radius_um: float) -> float:
    aperture_um = min(pupil_radius_um, generator.Lx_slm / 2, generator.Ly_slm / 2)
    return generator.lam * (generator.f_um / aperture_um) ** 2


def simulate(
    generator, phase, x_um: np.ndarray, y_um: np.ndarray, z_um: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity from ``phase`` on the SLM, in the trap plane and viewed top-down.

    Propagates with the optimiser's own Fourier convention, adding paraxial
    defocus for planes ``z_um`` beyond the trap plane.

    Returns:
        The trap-plane intensity on the ``y_um`` x ``x_um`` grid, and its sum
        over ``y_um`` at each ``z_um`` (row index along z), each peaking at 1.
    """
    import cupy as cp

    x_slm, y_slm = generator.x_slm, generator.y_slm
    field_slm = generator.generate_source_amplitude() * cp.exp(1j * cp.asarray(phase))
    to_x = cp.exp(-generator.j_k * x_slm[:, None] * cp.asarray(x_um, cp.float32))
    to_y = cp.exp(-generator.j_k * cp.asarray(y_um, cp.float32)[:, None] * y_slm)
    r2_slm = x_slm[None, :] ** 2 + y_slm[:, None] ** 2

    def intensity(z_um: float):
        defocus = cp.exp(
            -1j * (math.pi * z_um / (generator.lam * generator.f_um**2)) * r2_slm
        )
        return cp.abs(to_y @ (field_slm * defocus) @ to_x) ** 2

    trap_plane = intensity(0.0)
    top_down = cp.stack([intensity(z).sum(axis=0) for z in z_um])
    return (
        (trap_plane / trap_plane.max()).get(),
        (top_down / top_down.max()).get(),
    )


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
    Also simulates the light the mask makes (see :func:`simulate`), on a grid
    of pixel centres ``grid_x_um``/``grid_y_um`` around the traps and planes
    ``grid_z_um`` either side of them: ``simulated_trap_plane`` (row index
    along y) and ``simulated_top_down`` (row index along z).
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

    x_um = np.asarray(x_um, dtype=float)
    y_um = np.asarray(y_um, dtype=float)
    grid_x_um, grid_y_um = focal_grid(
        x_um,
        y_um,
        generator.dx_f / OVERSAMPLING,
        margin_um=2 * spacing_um + 3 * generator.dx_f,
    )
    planes_per_side = 4 * DEPTHS_OF_FOCUS
    plane_spacing_um = depth_of_focus_um(generator, pupil_radius_um) / 4
    grid_z_um = plane_spacing_um * np.arange(-planes_per_side, planes_per_side + 1)
    # The Fresnel lens and Zernike correction compensate optics the model takes
    # as ideal, so they are left out: the simulation shows the plane the traps
    # focus in.
    simulated_phase = apply_pupil(
        generator,
        generator.superimpose(
            [array_phase, generator.generate_blazed_grating_phasemask(*blaze_um)]
        ),
        pupil_radius_um,
        *pupil_dump_um,
    )
    trap_plane, top_down = simulate(
        generator,
        simulated_phase,
        grid_x_um + blaze_um[0],
        grid_y_um + blaze_um[1],
        grid_z_um,
    )
    return {
        "mask": generator.transform_phase_8bit(phase).get(),
        "x_um": x_um,
        "y_um": y_um,
        "weights": np.asarray(weights, dtype=float),
        "focal_pixel_um": float(generator.dx_f),
        "grid_x_um": grid_x_um,
        "grid_y_um": grid_y_um,
        "grid_z_um": grid_z_um,
        "simulated_trap_plane": trap_plane,
        "simulated_top_down": top_down,
        "uniformity": float(uniformity[-1]),
        "min_max_ratio": float(min_max[-1]),
        "mse": float(mse[-1]),
    }
