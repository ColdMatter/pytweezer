"""The cloud fits a measurement file recorded, for the Results tab.

Reads the cloud fits an experiment records per point for each camera, as
:meth:`~pytweezer.experiments.load_tweezers.LoadTweezersMOTImage.record_cloud_fit`
does for the ORCA: ``<camera>_<field>`` for every field of
:data:`~pytweezer.analysis.cloud_fit.FIELDS`, ``<camera>_<field>_error`` for
each of :data:`~pytweezer.analysis.cloud_fit.ERROR_FIELDS`, and optionally
``<camera>_mean_image``.

The file does not record where the mean image sits in the camera frame, so
:func:`point_image` refits it in its own pixel coordinates: the same fit of the
same image as the one recorded, placed where the image is drawn.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import atomic_mass, k

from pytweezer.analysis.cloud_fit import (
    ERROR_FIELDS,
    FIELDS,
    CloudFit,
    fit_cloud,
)
from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import aggregate

SIZES = ("sigma_x", "sigma_y")
CENTRES = ("x0", "y0")
TIME_ARGUMENT_HINTS = ("freeexp", "freefall", "drop")
MIN_TIMES = 3


def cloud_cameras(measurement: Measurement) -> list[str]:
    """The cameras whose cloud fits the file recorded, in recording order."""
    return [
        name.removesuffix("_sigma_x")
        for name, shape in measurement.result_shapes.items()
        if name.endswith("_sigma_x") and shape == ()
    ]


def scanned_arguments(measurement: Measurement) -> list[str]:
    """The numeric scanned arguments, in scan order."""
    return [
        axis.argument
        for axis in measurement.scan.axes
        if np.asarray(measurement.points[axis.argument]).dtype.kind in "biuf"
    ]


def needed_results(measurement: Measurement) -> list[str]:
    """Every camera's recorded fit fields and their errors; no mean images."""
    names = []
    for camera in cloud_cameras(measurement):
        names += [f"{camera}_{field}" for field in FIELDS]
        names += [f"{camera}_{field}_error" for field in ERROR_FIELDS]
    return [name for name in names if name in measurement.result_shapes]


def pooled_field(
    measurement: Measurement, camera: str, field: str, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(axis values, mean, error)`` of the recorded ``<camera>_<field>`` against ``axis``.

    Points sharing a value of ``axis`` are pooled: the mean and standard error
    over those whose fit succeeded. A value with one such point keeps that
    point's recorded per-frame error (NaN when it had a single frame).
    """
    x = np.asarray(measurement.points[axis])
    y = np.asarray(measurement.results[f"{camera}_{field}"], dtype=float)
    recorded_error = np.asarray(
        measurement.results[f"{camera}_{field}_error"], dtype=float
    )
    empty = (x[:0], np.empty(0), np.empty(0), np.empty(0, dtype=int))
    xs, means, errors, counts = next(iter(aggregate(x, y).values()), empty)
    for i in np.flatnonzero(counts == 1):
        errors[i] = recorded_error[(x == xs[i]) & np.isfinite(y)][0]
    return xs, means, errors


def cloud_sizes(
    measurement: Measurement, camera: str, axis: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{"sigma_x": (axis values, mean, error), "sigma_y": ...}`` against ``axis``.

    Pooled as :func:`pooled_field` does.

    Args:
        measurement: Loaded with the camera's ``sigma_x``, ``sigma_y`` and
            their errors.
        camera: As :func:`cloud_cameras` gives.
        axis: A ``/points`` column, such as a scanned argument or ``"index"``.
    """
    return {field: pooled_field(measurement, camera, field, axis) for field in SIZES}


def point_image(
    measurement: Measurement, camera: str, index: int, mean_image: np.ndarray | None
) -> tuple[np.ndarray | None, CloudFit]:
    """Point ``index``'s mean image and its cloud fit.

    With an image, the fit is in the image's pixel coordinates (pixel
    ``(row, col)`` centred at ``x = col + 0.5``, ``y = row + 0.5``), so it
    overlays the image drawn row-major. The image is ``None`` when the file
    recorded none for this point; the fit is then the recorded one, in the
    file's units.

    Args:
        measurement: Loaded with the camera's fields.
        camera: As :func:`cloud_cameras` gives.
        index: Row of the point in ``measurement.points``.
        mean_image: The point's recorded ``<camera>_mean_image``, read on its
            own as it is large; ``None`` if the file has none.
    """
    image = None if mean_image is None else np.asarray(mean_image)
    if image is not None and np.isfinite(image).any():
        return image, fit_cloud(image)
    values = {
        field: float(measurement.results[f"{camera}_{field}"][index])
        for field in FIELDS
    }
    return None, CloudFit(
        **values,
        units=measurement.units.get(f"{camera}_x0", ""),
        success=bool(np.isfinite(values["sigma_x"])),
    )


@dataclass(frozen=True)
class TofFit:
    """Result of :func:`tof_fit`: ``sigma**2 = sigma0**2 + (k_B T / m) t**2``.

    ``slope`` and ``intercept`` are the line in ``(t**2, sigma**2)`` in SI;
    ``sigma0`` is NaN when the intercept is negative. When ``success`` is False
    ``message`` says why; otherwise it says how the errors were found.
    """

    temperature_K: float
    temperature_err_K: float
    sigma0: float
    sigma0_err: float
    success: bool
    message: str
    slope: float = np.nan
    intercept: float = np.nan


@dataclass(frozen=True)
class GravityFit:
    """Result of :func:`fit_gravity`: ``centre = offset + g t**2 / 2``."""

    g: float
    g_err: float
    offset: float
    success: bool
    message: str


@dataclass(frozen=True)
class _LineFit:
    intercept: float
    slope: float
    intercept_err: float
    slope_err: float
    description: str


def guess_time_axis(names: list[str]) -> str | None:
    """The first of ``names`` that looks like a free-fall time, or ``None``.

    Matches ``FreeExp``, ``FreeFall`` or ``Drop`` in any case.
    """
    for name in names:
        if any(hint in name.lower() for hint in TIME_ARGUMENT_HINTS):
            return name
    return None


def _line_fit(u: np.ndarray, v: np.ndarray, v_err: np.ndarray) -> _LineFit:
    """Least-squares line ``v = intercept + slope u`` and its covariance errors.

    Weighted by ``v_err`` when every error is finite and positive, with the
    covariance taken as absolute and then widened by ``sqrt(chi2 / dof)`` if the
    scatter exceeds what those errors allow (they are often few-shot standard
    errors, and an understated error is worse than an overstated one). Otherwise
    unweighted, with the covariance scaled to the residual scatter.
    """
    weighted = bool(np.all(np.isfinite(v_err) & (v_err > 0)))
    u_scale = np.max(np.abs(u))
    design = np.column_stack([np.ones_like(u), u / u_scale])
    if weighted:
        design, target = design / v_err[:, None], v / v_err
    else:
        target = v
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    dof = len(u) - 2
    chi2_per_dof = float(residual @ residual) / dof
    covariance = np.linalg.inv(design.T @ design)
    if weighted:
        factor = max(1.0, chi2_per_dof)
        description = "weighted by the recorded errors" + (
            f", errors widened by {np.sqrt(factor):.2g} to the scatter"
            if factor > 1.0 + 1e-9
            else ""
        )
    else:
        factor = chi2_per_dof
        description = "unweighted: no usable errors, errors from the scatter"
    errors = np.sqrt(np.diag(covariance) * factor)
    return _LineFit(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1] / u_scale),
        intercept_err=float(errors[0]),
        slope_err=float(errors[1] / u_scale),
        description=description,
    )


def _finite_rows(*columns: np.ndarray) -> np.ndarray:
    return np.all([np.isfinite(column) for column in columns], axis=0)


def _too_few_times(t: np.ndarray) -> str | None:
    distinct = len(np.unique(t))
    if distinct < MIN_TIMES:
        return f"Needs at least three distinct times; this file has {distinct}."
    return None


def _failed_fit(message: str) -> TofFit:
    return TofFit(np.nan, np.nan, np.nan, np.nan, False, message)


def tof_fit(
    t_s: np.ndarray, sigma: np.ndarray, sigma_err: np.ndarray, mass_amu: float
) -> TofFit:
    """Temperature of a freely expanding cloud from its size against time.

    Fits ``sigma**2 = sigma0**2 + (k_B T / m) t**2``. Each ``sigma**2`` carries
    ``2 sigma sigma_err``; the fit is weighted by those if all are finite and
    positive, and unweighted otherwise (see :func:`_line_fit` for the errors).

    Args:
        t_s: Free-fall times in seconds.
        sigma: Cloud sizes in metres.
        sigma_err: Their errors in metres; NaN where unknown.
        mass_amu: Mass of the species in atomic mass units.
    """
    t_s, sigma, sigma_err = (
        np.asarray(a, dtype=float) for a in (t_s, sigma, sigma_err)
    )
    keep = _finite_rows(t_s, sigma) & (sigma > 0)
    t_s, sigma, sigma_err = t_s[keep], sigma[keep], sigma_err[keep]
    if (message := _too_few_times(t_s)) is not None:
        return _failed_fit(message)
    line = _line_fit(t_s**2, sigma**2, 2 * sigma * sigma_err)
    if line.slope <= 0:
        return TofFit(
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            "The cloud does not grow with time: no temperature.",
            line.slope,
            line.intercept,
        )
    kelvin_per_slope = mass_amu * atomic_mass / k
    sigma0 = np.sqrt(line.intercept) if line.intercept > 0 else np.nan
    return TofFit(
        temperature_K=line.slope * kelvin_per_slope,
        temperature_err_K=line.slope_err * kelvin_per_slope,
        sigma0=float(sigma0),
        sigma0_err=float(line.intercept_err / (2 * sigma0)),
        success=True,
        message=line.description,
        slope=line.slope,
        intercept=line.intercept,
    )


def fit_gravity(
    t_s: np.ndarray, centre: np.ndarray, centre_err: np.ndarray
) -> GravityFit:
    """``g`` from the cloud centre falling as ``centre = offset + g t**2 / 2``.

    Weighted and with errors as :func:`tof_fit`. The sign is that of the
    centre's coordinate: positive when the cloud moves towards larger values.
    """
    t_s, centre, centre_err = (
        np.asarray(a, dtype=float) for a in (t_s, centre, centre_err)
    )
    keep = _finite_rows(t_s, centre)
    t_s, centre, centre_err = t_s[keep], centre[keep], centre_err[keep]
    if (message := _too_few_times(t_s)) is not None:
        return GravityFit(np.nan, np.nan, np.nan, False, message)
    line = _line_fit(t_s**2, centre, centre_err)
    return GravityFit(
        2 * line.slope, 2 * line.slope_err, line.intercept, True, line.description
    )


def cloud_temperature(
    measurement: Measurement, camera: str, axis: str, seconds_per_unit: float
) -> dict:
    """Time-of-flight temperature of a camera's cloud against scanned argument ``axis``.

    The values of ``axis`` times ``seconds_per_unit`` are the free-fall times.
    Sizes and centres are pooled as :func:`pooled_field` does. The mass is the
    file's constant ``cloud_mass_amu``. Sizes recorded in pixels give failed
    fits: a temperature needs metres.

    Args:
        measurement: Loaded with :func:`needed_results` and the constant.
        camera: As :func:`cloud_cameras` gives.
        axis: A scanned argument holding the free-fall time.
        seconds_per_unit: Seconds in one unit of the recorded values of ``axis``.

    Returns:
        A dict. ``"unit"`` is the length unit recorded. ``"fits"`` maps
        ``"sigma_x"`` and ``"sigma_y"`` to a :class:`TofFit`. ``"sigma2"`` maps
        each size field to ``(t**2, sigma**2, error)`` in SI. ``"centres"`` maps
        ``"x0"`` and ``"y0"`` to ``(t**2, mean, error)``. ``"gravity"`` is the
        :class:`GravityFit` of ``y0``.
    """
    unit = measurement.units.get(f"{camera}_sigma_x", "")
    mass_amu = float(measurement.constants["cloud_mass_amu"])
    sigma2, fits = {}, {}
    for field in SIZES:
        x, sigma, error = pooled_field(measurement, camera, field, axis)
        t = x * seconds_per_unit
        sigma2[field] = (t**2, sigma**2, 2 * sigma * error)
        fits[field] = (
            tof_fit(t, sigma, error, mass_amu)
            if unit == "m"
            else _failed_fit(
                "Sizes were recorded in pixels (the camera has no pixel scale): "
                "no temperature."
            )
        )
    centres = {}
    for field in CENTRES:
        x, mean, error = pooled_field(measurement, camera, field, axis)
        centres[field] = ((x * seconds_per_unit) ** 2, mean, error)
    t2, y0, y0_err = centres["y0"]
    gravity = (
        fit_gravity(np.sqrt(t2), y0, y0_err)
        if unit == "m"
        else GravityFit(
            np.nan, np.nan, np.nan, False, "Centres were recorded in pixels."
        )
    )
    return {
        "unit": unit,
        "fits": fits,
        "sigma2": sigma2,
        "centres": centres,
        "gravity": gravity,
    }
