"""The radial trap frequency from tweezer survival against a parametric modulation frequency.

Modulating the trap depth at ``2 ν / n`` heats atoms out of a trap of frequency
``ν``, so a scan shows several survival dips: the parametric resonances of the
radial and axial motion and their subharmonics. The radial ``2 ν_r`` is the
highest-frequency dip that is still one of the most pronounced, since the axial
frequency is lower and the subharmonics are weaker: :func:`trap_frequency`
fits a Gaussian to each dip alone, then all of them together on one baseline,
takes the highest-frequency dip at least ``min_relative_depth`` as prominent as
the deepest, and halves its centre from the joint fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pytweezer.experiment.storage import Measurement
from pytweezer.experiment.summary import aggregate

FREQUENCY_ARGUMENT = "modulation_frequency"
OCCUPANCY = ("first_occupied", "second_occupied")
#: Dips shallower than this are noise, whatever the error bars say.
MIN_PROMINENCE = 0.1


@dataclass
class DipFit:
    """One survival dip; frequencies in Hz.

    ``curve`` is the Gaussian fitted to this dip alone, as ``(baseline, depth,
    centre, sigma)``, and ``fit_range_hz`` the span of the points it fitted.
    ``centre_hz`` comes from the joint fit of every dip if ``joint``, else from
    ``curve``, else it is the dip's lowest point.
    """

    prominence: float
    centre_hz: float = np.nan
    centre_err_hz: float = np.nan
    curve: tuple[float, float, float, float] | None = None
    fit_range_hz: tuple[float, float] | None = None
    joint: bool = False
    message: str = ""

    @property
    def fitted(self) -> bool:
        return self.joint or self.curve is not None

    @property
    def trap_hz(self) -> float:
        """The trap frequency if this dip is a ``2 ν`` resonance."""
        return self.centre_hz / 2

    @property
    def trap_err_hz(self) -> float:
        return self.centre_err_hz / 2


@dataclass
class TrapFrequencyFit:
    """The radial trap frequency found in one survival spectrum; frequencies in Hz.

    ``dips`` are the detected dips in order of frequency, each fitted, and
    ``chosen`` indexes the one taken as ``2 ν_r`` (``None`` when there is none).
    ``joint_curve`` is the :func:`gaussian_dips` parameters of all the dips
    fitted to the whole spectrum together, ``None`` if that fit failed.
    """

    frequency: np.ndarray
    survival: np.ndarray
    error: np.ndarray
    dips: list[DipFit] = field(default_factory=list)
    chosen: int | None = None
    joint_curve: tuple[float, ...] | None = None
    message: str = ""

    @property
    def success(self) -> bool:
        return self.chosen is not None

    @property
    def radial_dip(self) -> DipFit | None:
        return None if self.chosen is None else self.dips[self.chosen]

    @property
    def other_dips(self) -> list[DipFit]:
        return [dip for index, dip in enumerate(self.dips) if index != self.chosen]

    @property
    def radial_hz(self) -> float:
        return np.nan if self.chosen is None else self.radial_dip.trap_hz

    @property
    def radial_err_hz(self) -> float:
        return np.nan if self.chosen is None else self.radial_dip.trap_err_hz


def gaussian_dip(f, baseline, depth, centre, sigma):
    return baseline - depth * np.exp(-((f - centre) ** 2) / (2 * sigma**2))


def gaussian_dips(f, baseline, *dips):
    """One baseline minus a Gaussian for each ``(depth, centre, sigma)`` in ``dips``."""
    return baseline + sum(
        gaussian_dip(f, 0.0, depth, centre, sigma)
        for depth, centre, sigma in np.reshape(dips, (-1, 3))
    )


def needed_results(measurement: Measurement) -> list[str]:
    """Each site's occupancy in both images if recorded, else the survival per shot."""
    if all(name in measurement.result_shapes for name in OCCUPANCY):
        return list(OCCUPANCY)
    return ["survival"]


def survival_spectrum(
    measurement: Measurement, argument: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(frequency, survival, error)`` against scanned ``argument``, sorted by frequency.

    With the site occupancies recorded, the survival at a frequency is the
    fraction of all atoms loaded in its shots that survived, with its binomial
    error; otherwise the mean and standard error of the shots' ``survival``.
    """
    x = np.asarray(measurement.points[argument], dtype=float)
    if all(name in measurement.results for name in OCCUPANCY):
        first = np.asarray(measurement.results["first_occupied"], dtype=bool)
        second = np.asarray(measurement.results["second_occupied"], dtype=bool)
        recorded = np.asarray(measurement.recorded["first_occupied"], dtype=bool)
        loaded = first.reshape(len(x), -1).sum(axis=1)
        survived = (first & second).reshape(len(x), -1).sum(axis=1)
        frequencies = np.unique(x[recorded])
        survival, error = [], []
        for frequency in frequencies:
            shots = recorded & (x == frequency)
            n = loaded[shots].sum()
            p = survived[shots].sum() / n if n else np.nan
            survival.append(p)
            error.append(np.sqrt(p * (1 - p) / n) if n else np.nan)
        return frequencies, np.array(survival), np.array(error)
    empty = (x[:0], np.empty(0), np.empty(0), np.empty(0, dtype=int))
    xs, means, errors, _ = next(
        iter(aggregate(x, measurement.results["survival"]).values()), empty
    )
    return np.asarray(xs, dtype=float), np.asarray(means), np.asarray(errors)


def find_dips(survival: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the survival dips and their prominences.

    A dip needs a measured point on each side. Both its prominence and its
    depth below the median survival must exceed :data:`MIN_PROMINENCE` and
    five times the typical error bar: prominence alone is measured from the
    maxima beside it, which noise raises.
    """
    typical_error = np.nanmedian(error) if np.isfinite(error).any() else 0.0
    threshold = max(MIN_PROMINENCE, 5 * typical_error)
    peaks, properties = find_peaks(-survival, prominence=threshold)
    deep = np.nanmedian(survival) - survival[peaks] >= threshold
    return peaks[deep], properties["prominences"][deep]


def trap_frequency(
    measurement: Measurement,
    argument: str = FREQUENCY_ARGUMENT,
    min_relative_depth: float = 0.5,
) -> TrapFrequencyFit:
    """The radial trap frequency from a survival spectrum against ``argument`` in Hz.

    Args:
        measurement: Loaded with :func:`needed_results`.
        argument: The scanned modulation frequency.
        min_relative_depth: A dip is a candidate for ``2 ν_r`` if its
            prominence is at least this fraction of the deepest dip's.
    """
    frequency, survival, error = survival_spectrum(measurement, argument)
    keep = np.isfinite(survival)
    result = TrapFrequencyFit(frequency[keep], survival[keep], error[keep])
    if result.frequency.size < 3:
        result.message = "Too few frequencies measured to find a dip."
        return result
    indices, prominences = find_dips(result.survival, result.error)
    if not indices.size:
        result.message = "No dip in survival stands out from the noise."
        return result
    result.dips = [
        fit_dip(result.frequency, result.survival, result.error, indices, n, prominence)
        for n, prominence in enumerate(prominences)
    ]
    joint = fit_all_dips(result.frequency, result.survival, result.error, result.dips)
    if joint is not None:
        result.joint_curve, joint_errors = joint
        for dip, centre, centre_err in zip(
            result.dips, result.joint_curve[2::3], joint_errors[2::3]
        ):
            if np.isfinite(centre_err):
                dip.centre_hz, dip.centre_err_hz = centre, centre_err
                dip.joint = True
                dip.message = "joint Gaussian fit of every dip"
    candidates = np.flatnonzero(prominences >= min_relative_depth * prominences.max())
    result.chosen = int(candidates[np.argmax(result.frequency[indices[candidates]])])
    result.message = result.radial_dip.message
    return result


def fit_dip(
    f: np.ndarray,
    p: np.ndarray,
    error: np.ndarray,
    indices: np.ndarray,
    n: int,
    prominence: float,
) -> DipFit:
    """Fit a Gaussian to dip ``n`` of ``indices`` between its neighbours; else take its lowest point."""
    centre_index = indices[n]
    low = (indices[n - 1] + centre_index) // 2 + 1 if n > 0 else 0
    high = (centre_index + indices[n + 1] + 1) // 2 if n + 1 < indices.size else f.size
    region = slice(low, high)
    step = np.median(np.diff(f))
    dip = DipFit(float(prominence), float(f[centre_index]), float(step / 2))
    if high - low < 5:
        dip.message = (
            "Too few points across the dip to fit it: its lowest point is taken."
        )
        return dip
    fx, fy = f[region], p[region]
    sigma_y = error[region]
    weighted = np.isfinite(sigma_y).all() and (sigma_y > 0).all()
    try:
        parameters, covariance = curve_fit(
            gaussian_dip,
            fx,
            fy,
            p0=(fy.max(), prominence, f[centre_index], step),
            sigma=sigma_y if weighted else None,
            absolute_sigma=weighted,
            bounds=(
                (0, 0, fx[0], step / 4),
                (1.5, 1.5, fx[-1], fx[-1] - fx[0]),
            ),
        )
    except (RuntimeError, ValueError):
        dip.message = "The dip could not be fitted: its lowest point is taken."
        return dip
    centre_err = float(np.sqrt(covariance[2, 2]))
    if not np.isfinite(centre_err):
        dip.message = "The dip's fit is unconstrained: its lowest point is taken."
        return dip
    dip.curve = tuple(float(value) for value in parameters)
    dip.fit_range_hz = (float(fx[0]), float(fx[-1]))
    dip.centre_hz = dip.curve[2]
    dip.centre_err_hz = centre_err
    dip.message = "Gaussian fit to the dip"
    return dip


def fit_all_dips(
    f: np.ndarray, p: np.ndarray, error: np.ndarray, dips: list[DipFit]
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Fit one baseline and a Gaussian per dip to the whole spectrum at once.

    Starts from each dip's own fit, or its lowest point if it had none. Each
    centre is kept between the midpoints to its neighbouring dips, so the
    Gaussians cannot swap or merge. Returns the :func:`gaussian_dips`
    parameters and their errors, or ``None`` if the fit fails.
    """
    step = np.median(np.diff(f))
    centres = [dip.centre_hz for dip in dips]
    edges = [f[0], *((a + b) / 2 for a, b in pairwise(centres)), f[-1]]
    start, lower, upper = [np.nanmax(p)], [0.0], [1.5]
    for dip, (low, high) in zip(dips, pairwise(edges)):
        depth, centre, sigma = (
            dip.curve[1:] if dip.curve else (dip.prominence, dip.centre_hz, step)
        )
        start += [depth, centre, sigma]
        lower += [0.0, low, step / 4]
        upper += [1.5, high, f[-1] - f[0]]
    weighted = np.isfinite(error).all() and (error > 0).all()
    try:
        parameters, covariance = curve_fit(
            gaussian_dips,
            f,
            p,
            p0=np.clip(start, lower, upper),
            sigma=error if weighted else None,
            absolute_sigma=weighted,
            bounds=(lower, upper),
        )
    except (RuntimeError, ValueError):
        return None
    errors = np.sqrt(np.diag(covariance))
    return tuple(map(float, parameters)), tuple(map(float, errors))
