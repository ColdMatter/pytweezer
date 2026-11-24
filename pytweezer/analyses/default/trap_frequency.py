"""The radial trap frequency from tweezer survival against a parametric modulation frequency.

Modulating the trap depth at ``2 ν / n`` heats atoms out of a trap of frequency
``ν``, so a scan shows several survival dips: the parametric resonances of the
radial and axial motion and their subharmonics. The radial ``2 ν_r`` is the
highest-frequency dip that is still one of the most pronounced, since the axial
frequency is lower and the subharmonics are weaker: :func:`trap_frequency`
takes the highest-frequency dip at least ``min_relative_depth`` as prominent as
the deepest, fits a Gaussian dip to it and halves its centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
class TrapFrequencyFit:
    """The radial trap frequency found in one survival spectrum; frequencies in Hz.

    ``dips`` are the detected dips' frequencies and ``prominences`` their
    depths below the surrounding survival. ``chosen`` indexes the one taken as
    ``2 ν_r`` (``None`` when there is none), and ``curve`` is the fitted
    Gaussian dip as ``(baseline, depth, centre, sigma)``, ``None`` when the
    centre is the lowest point of the dip instead; ``fit_range_hz`` is
    the span of the points fitted.
    """

    frequency: np.ndarray
    survival: np.ndarray
    error: np.ndarray
    dips: np.ndarray = field(default_factory=lambda: np.empty(0))
    prominences: np.ndarray = field(default_factory=lambda: np.empty(0))
    chosen: int | None = None
    resonance_hz: float = np.nan
    resonance_err_hz: float = np.nan
    curve: tuple[float, float, float, float] | None = None
    fit_range_hz: tuple[float, float] | None = None
    message: str = ""

    @property
    def success(self) -> bool:
        return self.chosen is not None

    @property
    def radial_hz(self) -> float:
        return self.resonance_hz / 2

    @property
    def radial_err_hz(self) -> float:
        return self.resonance_err_hz / 2


def gaussian_dip(f, baseline, depth, centre, sigma):
    return baseline - depth * np.exp(-((f - centre) ** 2) / (2 * sigma**2))


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
    result.dips = result.frequency[indices]
    result.prominences = prominences
    if not indices.size:
        result.message = "No dip in survival stands out from the noise."
        return result
    candidates = np.flatnonzero(prominences >= min_relative_depth * prominences.max())
    result.chosen = int(candidates[np.argmax(result.dips[candidates])])
    _fit_dip(result, indices)
    return result


def _fit_dip(result: TrapFrequencyFit, indices: np.ndarray) -> None:
    """Fit a Gaussian dip between the chosen dip's neighbours; else take its lowest point."""
    f, p, chosen = result.frequency, result.survival, result.chosen
    centre_index = indices[chosen]
    low = (indices[chosen - 1] + centre_index) // 2 + 1 if chosen > 0 else 0
    high = (
        (centre_index + indices[chosen + 1] + 1) // 2
        if chosen + 1 < indices.size
        else f.size
    )
    region = slice(low, high)
    step = np.median(np.diff(f))
    result.resonance_hz = float(f[centre_index])
    result.resonance_err_hz = float(step / 2)
    if high - low < 5:
        result.message = (
            "Too few points across the dip to fit it: its lowest point is taken."
        )
        return
    fx, fy = f[region], p[region]
    sigma_y = result.error[region]
    weighted = np.isfinite(sigma_y).all() and (sigma_y > 0).all()
    try:
        parameters, covariance = curve_fit(
            gaussian_dip,
            fx,
            fy,
            p0=(fy.max(), result.prominences[chosen], f[centre_index], step),
            sigma=sigma_y if weighted else None,
            absolute_sigma=weighted,
            bounds=(
                (0, 0, fx[0], step / 4),
                (1.5, 1.5, fx[-1], fx[-1] - fx[0]),
            ),
        )
    except (RuntimeError, ValueError):
        result.message = "The dip could not be fitted: its lowest point is taken."
        return
    centre_err = float(np.sqrt(covariance[2, 2]))
    if not np.isfinite(centre_err):
        result.message = "The dip's fit is unconstrained: its lowest point is taken."
        return
    result.curve = tuple(float(value) for value in parameters)
    result.fit_range_hz = (float(fx[0]), float(fx[-1]))
    result.resonance_hz = result.curve[2]
    result.resonance_err_hz = centre_err
    result.message = "Gaussian fit to the dip"
