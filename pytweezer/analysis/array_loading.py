"""Tweezer-array loading statistics from photon counts at the trap sites.

Shared by the live ``tweezer_loading`` analysis and the Results tab's
histogram of a measurement file. The numbers match
``analysis.get_array_loading_statistics`` with ``method="box"``, which is the
reference for how the array is scored.
"""

from collections.abc import Iterable

import numpy as np
from scipy.stats import norm

from pytweezer.analysis import analysis as an


def find_sites(image, n_sites, step):
    """``n_sites`` bright spots in ``image`` as ``{index: (row, col)}``, or ``None``."""
    level = float(image.max())
    while level >= 1:
        centres = an.detect_bright_points(image, threshold=level)
        if len(centres) == n_sites:
            return {i: (int(row), int(col)) for i, (row, col) in enumerate(centres)}
        level -= step
    return None


def photons_at_sites(
    frames: Iterable[np.ndarray], sites: dict[int, tuple[int, int]], window_size: int
) -> np.ndarray:
    """Photons in a ``window_size`` box at every site of every top-hat filtered frame, ``(shots, sites)``."""
    counts = np.array(
        [an.sum_pixel_values(frame, sites, window_size=window_size) for frame in frames]
    )
    return an.convert_counts_to_photons(counts).astype(np.float64)


def occupied_sites(
    frame: np.ndarray,
    sites: np.ndarray,
    *,
    window_size: int,
    feature_size: int,
    threshold: float,
) -> np.ndarray:
    """Whether each of ``sites`` (``(N, 2)`` ``(y, x)`` pixels) holds an atom in ``frame``.

    A site is occupied when its photons in the top-hat filtered frame exceed
    ``threshold``, as :func:`photons_at_sites` counts them.
    """
    filtered = an.morphological_tophat_high_pass(
        np.asarray(frame, dtype=float), feature_size=feature_size
    )
    site_pixels = {index: (int(y), int(x)) for index, (y, x) in enumerate(sites)}
    return photons_at_sites([filtered], site_pixels, window_size)[0] > threshold


def loading_statistics(
    photons: np.ndarray, sites: dict[int, tuple[int, int]], fallback_threshold: float
) -> dict:
    """Threshold, fidelity and per-site loading of ``photons`` (``(shots, sites)``).

    The threshold is where the two-Gaussian fit to all the photon counts
    crosses; if the fit fails, ``fallback_threshold`` is used and
    ``background``/``signal`` are ``None``.
    """
    counts = photons.ravel()
    background = signal = None
    try:
        threshold, background, signal = an.detect_loading_threshold(counts)
    except (ValueError, TypeError, StopIteration, RuntimeError):
        threshold = fallback_threshold
        try:
            empty, full = counts[counts < threshold], counts[counts >= threshold]
            # TODO: check the fallback fidelity: this is the error probability,
            # not 1 minus it (as in analysis.get_array_loading_statistics)
            fidelity = norm.cdf((threshold - full.mean()) / np.sqrt(full.var())) + (
                1 - norm.cdf((threshold - empty.mean()) / np.sqrt(empty.var()))
            )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            fidelity = np.nan
    else:
        (mu_bg, var_bg, weight_bg), (mu_sig, var_sig, weight_sig) = background, signal
        false_positive = 1.0 - norm.cdf(threshold, mu_bg, np.sqrt(var_bg))
        false_negative = norm.cdf(threshold, mu_sig, np.sqrt(var_sig))
        fidelity = 1.0 - (weight_bg * false_positive + weight_sig * false_negative)
        background = [float(value) for value in background]
        signal = [float(value) for value in signal]

    site_loading = (photons > threshold).mean(axis=0)
    mean_loading = float(site_loading.mean())
    rows, cols = zip(*(sites[i] for i in range(len(sites))), strict=True)
    return {
        "sites": len(sites),
        "threshold": float(threshold),
        "fidelity": float(fidelity) if np.isfinite(fidelity) else None,
        "loading": mean_loading,
        "loading_spread": float(site_loading.std() / mean_loading)
        if mean_loading > 0
        else None,
        "site_loading": [float(value) for value in site_loading],
        "site_rows": list(rows),
        "site_cols": list(cols),
        "background": background,
        "signal": signal,
    }
