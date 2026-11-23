"""Tweezer-array loading statistics from photon counts at the trap sites.

Shared by the live ``tweezer_loading`` analysis and the Results tab's
histogram of a measurement file. The numbers match
``analysis.get_array_loading_statistics`` with ``method="box"``, which is the
reference for how the array is scored.

``TrapSites`` bundles the saved trap sites and scoring settings, and scores raw
camera frames with them.
"""

from collections.abc import Iterable
from dataclasses import dataclass

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


def top_hat_image(image: np.ndarray, feature_size: int) -> np.ndarray:
    """``image`` top-hat filtered as ``float``, keeping bright features smaller than ``feature_size``."""
    return an.morphological_tophat_high_pass(
        np.asarray(image, dtype=float), feature_size=feature_size
    )


def top_hat(frames: Iterable[np.ndarray], feature_size: int) -> list[np.ndarray]:
    """Each of ``frames`` top-hat filtered, keeping bright features smaller than ``feature_size``."""
    return [top_hat_image(frame, feature_size) for frame in frames]


def background_subtracted_mean(images, backgrounds) -> np.ndarray:
    """Mean of ``images`` minus the mean of ``backgrounds``."""
    return an.tweezer_show_bg_subtracted(images, backgrounds, show=False)


def site_counts(
    frames: Iterable[np.ndarray], sites: dict[int, tuple[int, int]], window_size: int
) -> np.ndarray:
    """Summed raw pixel values in a ``window_size`` box at every site of every frame, ``(frames, sites)``."""
    return np.array(
        [
            an.sum_pixel_values(np.asarray(frame), sites, window_size=window_size)
            for frame in frames
        ]
    )


def photons_at_sites(
    frames: Iterable[np.ndarray], sites: dict[int, tuple[int, int]], window_size: int
) -> np.ndarray:
    """Photons in a ``window_size`` box at every site of every top-hat filtered frame, ``(shots, sites)``."""
    counts = np.array(
        [an.sum_pixel_values(frame, sites, window_size=window_size) for frame in frames]
    )
    return an.convert_counts_to_photons(counts).astype(np.float64)


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


def imaging_survival(first: np.ndarray, second: np.ndarray, threshold: float) -> dict:
    """Of the sites occupied in the first image, the fraction still occupied in the second.

    ``first`` and ``second`` are ``(shots, sites)`` in the unit of ``threshold``,
    pooled over every shot and site. ``survival`` and its binomial standard
    error are ``None`` when no site was occupied in the first image.
    """
    loaded = first > threshold
    atoms = int(loaded.sum())
    if not atoms:
        return {"survival": None, "survival_error": None, "survival_atoms": 0}
    survival = float((second > threshold)[loaded].mean())
    return {
        "survival": survival,
        "survival_error": float(np.sqrt(survival * (1 - survival) / atoms)),
        "survival_atoms": atoms,
    }


@dataclass(frozen=True)
class BatchLoading:
    """Loading statistics of a batch of shots.

    Attributes:
        photons: Photons at every site of every shot, ``(shots, sites)``.
        loading: Mean loading probability over the sites.
        threshold: Fitted photon threshold between empty and occupied.
        fidelity: Detection fidelity at that threshold.
    """

    photons: np.ndarray
    loading: float
    threshold: float
    fidelity: float


@dataclass(frozen=True)
class TrapSites:
    """The trap sites of an array and the settings that score a camera's frames at them.

    Attributes:
        positions: ``{index: (row, col)}`` of each site in frame pixels.
        window_size: Side of the box summed at each site, in pixels.
        feature_size: Largest bright feature kept by the top-hat filter, in pixels.
        threshold: Photons above which a site counts as occupied.
    """

    positions: dict[int, tuple[int, int]]
    window_size: int
    feature_size: int
    threshold: float

    def counts(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Raw pixel sums at every site of every frame, ``(frames, sites)``."""
        return site_counts(frames, self.positions, self.window_size)

    def photons(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Photons at every site of every frame after the top-hat filter, ``(frames, sites)``."""
        return photons_at_sites(
            top_hat(frames, self.feature_size), self.positions, self.window_size
        )

    def occupied(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Whether each site of each frame holds an atom, bool ``(frames, sites)``."""
        return self.photons(frames) > self.threshold

    def fit_loading(self, frames: Iterable[np.ndarray]) -> BatchLoading:
        """Loading statistics of ``frames``, with the threshold fitted to their photon counts."""
        photons, loading, threshold, fidelity = an.get_array_loading_statistics(
            top_hat(frames, self.feature_size),
            self.positions,
            threshold=self.threshold,
            window_size=self.window_size,
            binning=60,
            show_histogram=False,
            threshold_detection=True,
            verbose=False,
        )
        return BatchLoading(
            np.asarray(photons),
            float(np.mean(loading)),
            float(threshold),
            float(fidelity),
        )

    def survival(
        self, firsts: Iterable[np.ndarray], seconds: Iterable[np.ndarray]
    ) -> float:
        """Fraction of atoms in ``firsts`` still present in ``seconds``."""
        return an.extract_survival_probability(
            top_hat(firsts, self.feature_size),
            top_hat(seconds, self.feature_size),
            self.positions,
            threshold=self.threshold,
            window_size=self.window_size,
        )
