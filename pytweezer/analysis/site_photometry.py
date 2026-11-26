"""Light at expected sites in camera frames, scored against the same measurement at empty positions.

A site's signal is the sum over a disc of ``aperture`` pixels about it, less
the disc's area times the mean of a ring ``ring_inner`` to ``ring_outer``
pixels out, so a uniform offset cancels. The same signal at empty positions of
the same frame, a grid at least ``aperture + ring_outer`` from every site and
``ring_outer`` from the edge, gives the frame's noise: a site's score ``z`` is
its signal less the empty positions' mean, over their standard deviation.
Positions are ``(y, x)`` pixels; a site is measured about the pixel nearest it.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

#: Random sets of empty positions drawn to estimate the combined score's chance level.
RANDOM_SETS = 2000


@dataclass(frozen=True)
class SiteScores:
    """Per-shot site signals and scores, and the empty positions they are judged against.

    Attributes:
        counts: ``(shots, sites)`` signal at each site, in the frames' units.
        z: ``(shots, sites)`` score at each site.
        empty_z: ``(shots, empty)`` score at each empty position.
        empty: ``(empty, 2)`` the empty positions, ``(y, x)``.
    """

    counts: np.ndarray
    z: np.ndarray
    empty_z: np.ndarray
    empty: np.ndarray

    def combined(self) -> np.ndarray:
        """Per shot, the sites' scores summed and scaled to unit variance for independent sites."""
        return self.z.sum(axis=1) / np.sqrt(self.z.shape[1])

    def combined_chance(self, rng=None) -> np.ndarray:
        """Mean :meth:`combined` over the shots for random sets of as many empty positions."""
        rng = np.random.default_rng(0) if rng is None else rng
        n_sites, n_empty = self.z.shape[1], self.empty_z.shape[1]
        if n_empty < n_sites:
            return np.array([])
        picks = np.array(
            [rng.choice(n_empty, n_sites, replace=False) for _ in range(RANDOM_SETS)]
        )
        sums = self.empty_z[:, picks].sum(axis=2) / np.sqrt(n_sites)
        return sums.mean(axis=0)


def merge_overlapping(
    sites: np.ndarray, aperture: float
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Sites whose apertures overlap, chained, as one site at their mean position.

    Returns:
        The ``(merged, 2)`` positions, and for each the indices into ``sites``
        of the sites it holds, in order.
    """
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    group = list(range(len(sites)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if np.hypot(*(sites[i] - sites[j])) < 2 * aperture:
                group[root(j)] = root(i)
    members: dict[int, list[int]] = {}
    for i in range(len(sites)):
        members.setdefault(root(i), []).append(i)
    held = [tuple(indices) for indices in members.values()]
    return np.array([sites[list(indices)].mean(axis=0) for indices in held]), held


def disc(radius: float) -> np.ndarray:
    """A 0/1 kernel of the pixels within ``radius`` of its centre pixel."""
    r = int(np.floor(radius))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (np.hypot(x, y) <= radius).astype(float)


def ring(inner: float, outer: float) -> np.ndarray:
    """A 0/1 kernel of the pixels from ``inner`` to below ``outer`` from its centre pixel."""
    r = int(np.ceil(outer))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    distance = np.hypot(x, y)
    return ((distance >= inner) & (distance < outer)).astype(float)


def signal_maps(
    frames: np.ndarray, aperture: float, ring_inner: float, ring_outer: float
) -> np.ndarray:
    """``(shots, H, W)`` signal with each pixel as the site: disc sum less area × ring mean."""
    frames = np.asarray(frames, dtype=float)
    disc_kernel, ring_kernel = disc(aperture), ring(ring_inner, ring_outer)
    ring_kernel = ring_kernel / ring_kernel.sum()
    sums = signal.fftconvolve(frames, disc_kernel[None], mode="same", axes=(1, 2))
    rings = signal.fftconvolve(frames, ring_kernel[None], mode="same", axes=(1, 2))
    return sums - disc_kernel.sum() * rings


def empty_positions(
    shape: tuple[int, int], sites: np.ndarray, aperture: float, ring_outer: float
) -> np.ndarray:
    """``(n, 2)`` grid positions, a disc apart, clear of the sites and of the edge by a ring."""
    margin = int(np.ceil(ring_outer))
    step = max(int(np.ceil(aperture)), 1)
    ys = np.arange(margin, shape[0] - margin, step)
    xs = np.arange(margin, shape[1] - margin, step)
    grid = np.stack(np.meshgrid(ys, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    if len(sites):
        distance = np.hypot(*(grid[:, None, :] - np.asarray(sites)[None]).T).T
        grid = grid[distance.min(axis=1) > aperture + ring_outer]
    return grid


def score_sites(
    frames: np.ndarray,
    sites: np.ndarray,
    aperture: float = 4,
    ring_inner: float = 8,
    ring_outer: float = 12,
) -> SiteScores:
    """Each frame's signal and score at ``sites`` (``(y, x)`` pixels of the frames).

    Raises:
        ValueError: If a site's ring leaves the frame, or the frame has fewer
            than two empty positions to judge against.
    """
    frames = np.asarray(frames, dtype=float)
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    shape = frames.shape[1:]
    pixels = np.rint(sites).astype(int)
    margin = int(np.ceil(ring_outer))
    if (pixels < margin).any() or (pixels >= np.array(shape) - margin).any():
        raise ValueError(
            f"a site's background ring ({ring_outer:g} px) leaves the "
            f"{shape[1]}×{shape[0]} px image"
        )
    empty = empty_positions(shape, sites, aperture, ring_outer)
    if len(empty) < 2:
        raise ValueError("too few empty positions in the image to estimate the noise")
    maps = signal_maps(frames, aperture, ring_inner, ring_outer)
    counts = maps[:, pixels[:, 0], pixels[:, 1]]
    noise = maps[:, empty[:, 0], empty[:, 1]]
    mean, sd = noise.mean(axis=1, keepdims=True), noise.std(axis=1, keepdims=True)
    return SiteScores(
        counts=counts,
        z=(counts - mean) / sd,
        empty_z=(noise - mean) / sd,
        empty=empty,
    )
