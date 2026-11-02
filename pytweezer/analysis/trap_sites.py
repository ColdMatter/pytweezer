"""Place a tweezer array's sites on a camera image by fitting its geometry.

The geometry is each site's ``(x_um, y_um)`` in the focal plane, as the
rearrangement rig reports it. The camera sees it mirrored in x, so a site lands
on the sensor at::

    x_img = -s * x' + x0
    y_img =  s * y' + y0

where ``(x', y')`` is the geometry rotated by θ. :func:`fit_trap_sites` fits
``s``, θ, ``x0`` and ``y0``; the mirror is fixed, because a square grid fits
equally well mirrored and only the site indexing would be wrong. Sites are
``(y, x)`` pixels and keep the geometry's order, bright or not.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize, signal

from pytweezer.analysis.analysis import hamcam_px_per_um

MAX_SCALE_ERROR = 0.2
MIN_CONTRAST = 5.0
MIN_EVENNESS = 0.5


@dataclass(frozen=True)
class TrapSiteFit:
    """Fitted ``(y, x)`` site positions and the transform that put them there."""

    sites: np.ndarray
    scale: float
    angle_deg: float
    x0: float
    y0: float


def default_geometry(
    rows: int = 8, columns: int = 8, spacing_um: float = 5.0
) -> np.ndarray:
    """A square grid's ``(x_um, y_um)``, centred on 0, row-major from the image's top-left.

    Columns run towards decreasing ``x_um`` because the camera mirrors x.
    """
    row, column = np.mgrid[:rows, :columns]
    x_um = -(column - (columns - 1) / 2) * spacing_um
    y_um = (row - (rows - 1) / 2) * spacing_um
    return np.column_stack([x_um.ravel(), y_um.ravel()])


def geometry_to_pixels(
    geometry: np.ndarray, scale: float, angle_deg: float, x0: float, y0: float
) -> np.ndarray:
    """Each site of ``geometry`` as ``(y, x)`` pixels under the given transform."""
    x_um, y_um = np.asarray(geometry, dtype=float).T
    angle = np.deg2rad(angle_deg)
    x_rotated = x_um * np.cos(angle) - y_um * np.sin(angle)
    y_rotated = x_um * np.sin(angle) + y_um * np.cos(angle)
    return np.column_stack([scale * y_rotated + y0, -scale * x_rotated + x0])


def fit_trap_sites(
    image: np.ndarray,
    geometry: np.ndarray,
    *,
    px_per_um: float = hamcam_px_per_um,
    angle_deg: float = -2.5,
    window_size: int = 5,
) -> TrapSiteFit:
    """Fit ``geometry`` onto the bright spots of a background-free ``image``.

    Starts from ``px_per_um`` and ``angle_deg``, finds the offset by
    cross-correlating the image with the geometry, then refines scale, angle
    and offset together on the image smoothed over about ``window_size``.

    Raises:
        ValueError: If a fitted site is off the image, the scale is more than
            ``MAX_SCALE_ERROR`` from ``px_per_um``, the median site is not
            ``MIN_CONTRAST`` noise widths above the background, or it is less
            than ``MIN_EVENNESS`` of the brightest tenth of sites.
    """
    smoothed = ndimage.gaussian_filter(
        np.asarray(image, dtype=float), max(1.0, window_size / 3)
    )
    y0, x0 = _coarse_offset(smoothed, geometry, px_per_um, angle_deg)

    def brightness(params):
        sites = geometry_to_pixels(geometry, *params)
        return -ndimage.map_coordinates(smoothed, sites.T, order=1, cval=0.0).sum()

    result = optimize.minimize(
        brightness,
        [px_per_um, angle_deg, x0, y0],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 4000},
    )
    scale, angle, x0, y0 = result.x
    sites = geometry_to_pixels(geometry, scale, angle, x0, y0)

    if abs(scale / px_per_um - 1) > MAX_SCALE_ERROR:
        raise ValueError(
            f"fitted scale {scale:.3g} px/µm is more than "
            f"{MAX_SCALE_ERROR:.0%} from {px_per_um:.3g}"
        )
    height, width = smoothed.shape
    if (
        (sites < 0).any()
        or (sites[:, 0] > height - 1).any()
        or (sites[:, 1] > width - 1).any()
    ):
        raise ValueError("the fitted array runs off the image")
    background = np.median(smoothed)
    noise = 1.4826 * np.median(np.abs(smoothed - background))
    signal_at_sites = ndimage.map_coordinates(smoothed, sites.T, order=1) - background
    typical = np.median(signal_at_sites)
    if not typical > MIN_CONTRAST * noise:
        raise ValueError(
            f"no array found: the typical site is {typical / noise:.1f} noise "
            f"widths above the background, not {MIN_CONTRAST:g}"
        )
    # A fit that missed the array still lands a few sites on spots; the
    # averaged array lights its sites roughly evenly.
    brightest = np.percentile(signal_at_sites, 90)
    if typical < MIN_EVENNESS * brightest:
        raise ValueError(
            f"the fit missed the array: the typical site is "
            f"{typical / brightest:.0%} as bright as the brightest tenth"
        )
    return TrapSiteFit(sites, float(scale), float(angle), float(x0), float(y0))


def _coarse_offset(smoothed, geometry, px_per_um, angle_deg) -> tuple[float, float]:
    """The ``(y0, x0)`` maximising the cross-correlation with the geometry at the start transform."""
    relative = np.rint(geometry_to_pixels(geometry, px_per_um, angle_deg, 0, 0))
    half = np.abs(relative).max(axis=0).astype(int)
    template = np.zeros(2 * half + 1)
    rows, columns = (relative + half).astype(int).T
    template[rows, columns] = 1.0
    correlation = signal.correlate(smoothed, template, mode="same", method="fft")
    y0, x0 = np.unravel_index(np.argmax(correlation), correlation.shape)
    return float(y0), float(x0)


def to_roi(sites: np.ndarray, roi_y0: int, roi_x0: int) -> np.ndarray:
    """Full-sensor ``(y, x)`` sites in the pixels of an image read from ROI origin ``(roi_y0, roi_x0)``."""
    return np.asarray(sites, dtype=float) - [roi_y0, roi_x0]


def same_geometry(saved: np.ndarray, x_um, y_um, tol_um: float = 0.1) -> bool:
    """Whether ``(x_um, y_um)`` are the sites of ``saved``, in order, each within ``tol_um``."""
    current = np.column_stack([np.ravel(x_um), np.ravel(y_um)])
    saved = np.asarray(saved, dtype=float)
    return saved.shape == current.shape and bool(
        np.all(np.abs(saved - current) <= tol_um)
    )
