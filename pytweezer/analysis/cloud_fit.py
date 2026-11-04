"""Fit a rotated 2-D Gaussian with offset to a cloud in a camera frame.

Coordinates: pixel column ``c`` and row ``r`` sit at
``x = (origin_x + c + 0.5) * res_x``, ``y = (origin_y + r + 0.5) * res_y``, where
``res`` is the stream header's ``_imgresolution`` and ``origin`` its ``_offset``
(binned pixels). Without them ``res = (1, 1)``, ``origin = (0, 0)`` and the
units are ``"px"``; with them ``"m"``. ``angle`` is that of the major axis,
measured from +x towards +y in these coordinates, in (-pi/2, pi/2].

The fit is cheap enough for full 2048 x 2048 frames: a multi-scale blob detection
and moments seed on a coarsely binned copy of the frame locates the cloud, the
least-squares fit then runs on a binned window of +-4 sigma about it with an
analytic Jacobian.

Run as an analysis (:class:`CloudFitAnalysis`):

Input:
    One image stream, raw or background-subtracted.

Output:
    A data message (no array) on ``<name>``: the fields of :class:`CloudFit`
    plus ``rid``, ``index``, ``_imgresolution`` and ``_offset`` from the input
    header. A frame that cannot be fitted publishes ``success`` False and
    those pass-through fields only.

Properties:
    *   imagestreams: ([str]) input image streams.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, least_squares

from pytweezer.analysis.analysis_base import ImageAnalysis, run_analysis
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient

LOGGER = get_logger(__name__)

FIELDS = (
    "x0",
    "y0",
    "sigma_x",
    "sigma_y",
    "sigma_major",
    "sigma_minor",
    "angle",
    "amplitude",
    "offset",
    "integral",
)
SPECIES_MASS_AMU = {"CaF": 59.0, "Rb": 86.909}
ERROR_FIELDS = ("x0", "y0", "sigma_x", "sigma_y", "sigma_major", "sigma_minor")

_SEED_BINS = 256
_QUANTILE_SAMPLES = 16384
_MIN_LEVEL_BINS = 32  # fewer and the noise floor is too poorly estimated
_SEED_MIN_SIGNIFICANCE = 7.0
_WINDOW_SIGMAS = 4.0
_MIN_HALF_WINDOW_PX = 8
_BINS_PER_SIGMA_MINOR = 3
_MAX_WINDOW_PASSES = 3
_SEED_MOMENT_PASSES = 4
_MAX_NFEV = 200
_MIN_SIGMA_PX = 1.0
_MIN_MATCHED_FILTER_SNR = 6.0


@dataclass(frozen=True)
class CloudFit:
    """Result of :func:`fit_cloud`; see the module docstring for the coordinates."""

    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    sigma_major: float
    sigma_minor: float
    angle: float
    amplitude: float
    offset: float
    integral: float
    units: str
    success: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProfileFit:
    """Result of :func:`fit_profile`: ``offset + amplitude * exp(-(x - center)**2 / 2 sigma**2)``."""

    center: float
    sigma: float
    amplitude: float
    offset: float
    success: bool


def frame_geometry(
    head: dict,
) -> tuple[tuple[float, float] | None, tuple[float, float]]:
    """Return ``(resolution or None, origin)`` from a stream header.

    The origin is ``_offset`` when present, else zero. The resolution is
    ``_imgresolution`` when present, else ``None`` (pixels).
    """
    resolution = head.get("_imgresolution")
    origin = head.get("_offset")
    return (
        None if resolution is None else (float(resolution[0]), float(resolution[1])),
        (0.0, 0.0) if origin is None else (float(origin[0]), float(origin[1])),
    )


# Pixel-space parameters throughout: (amplitude, x0, y0, sigma_a, sigma_b, theta, offset)
# in full-frame coordinates where pixel c sits at c + 0.5; sigma_a is along theta.


def _model(p, x, y):
    amplitude, x0, y0, sigma_a, sigma_b, theta, offset = p
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = x - x0, y - y0
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return offset + amplitude * np.exp(-0.5 * ((u / sigma_a) ** 2 + (v / sigma_b) ** 2))


def _residuals(p, x, y, data):
    return _model(p, x, y) - data


def _jacobian(p, x, y, data):
    amplitude, x0, y0, sigma_a, sigma_b, theta, _ = p
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = x - x0, y - y0
    u = c * dx + s * dy
    v = -s * dx + c * dy
    g = np.exp(-0.5 * ((u / sigma_a) ** 2 + (v / sigma_b) ** 2))
    ag = amplitude * g
    jac = np.empty((x.size, 7))
    jac[:, 0] = g
    jac[:, 1] = ag * (u * c / sigma_a**2 - v * s / sigma_b**2)
    jac[:, 2] = ag * (u * s / sigma_a**2 + v * c / sigma_b**2)
    jac[:, 3] = ag * u**2 / sigma_a**3
    jac[:, 4] = ag * v**2 / sigma_b**3
    jac[:, 5] = -ag * u * v * (1 / sigma_a**2 - 1 / sigma_b**2)
    jac[:, 6] = 1.0
    return jac


def _block_mean(block: np.ndarray, factor: int) -> np.ndarray:
    rows, cols = block.shape[0] // factor, block.shape[1] // factor
    block = block[: rows * factor, : cols * factor]
    return block.reshape(rows, factor, cols, factor).sum(axis=3, dtype=np.float64).sum(
        axis=1
    ) / (factor * factor)


def _baseline_and_noise(image: np.ndarray) -> tuple[float, float]:
    """Mean and width of the noise floor from the lower tail, which a cloud does not reach."""
    flat = image.ravel()
    p10, p25 = np.percentile(flat[:: max(1, flat.size // _QUANTILE_SAMPLES)], (10, 25))
    noise = float(p25 - p10) / 0.607  # lower-quartile spread of Gaussian noise
    noise = max(noise, 1e-9 * max(float(np.abs(image).max()), 1.0))
    return float(p10) + 1.2816 * noise, noise  # p10 sits 1.28 sigma below the mean


def _detect(coarse: np.ndarray):
    """Find the most significant blob over a ladder of scales.

    Each level of a block-averaged pyramid is tested raw and smoothed by one
    bin, so every scale is within about 1.4x of a matched filter. Returns
    ``(z, row, col, bins, smoothed, baseline, noise, sigma_bins)`` for the
    strongest response, where ``bins`` is the level's bin size in coarse bins and
    ``sigma_bins`` the filter's width in coarse bins.
    """
    best = None
    level, bins = coarse, 1
    while min(level.shape) >= _MIN_LEVEL_BINS:
        for smoothing in (0.0, 1.0):
            smoothed = (
                gaussian_filter(level, smoothing, mode="nearest")
                if smoothing
                else level
            )
            baseline, noise = _baseline_and_noise(smoothed)
            z = (smoothed - baseline) / noise
            row, col = np.unravel_index(np.argmax(z), z.shape)
            if best is None or z[row, col] > best[0]:
                sigma_bins = bins * math.sqrt(smoothing**2 + 1 / 12)
                best = (
                    float(z[row, col]),
                    row,
                    col,
                    bins,
                    smoothed,
                    baseline,
                    noise,
                    sigma_bins,
                )
        level, bins = _block_mean(level, 2), bins * 2
    return best


def _moments_seed(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Seed from the strongest blob of a coarsely binned copy of the frame.

    Detection is by matched-filter-like significance over several scales, so a
    faint, broad cloud is found as readily as a bright, small one; the second
    moments of the blob then give the widths and the angle. Fails when nothing
    stands ``_SEED_MIN_SIGNIFICANCE`` noise widths above the floor.
    """
    rows, cols = image.shape
    factor = max(1, min(math.ceil(max(rows, cols) / _SEED_BINS), rows, cols))
    coarse = _block_mean(image, factor)
    baseline, _ = _baseline_and_noise(coarse)
    detected = _detect(coarse)
    if detected is None:
        return np.array([0.0, cols / 2, rows / 2, 0.0, 0.0, 0.0, baseline]), False
    z_max, row, col, bins, smoothed, level_baseline, noise, sigma_bins = detected
    step = bins * factor  # pixels per bin of the detecting level
    centre_x, centre_y = (col + 0.5) * step, (row + 0.5) * step
    if z_max < _SEED_MIN_SIGNIFICANCE:
        return np.array([0.0, centre_x, centre_y, 0.0, 0.0, 0.0, baseline]), False

    filter_variance = (sigma_bins * factor) ** 2  # the detecting filter widens the blob
    sigma_px = max(sigma_bins * factor, 1.0)
    x0, y0 = centre_x, centre_y
    for _ in range(_SEED_MOMENT_PASSES):
        # The detecting scale can be far narrower than the cloud: widen the
        # moment window with the width found until it stops growing.
        reach = math.ceil(3 * sigma_px / step) + 1
        r0, r1 = (
            max(0, int(y0 // step) - reach),
            min(smoothed.shape[0], int(y0 // step) + reach + 1),
        )
        c0, c1 = (
            max(0, int(x0 // step) - reach),
            min(smoothed.shape[1], int(x0 // step) + reach + 1),
        )
        signal = smoothed[r0:r1, c0:c1] - level_baseline
        weights = np.where(signal > 2 * noise, signal, 0.0)
        total = weights.sum()
        if total <= 0:
            break
        xs = (np.arange(c0, c1) + 0.5) * step
        ys = (np.arange(r0, r1) + 0.5) * step
        wx, wy = weights.sum(axis=0), weights.sum(axis=1)
        x0, y0 = float(wx @ xs / total), float(wy @ ys / total)
        xc, yc = xs - x0, ys - y0
        vxx = float(wx @ xc**2 / total) - filter_variance
        vyy = float(wy @ yc**2 / total) - filter_variance
        vxy = float(yc @ weights @ xc / total)
        mean, radius = 0.5 * (vxx + vyy), math.hypot(0.5 * (vxx - vyy), vxy)
        widest = math.sqrt(max(mean + radius, 1.0))
        grown = widest > 1.2 * sigma_px
        sigma_px = max(sigma_px, widest)
        if not grown:
            break
    theta = 0.5 * math.atan2(2 * vxy, vxx - vyy)
    sigma_a = math.sqrt(max(mean + radius, 1.0))
    sigma_b = math.sqrt(max(mean - radius, 1.0))
    amplitude = float((smoothed[r0:r1, c0:c1] - level_baseline).max())
    return np.array([amplitude, x0, y0, sigma_a, sigma_b, theta, baseline]), True


def _window(shape, p):
    """Return ``(x_start, x_stop, y_start, y_stop, bin)`` for the fit window about ``p``."""
    sigma_major, sigma_minor = max(abs(p[3]), abs(p[4])), min(abs(p[3]), abs(p[4]))
    half = max(_WINDOW_SIGMAS * sigma_major, _MIN_HALF_WINDOW_PX)
    rows, cols = shape
    x_start, x_stop = max(0, math.floor(p[1] - half)), min(cols, math.ceil(p[1] + half))
    y_start, y_stop = max(0, math.floor(p[2] - half)), min(rows, math.ceil(p[2] + half))
    if x_stop <= x_start or y_stop <= y_start:
        return None
    factor = int(sigma_minor // _BINS_PER_SIGMA_MINOR)
    factor = max(1, min(factor, (x_stop - x_start) // 4, (y_stop - y_start) // 4))
    return x_start, x_stop, y_start, y_stop, factor


def _fit_window(image, p, window):
    """Least-squares fit on the binned window; return ``(params, ok)``."""
    x_start, x_stop, y_start, y_stop, factor = window
    binned = _block_mean(image[y_start:y_stop, x_start:x_stop], factor)
    ny, nx = binned.shape
    xs = (np.arange(nx) + 0.5) * factor
    ys = (np.arange(ny) + 0.5) * factor
    x, y = np.meshgrid(xs, ys)
    x, y, data = x.ravel(), y.ravel(), binned.ravel()

    start = np.array(p, dtype=float)
    start[1] -= x_start
    start[2] -= y_start
    try:
        solution = least_squares(
            _residuals,
            start,
            jac=_jacobian,
            method="lm",
            x_scale="jac",
            max_nfev=_MAX_NFEV,
            args=(x, y, data),
        )
    except (ValueError, FloatingPointError):
        return p, False

    q = solution.x.copy()
    q[1] += x_start
    q[2] += y_start
    q[3], q[4] = abs(q[3]), abs(q[4])
    if solution.status <= 0 or not np.all(np.isfinite(q)):
        return p, False

    rms = math.sqrt(float(np.mean(solution.fun**2)))
    matched_filter_snr = (
        q[0] * math.sqrt(math.pi * q[3] * q[4]) / (factor * rms)
        if rms > 0
        else math.inf
    )
    rows, cols = image.shape
    plausible = (
        q[0] > 0
        and min(q[3], q[4]) >= _MIN_SIGMA_PX
        and max(q[3], q[4]) <= max(rows, cols)
        and 0 <= q[1] <= cols
        and 0 <= q[2] <= rows
        and matched_filter_snr >= _MIN_MATCHED_FILTER_SNR
    )
    return q, plausible


def _fit_pixels(
    image: np.ndarray, seed: tuple[np.ndarray, bool] | None = None
) -> tuple[np.ndarray, bool]:
    """Fit the cloud in pixel space, from ``seed`` or the moments seed.

    An image that is not 2-D, is empty or too small to search, or holds a NaN or
    infinity fails (``ok`` False) rather than raising.
    """
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        return np.zeros(7), False
    p, ok = _moments_seed(image) if seed is None else seed
    if not ok:
        return p, False
    seed_params = p
    for _ in range(_MAX_WINDOW_PASSES):
        window = _window(image.shape, p)
        if window is None:
            return seed_params, False
        p, ok = _fit_window(image, p, window)
        if not ok:
            return seed_params, False
        # A seed with a biased width gives a wrong window: refit on the fitted cloud's.
        refined = _window(image.shape, p)
        if (
            refined is not None
            and refined[4] == window[4]
            and abs((refined[1] - refined[0]) / (window[1] - window[0]) - 1) < 0.2
        ):
            break
    return p, True


def _cloud_fit(
    p: np.ndarray,
    success: bool,
    resolution: tuple[float, float] | None,
    origin: tuple[float, float],
) -> CloudFit:
    amplitude, x0, y0, sigma_a, sigma_b, theta, offset = (float(v) for v in p)
    res_x, res_y = resolution if resolution is not None else (1.0, 1.0)
    c, s = math.cos(theta), math.sin(theta)
    var_a, var_b = sigma_a**2, sigma_b**2
    sxx = (var_a * c**2 + var_b * s**2) * res_x**2
    syy = (var_a * s**2 + var_b * c**2) * res_y**2
    sxy = (var_a - var_b) * c * s * res_x * res_y
    mean, radius = 0.5 * (sxx + syy), math.hypot(0.5 * (sxx - syy), sxy)
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    if angle <= -math.pi / 2:
        angle += math.pi
    return CloudFit(
        x0=(origin[0] + x0) * res_x,
        y0=(origin[1] + y0) * res_y,
        sigma_x=math.sqrt(sxx),
        sigma_y=math.sqrt(syy),
        sigma_major=math.sqrt(mean + radius),
        sigma_minor=math.sqrt(max(mean - radius, 0.0)),
        angle=angle,
        amplitude=amplitude,
        offset=offset,
        integral=2 * math.pi * amplitude * abs(sigma_a) * abs(sigma_b),
        units="px" if resolution is None else "m",
        success=bool(success),
    )


def fit_cloud(
    image: np.ndarray,
    resolution: tuple[float, float] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> CloudFit:
    """Fit a rotated 2-D Gaussian with offset to ``image``.

    Args:
        image: 2-D frame, rows along y.
        resolution: ``(res_x, res_y)`` per binned pixel, or ``None`` for pixels.
        origin: ``(origin_x, origin_y)`` of the frame in binned pixels.

    Returns:
        The fit. When no cloud is found or the fit fails, ``success`` is False
        and the other fields hold the moments seed.
    """
    p, ok = _fit_pixels(np.asarray(image))
    return _cloud_fit(p, ok, resolution, origin)


def fit_profile(x: np.ndarray, y: np.ndarray) -> ProfileFit:
    """Fit ``offset + amplitude * exp(-(x - center)**2 / 2 sigma**2)`` to a 1-D trace."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    baseline = float(np.percentile(y, 10))
    weights = np.clip(y - baseline, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return ProfileFit(float(x[int(np.argmax(y))]), 0.0, 0.0, baseline, False)

    center = float(weights @ x / total)
    spacing = float(np.ptp(x)) / max(x.size - 1, 1)
    sigma = max(math.sqrt(float(weights @ (x - center) ** 2 / total)), spacing)
    amplitude = float(y.max()) - baseline
    seed = ProfileFit(center, sigma, amplitude, baseline, False)
    try:
        popt, _ = curve_fit(
            _profile, x, y, p0=[amplitude, center, sigma, baseline], maxfev=2000
        )
    except (RuntimeError, ValueError):
        return seed
    return ProfileFit(
        float(popt[1]), abs(float(popt[2])), float(popt[0]), float(popt[3]), True
    )


def _profile(x, amplitude, center, sigma, offset):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def profile_curve(x: np.ndarray, fit: ProfileFit) -> np.ndarray:
    """Evaluate a :class:`ProfileFit` at ``x``."""
    return _profile(
        np.asarray(x, dtype=np.float64),
        fit.amplitude,
        fit.center,
        fit.sigma,
        fit.offset,
    )


def contour_ellipses(fit: CloudFit, n: int = 8, points: int = 100) -> list[np.ndarray]:
    """Closed outlines of the fitted cloud at ``n`` levels, in the fit's coordinates.

    Level ``k`` (``k = 1..n``) is at ``k / (n + 1)`` of the amplitude above the
    offset, an ellipse of radius ``sqrt(-2 ln f)`` sigma. Each outline is a
    ``(2, points)`` array ``[x, y]``.
    """
    phase = np.linspace(0.0, 2 * np.pi, points)
    c, s = math.cos(fit.angle), math.sin(fit.angle)
    along = fit.sigma_major * np.cos(phase)
    across = fit.sigma_minor * np.sin(phase)
    outlines = []
    for k in range(1, n + 1):
        radius = math.sqrt(-2 * math.log(k / (n + 1)))
        outlines.append(
            np.array(
                [
                    fit.x0 + radius * (c * along - s * across),
                    fit.y0 + radius * (s * along + c * across),
                ]
            )
        )
    return outlines


def fit_frames(
    frames: np.ndarray,
    background: np.ndarray | None,
    resolution: tuple[float, float] | None,
    origin: tuple[float, float],
) -> tuple[dict[str, float], np.ndarray]:
    """Fit the mean background-subtracted image of ``frames``, with errors.

    Args:
        frames: ``(n, rows, cols)`` stack, or a single 2-D frame.
        background: frame to subtract from the mean, or ``None``.
        resolution: as :func:`fit_cloud`.
        origin: as :func:`fit_cloud`.

    Returns:
        ``({field: value}, mean_image)``. The dict has every name in
        :data:`FIELDS` of the mean image's fit (NaN where it failed) and
        ``<field>_error`` for each of :data:`ERROR_FIELDS`: the standard error
        of the mean over the frames that fitted, each fit seeded from the mean
        fit, NaN with fewer than two. ``mean_image`` is float32.
    """
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    mean_image = frames.mean(axis=0, dtype=np.float32)
    if background is not None:
        mean_image -= np.asarray(background, dtype=np.float32)

    p, ok = _fit_pixels(mean_image)
    mean_fit = _cloud_fit(p, ok, resolution, origin)
    values = {name: getattr(mean_fit, name) if ok else math.nan for name in FIELDS}

    per_frame = []
    if ok:
        for frame in frames:
            corrected = frame.astype(np.float32)
            if background is not None:
                corrected -= np.asarray(background, dtype=np.float32)
            frame_p, frame_ok = _fit_pixels(corrected, (p, True))
            if frame_ok:
                per_frame.append(_cloud_fit(frame_p, True, resolution, origin))
    for name in ERROR_FIELDS:
        if len(per_frame) < 2:
            values[f"{name}_error"] = math.nan
        else:
            samples = np.array([getattr(fit, name) for fit in per_frame])
            values[f"{name}_error"] = float(
                samples.std(ddof=1) / math.sqrt(len(samples))
            )
    return values, mean_image


_PASS_THROUGH = ("rid", "index", "_imgresolution", "_offset")


class CloudFitAnalysis(ImageAnalysis):
    """Publish :class:`CloudFit` of every frame on the Data hub."""

    def __init__(self, name):
        super().__init__(name)
        self.dataq = DataClient(name.split("/")[-1])

    def process(self, head, data):
        if data is None or data.ndim != 2:
            return
        out = {key: head[key] for key in _PASS_THROUGH if key in head}
        resolution, origin = frame_geometry(head)
        try:
            out.update(fit_cloud(data, resolution, origin).as_dict())
        except Exception:
            LOGGER.warning("cloud fit failed on a frame", exc_info=True)
            out["success"] = False
        self.dataq.send(out, None)
        return


if __name__ == "__main__":
    run_analysis(CloudFitAnalysis)
