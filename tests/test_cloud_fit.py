"""Cloud fit: a rotated 2-D Gaussian recovered from noisy frames, in pixel and physical coordinates."""

import time

import numpy as np
import pytest

from pytweezer.analysis.cloud_fit import (
    contour_ellipses,
    fit_cloud,
    fit_frames,
    fit_profile,
)


def gaussian_image(
    shape, x0, y0, s_major, s_minor, angle, amplitude, offset, seed=0, noisy=True
):
    """Frame of the model at pixel centres (pixel ``c`` sits at ``x = c``), with Poisson noise."""
    rows, cols = np.indices(shape, dtype=np.float32)
    dx, dy = cols - x0, rows - y0
    u = np.cos(angle) * dx + np.sin(angle) * dy
    v = -np.sin(angle) * dx + np.cos(angle) * dy
    model = offset + amplitude * np.exp(
        -0.5 * ((u / s_major) ** 2 + (v / s_minor) ** 2)
    )
    if not noisy:
        return model
    return np.random.default_rng(seed).poisson(model).astype(np.float64)


def rotated_cloud(**overrides):
    return {
        "shape": (200, 300),
        "x0": 140.3,
        "y0": 80.7,
        "s_major": 12.0,
        "s_minor": 6.0,
        "angle": 0.5,
        "amplitude": 500.0,
        "offset": 100.0,
    } | overrides


def test_rotated_cloud_parameters_recovered():
    cloud = rotated_cloud()
    fit = fit_cloud(gaussian_image(**cloud))

    assert fit.success is True
    assert fit.units == "px"
    assert fit.x0 - 0.5 == pytest.approx(cloud["x0"], abs=0.1)
    assert fit.y0 - 0.5 == pytest.approx(cloud["y0"], abs=0.1)
    assert fit.sigma_major == pytest.approx(12.0, rel=0.02)
    assert fit.sigma_minor == pytest.approx(6.0, rel=0.02)
    assert fit.angle == pytest.approx(0.5, abs=0.02)
    assert fit.amplitude == pytest.approx(500.0, rel=0.02)
    assert fit.offset == pytest.approx(100.0, rel=0.02)
    assert fit.integral == pytest.approx(2 * np.pi * 500.0 * 12.0 * 6.0, rel=0.03)
    expected_sigma_x = np.hypot(12.0 * np.cos(0.5), 6.0 * np.sin(0.5))
    expected_sigma_y = np.hypot(12.0 * np.sin(0.5), 6.0 * np.cos(0.5))
    assert fit.sigma_x == pytest.approx(expected_sigma_x, rel=0.02)
    assert fit.sigma_y == pytest.approx(expected_sigma_y, rel=0.02)


def test_fit_reports_in_frame_coordinates():
    cloud = rotated_cloud(angle=0.0)
    resolution = (4e-6, 4e-6)
    fit = fit_cloud(gaussian_image(**cloud), resolution=resolution, origin=(10, 20))

    assert fit.units == "m"
    assert fit.x0 == pytest.approx((10 + cloud["x0"] + 0.5) * 4e-6, abs=0.1 * 4e-6)
    assert fit.y0 == pytest.approx((20 + cloud["y0"] + 0.5) * 4e-6, abs=0.1 * 4e-6)
    assert fit.sigma_x == pytest.approx(12.0 * 4e-6, rel=0.02)
    assert fit.sigma_y == pytest.approx(6.0 * 4e-6, rel=0.02)


def test_anisotropic_resolution_gives_angle_in_metres():
    # 14 px major axis at 45 deg and 5 px minor, pixels 1 um wide and 3 um tall
    cloud = rotated_cloud(angle=np.pi / 4, s_major=14.0, s_minor=5.0)
    fit = fit_cloud(gaussian_image(**cloud), resolution=(1e-6, 3e-6))

    sxx, syy = 110.5e-12, 110.5e-12 * 9
    sxy = 85.5 * 3e-12
    assert fit.sigma_x == pytest.approx(np.sqrt(sxx), rel=0.02)
    assert fit.sigma_y == pytest.approx(np.sqrt(syy), rel=0.02)
    assert fit.angle == pytest.approx(0.5 * np.arctan2(2 * sxy, sxx - syy), abs=0.03)
    assert fit.sigma_major >= fit.sigma_minor
    assert -np.pi / 2 < fit.angle <= np.pi / 2


def test_blank_frame_fails_quietly():
    frame = np.random.default_rng(1).poisson(100.0, (400, 400)).astype(np.float64)
    started = time.perf_counter()
    fit = fit_cloud(frame)

    assert fit.success is False
    assert time.perf_counter() - started < 1.0


def test_cloud_at_the_edge_is_fitted():
    cloud = rotated_cloud(x0=3.0, y0=100.0, s_major=10.0, s_minor=10.0, angle=0.0)
    fit = fit_cloud(gaussian_image(**cloud))

    assert fit.success is True
    assert fit.x0 - 0.5 == pytest.approx(3.0, abs=1.0)
    assert fit.y0 - 0.5 == pytest.approx(100.0, abs=1.0)


def test_large_frame_fits_through_window():
    cloud = rotated_cloud(
        shape=(2048, 2048), x0=900.2, y0=1100.6, s_major=8.0, s_minor=8.0, angle=0.0
    )
    fit = fit_cloud(gaussian_image(**cloud))

    assert fit.success is True
    assert fit.x0 - 0.5 == pytest.approx(900.2, abs=0.1)
    assert fit.y0 - 0.5 == pytest.approx(1100.6, abs=0.1)
    assert fit.sigma_x == pytest.approx(8.0, rel=0.02)
    assert fit.sigma_y == pytest.approx(8.0, rel=0.02)


def test_profile_fit_recovers_gaussian():
    x = np.linspace(0.0, 80.0, 161)
    clean = 20.0 + 300.0 * np.exp(-0.5 * ((x - 31.0) / 7.0) ** 2)
    y = np.random.default_rng(2).poisson(clean).astype(np.float64)
    fit = fit_profile(x, y)

    assert fit.success is True
    assert fit.sigma == pytest.approx(7.0, rel=0.02)
    assert fit.center == pytest.approx(31.0, abs=0.2)


def test_contour_ellipse_lies_on_its_level():
    cloud = rotated_cloud()
    fit = fit_cloud(gaussian_image(**cloud), resolution=(2e-6, 3e-6), origin=(5, 7))
    n = 5
    ellipses = contour_ellipses(fit, n=n, points=50)

    assert len(ellipses) == n
    c, s = np.cos(fit.angle), np.sin(fit.angle)
    for k, (x, y) in enumerate(ellipses, start=1):
        dx, dy = x - fit.x0, y - fit.y0
        u, v = c * dx + s * dy, -s * dx + c * dy
        height = fit.offset + fit.amplitude * np.exp(
            -0.5 * ((u / fit.sigma_major) ** 2 + (v / fit.sigma_minor) ** 2)
        )
        expected = fit.offset + fit.amplitude * k / (n + 1)
        assert height == pytest.approx(expected, rel=1e-6)


def test_frame_errors_are_sem_of_frame_fits():
    jitters = (0.9, 1.05, 1.0, 1.12, 0.95)
    frames = np.stack(
        [
            gaussian_image(
                **rotated_cloud(s_major=12.0 * j, s_minor=6.0 * j, seed=i + 10)
            )
            for i, j in enumerate(jitters)
        ]
    )
    individual = np.array([fit_cloud(frame).sigma_x for frame in frames])
    expected_sem = individual.std(ddof=1) / np.sqrt(len(frames))

    result, mean_image = fit_frames(frames, None, None, (0.0, 0.0))

    assert result["sigma_x_error"] == pytest.approx(expected_sem, rel=0.1)
    assert result["sigma_x"] == pytest.approx(individual.mean(), rel=0.05)
    assert mean_image.dtype == np.float32
    assert mean_image.shape == frames.shape[1:]
