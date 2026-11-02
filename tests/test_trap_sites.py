import numpy as np
import pytest

from pytweezer.analysis.trap_sites import (
    default_geometry,
    fit_trap_sites,
    geometry_to_pixels,
    same_geometry,
    to_roi,
)

SHAPE = (128, 160)


def array_image(sites, dark=(), amplitude=300.0, seed=0):
    """Gaussian spots at ``sites`` ((y, x) pixels), except those indexed by ``dark``, on noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: SHAPE[0], : SHAPE[1]]
    image = rng.normal(0, 5, SHAPE)
    for index, (sy, sx) in enumerate(sites):
        if index not in dark:
            image += amplitude * np.exp(-((y - sy) ** 2 + (x - sx) ** 2) / (2 * 1.5**2))
    return image


def test_the_default_grid_is_row_major_from_the_top_left_of_the_image():
    sites = geometry_to_pixels(default_geometry(), 2.0, 0.0, 50.0, 50.0)

    assert len(sites) == 64
    np.testing.assert_allclose(sites[0], [15.0, 15.0])
    np.testing.assert_allclose(sites[1], [15.0, 25.0])
    np.testing.assert_allclose(sites[8], [25.0, 15.0])


def test_the_fit_recovers_a_known_transform_with_dark_sites():
    geometry = default_geometry()
    truth = geometry_to_pixels(geometry, 2.0, -1.0, 70.7, 60.3)
    image = array_image(truth, dark=range(0, 60, 3))

    fit = fit_trap_sites(image, geometry)

    assert np.abs(fit.sites - truth).max() < 0.5
    assert fit.scale == pytest.approx(2.0, rel=0.01)
    assert fit.angle_deg == pytest.approx(-1.0, abs=0.3)


def test_a_rig_geometry_keeps_its_own_order():
    geometry = default_geometry(3, 3)[[4, 0, 8, 2, 6, 1, 3, 5, 7]]
    truth = geometry_to_pixels(geometry, 2.07, -2.5, 80.0, 64.0)
    image = array_image(truth)

    fit = fit_trap_sites(image, geometry)

    assert np.abs(fit.sites - truth).max() < 0.5


@pytest.mark.parametrize(
    "image",
    [
        np.random.default_rng(3).normal(0, 5, SHAPE),
        array_image(geometry_to_pixels(default_geometry(5, 5), 3.0, 0.0, 80.0, 64.0)),
    ],
    ids=["blank", "scale far from the camera's"],
)
def test_an_image_without_the_array_at_the_expected_scale_is_refused(image):
    with pytest.raises(ValueError):
        fit_trap_sites(image, default_geometry(5, 5))


def test_to_roi_subtracts_the_roi_origin():
    sites = np.array([[230.0, 125.5], [240.0, 135.0]])
    np.testing.assert_allclose(to_roi(sites, 220, 110), [[10.0, 15.5], [20.0, 25.0]])


def test_same_geometry_allows_only_a_small_move_of_the_same_sites():
    saved = default_geometry(2, 2)
    x_um, y_um = saved.T
    assert same_geometry(saved, x_um + 0.05, y_um)
    assert not same_geometry(saved, x_um + 0.2, y_um)
    assert not same_geometry(saved, x_um[:3], y_um[:3])
