import numpy as np
import pytest
from scipy.constants import atomic_mass, k

from pytweezer.analysis.cloud_results import (
    cloud_temperature,
    guess_time_axis,
    tof_fit,
)
from pytweezer.experiment import ListAxis, Scan
from pytweezer.experiment.archive import ResultsArchive
from tests.helpers import write_run

RB_MASS_AMU = 86.909
TEMPERATURE_K = 20e-6
SIGMA0_M = 0.5e-3
TIMES_S = np.linspace(1e-3, 8e-3, 8)
TICKS = np.array([100.0, 200.0, 300.0, 500.0, 800.0])


def expansion(times, seed, noise=0.01, temperature=TEMPERATURE_K):
    """``(sigma, sigma_err)`` of a Rb cloud released from rest, ``noise`` fractional."""
    sigma = np.sqrt(
        SIGMA0_M**2 + k * temperature / (RB_MASS_AMU * atomic_mass) * times**2
    )
    measured = sigma * (1 + noise * np.random.default_rng(seed).normal(size=len(times)))
    return measured, noise * sigma


def test_tof_fit_recovers_temperature():
    sigma, error = expansion(TIMES_S, seed=1)

    fit = tof_fit(TIMES_S, sigma, error, RB_MASS_AMU)

    assert fit.success
    assert fit.temperature_K == pytest.approx(TEMPERATURE_K, rel=0.05)
    assert abs(fit.temperature_K - TEMPERATURE_K) < 2 * fit.temperature_err_K
    assert fit.sigma0 == pytest.approx(SIGMA0_M, rel=0.05)


def test_tof_fit_error_is_the_scatter_of_the_temperature():
    pulls = []
    for seed in range(300):
        sigma, error = expansion(TIMES_S, seed)
        fit = tof_fit(TIMES_S, sigma, error, RB_MASS_AMU)
        pulls.append((fit.temperature_K - TEMPERATURE_K) / fit.temperature_err_K)

    assert np.std(pulls) == pytest.approx(1.0, abs=0.2)
    assert abs(np.mean(pulls)) < 0.2


def test_tof_fit_without_usable_errors_is_unweighted_and_scales_the_error_to_the_scatter():
    sigma, _ = expansion(TIMES_S, seed=2, noise=0.03)
    no_errors = np.full(len(sigma), np.nan)

    unweighted = tof_fit(TIMES_S, sigma, no_errors, RB_MASS_AMU)
    one_unknown = tof_fit(TIMES_S, sigma, np.r_[np.nan, 0.03 * sigma[1:]], RB_MASS_AMU)

    assert unweighted.success
    assert unweighted.temperature_K == pytest.approx(TEMPERATURE_K, rel=0.2)
    assert 0 < unweighted.temperature_err_K < 0.3 * TEMPERATURE_K
    assert one_unknown.temperature_K == unweighted.temperature_K


def test_stated_errors_are_never_trusted_below_the_scatter():
    sigma, error = expansion(TIMES_S, seed=3)

    honest = tof_fit(TIMES_S, sigma, error, RB_MASS_AMU)
    over = tof_fit(TIMES_S, sigma, 10 * error, RB_MASS_AMU)
    under = tof_fit(TIMES_S, sigma, error / 100, RB_MASS_AMU)

    assert over.temperature_err_K > honest.temperature_err_K
    assert under.temperature_err_K == pytest.approx(honest.temperature_err_K, rel=0.5)


@pytest.mark.parametrize("times", [[1e-3, 2e-3], [1e-3, 1e-3, 2e-3, 2e-3]])
def test_too_few_times_is_refused(times):
    times = np.array(times)
    sigma, error = expansion(times, seed=0)

    fit = tof_fit(times, sigma, error, RB_MASS_AMU)

    assert not fit.success
    assert "three" in fit.message


def test_a_cloud_that_does_not_expand_has_no_temperature():
    sigma = SIGMA0_M * np.array([1.0, 1.0, 1.0, 0.9, 0.8])
    times = np.linspace(1e-3, 5e-3, 5)

    fit = tof_fit(times, sigma, 0.01 * sigma, RB_MASS_AMU)

    assert not fit.success
    assert np.isnan(fit.temperature_K)


@pytest.mark.parametrize(
    ("names", "guess"),
    [
        (["tFreeFallDuration", "x"], "tFreeFallDuration"),
        (["MOTCoilsCurrentValue", "FreeExpTime"], "FreeExpTime"),
        (["tDropDuration"], "tDropDuration"),
        (["freeexp_time"], "freeexp_time"),
        (["x", "MOTCoilsCurrentValue"], None),
    ],
)
def test_guess_time_axis(names, guess):
    assert guess_time_axis(names) == guess


def tof_file(tmp_path, unit="m", species=True):
    """Rb cloud released at 20 µK with ``ticks`` of 10 µs, falling with g along y."""
    seconds = TICKS * 1e-5
    sigma_x, error_x = expansion(seconds, seed=4)
    sigma_y, error_y = expansion(seconds, seed=5)
    drop = 0.5 * 9.81 * seconds**2

    def record(point):
        i = point.index
        return {
            "cam_x0": 1e-3,
            "cam_y0": 2e-3 + drop[i],
            "cam_x0_error": 1e-6,
            "cam_y0_error": 1e-6,
            "cam_sigma_x": sigma_x[i],
            "cam_sigma_y": sigma_y[i],
            "cam_sigma_x_error": error_x[i],
            "cam_sigma_y_error": error_y[i],
        }

    scan = Scan(axes=[ListAxis(argument="x", values=list(TICKS))])
    names = [
        f"cam_{field}{suffix}"
        for field in ("x0", "y0", "sigma_x", "sigma_y")
        for suffix in ("", "_error")
    ]
    write_run(
        tmp_path / "tof.h5",
        scan=scan,
        record=record,
        constants={"cloud_mass_amu": (RB_MASS_AMU, "u")} if species else None,
        units={name: unit for name in names},
    )
    return ResultsArchive(tmp_path).load("tof.h5")


def test_cloud_temperature_converts_clock_ticks_to_seconds(tmp_path):
    measurement = tof_file(tmp_path)

    result = cloud_temperature(measurement, "cam", "x", 1e-5)

    for axis in ("sigma_x", "sigma_y"):
        fit = result["fits"][axis]
        assert fit.success
        assert abs(fit.temperature_K - TEMPERATURE_K) < 3 * fit.temperature_err_K
    assert result["gravity"].g == pytest.approx(9.81, rel=0.01)
    assert result["gravity"].g_err > 0
    t2, _, _ = result["sigma2"]["sigma_x"]
    np.testing.assert_allclose(t2, (TICKS * 1e-5) ** 2)


def test_sizes_in_pixels_give_no_temperature(tmp_path):
    measurement = tof_file(tmp_path, unit="px")

    result = cloud_temperature(measurement, "cam", "x", 1e-5)

    for fit in result["fits"].values():
        assert not fit.success
        assert "pixel" in fit.message
    assert not result["gravity"].success
