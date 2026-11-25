import numpy as np
import pytest

from pytweezer.analyses.default.trap_frequency import trap_frequency
from pytweezer.experiment import ListAxis, Scan
from pytweezer.experiment.archive import ResultsArchive
from pytweezer.experiments.trap_frequency import TrapFrequencyScan
from pytweezer.GUI.default_analyses import ANALYSES
from pytweezer.GUI.default_analyses.trap_frequency import TrapFrequencyView
from tests.helpers import until, write_run

RADIAL_HZ = 47.5e3
FREQUENCIES_HZ = np.linspace(10e3, 300e3, 59)
SITES = 100
BASELINE = 0.92
#: (centre, depth, width) in Hz: 2 ν_r, ν_r, 2 ν_z, ν_z and a weak dip above 2 ν_r.
DIPS = (
    (2 * RADIAL_HZ, 0.55, 6e3),
    (RADIAL_HZ, 0.35, 4e3),
    (30e3, 0.3, 3e3),
    (15e3, 0.15, 2e3),
    (190e3, 0.12, 5e3),
)


def survival(frequency_hz, dips=DIPS):
    return BASELINE - sum(
        depth * np.exp(-((frequency_hz - centre) ** 2) / (2 * width**2))
        for centre, depth, width in dips
    )


# The helpers' Sweep scans "x"; the frequency is in Hz as modulation_frequency would be.
def spectrum_file(tmp_path, occupancy=True, dips=DIPS, repetitions=4, seed=0):
    """A trap frequency scan with each site loaded half the time and surviving as :func:`survival`."""
    rng = np.random.default_rng(seed)

    def record(point):
        first = rng.random(SITES) < 0.5
        second = first & (rng.random(SITES) < survival(point.values["x"], dips))
        if occupancy:
            return {"first_occupied": first, "second_occupied": second}
        return {"survival": second[first].mean()}

    scan = Scan(
        axes=[ListAxis(argument="x", values=list(FREQUENCIES_HZ))],
        repetitions=repetitions,
    )
    write_run(tmp_path / "trap.h5", scan=scan, record=record)
    return ResultsArchive(tmp_path).load("trap.h5")


@pytest.mark.parametrize("occupancy", [True, False])
def test_the_radial_frequency_is_half_the_highest_pronounced_dip(tmp_path, occupancy):
    fit = trap_frequency(spectrum_file(tmp_path, occupancy), "x")

    assert fit.success, fit.message
    assert fit.radial_dip.joint
    assert fit.radial_hz == pytest.approx(RADIAL_HZ, rel=0.02)
    assert abs(fit.radial_hz - RADIAL_HZ) < 4 * fit.radial_err_hz
    assert len(fit.dips) >= 3


def test_every_dip_is_found_and_fitted_where_wide_enough(tmp_path):
    fit = trap_frequency(spectrum_file(tmp_path), "x")

    expected = np.array([15e3, 30e3, RADIAL_HZ, 2 * RADIAL_HZ])
    centres = {
        expected[np.argmin(abs(expected - dip.centre_hz))]: dip for dip in fit.dips
    }
    assert set(centres) == set(expected)
    # 15 and 30 kHz are three points apart: too few to fit either alone.
    assert centres[30e3].curve is None
    assert centres[RADIAL_HZ].centre_hz == pytest.approx(RADIAL_HZ, rel=0.02)
    assert centres[RADIAL_HZ].curve is not None
    assert fit.radial_dip not in fit.other_dips
    assert len(fit.other_dips) == len(fit.dips) - 1


def test_the_joint_fit_shares_one_baseline_across_every_dip(tmp_path):
    fit = trap_frequency(spectrum_file(tmp_path), "x")

    baseline, *dips = fit.joint_curve
    depths, centres, _ = np.reshape(dips, (-1, 3)).T
    assert baseline == pytest.approx(BASELINE, abs=0.05)
    # Including the 15 and 30 kHz dips too close together to fit one by one.
    assert centres == pytest.approx([15e3, 30e3, RADIAL_HZ, 2 * RADIAL_HZ], rel=0.15)
    # The 5 kHz step barely samples the narrow low-frequency dips' depths.
    assert depths[2:] == pytest.approx([0.35, 0.55], abs=0.05)


def test_a_dip_above_the_radial_one_is_taken_once_deep_enough(tmp_path):
    dips = (*DIPS[:-1], (190e3, 0.25, 5e3))
    measurement = spectrum_file(tmp_path, dips=dips)

    assert trap_frequency(measurement, "x").radial_hz == pytest.approx(
        RADIAL_HZ, rel=0.02
    )
    fit = trap_frequency(measurement, "x", min_relative_depth=0.1)
    assert fit.radial_hz == pytest.approx(95e3, rel=0.03)


def test_a_flat_spectrum_has_no_trap_frequency(tmp_path):
    fit = trap_frequency(spectrum_file(tmp_path, dips=()), "x")

    assert not fit.success
    assert np.isnan(fit.radial_hz)
    assert "No dip" in fit.message


def test_the_trap_frequency_scan_offers_its_analysis_first():
    assert TrapFrequencyScan.analyses[0] == "trap_frequency"
    assert "trap_frequency" in ANALYSES


def test_the_view_shows_the_radial_frequency_and_lists_every_dip(qapp, tmp_path):
    measurement = spectrum_file(tmp_path)
    view = TrapFrequencyView()
    view.set_measurement(ResultsArchive(tmp_path), "trap.h5", measurement)
    until(lambda: view.readout.text(), qapp=qapp)

    assert view.readout.text().startswith("ν_r = 47.")
    assert len(view.readout.text().splitlines()) >= 5
    assert len(view.dip_items) >= 6
    assert len(view.points.getData()[0]) == len(FREQUENCIES_HZ)
    joint_khz, _ = view.joint_fit.getData()
    assert joint_khz[0] == pytest.approx(FREQUENCIES_HZ[0] / 1e3)
    assert joint_khz[-1] == pytest.approx(FREQUENCIES_HZ[-1] / 1e3)
