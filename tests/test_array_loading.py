import numpy as np
import pytest

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import TrapSites, top_hat

SITES = {0: (10, 10), 1: (10, 30)}
WINDOW_SIZE = 5
FEATURE_SIZE = 7


class FakeParameters:
    def get_parameter(self, name):
        return {"conversion_factor": 0.7, "conversion_offset": 0.0}[name]


@pytest.fixture(autouse=True)
def known_photon_conversion(monkeypatch):
    monkeypatch.setattr(an, "ExpParameterManager", FakeParameters)


def frame_with_atoms(*sites, amplitude=100.0):
    frame = np.full((40, 40), 5.0)
    for site in sites:
        row, col = SITES[site]
        frame[row - 1 : row + 2, col - 1 : col + 2] += amplitude
    return frame


@pytest.fixture
def trap_sites():
    return TrapSites(
        positions=SITES,
        window_size=WINDOW_SIZE,
        feature_size=FEATURE_SIZE,
        threshold=100.0,
    )


@pytest.fixture
def frames():
    return [frame_with_atoms(0, 1), frame_with_atoms(0)]


def test_occupied_marks_the_sites_with_an_atom_in_each_frame(trap_sites, frames):
    assert trap_sites.occupied(frames).tolist() == [[True, True], [True, False]]


def test_counts_are_raw_pixel_sums_at_each_site(trap_sites, frames):
    row, col = SITES[0]
    half = WINDOW_SIZE // 2
    box = frames[0][row - half : row + half + 1, col - half : col + half + 1]
    assert trap_sites.counts(frames)[0, 0] == box.sum()


def test_fit_loading_matches_the_notebook_statistics(trap_sites):
    rng = np.random.default_rng(1)
    shots = [
        frame_with_atoms(*(s for s in SITES if rng.random() < 0.5))
        + rng.normal(0, 1, (40, 40))
        for _ in range(60)
    ]
    photons, loading, threshold, fidelity = an.get_array_loading_statistics(
        top_hat(shots, FEATURE_SIZE),
        SITES,
        threshold=trap_sites.threshold,
        window_size=WINDOW_SIZE,
        binning=60,
        show_histogram=False,
        threshold_detection=True,
        verbose=False,
    )

    fitted = trap_sites.fit_loading(shots)

    np.testing.assert_array_equal(fitted.photons, photons)
    assert fitted.loading == float(np.mean(loading))
    assert fitted.threshold == float(threshold)
    assert fitted.fidelity == float(fidelity)
