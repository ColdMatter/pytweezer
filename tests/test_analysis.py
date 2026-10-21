"""Numerical analysis helpers in ``pytweezer/analysis/analysis.py``."""

import numpy as np
import pytest

from pytweezer.analysis import analysis as an


class FixedScorer:
    """Stands in for :class:`~pytweezer.analysis.analysis.SiteScorer`: returns given photons."""

    def __init__(self, photons):
        self.photons = np.asarray(photons, dtype=float)

    def score_stack(self, images):
        return self.photons


@pytest.mark.parametrize(
    "loading_statistics",
    [
        an.get_array_loading_statistics,
        lambda *args, **kwargs: an.get_array_loading_statistics_grid(
            *args, grid_shape=(1, 2), **kwargs
        ),
    ],
    ids=["sites", "grid"],
)
def test_well_separated_counts_at_a_fixed_threshold_have_fidelity_near_one(
    loading_statistics,
):
    photons = [[-1.0, 99.0], [1.0, 101.0]]
    *_, fidelity = loading_statistics(
        [None, None],
        {0: (0, 0), 1: (0, 1)},
        threshold=50,
        scorer=FixedScorer(photons),
        threshold_detection=False,
        show_histogram=False,
        verbose=False,
    )
    assert fidelity == pytest.approx(1.0, abs=1e-6)
