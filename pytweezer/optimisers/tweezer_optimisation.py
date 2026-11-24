"""The base of the Bayesian optimisers that score the Rb tweezer array over a batch of shots."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pytweezer.experiment.arguments import Bool, Integer
from pytweezer.experiment.optimisers import BayesianOptimisation, Dimension

#: The molasses time searched by the optimisers that minimise it.
#: FIXME: placeholder limits (1, 20), to check against the script
MOLASSES_TIME = Dimension(
    "molasses_time", save_as="tMolassesDuration", low=1, high=20, whole=True
)


class TweezerOptimisation(BayesianOptimisation):
    """A Bayesian optimisation whose evaluations score a batch of tweezer shots, mixed with :class:`~pytweezer.experiments.load_tweezers.LoadTweezers`.

    An optimiser subclasses ``(TweezerOptimisation, LoadTweezers)``. The
    trap sites, summing window, top-hat size and occupancy threshold are
    those accepted from :class:`~pytweezer.optimisers.find_trap_sites.FindTrapSites`;
    an optimisation refuses to run without them.

    :meth:`measure` maps each setting to its dimension's ``save_as`` script
    parameter and returns :meth:`score_batch` at them; an optimiser overrides
    it only when a setting maps to something else. ``drop`` names the argument
    holding the drop duration between the two images, or ``None`` for no drop.
    """

    drop: ClassVar[str | None] = None

    shots = Integer(20, min=1, group="Analysis", tooltip="shots per evaluation")
    save_raw_images = Bool(
        False,
        group="Camera",
        tooltip="record every shot's camera frames as hamcam_images",
    )

    def prepare(self) -> None:
        super().prepare()
        if self.trap_sites is None:
            raise RuntimeError(self.no_trap_sites_reason)

    def drop_overrides(self) -> dict:
        """The script parameters that set the drop between the two images."""
        if self.drop is None:
            return {"BackgroundDrop": 0}
        return {"BackgroundDrop": 1, "tDropDuration": getattr(self, self.drop)}

    def measure(self, **settings: float | None) -> dict[str, float]:
        """Score the batch at ``settings``; ``None`` keeps a saved value."""
        mapped = {
            dimension.save_as: settings[dimension.setting]
            for dimension in self.dimensions
            if settings.get(dimension.setting) is not None
        }
        return self.score_batch(**self.drop_overrides(), **mapped)

    def take_batch(self, **overrides) -> tuple[list, list]:
        """``shots`` shots with script-parameter ``overrides``; their first and second frames.

        With ``save_raw_images`` the frames are recorded as ``hamcam_images``,
        ``(shots, 2, height, width)`` per point.
        """
        stacks = []
        for _ in range(self.shots):
            mark = self.hamcam.start_shot({"rid": self.rid})
            self.run_motmaster_sequence(**overrides)
            stacks.append(self.hamcam.wait_frames(mark, 2))
        stacks = np.array(stacks)
        if self.save_raw_images:
            self.record("hamcam_images", stacks)
        return list(stacks[:, 0]), list(stacks[:, 1])

    def score_batch(self, **overrides) -> dict[str, float]:
        """Loading, survival from the first image to the second, and fidelity, over ``shots`` shots.

        Also records whether every site was occupied in each shot's images, as
        ``first_occupied`` and ``second_occupied`` and, with
        ``save_tweezer_counts``, their raw pixel sums as ``first_counts`` and
        ``second_counts``.
        """
        firsts, seconds = self.take_batch(**overrides)
        if self.save_tweezer_counts:
            self.record("first_counts", self.trap_sites.counts(firsts), unit="counts")
            self.record("second_counts", self.trap_sites.counts(seconds), unit="counts")
        # Loading is judged against a threshold refitted on this batch (the
        # accepted one only if the fit fails), survival against the accepted one.
        # TODO: check loading is judged against a per-batch refit while survival
        # uses the calibration threshold (as on rearrangement-optimisation-RnD)
        loading = self.trap_sites.fit_loading(firsts)
        survival = self.trap_sites.survival(firsts, seconds)
        self.record("first_occupied", loading.photons > self.trap_sites.threshold)
        self.record("second_occupied", self.trap_sites.occupied(seconds))
        return {
            "loading": loading.loading,
            "survival": float(survival),
            "fidelity": loading.fidelity,
        }
