"""Score the Rb tweezer array over batches of shots, by loading and survival."""

from __future__ import annotations

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.experiment import Experiment
from pytweezer.experiment.arguments import Bool, Integer
from pytweezer.experiments.load_tweezers import LoadTweezers


class TweezerArrayMeasurement(LoadTweezers):
    """Score batches of shots at the accepted trap sites, by loading and survival.

    The trap sites, summing window, top-hat size and occupancy threshold are
    those accepted from :class:`~pytweezer.optimisers.find_trap_sites.FindTrapSites`; a measurement refuses to run
    without them, or when they were fitted to another array than the rig has
    on the SLM.
    """

    shots = Integer(20, min=1, group="Analysis", tooltip="shots per point")
    save_raw_images = Bool(
        False,
        group="Camera",
        tooltip="record every shot's camera frames as hamcam_images",
    )

    # Not LoadTweezers' point: each optimiser measures its own, and this keeps
    # the class out of the Experiments list.
    run_point = Experiment.run_point

    def prepare(self) -> None:
        super().prepare()
        if self.trap_sites is None:
            raise RuntimeError(self.no_trap_sites_reason)
        self.threshold = self.trap_threshold

    def take_batch(self, **sequence) -> tuple[list, list]:
        """``shots`` shots with script parameters ``sequence``; their first and second frames.

        With ``save_raw_images`` the frames are recorded as ``hamcam_images``,
        ``(shots, 2, height, width)`` per point.
        """
        stacks = np.array([self.take_shot(**sequence) for _ in range(self.shots)])
        if self.save_raw_images:
            self.record("hamcam_images", stacks)
        return list(stacks[:, 0]), list(stacks[:, 1])

    def score_batch(self, **sequence) -> dict[str, float]:
        """Loading, survival from the first image to the second, and fidelity, over ``shots`` shots.

        Also records whether every site was occupied in each shot's images, as
        ``first_occupied`` and ``second_occupied`` and, with
        ``save_tweezer_counts``, their raw pixel sums as ``first_counts`` and
        ``second_counts``.
        """
        firsts, seconds = self.take_batch(**sequence)
        if self.save_tweezer_counts:
            self.record("first_counts", self.site_counts(firsts), unit="counts")
            self.record("second_counts", self.site_counts(seconds), unit="counts")
        firsts = self.top_hat(firsts, self.trap_feature_size)
        seconds = self.top_hat(seconds, self.trap_feature_size)
        # Loading is judged against a threshold refitted on this batch (the
        # accepted one only if the fit fails), survival against the accepted one.
        # TODO: check loading is judged against a per-batch refit while survival
        # uses the calibration threshold (as on rearrangement-optimisation-RnD)
        first_photons, loading, _, fidelity = an.get_array_loading_statistics(
            firsts,
            self.trap_sites,
            threshold=self.threshold,
            window_size=self.trap_window_size,
            binning=60,
            show_histogram=False,
            threshold_detection=True,
            verbose=False,
        )
        survival = an.extract_survival_probability(
            firsts,
            seconds,
            self.trap_sites,
            threshold=self.threshold,
            window_size=self.trap_window_size,
        )
        second_photons = photons_at_sites(
            seconds, self.trap_sites, self.trap_window_size
        )
        self.record("first_occupied", np.asarray(first_photons) > self.threshold)
        self.record("second_occupied", second_photons > self.threshold)
        return {
            "loading": float(np.mean(loading)),
            "survival": float(survival),
            "fidelity": float(fidelity),
        }
