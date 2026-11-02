"""Batches of shots of the Rb tweezer array at the saved sequence parameters."""

from __future__ import annotations

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.experiment import Experiment
from pytweezer.experiment.arguments import Bool
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.experiments.load_tweezers import LoadTweezers

_SAVED_UNDER_OWN_NAME = (
    "tMOTLoadRepVCO",
    "tMOTLoadCoolVCO",
    "tMOTLoadRepVVA",
    "tMOTLoadCoolVVA",
    "tMolCoolVCO1",
    "tMolRepVCO1",
    "tMolCoolVVA1",
    "tMolRepVVA1",
    "tMolCoolVCO2",
    "tMolRepVCO2",
    "tMolCoolVVA2",
    "tMolRepVVA2",
    "tMOTShimXccValue",
    "tMOTShimYccValue",
    "tMOTShimZccValue",
    "tMolassesShimXccValue",
    "tMolassesShimYccValue",
    "tMolassesShimZccValue",
)

#: Script parameter of RbTweezerBasic2026_2 -> the saved parameter it is set from.
SAVED_SEQUENCE_PARAMETERS = {
    **{name: name for name in _SAVED_UNDER_OWN_NAME},
    "tImgCoolVCO1": "tImgCoolVCO",
    "tImgCoolVCO2": "tImgCoolVCO",
    "tImgCoolVVA1": "tImgCoolVVA",
    "tImgCoolVVA2": "tImgCoolVVA",
}


class TweezerArrayShots(LoadTweezers):
    """Batches of shots of the tweezer array at the saved sequence parameters.

    Every shot runs the sequence at the saved parameters
    (:data:`SAVED_SEQUENCE_PARAMETERS`); a script parameter set in the form
    wins over its saved value.
    """

    save_raw_images = Bool(
        False,
        group="Camera",
        tooltip="record every shot's camera frames as first_images and second_images",
    )

    # Not LoadTweezers' point: each measurement defines its own, and this keeps
    # the bases out of the Experiments list.
    run_point = Experiment.run_point

    def prepare(self) -> None:
        super().prepare()
        saved = get_saved_parameters()
        missing = sorted(set(SAVED_SEQUENCE_PARAMETERS.values()) - set(saved))
        if missing:
            raise KeyError(f"no saved parameter {', '.join(map(repr, missing))}")
        set_in_form = self.motmaster_parameters()["rb"]
        self.saved_sequence = {
            parameter: saved[key]
            for parameter, key in SAVED_SEQUENCE_PARAMETERS.items()
            if parameter not in set_in_form
        }
        self.record("saved_sequence", self.saved_sequence)

    def _batch(self, n_shots: int, **sequence) -> tuple[list, list]:
        """``n_shots`` shots at the saved parameters plus ``sequence``; their first and second frames."""
        overrides = {
            f"rb.{parameter}": value
            for parameter, value in (self.saved_sequence | sequence).items()
        }
        frames = [self.shot(**overrides) for _ in range(n_shots)]
        firsts, seconds = zip(*frames, strict=True)
        if self.save_raw_images:
            self.record("first_images", np.array(firsts))
            self.record("second_images", np.array(seconds))
        return list(firsts), list(seconds)

    @staticmethod
    def _filtered(images: list, feature_size: int) -> list:
        return [
            an.morphological_tophat_high_pass(image, feature_size=feature_size)
            for image in images
        ]
