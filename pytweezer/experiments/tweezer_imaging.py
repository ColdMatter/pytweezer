"""Score the Rb tweezer imaging over a batch of shots, and optimise the imaging light."""

from typing import ClassVar

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.experiment.optimisers import BayesianOptimisation, Dimension
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


class TweezerImaging(LoadTweezers):
    """Score the tweezer imaging by loading, survival and detection fidelity.

    Every shot runs the sequence at the saved parameters
    (:data:`SAVED_SEQUENCE_PARAMETERS`); a script parameter set in the form
    wins over its saved value. :meth:`prepare` first takes
    ``calibration_shots`` with a background drop, finds the ``sites`` trap
    sites in their mean background-subtracted image and fits the occupancy
    threshold. Each point then takes ``shots`` shots imaging the atoms twice
    and records the mean loading probability, the survival from the first
    image to the second, and the detection fidelity of the first image.
    """

    shots = Integer(20, min=1, group="Analysis", tooltip="shots per point")
    calibration_shots = Integer(100, min=1, group="Analysis")
    sites = Integer(64, min=1, group="Analysis", tooltip="trap sites in the array")
    feature_size = Integer(
        10, min=1, unit="px", group="Analysis", tooltip="top-hat filter size"
    )
    window_size = Integer(
        5, min=1, unit="px", group="Analysis", tooltip="box summed per site"
    )
    initial_threshold = Number(
        100,
        unit="photons",
        group="Analysis",
        tooltip="starting guess for the calibration's occupancy threshold fit",
    )

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
        self._calibrate()

    def _calibrate(self) -> None:
        firsts, backgrounds = self._batch(
            self.calibration_shots,
            BackgroundDrop=1,
            tTweezerSetVVA0=0.0,
            tTweezerSetVVA1=0.0,
        )
        average = an.tweezer_show_bg_subtracted(firsts, backgrounds, show=False)
        positions, _ = an.detect_trap_sites(average, self.sites, detection_step=100)
        if len(positions) != self.sites:
            raise RuntimeError(
                f"found {len(positions)} trap sites in the calibration image, "
                f"not {self.sites}"
            )
        # Sites are summed by slicing the image, so they must be whole pixels.
        self.trap_sites = {
            index: (round(y), round(x)) for index, (y, x) in positions.items()
        }
        _, _, self.threshold, fidelity = an.get_array_loading_statistics(
            self._filtered(firsts),
            self.trap_sites,
            threshold=self.initial_threshold,
            window_size=self.window_size,
            binning=60,
            show_histogram=False,
            threshold_detection=True,
            verbose=False,
        )
        self.record("calibration_image", average)
        self.record("trap_sites", np.array(list(self.trap_sites.values())))
        self.record("threshold", float(self.threshold), unit="photons")
        self.record("calibration_fidelity", float(fidelity))

    def _batch(self, n_shots: int, **sequence) -> tuple[list, list]:
        """``n_shots`` shots at the saved parameters plus ``sequence``; their first and second frames."""
        overrides = {
            f"rb.{parameter}": value
            for parameter, value in (self.saved_sequence | sequence).items()
        }
        frames = [self.shot(**overrides) for _ in range(n_shots)]
        firsts, seconds = zip(*frames, strict=True)
        return list(firsts), list(seconds)

    def _filtered(self, images: list) -> list:
        return [
            an.morphological_tophat_high_pass(image, feature_size=self.feature_size)
            for image in images
        ]

    def measure(
        self, *, imaging_vco: float | None = None, imaging_vva: float | None = None
    ) -> dict[str, float]:
        """Loading, survival and fidelity over ``shots`` shots; ``None`` keeps a saved value."""
        imaging = {}
        if imaging_vco is not None:
            imaging |= {"tImgCoolVCO1": imaging_vco, "tImgCoolVCO2": imaging_vco}
        if imaging_vva is not None:
            imaging |= {"tImgCoolVVA1": imaging_vva, "tImgCoolVVA2": imaging_vva}
        firsts, seconds = self._batch(self.shots, BackgroundDrop=0, **imaging)
        firsts, seconds = self._filtered(firsts), self._filtered(seconds)
        # Refitting the threshold here only measures the fidelity; occupancy is
        # always judged against the calibration's threshold.
        _, loading, _, fidelity = an.get_array_loading_statistics(
            firsts,
            self.trap_sites,
            threshold=self.threshold,
            window_size=self.window_size,
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
            window_size=self.window_size,
        )
        return {
            "loading": float(np.mean(loading)),
            "survival": float(survival),
            "fidelity": float(fidelity),
        }

    def run_point(self) -> None:
        for name, value in self.measure().items():
            self.record(name, value)


class OptimiseTweezerImaging(BayesianOptimisation, TweezerImaging):
    """Find the imaging cooling-light VCO and VVA that keep the most atoms through imaging.

    The score is the survival while the fidelity exceeds ``fidelity_floor``,
    and 0.8 × fidelity below it, so a setting too dim to tell full sites from
    empty ones scores low however many atoms it appears to keep.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("imaging_vco", low=0.2, high=4.0, save_as="tImgCoolVCO"),
        Dimension("imaging_vva", low=0.4, high=2.0, save_as="tImgCoolVVA"),
    ]
    n_initial = Integer(10, min=1)
    fidelity_floor = Number(0.99, min=0, max=1, group="Analysis")

    def objective_of(self, results: dict[str, float]) -> float:
        if results["fidelity"] > self.fidelity_floor:
            return results["survival"]
        return 0.8 * results["fidelity"]
