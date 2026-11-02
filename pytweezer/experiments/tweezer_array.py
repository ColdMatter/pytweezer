"""Find the Rb tweezer array's trap sites, score it over batches of shots, and optimise its imaging and molasses."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.trap_sites import default_geometry, fit_trap_sites
from pytweezer.experiment import Experiment
from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.experiment.optimisers import (
    SEARCH,
    BayesianOptimisation,
    Dimension,
    Fit,
)
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
        return list(firsts), list(seconds)

    @staticmethod
    def _filtered(images: list, feature_size: int) -> list:
        return [
            an.morphological_tophat_high_pass(image, feature_size=feature_size)
            for image in images
        ]


class FindTrapSites(Fit, TweezerArrayShots):
    """Find the trap sites by fitting the array's geometry to the atoms, and propose them.

    Takes ``calibration_shots`` with the tweezers off during the background
    drop, and fits the geometry of the array the rig has on the SLM (or, when
    it does not know, a ``default_rows`` x ``default_columns`` grid
    ``default_spacing`` apart) to their mean background-subtracted, top-hat
    filtered image, free in position, rotation and scale (see
    :mod:`pytweezer.analysis.trap_sites`). It then fits the occupancy
    threshold at those sites, and proposes both for every tweezer measurement
    to use.
    """

    analyses = ()
    proposal_units: ClassVar[dict[str, str]] = {
        "trap_sites": "px",
        "trap_threshold": "photons",
        "trap_window_size": "px",
        "trap_feature_size": "px",
        "trap_geometry": "µm",
    }
    proposal_group = "Tweezer array"

    calibration_shots = Integer(100, min=1, group=SEARCH)
    window_size = Integer(
        5, min=1, unit="px", group=SEARCH, tooltip="box summed per site"
    )
    feature_size = Integer(
        10, min=1, unit="px", group=SEARCH, tooltip="top-hat filter size"
    )
    initial_threshold = Number(
        100,
        unit="photons",
        group=SEARCH,
        tooltip="starting guess for the occupancy threshold fit",
    )
    default_rows = Integer(8, min=1, group="Default array")
    default_columns = Integer(8, min=1, group="Default array")
    default_spacing = Number(5e-6, unit="um", scale=1e-6, min=0, group="Default array")

    def saved_trap_sites(self):
        return None, ""

    def prepare(self) -> None:
        super().prepare()
        array = self.rig.get_array()
        if array:
            self.geometry = np.column_stack([array["x_um"], array["y_um"]])
        else:
            self.geometry = default_geometry(
                self.default_rows, self.default_columns, self.default_spacing * 1e6
            )
        self.record("sites_from", "rig" if array else "default")

    def fit(self) -> dict:
        firsts, backgrounds = self._batch(
            self.calibration_shots,
            BackgroundDrop=1,
            tTweezerSetVVA0=0.0,
            tTweezerSetVVA1=0.0,
        )
        average = an.tweezer_show_bg_subtracted(firsts, backgrounds, show=False)
        self.record("calibration_image", average)
        filtered = an.morphological_tophat_high_pass(
            np.asarray(average, dtype=float), feature_size=self.feature_size
        )
        found = fit_trap_sites(filtered, self.geometry, window_size=self.window_size)
        self.record("roi_sites", found.sites, unit="px")
        self.record("scale", found.scale, unit="px/µm")
        self.record("angle", found.angle_deg, unit="deg")
        whole_pixels = {
            index: (round(y), round(x)) for index, (y, x) in enumerate(found.sites)
        }
        _, _, threshold, fidelity = an.get_array_loading_statistics(
            self._filtered(firsts, self.feature_size),
            whole_pixels,
            threshold=self.initial_threshold,
            window_size=self.window_size,
            binning=60,
            show_histogram=False,
            threshold_detection=True,
            verbose=False,
        )
        self.record("threshold", float(threshold), unit="photons")
        self.record("calibration_fidelity", float(fidelity))
        return {
            "trap_sites": found.sites + [self.roi_y0, self.roi_x0],
            "trap_threshold": float(threshold),
            "trap_window_size": self.window_size,
            "trap_feature_size": self.feature_size,
            "trap_geometry": self.geometry,
        }


class TweezerArrayMeasurement(TweezerArrayShots):
    """Score batches of shots at the accepted trap sites, by loading and survival.

    The trap sites, summing window, top-hat size and occupancy threshold are
    those accepted from :class:`FindTrapSites`; a measurement refuses to run
    without them, or when they were fitted to another array than the rig has
    on the SLM.
    """

    shots = Integer(20, min=1, group="Analysis", tooltip="shots per point")

    def prepare(self) -> None:
        super().prepare()
        if self.trap_sites is None:
            raise RuntimeError(self.no_trap_sites_reason)
        self.threshold = self.trap_threshold
        self.record("threshold", self.threshold, unit="photons")

    def score_batch(self, **sequence) -> dict[str, float]:
        """Loading, survival from the first image to the second, and fidelity, over ``shots`` shots.

        Also records every shot's photons at every site of each image, as
        ``first_photons`` and ``second_photons``.
        """
        firsts, seconds = self._batch(self.shots, **sequence)
        firsts = self._filtered(firsts, self.trap_feature_size)
        seconds = self._filtered(seconds, self.trap_feature_size)
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
        self.record("first_photons", first_photons, unit="photons")
        self.record(
            "second_photons",
            photons_at_sites(seconds, self.trap_sites, self.trap_window_size),
            unit="photons",
        )
        return {
            "loading": float(np.mean(loading)),
            "survival": float(survival),
            "fidelity": float(fidelity),
        }


class TweezerImaging(TweezerArrayMeasurement):
    """Score the tweezer imaging: each point images the atoms twice with no drop between."""

    def measure(
        self, *, imaging_vco: float | None = None, imaging_vva: float | None = None
    ) -> dict[str, float]:
        """Score the imaging light; ``None`` keeps a saved value."""
        imaging = {}
        if imaging_vco is not None:
            imaging |= {"tImgCoolVCO1": imaging_vco, "tImgCoolVCO2": imaging_vco}
        if imaging_vva is not None:
            imaging |= {"tImgCoolVVA1": imaging_vva, "tImgCoolVVA2": imaging_vva}
        return self.score_batch(BackgroundDrop=0, **imaging)

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
        Dimension("imaging_vco", span=0.5, save_as="tImgCoolVCO"),
        Dimension("imaging_vva", span=0.2, save_as="tImgCoolVVA"),
    ]
    n_initial = Integer(10, min=1, group=SEARCH)
    fidelity_floor = Number(0.99, min=0, max=1, group=SEARCH)

    def objective_of(self, results: dict[str, float]) -> float:
        if results["fidelity"] > self.fidelity_floor:
            return results["survival"]
        return 0.8 * results["fidelity"]


class MolassesRecapture(TweezerArrayMeasurement):
    """Score the molasses by release and recapture: each point drops the atoms for ``drop_duration`` between images."""

    drop_duration = Integer(
        3, min=0, group="Analysis", tooltip="the script's tDropDuration"
    )

    def measure(
        self,
        *,
        shim_x: float | None = None,
        shim_y: float | None = None,
        shim_z: float | None = None,
    ) -> dict[str, float]:
        """Score the molasses shims; ``None`` keeps a saved value."""
        shims = {
            parameter: value
            for parameter, value in (
                ("tMolassesShimXccValue", shim_x),
                ("tMolassesShimYccValue", shim_y),
                ("tMolassesShimZccValue", shim_z),
            )
            if value is not None
        }
        return self.score_batch(
            BackgroundDrop=1, tDropDuration=self.drop_duration, **shims
        )

    def run_point(self) -> None:
        for name, value in self.measure().items():
            self.record(name, value)


class OptimiseMolassesShims(BayesianOptimisation, MolassesRecapture):
    """Find the molasses shim currents that recapture the most atoms after a drop.

    Below ``loading_floor`` the score is the loading; above it, the floor plus
    the recapture probability, so every setting that loads the array outscores
    every one that does not, and among those the coldest wins.
    """

    objective = "score"
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("shim_x", span=1.0, save_as="tMolassesShimXccValue"),
        Dimension("shim_y", span=1.0, save_as="tMolassesShimYccValue"),
        Dimension("shim_z", span=0.25, save_as="tMolassesShimZccValue"),
    ]
    n_calls = Integer(50, min=1, group=SEARCH)
    n_initial = Integer(10, min=1, group=SEARCH)
    loading_floor = Number(0.5, min=0, max=1, group=SEARCH)

    def objective_of(self, results: dict[str, float]) -> float:
        if results["loading"] < self.loading_floor:
            return results["loading"]
        return self.loading_floor + results["survival"]
