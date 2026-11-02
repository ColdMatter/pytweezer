"""Load the Rb tweezers at the saved sequence parameters and image them twice on the HamCam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Bool, Device, Integer
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
)
from pytweezer.logging_utils import get_logger

if TYPE_CHECKING:
    # Imports cupy where it is installed, so only for the annotation.
    from pytweezer.coordinators.rearrangement import Rearrangement

logger = get_logger(__name__)

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

#: The saved parameters a trap-site fit is accepted into.
TRAP_SITE_KEYS = (
    "trap_sites",
    "trap_threshold",
    "trap_window_size",
    "trap_feature_size",
    "trap_geometry",
)


class LoadTweezers(MotMasterExperiment):
    """Run the Rb tweezer-loading sequence and image the tweezers twice per point.

    Every shot runs the sequence at the saved parameters
    (:data:`SAVED_SEQUENCE_PARAMETERS`); a script parameter set in the form
    wins over its saved value, and the values used are recorded as
    ``saved_sequence``. :meth:`take_shots` runs a batch of shots, for
    subclasses whose point is one.

    ``signal_counts`` is the summed first frame minus the summed second.
    :meth:`prepare` also loads the accepted trap sites (see
    :meth:`saved_trap_sites`) and records them in this ROI's pixels as
    ``trap_sites``, with the ``trap_window_size``, ``trap_feature_size`` and
    occupancy ``threshold`` they are scored with. With them, each point records
    whether each site was occupied in each frame (``first_occupied``,
    ``second_occupied``: its top-hat filtered photons above the threshold), the
    fraction of sites occupied in the first frame (``loading``) and,
    with ``save_tweezer_counts``, the sum of the camera's raw pixel values in
    its window (``first_counts``, ``second_counts``). The frames themselves
    (``first_image``, ``second_image``) are recorded only with
    ``save_raw_images``, which needs no sites.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2")
    analyses = ("tweezer_histogram", "tweezer_photon_histogram")

    camera: ImagEMX2Camera = Device("Rb HamCam")
    rig: Rearrangement = Device("Rb Rearrangement Rig")

    sensitivity = Integer(600, min=0, group="Camera", tooltip="EM gain sensitivity")
    roi_x0 = Integer(110, min=0, group="Camera")
    roi_y0 = Integer(220, min=0, group="Camera")
    roi_width = Integer(256, min=1, group="Camera")
    roi_height = Integer(256, min=1, group="Camera")
    save_raw_images = Bool(
        True,
        group="Camera",
        tooltip="record the camera frames; without them only each site's "
        "occupancy and counts are kept",
    )
    save_tweezer_counts = Bool(
        True,
        group="Camera",
        tooltip="record the sum of the raw pixel values in each site's window, "
        "beside its occupancy",
    )

    def prepare(self) -> None:
        super().prepare()
        try:
            saved = get_saved_parameters()
        except (ManagerUnavailable, ManagerError) as error:
            raise RuntimeError(
                f"the saved parameters could not be read: {error}"
            ) from error
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

        # The camera may round the ROI's corner; sites are placed by the one it used.
        self.roi_origin = tuple(
            self.camera.set_roi(
                self.roi_x0, self.roi_width, self.roi_y0, self.roi_height
            )
        )
        self.camera.set_external_exposure_mode()
        self.camera.enable_em_gain(True)
        self.camera.enable_direct_em_gain(True)
        self.camera.set_sensitivity(self.sensitivity)
        self.camera.setup_acquisition(2, mode="snap")

        self.trap_sites, self.no_trap_sites_reason = self.saved_trap_sites(saved)
        if self.trap_sites is None:
            if not self.no_trap_sites_reason:
                return
            if not self.save_raw_images:
                raise RuntimeError(
                    "without raw images only the site occupancy is recorded, "
                    f"which needs trap sites: {self.no_trap_sites_reason}"
                )
            logger.info("Recording no trap sites: %s", self.no_trap_sites_reason)
            return
        self.record("trap_sites", np.array(list(self.trap_sites.values())))
        self.record("trap_window_size", self.trap_window_size, unit="px")
        self.record("trap_feature_size", self.trap_feature_size, unit="px")
        self.record("threshold", self.trap_threshold, unit="photons")

    def saved_trap_sites(
        self, saved: dict
    ) -> tuple[dict[int, tuple[int, int]] | None, str]:
        """The accepted trap sites in ``saved`` as whole pixels of this ROI, by index; or ``None`` and why.

        An empty reason means the experiment wants none.

        Also sets ``trap_window_size``, ``trap_feature_size`` and
        ``trap_threshold`` from ``saved``. Sites fitted to another
        array than the rig has on the SLM are refused; if the rig does not know
        its array, or cannot be reached, the saved sites are used unchecked.
        """
        if any(key not in saved for key in TRAP_SITE_KEYS):
            return None, (
                "no trap sites are saved; run FindTrapSites and accept its proposal"
            )
        try:
            array = self.rig.get_array()
        except Exception as error:  # an unreachable rig must not stop the run
            logger.warning("Could not ask the rig for its array: %r", error)
            array = None
        if array and not same_geometry(
            saved["trap_geometry"], array["x_um"], array["y_um"]
        ):
            return None, (
                "the saved trap sites were fitted to another array than the rig "
                "has on the SLM; run FindTrapSites again and accept its proposal"
            )
        self.trap_window_size = int(saved["trap_window_size"])
        self.trap_feature_size = int(saved["trap_feature_size"])
        self.trap_threshold = float(saved["trap_threshold"])
        roi_x0, roi_y0 = self.roi_origin
        sites = np.rint(to_roi(saved["trap_sites"], roi_y0, roi_x0)).astype(int)
        half = self.trap_window_size // 2
        outside = [
            index
            for index, (y, x) in enumerate(sites)
            if min(y, x) < half
            or y + half >= self.roi_height
            or x + half >= self.roi_width
        ]
        if outside:
            return None, (
                f"the summing windows of trap sites {outside} leave the camera's ROI"
            )
        return {index: (int(y), int(x)) for index, (y, x) in enumerate(sites)}, ""

    def shot(self, **overrides) -> tuple:
        """Run the sequence once at the saved parameters plus ``overrides``; the two HamCam frames.

        ``overrides`` are keyed as for :meth:`run_sequences` (``rb.<parameter>``).
        """
        saved = {f"rb.{name}": value for name, value in self.saved_sequence.items()}
        self.camera.start_acquisition()
        self.run_sequences(**(saved | overrides))
        first_image, second_image = self.camera.acquire_n_frames(
            2, broadcast=True, header={"rid": self.rid}
        )
        self.camera.stop_acquisition()
        return first_image, second_image

    def take_shots(self, n_shots: int, **sequence) -> tuple[list, list]:
        """``n_shots`` shots with script parameters ``sequence``; their first and second frames.

        With ``save_raw_images`` the frames are recorded as ``first_images`` and
        ``second_images``, ``(shots, height, width)`` per point.
        """
        overrides = {f"rb.{name}": value for name, value in sequence.items()}
        frames = [self.shot(**overrides) for _ in range(n_shots)]
        firsts, seconds = zip(*frames, strict=True)
        if self.save_raw_images:
            self.record("first_images", np.array(firsts))
            self.record("second_images", np.array(seconds))
        return list(firsts), list(seconds)

    @staticmethod
    def top_hat(frames, feature_size: int) -> list[np.ndarray]:
        return [
            an.morphological_tophat_high_pass(
                np.asarray(frame, dtype=float), feature_size=feature_size
            )
            for frame in frames
        ]

    def site_photons(self, frames) -> np.ndarray:
        """Every site's photons in each of ``frames``, top-hat filtered: ``(frames, sites)``."""
        filtered = self.top_hat(frames, self.trap_feature_size)
        return photons_at_sites(filtered, self.trap_sites, self.trap_window_size)

    def site_counts(self, frames) -> np.ndarray:
        """Every site's summed raw pixel values in each of ``frames``: ``(frames, sites)``."""
        return np.array(
            [
                an.sum_pixel_values(
                    np.asarray(frame),
                    self.trap_sites,
                    window_size=self.trap_window_size,
                )
                for frame in frames
            ]
        )

    def run_point(self):
        first_image, second_image = self.shot()
        if self.save_raw_images:
            self.record("first_image", first_image)
            self.record("second_image", second_image)
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )
        if self.trap_sites is not None:
            frames = [first_image, second_image]
            first, second = self.site_photons(frames) > self.trap_threshold
            self.record("first_occupied", first)
            self.record("second_occupied", second)
            self.record("loading", float(first.mean()))
            if self.save_tweezer_counts:
                first, second = self.site_counts(frames)
                self.record("first_counts", first, unit="counts")
                self.record("second_counts", second, unit="counts")

    def finish(self) -> None:
        self.camera.stop_acquisition()
