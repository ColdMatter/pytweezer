"""Load the Rb tweezers at the saved sequence parameters and image them twice on the HamCam."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.cloud_fit import SPECIES_MASS_AMU, fit_frames
from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Bool, Device, Integer, Number
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.experiment.scan import Point, Scan
from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_client import get_device_config

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

#: The saved parameters a trap-site fit is accepted into.
TRAP_SITE_KEYS = (
    "trap_sites",
    "trap_threshold",
    "trap_window_size",
    "trap_feature_size",
    "trap_geometry",
)


_ROI_ARGUMENTS = (
    "hamcam_roi_x0",
    "hamcam_roi_width",
    "hamcam_roi_y0",
    "hamcam_roi_height",
)


def _roi_argument(default: int, *, size: bool) -> Integer:
    return Integer(
        default,
        min=1 if size else 0,
        unit="px",
        group="Camera",
        tooltip="Unbinned sensor pixels",
    )


class LoadTweezers(MotMasterExperiment):
    """Run the Rb tweezer-loading sequence and image the tweezers twice per point.

    Every shot runs the sequence at the saved parameters ``rb`` binds
    (``LoadTweezers.rb.saved``); a script parameter set in the form wins over
    its saved value, and the values used are recorded as ``saved_sequence``.

    Each shot takes two frames on ``hamcam``, exposed by the trigger pulse,
    with EM gain on and ``hamcam_gain`` as its sensitivity. Binning is fixed at
    1, because the trap sites are saved in sensor pixels; the ROI cannot be
    scanned, because the sites are placed in the ROI set at :meth:`prepare`.

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
    (``hamcam_images``, ``(2, H, W)`` per point) are recorded only with
    ``save_raw_images``, which needs no sites.
    """

    rb = MotMaster(
        "Rb MotMaster",
        script="RbTweezerBasic2026_2",
        saved=(
            *_SAVED_UNDER_OWN_NAME,
            {
                "tImgCoolVCO1": "tImgCoolVCO",
                "tImgCoolVCO2": "tImgCoolVCO",
                "tImgCoolVVA1": "tImgCoolVVA",
                "tImgCoolVVA2": "tImgCoolVVA",
            },
        ),
    )
    analyses = ("tweezer_histogram", "tweezer_photon_histogram")

    hamcam: ImagEMX2Camera = Device("Rb HamCam")
    rig: Rearrangement = Device("Rb Rearrangement Rig")

    hamcam_gain = Number(
        600, min=0, group="Camera", tooltip="EM-gain sensitivity of the HamCam"
    )
    hamcam_roi_x0 = _roi_argument(110, size=False)
    hamcam_roi_width = _roi_argument(256, size=True)
    hamcam_roi_y0 = _roi_argument(220, size=False)
    hamcam_roi_height = _roi_argument(256, size=True)
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

    def planned_points(self, scan: Scan) -> list[Point]:
        for axis in scan.axes:
            if axis.argument in _ROI_ARGUMENTS:
                raise ValueError(
                    f"{axis.argument} cannot be scanned: the trap sites are placed "
                    "in the ROI set before the first point"
                )
        return super().planned_points(scan)

    def prepare(self) -> None:
        super().prepare()
        self.hamcam_was_live = self.hamcam.is_live()
        self.hamcam.stop_live()
        self.hamcam.set_binning(1)
        self.roi_origin = tuple(
            self.hamcam.set_roi(
                self.hamcam_roi_x0,
                self.hamcam_roi_width,
                self.hamcam_roi_y0,
                self.hamcam_roi_height,
            )
        )
        self.hamcam.set_trigger_source("ext")
        self.hamcam.set_external_exposure_mode()
        self.hamcam.enable_em_gain(True)
        self.hamcam.enable_direct_em_gain(True)
        self.hamcam.set_gain(self.hamcam_gain)
        self.sent_gain = self.hamcam_gain
        self.hamcam.setup_acquisition(2, mode="snap")

        self.trap_sites, self.no_trap_sites_reason = self.saved_trap_sites(
            self.saved_parameters
        )
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

    def finish(self) -> None:
        try:
            if getattr(self, "hamcam_was_live", False):
                self.hamcam.start_live()
        finally:
            super().finish()

    def saved_trap_sites(
        self, saved: dict
    ) -> tuple[dict[int, tuple[int, int]] | None, str]:
        """The accepted trap sites in ``saved`` as whole pixels of this ROI, by index; or ``None`` and why.

        An empty reason means the experiment wants none.

        Also sets ``trap_window_size``, ``trap_feature_size`` and
        ``trap_threshold`` from ``saved``. Sites fitted to another
        array than the rig has on the SLM are used with a warning; if the rig
        does not know its array, or cannot be reached, they are used unchecked.
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
            logger.warning(
                "The saved trap sites were fitted to another array than the rig "
                "has on the SLM; using them anyway. Run FindTrapSites again and "
                "accept its proposal to fit the current array."
            )
        self.trap_window_size = int(saved["trap_window_size"])
        self.trap_feature_size = int(saved["trap_feature_size"])
        self.trap_threshold = float(saved["trap_threshold"])
        # The camera may round the ROI's corner; sites are placed by the one it used.
        roi_x0, roi_y0 = self.roi_origin
        roi_width, roi_height = self.hamcam_roi_width, self.hamcam_roi_height
        sites = np.rint(to_roi(saved["trap_sites"], roi_y0, roi_x0)).astype(int)
        half = self.trap_window_size // 2
        outside = [
            index
            for index, (y, x) in enumerate(sites)
            if min(y, x) < half or y + half >= roi_height or x + half >= roi_width
        ]
        if outside:
            return None, (
                f"the summing windows of trap sites {outside} leave the camera's ROI"
            )
        return {index: (int(y), int(x)) for index, (y, x) in enumerate(sites)}, ""

    def take_shot(self, **sequence: Any) -> np.ndarray:
        """Run the sequence once with script parameters ``sequence``; the HamCam's two frames.

        The frames are broadcast on the camera's stream as this task's.
        """
        if self.hamcam_gain != self.sent_gain:
            self.hamcam.set_gain(self.hamcam_gain)
            self.sent_gain = self.hamcam_gain
        self.hamcam.start_acquisition()
        try:
            self.run_sequences(**sequence)
            return np.asarray(
                self.hamcam.acquire_n_frames(
                    2, broadcast=True, header={"rid": self.rid}
                )
            )
        finally:
            self.hamcam.stop_acquisition()

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

    def run_point(self) -> None:
        self.record_tweezers(self.take_shot())

    def record_tweezers(self, frames: np.ndarray) -> None:
        """Record the tweezer results of one shot's two HamCam ``frames``."""
        if self.save_raw_images:
            self.record("hamcam_images", frames)
        first_image, second_image = frames
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )
        if self.trap_sites is not None:
            images = [first_image, second_image]
            first, second = self.site_photons(images) > self.trap_threshold
            self.record("first_occupied", first)
            self.record("second_occupied", second)
            self.record("loading", float(first.mean()))
            if self.save_tweezer_counts:
                first, second = self.site_counts(images)
                self.record("first_counts", first, unit="counts")
                self.record("second_counts", second, unit="counts")


class LoadTweezersMOTImage(LoadTweezers):
    """:class:`LoadTweezers` that also fits the Rb MOT imaged on the ORCA.

    The ORCA takes one frame in the same MOTMaster run as the HamCam's two, so
    each point is still one shot. A Gaussian fit to it is recorded as
    ``orca_<field>`` for every field of
    :data:`~pytweezer.analysis.cloud_fit.FIELDS` and ``orca_<field>_error`` for
    each of :data:`~pytweezer.analysis.cloud_fit.ERROR_FIELDS`, with no
    background subtracted: in metres when the ORCA's CONFIG entry gives
    ``object_pixel_m``, in pixels otherwise, and NaN where the fit found
    nothing. The frame itself is recorded as ``orca_mean_image`` only with
    ``save_mean_image``. The Rb mass is recorded as ``cloud_mass_amu``.
    """

    analyses = (*LoadTweezers.analyses, "cloud_fit", "cloud_temperature")

    orca = Device("ORCA")

    save_mean_image = Bool(
        True,
        group="Camera",
        tooltip="record the ORCA's frame, beside the fit to it",
    )

    def prepare(self) -> None:
        super().prepare()
        self.orca_was_live = self.orca.is_live()
        self.orca.stop_live()
        self.orca.set_trigger_source("ext")
        self.orca.setup_acquisition(1, mode="snap")
        self.record("cloud_mass_amu", SPECIES_MASS_AMU["Rb"], unit="u")

    def finish(self) -> None:
        try:
            if getattr(self, "orca_was_live", False):
                self.orca.start_live()
        finally:
            super().finish()

    def run_point(self) -> None:
        self.orca.start_acquisition()
        try:
            hamcam_frames = self.take_shot()
            orca_frames = np.asarray(
                self.orca.acquire_n_frames(1, broadcast=True, header={"rid": self.rid})
            )
        finally:
            self.orca.stop_acquisition()
        self.record_tweezers(hamcam_frames)
        self.record_cloud_fit(orca_frames)

    def record_cloud_fit(self, frames: np.ndarray) -> None:
        """Fit the mean of the ORCA's ``frames`` and record the cloud as ``orca_*``."""
        pixel_m = get_device_config("ORCA").get("object_pixel_m")
        # Read from the camera, which may have been set up outside this experiment.
        binning, (x0, y0) = self.orca.frame_geometry()
        resolution = None if pixel_m is None else (pixel_m * binning,) * 2
        values, mean_image = fit_frames(
            frames, None, resolution, (x0 / binning, y0 / binning)
        )
        length = "px" if resolution is None else "m"
        for field, value in values.items():
            quantity = field.removesuffix("_error")
            unit = {"angle": "rad", "amplitude": "counts", "offset": "counts"}.get(
                quantity, "counts" if quantity == "integral" else length
            )
            self.record(f"orca_{field}", value, unit=unit)
        if self.save_mean_image:
            self.record("orca_mean_image", mean_image, unit="counts")
