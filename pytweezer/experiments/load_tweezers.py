"""Load the Rb tweezers at the saved sequence parameters and image them twice on the HamCam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytweezer.analysis.array_loading import TrapSites
from pytweezer.analysis.cloud_fit import SPECIES_MASS_AMU, fit_frames
from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Bool, Device, Integer, Number
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
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


def _roi_argument(default: int, *, size: bool) -> Integer:
    return Integer(
        default,
        min=1 if size else 0,
        unit="px",
        group="Camera",
        tooltip="Unbinned sensor pixels; the trap sites are placed in the ROI set "
        "before the first point, so it cannot be scanned",
        scannable=False,
    )


class LoadTweezers(MotMasterExperiment):
    """Run the Rb tweezer-loading sequence and image the tweezers twice per point.

    Every shot runs the sequence at the saved parameters ``rb`` binds
    (``LoadTweezers.rb.saved``); a script parameter set in the form wins over
    its saved value, and the values used are recorded as ``saved_sequence``.
    With ``use_motmaster_parameter_overrides`` off it sends none of them, so the
    script's own values run; the trap sites are still the saved ones.

    Each shot takes two frames on ``hamcam``, exposed by the trigger pulse,
    with EM gain on and ``hamcam_gain`` as its sensitivity. The HamCam stays
    live, during the task and after it: its live mode broadcasts each shot's
    frames, tagged with the task's ``rid`` and their ``index`` in the shot. Binning is fixed at
    1, because the trap sites are saved in sensor pixels; the ROI cannot be
    scanned, because the sites are placed in the ROI set at :meth:`prepare`.

    ``signal_counts`` is the summed first frame minus the summed second.
    :meth:`prepare` also loads the accepted trap sites (see
    :meth:`get_trap_site_positions`) and records them in this ROI's pixels as
    ``trap_sites``, with the ``trap_window_size``, ``trap_feature_size`` and
    occupancy ``threshold`` they are scored with. With them, each point records
    whether each site was occupied in each frame (``first_occupied``,
    ``second_occupied``: its top-hat filtered photons above the threshold), the
    fraction of sites occupied in the first frame (``loading``), the fraction
    of those still occupied in the second (``survival``, NaN if none were) and,
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
        600,
        min=0,
        group="Camera",
        tooltip="EM-gain sensitivity of the HamCam, set once before the first point",
        scannable=False,
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

    def prepare(self) -> None:
        super().prepare()
        self.set_up_hamcam()
        self.read_trap_sites()

    def run_point(self) -> None:
        mark = self.hamcam.start_shot({"rid": self.rid})
        self.run_motmaster_sequence()
        frames = np.asarray(self.hamcam.wait_frames(mark, 2))
        self.record_data(frames)

    def finish(self) -> None:
        try:
            self.hamcam.start_live()  # does nothing unless a failure left it stopped
            self.hamcam.end_shots()

        finally:
            super().finish()

    def set_up_hamcam(self) -> None:
        """Put the HamCam in external exposure with EM gain, binning 1 and the chosen ROI, live."""
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
        self.hamcam.start_live()

    def read_trap_sites(self) -> None:
        """Load the accepted trap sites into ``trap_sites`` and record them, or record why there are none."""
        # Fetched only for sequence bindings, so not with the overrides off.
        saved = self.saved_parameters or get_saved_parameters()
        self.trap_sites, self.no_trap_sites_reason = self.get_trap_site_positions(saved)
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
        self.record("trap_sites", np.array(list(self.trap_sites.positions.values())))
        self.record("trap_window_size", self.trap_sites.window_size, unit="px")
        self.record("trap_feature_size", self.trap_sites.feature_size, unit="px")
        self.record("threshold", self.trap_sites.threshold, unit="photons")

    def record_data(self, frames: np.ndarray) -> None:
        """Record the frames, signal and per-site results of one shot's two HamCam ``frames``."""
        if self.save_raw_images:
            self.record("hamcam_images", frames)
        first_image, second_image = frames
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )
        if self.trap_sites is None:
            return

        first, second = self.trap_sites.occupied(frames)
        self.record("first_occupied", first)
        self.record("second_occupied", second)
        self.record("loading", float(first.mean()))
        self.record("survival", float(second[first].mean()) if first.any() else np.nan)
        if self.save_tweezer_counts:
            first_counts, second_counts = self.trap_sites.counts(frames)
            self.record("first_counts", first_counts, unit="counts")
            self.record("second_counts", second_counts, unit="counts")

    def get_trap_site_positions(self, saved: dict) -> tuple[TrapSites | None, str]:
        """The accepted trap sites in ``saved``, placed in whole pixels of this ROI; or ``None`` and why.

        An empty reason means the experiment wants none. Sites fitted to another
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
        window_size = int(saved["trap_window_size"])
        # The camera may round the ROI's corner; sites are placed by the one it used.
        roi_x0, roi_y0 = self.roi_origin
        roi_width, roi_height = self.hamcam_roi_width, self.hamcam_roi_height
        sites = np.rint(to_roi(saved["trap_sites"], roi_y0, roi_x0)).astype(int)
        half = window_size // 2
        outside = [
            index
            for index, (y, x) in enumerate(sites)
            if min(y, x) < half or y + half >= roi_height or x + half >= roi_width
        ]
        if outside:
            return None, (
                f"the summing windows of trap sites {outside} leave the camera's ROI"
            )
        positions = {index: (int(y), int(x)) for index, (y, x) in enumerate(sites)}
        return TrapSites(
            positions,
            window_size,
            int(saved["trap_feature_size"]),
            float(saved["trap_threshold"]),
        ), ""


class LoadTweezersNImages(LoadTweezers):
    """Load the tweezers and image them ``n_images`` times per point."""

    rb = MotMaster(
        "Rb MotMaster",
        script="RbTweezerBasic2026_multiple_images",
    )

    def run_point(self) -> None:
        # The live mode broadcasts the frames; nothing is recorded.
        mark = self.hamcam.start_shot({"rid": self.rid})
        self.run_motmaster_sequence()
        self.hamcam.wait_frames(mark, 1)


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
        self.orca.stop_live()
        self.orca.set_trigger_source("ext")
        self.orca.start_live()
        self.record("cloud_mass_amu", SPECIES_MASS_AMU["Rb"], unit="u")

    def run_point(self) -> None:
        orca_mark = self.orca.start_shot({"rid": self.rid})
        super().run_point()
        self.record_cloud_fit(np.asarray(self.orca.wait_frames(orca_mark, 1)))

    def finish(self) -> None:
        try:
            self.orca.start_live()  # does nothing unless a failure left it stopped
            self.orca.end_shots()
        finally:
            super().finish()

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
