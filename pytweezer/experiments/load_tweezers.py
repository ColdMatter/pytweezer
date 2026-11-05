"""Load the Rb tweezers at the saved sequence parameters and image them twice on the HamCam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytweezer.analysis import analysis as an
from pytweezer.analysis.array_loading import photons_at_sites
from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Bool, Device
from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.experiment.scan import Point, Scan
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

#: The saved parameters a trap-site fit is accepted into.
TRAP_SITE_KEYS = (
    "trap_sites",
    "trap_threshold",
    "trap_window_size",
    "trap_feature_size",
    "trap_geometry",
)


class LoadTweezers(CameraExperiment):
    """Run the Rb tweezer-loading sequence and image the tweezers twice per point.

    Every shot runs the sequence at the saved parameters ``rb`` binds
    (``LoadTweezers.rb.saved``); a script parameter set in the form wins over
    its saved value, and the values used are recorded as ``saved_sequence``.
    :meth:`take_shots` runs a batch of shots, for subclasses whose point is one.

    Each shot takes two frames on ``hamcam``, exposed by the trigger pulse,
    with EM gain on and ``hamcam.gain`` as its sensitivity. Binning is fixed at
    1, because the trap sites are saved in sensor pixels; the ROI origin cannot
    be scanned, because the sites are placed in the ROI set at :meth:`prepare`.

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

    hamcam: ImagEMX2Camera = CameraShots(
        "Rb HamCam",
        frames=2,
        binning=1,
        gain=600,
        roi=(110, 256, 220, 256),
        locked=("frames", "binning"),
    )
    rig: Rearrangement = Device("Rb Rearrangement Rig")

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

    def configure_camera(self, name: str, camera: ImagEMX2Camera) -> None:
        camera.set_external_exposure_mode()
        camera.enable_em_gain(True)
        camera.enable_direct_em_gain(True)

    def planned_points(self, scan: Scan) -> list[Point]:
        for axis in scan.axes:
            if axis.argument in ("hamcam.roi_x0", "hamcam.roi_y0"):
                raise ValueError(
                    f"{axis.argument} cannot be scanned: the trap sites are placed "
                    "in the ROI set before the first point"
                )
        return super().planned_points(scan)

    def prepare(self) -> None:
        super().prepare()
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
        # The camera may round the ROI's corner; sites are placed by the one it used.
        roi_x0, roi_y0 = self.roi_origins["hamcam"]
        _, roi_width, _, roi_height = self.shot_settings("hamcam").roi
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

    def take_shots(self, n_shots: int, **sequence) -> tuple[list, list]:
        """``n_shots`` shots with script parameters ``sequence``; their first and second frames.

        With ``save_raw_images`` the frames are recorded as ``hamcam_images``,
        ``(shots, 2, height, width)`` per point.
        """
        overrides = {f"rb.{name}": value for name, value in sequence.items()}
        stacks = np.array(
            [self.take_shot(**overrides)["hamcam"] for _ in range(n_shots)]
        )
        if self.save_raw_images:
            self.record_frames({"hamcam": stacks})
        return list(stacks[:, 0]), list(stacks[:, 1])

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
        self.record_shot(self.take_shot())

    def record_shot(self, frames: dict[str, np.ndarray]) -> None:
        """Record the tweezer results of one shot's ``hamcam`` frames."""
        if self.save_raw_images:
            self.record_frames({"hamcam": frames["hamcam"]})
        first_image, second_image = frames["hamcam"]
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
    each point is still one shot. Its cloud fit is recorded as ``orca_*`` with no
    background subtracted
    (:meth:`~pytweezer.experiment.camera_experiment.CameraExperiment.record_cloud_fit`),
    and the ORCA's frame is not recorded itself unless ``save_mean_image`` is set.
    """

    cloud_species = "Rb"
    analyses = (*LoadTweezers.analyses, "cloud_fit", "cloud_temperature")

    orca = CameraShots("ORCA", frames=1, locked=("frames",))

    save_mean_image = Bool(
        True,
        group="Camera",
        tooltip="record the ORCA's frame, beside the fit to it",
    )

    def configure_camera(self, name: str, camera) -> None:
        # The HamCam's EM-gain calls don't exist on the ORCA.
        if name == "hamcam":
            super().configure_camera(name, camera)

    def run_point(self):
        frames = self.take_shot()
        self.record_shot(frames)
        self.record_cloud_fit("orca", frames["orca"])
