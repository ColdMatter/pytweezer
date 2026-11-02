"""Load the Rb tweezers and image them twice on the HamCam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pytweezer.analysis.trap_sites import same_geometry, to_roi
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Device, Integer
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
)
from pytweezer.logging_utils import get_logger

if TYPE_CHECKING:
    # Imports cupy where it is installed, so only for the annotation.
    from pytweezer.coordinators.rearrangement import Rearrangement

logger = get_logger(__name__)

#: The saved parameters a trap-site fit is accepted into.
TRAP_SITE_KEYS = (
    "trap_sites",
    "trap_threshold",
    "trap_window_size",
    "trap_feature_size",
    "trap_geometry",
)


class LoadTweezers(MotMasterExperiment):
    """Run the Rb tweezer-loading sequence and record two HamCam frames per point.

    ``signal_counts`` is the summed first frame minus the summed second.
    :meth:`prepare` also loads the accepted trap sites (see
    :meth:`saved_trap_sites`) and records them in this ROI's pixels as
    ``trap_sites``, with the ``trap_window_size`` and ``trap_feature_size``
    they are scored with.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2")
    analyses = ("tweezer_histogram",)

    camera: ImagEMX2Camera = Device("Rb HamCam")
    rig: Rearrangement = Device("Rb Rearrangement Rig")

    sensitivity = Integer(600, min=0, group="Camera", tooltip="EM gain sensitivity")
    roi_x0 = Integer(110, min=0, group="Camera")
    roi_y0 = Integer(220, min=0, group="Camera")
    roi_width = Integer(256, min=1, group="Camera")
    roi_height = Integer(256, min=1, group="Camera")

    def prepare(self) -> None:
        super().prepare()

        self.camera.set_roi(self.roi_x0, self.roi_width, self.roi_y0, self.roi_height)
        self.camera.set_external_exposure_mode()
        self.camera.enable_em_gain(True)
        self.camera.enable_direct_em_gain(True)
        self.camera.set_sensitivity(self.sensitivity)
        self.camera.setup_acquisition("snap", 2)

        self.trap_sites, self.no_trap_sites_reason = self.saved_trap_sites()
        if self.trap_sites is None:
            if self.no_trap_sites_reason:
                logger.info("Recording no trap sites: %s", self.no_trap_sites_reason)
            return
        self.record("trap_sites", np.array(list(self.trap_sites.values())))
        self.record("trap_window_size", self.trap_window_size, unit="px")
        self.record("trap_feature_size", self.trap_feature_size, unit="px")

    def saved_trap_sites(self) -> tuple[dict[int, tuple[int, int]] | None, str]:
        """The accepted trap sites as whole pixels of this ROI, by index; or ``None`` and why.

        Also sets ``trap_window_size``, ``trap_feature_size`` and
        ``trap_threshold`` from the saved parameters. Sites fitted to another
        array than the rig has on the SLM are refused; if the rig does not know
        its array, or cannot be reached, the saved sites are used unchecked.
        """
        saved = get_saved_parameters()
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
        sites = np.rint(to_roi(saved["trap_sites"], self.roi_y0, self.roi_x0)).astype(
            int
        )
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
        """Run the sequence once with ``overrides`` and return the two HamCam frames."""
        self.camera.start_acquisition()
        self.run_sequences(**overrides)
        first_image, second_image = self.camera.acquire_n_frames(
            2, broadcast=True, header={"rid": self.rid}
        )
        self.camera.stop_acquisition()
        return first_image, second_image

    def run_point(self):
        first_image, second_image = self.shot()
        self.record("first_image", first_image)
        self.record("second_image", second_image)
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )

    def finish(self) -> None:
        self.camera.stop_acquisition()
