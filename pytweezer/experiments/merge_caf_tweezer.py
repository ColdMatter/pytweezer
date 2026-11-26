"""CaF into the tweezers with MERGE.CaFTweezer, imaged on the HamCam and the CaFMOT Cam."""

from __future__ import annotations

import math

import numpy as np

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.orca import ORCACamera
from pytweezer.experiment import Device, Integer, Number
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.experiments.load_tweezers import (
    hamcam_roi_argument,
    set_up_hamcam,
)
from pytweezer.experiments.slm_tweezer_array import SlmArray


class MergeCaFTweezer(SlmArray, MotMasterExperiment):
    """Put the tweezer array on the SLM, run MERGE.CaFTweezer linked, and image each shot.

    The script runs on the CaF MOTMaster in linked mode, driving RbTweezer too,
    at its own parameter values; scan any of them by name. Its ``tHamCamTrig``
    and ``tMOTCamTrig`` pulses each take one frame, recorded per point as
    ``hamcam_image`` and ``cafmot_image`` with their sums ``hamcam_counts`` and
    ``cafmot_counts``.

    The mask is written before the first point and again whenever an SLM
    argument changes, so the array and the Fresnel lens can be scanned; each
    write waits ``slm_settle_time``. Every point records the quality of the
    mask on the SLM (``uniformity``, ``min_max_ratio``, ``mse``).

    The ``site_*`` arguments are where the CaF imaging puts the array on the
    HamCam: the blaze spot's sensor pixel, and the scale and rotation from
    focal-plane µm to pixels, with x mirrored as in
    :func:`~pytweezer.analysis.trap_sites.geometry_to_pixels`. They are separate
    from the Rb ``trap_*`` calibration because the CaF imaging differs. The
    run does not use them; the ``caf_tweezer_sites`` analysis places the sites
    with them, the ROI origin the camera set (``hamcam_roi_origin``, ``(x0, y0)``),
    ``hamcam_binning`` and the arrays on the SLM (``slm_arrays``).
    """

    analyses = ("caf_tweezer_sites",)

    caf = MotMaster("CaF MotMaster", script="MERGE.CaFTweezer", linked=True)

    hamcam: ImagEMX2Camera = Device("Rb HamCam")
    cafmot_cam: ORCACamera = Device("CaFMOT Cam")

    slm_settle_time = Number(1, unit="s", min=0, group="Array")
    hamcam_gain = Number(
        600,
        min=0,
        group="Camera",
        tooltip="EM-gain sensitivity of the HamCam, set once before the first point",
        scannable=False,
    )
    hamcam_binning = Integer(
        1,
        min=1,
        group="Camera",
        tooltip="Sensor pixels binned per side on the HamCam, set once before the first point",
        scannable=False,
    )
    hamcam_roi_x0 = hamcam_roi_argument(110, size=False)
    hamcam_roi_width = hamcam_roi_argument(256, size=True)
    hamcam_roi_y0 = hamcam_roi_argument(220, size=False)
    hamcam_roi_height = hamcam_roi_argument(256, size=True)

    site_centre_x = Number(
        saved="caf_hamcam_centre_x",
        unit="px",
        group="Sites",
        tooltip="Sensor x pixel of the blaze spot on the HamCam, for the analysis",
        scannable=False,
    )
    site_centre_y = Number(
        saved="caf_hamcam_centre_y",
        unit="px",
        group="Sites",
        tooltip="Sensor y pixel of the blaze spot on the HamCam, for the analysis",
        scannable=False,
    )
    site_scale = Number(
        saved="caf_hamcam_px_per_m",
        unit="px/µm",
        scale=1e6,
        min=0,
        group="Sites",
        tooltip="HamCam pixels per µm in the focal plane, for the analysis",
        scannable=False,
    )
    site_angle = Number(
        saved="caf_hamcam_angle",
        unit="deg",
        scale=math.pi / 180,
        group="Sites",
        tooltip="Rotation of the focal plane on the HamCam, for the analysis",
        scannable=False,
    )

    def prepare(self) -> None:
        super().prepare()
        self.read_slm_settings()
        roi_origin = set_up_hamcam(
            self.hamcam,
            self.hamcam_gain,
            (
                self.hamcam_roi_x0,
                self.hamcam_roi_width,
                self.hamcam_roi_y0,
                self.hamcam_roi_height,
            ),
            self.hamcam_binning,
        )
        self.record("hamcam_roi_origin", np.array(roi_origin), unit="px")
        self.set_up_cafmot_cam()

    def run_point(self) -> None:
        self.update_slm_array(self.slm_settle_time)
        hamcam_mark = self.hamcam.start_shot({"rid": self.rid})
        cafmot_mark = self.cafmot_cam.start_shot({"rid": self.rid})
        self.run_motmaster_sequence()
        self.record_frame("hamcam", self.hamcam.wait_frames(hamcam_mark, 1))
        self.record_frame("cafmot", self.cafmot_cam.wait_frames(cafmot_mark, 1))

    def finish(self) -> None:
        try:
            self.record_slm_arrays()
        finally:
            try:
                self.restore_cameras()
            finally:
                super().finish()

    def restore_cameras(self) -> None:
        """Leave both cameras live, broadcasting untagged frames."""
        try:
            self.hamcam.start_live()  # does nothing unless a failure left it stopped
            self.hamcam.end_shots()
        finally:
            self.cafmot_cam.start_live()
            self.cafmot_cam.end_shots()

    def set_up_cafmot_cam(self) -> None:
        """Expose the CaFMOT Cam on its external trigger, live."""
        self.cafmot_cam.stop_live()
        self.cafmot_cam.set_trigger_source("ext")
        self.cafmot_cam.start_live()

    def record_frame(self, camera: str, frames) -> None:
        image = np.asarray(frames)[0]
        self.record(f"{camera}_image", image)
        self.record(f"{camera}_counts", float(image.sum(dtype=float)), unit="counts")


class MergeCaFTweezerHamCamBackground(MergeCaFTweezer):
    """:class:`MergeCaFTweezer` with a HamCam background frame in every shot.

    MERGE.CaFTweezerHamCamBackground fires the HamCam a second time,
    ``HamCamBackgroundGap`` (10 us ticks, scannable by name) after the first
    frame ends. That frame is recorded as ``hamcam_background``, and
    ``hamcam_signal_counts`` is the summed image minus the summed background.
    """

    caf = MotMaster(
        "CaF MotMaster", script="MERGE.CaFTweezerHamCamBackground", linked=True
    )

    def run_point(self) -> None:
        self.update_slm_array(self.slm_settle_time)
        hamcam_mark = self.hamcam.start_shot({"rid": self.rid})
        cafmot_mark = self.cafmot_cam.start_shot({"rid": self.rid})
        self.run_motmaster_sequence()
        image, background = np.asarray(self.hamcam.wait_frames(hamcam_mark, 2))
        self.record("hamcam_image", image)
        self.record("hamcam_background", background)
        self.record(
            "hamcam_signal_counts",
            float(image.sum(dtype=float) - background.sum(dtype=float)),
            unit="counts",
        )
        self.record_frame("cafmot", self.cafmot_cam.wait_frames(cafmot_mark, 1))
