"""CaF into the tweezers with MERGE.CaFTweezer, imaged on the HamCam and the CaFMOT Cam."""

from __future__ import annotations

import numpy as np

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.orca import ORCACamera
from pytweezer.experiment import Device, Number
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
    """

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
    hamcam_roi_x0 = hamcam_roi_argument(110, size=False)
    hamcam_roi_width = hamcam_roi_argument(256, size=True)
    hamcam_roi_y0 = hamcam_roi_argument(220, size=False)
    hamcam_roi_height = hamcam_roi_argument(256, size=True)

    def prepare(self) -> None:
        super().prepare()
        self.read_slm_settings()
        set_up_hamcam(
            self.hamcam,
            self.hamcam_gain,
            (
                self.hamcam_roi_x0,
                self.hamcam_roi_width,
                self.hamcam_roi_y0,
                self.hamcam_roi_height,
            ),
        )
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
            self.hamcam.start_live()  # does nothing unless a failure left it stopped
            self.hamcam.end_shots()
        finally:
            try:
                self.cafmot_cam.start_live()
                self.cafmot_cam.end_shots()
            finally:
                super().finish()

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
