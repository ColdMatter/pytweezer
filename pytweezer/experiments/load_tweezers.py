"""Load the Rb tweezers and image them twice on the HamCam."""

from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.experiment.arguments import Device, Integer
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
)


class LoadTweezers(MotMasterExperiment):
    """Run the Rb tweezer-loading sequence and record two HamCam frames per point.

    ``signal_counts`` is the summed first frame minus the summed second.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic2026_2")

    camera: ImagEMX2Camera = Device("Rb HamCam")

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

    def run_point(self):
        self.camera.start_acquisition()
        self.run_sequences()
        first_image, second_image = self.camera.acquire_n_frames(2, broadcast=True)
        self.camera.stop_acquisition()
        self.record("first_image", first_image)
        self.record("second_image", second_image)
        self.record(
            "signal_counts",
            float(first_image.sum(dtype=float) - second_image.sum(dtype=float)),
            unit="counts",
        )

    def finish(self) -> None:
        self.camera.stop_acquisition()
