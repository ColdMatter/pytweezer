from pytweezer.experiment.experiment import Experiment
from pytweezer.drivers.imagemX2 import ImagEMX2Camera, ImagEMX2CameraClient

class RBBasicTest(Experiment):
    motmaster_script = "RbTweezerBasic2025"

    def build(self):
        super().build()
        self.cam_rpc = ImagEMX2CameraClient()
        self.direct_cam = None

    def run(self):
        # Tell the camera server to release DCAM so this process can open it directly.
        self.cam_rpc.relinquish_camera()
        try:
            self.direct_cam = ImagEMX2Camera()

            # Do your low-latency direct-camera sequence here.
            self.direct_cam.set_external_exposure_mode()
            self.direct_cam.start_acquisition()
            image = self.direct_cam.acquire_single_frame()
            self.direct_cam.stop_acquisition()

            # Continue experiment flow after camera readout.
            self._motmaster_client.start_experiment()
        finally:
            if self.direct_cam is not None:
                self.direct_cam.close()
                self.direct_cam = None
            # Give ownership back to the persistent camera server process.
            self.cam_rpc.reacquire_camera()

    def cleanup(self):
        super().cleanup()
        if getattr(self, "direct_cam", None) is not None:
            self.direct_cam.close()
            self.direct_cam = None
        if getattr(self, "cam_rpc", None) is not None:
            self.cam_rpc.close()