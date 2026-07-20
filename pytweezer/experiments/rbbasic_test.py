from pytweezer.experiment.experiment import Experiment
from pytweezer.drivers.imagemX2 import ImagEMX2Camera, ImagEMX2CameraClient

class RBBasicTest(Experiment):
    motmaster_script = "RbTweezerBasic2025"

    def build(self):
        super().build()
        self.cam_rpc = ImagEMX2CameraClient()
        self.direct_cam = None

    def run(self):
        print(f"sending params {self._experiment_params} to MotMaster and starting experiment...")
        self._motmaster_client.start_experiment(parameters=self._experiment_params)