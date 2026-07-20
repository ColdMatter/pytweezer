
import time
from typing import Union
from pytweezer.experiment.experiment import Experiment
from pytweezer.experiment.experiment import NumberValue
import numpy as np
from pytweezer.drivers.imagemX2 import ImagEMX2Camera


class DummyExperiment(Experiment):
    motmaster_script = "dummy_script"
    def build(self):
        super().build()
        self.setattr_device("dummy_camera", mode="test", n_frames=3, servers=True)
        self.dummy_camera: ImagEMX2Camera
        
        self.setattr_argument("sleep_time", NumberValue, display_multiplier=1.0, unit="s")
        self.sleep_time: float

    def pre_run(self):
        super().pre_run()
        self.dummy_camera.start_acquisition()
        
    def run(self):
        image_0 = np.array([[1 if j == i else 0 for j in range(5)] for i in range(5)])
        image_1 = np.array([[1 if j == 2 else 0 for j in range(5)] for i in range(5)])
        image_2 = np.array([[1 if j == 4 - i else 0 for j in range(5)] for i in range(5)])
        self.dummy_camera.dummy_images = [image_0, image_1, image_2]
        time.sleep(self.sleep_time.get())
        

    def post_run(self):
        exp_info = {"task": self._task, "run": self._run, "rep": self._rep}
        self.dummy_camera.acquire_n_frames(exp_info=exp_info, autosave=True)
        time.sleep(0.1)


def dummy_analysis(imgs):
    print(f"Analyzing data")
    # return list of random numbers for testing
    return [[sum(img) for img in imgs]]

    