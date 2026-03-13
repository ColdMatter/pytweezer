
import time
from typing import Union
from pytweezer.experiment.experiment import Experiment, ExperimentHandler
import numpy as np


class DummyExperiment(Experiment):
    motmaster_script = "dummy_script"
    def build(self):
        self.camera = self.setattr_device("dummy_camera", mode="test", n_frames=3)
        
    def pre_run(self):
        self.camera.start_acquisition()
        
    def run(self):
        image_0 = np.array([[1 if j == i else 0 for j in range(5)] for i in range(5)])
        image_1 = np.array([[1 if j == 2 else 0 for j in range(5)] for i in range(5)])
        image_2 = np.array([[1 if j == 4 - i else 0 for j in range(5)] for i in range(5)])
        self.camera.dummy_images = [image_0, image_1, image_2]

    def post_run(self):
        self.camera.acquire_n_frames(autosave=True)
        time.sleep(0.1)


def dummy_analysis(imgs):
    print(f"Analyzing data")
    # return list of random numbers for testing
    return [[sum(img) for img in imgs]]


class DummyCamera:
    def __init__(self, mode, n_frames):
        self.mode = mode
        self.n_frames = n_frames
        self.dummmy_images = None
        self.setup_acquisition()

    def setup_acquisition(self):
        print(f"Setting up camera in {self.mode} mode for {self.n_frames} frames.")

    def start_acquisition(self):
        print("Starting camera acquisition.")

    def acquire_n_frames(self, autosave=True):
        # print(f"Acquiring {n_frames} frames. Autosave={autosave}")
        # return random data for testing
        if self.dummmy_images is None:
            self.dummmy_images = [[i + j for j in range(5)] for i in range(self.n_frames)]
        return self.dummmy_images
    
    def close(self):
        print("Closing camera connection.")
    
class DummySynth:
    def __init__(self, args):
        self.args = args

class DummyAnalyser:
    def get_next_zipno(self):
        return 42

    def tweezer_inject(self, zip_no):
        print(f"Injecting data from zip number {zip_no} into analysis pipeline.")

    def tweezer_inject_double(self, zip_no):
        print(
            f"Injecting data from zip number {zip_no} (double) into analysis pipeline."
        )

class DummyMotMasterInterface:
        def __init__(self, interval: Union[int, float] = 0.1) -> None:
            self.script = None
            
        def set_motmaster_experiment(self, script: str):
            self.script = script
            print(f"MotMaster script set to {script}.")
            
        def start_motmaster_experiment(self):
            if self.script is None:
                raise ValueError("MotMaster script not set. Please call set_motmaster_experiment first.")
            print(f"Starting MotMaster experiment {self.script}.")
        
        def set_motmaster_dictionary(self):
            pass
        
if __name__ == "__main__":
    mm = DummyMotMasterInterface()
    handler = ExperimentHandler(mm)
    handler.run_experiment("DummyExperimment")
    