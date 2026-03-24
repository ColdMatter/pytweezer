import time
from typing import Union
from pytweezer.experiment.experiment import MotMasterInterface
from pytweezer.servers import ImageClient
from pytweezer.servers import CommandClient, DataClient
import numpy as np


class DummyMotMasterInterface(MotMasterInterface):
    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        self.script = None
        self.motmaster = DummyMotMaster()


    def set_motmaster_experiment(self, script: str):
        self.script = script
        print(f"MotMaster script set to {script}.")

    def start_motmaster_experiment(self):
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        print(f"Starting MotMaster experiment {self.script}")
        self.motmaster.Go(self.parameter_dictionary)
        
    def set_motmaster_dictionary(self):
        self.parameter_dictionary = self.get_params()


    


class DummyMotMaster:

    def GetParameters(self):
        return {"param1": 1, "param2": 2.0}
    
    def Go(self, parameter_dictionary):
        print("sending mm dictionary {parameter_dictionary}")




class DummyCamera:
    def __init__(self, mode, n_frames, servers=False):
        self.mode = mode
        self.n_frames = n_frames
        self.dummmy_images = None
        self.setup_acquisition()
        if servers:
            self._connect_clients()
            
    def _connect_clients(self):
        self.imstream = ImageClient('dummy_cam')
        self.cmdstream = CommandClient('dummy_cam')
        self.cmdstream.subscribe('dummy_cam')

    def setup_acquisition(self):
        print(f"Setting up camera in {self.mode} mode for {self.n_frames} frames.")

    def start_acquisition(self):
        print("Starting camera acquisition.")

    def acquire_n_frames(self, exp_info=None, autosave=True):
        # print(f"Acquiring {n_frames} frames. Autosave={autosave}")
        # return random data for testing
        if self.dummmy_images is None:
            # create dummy images as ndarrays with a simple pattern for testing
            self.dummmy_images = [np.array([[1 if j == i else 0 for j in range(5)] for i in range(5)]) for _ in range(self.n_frames)]
        if exp_info is not None:
            for i, img in enumerate(self.dummmy_images):
                self.broadcast_image(img, task=exp_info["task"], run=exp_info["run"], rep=exp_info["rep"], index=i, timestamp=time.time())
            

        return self.dummmy_images
    
    def broadcast_image(self, im, task, run, rep, index, timestamp):
    
        info = {
            "timestamp": timestamp,
            "task": task,
            "run": run,
            "rep": rep,
            "index": index
        }  # TODO all cam settings
        print(f"sening image {im}")

        self.imstream.send(im, info)
        # print('camera time stamp:',image_stamp)
    
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
3