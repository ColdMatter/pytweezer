import time
from pytweezer.servers import ImageClient
from pytweezer.servers import CommandClient
import numpy as np


class DummyMotMaster:

    def GetParameters(self):
        return {"param1": 1, "param2": 2.0}

    def Go(self, parameter_dictionary):
        print("sending mm dictionary {parameter_dictionary}")


class DummyCamera:
    def __init__(self, mode, n_frames, servers=False):
        self.mode = mode
        self.n_frames = n_frames
        self.dummy_images = None
        self.setup_acquisition()
        if servers:
            self._connect_clients()

    def _connect_clients(self):
        self.imstream = ImageClient("dummy_cam")
        self.cmdstream = CommandClient("dummy_cam")
        self.cmdstream.subscribe("dummy_cam")

    def setup_acquisition(self):
        print(f"Setting up camera in {self.mode} mode for {self.n_frames} frames.")

    def start_acquisition(self):
        print("Starting camera acquisition.")

    def acquire_n_frames(self, exp_info=None, autosave=True):
        # print(f"Acquiring {n_frames} frames. Autosave={autosave}")
        # return random data for testing
        if self.dummy_images is None:
            # create dummy images as ndarrays with a simple pattern for testing
            self.dummy_images = [
                np.array([[1 if j == i else 0 for j in range(5)] for i in range(5)])
                for _ in range(self.n_frames)
            ]
        if exp_info is not None:
            for i, img in enumerate(self.dummy_images):
                print(f"Broadcasting image {img} with exp_info: {exp_info}")
                self.broadcast_image(
                    img,
                    task=exp_info["task"],
                    run=exp_info["run"],
                    rep=exp_info["rep"],
                    index=i,
                    timestamp=time.time(),
                )

        return self.dummy_images

    def broadcast_image(self, im, task, run, rep, index, timestamp):

        info = {
            "timestamp": timestamp,
            "task": task,
            "run": run,
            "rep": rep,
            "_imgindex": index,
        }  # TODO all cam settings

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
