import time
from tkinter.font import names
from typing import Callable, Optional
from pytweezer.imagemX2 import ImagEMX2Camera
from tqdm import tqdm
from pytweezer.analysis import TweezerExperimentAnalysis


class Experiment:

    def __init__(
        self,
        experiment_runner,
        script: str,
        camera: ImagEMX2Camera,
        scan_param: str,
        scan_vals: list,
        analyser: TweezerExperimentAnalysis,
        n_images: int = 2,
        analysis_funcs: Optional[list[Callable]] = None,
    ):
        self.experiment_runner = experiment_runner
        self.script = script
        self.scan_param = scan_param
        self.scan_vals = scan_vals
        self.cam = camera
        self.n_images = n_images
        self.analyser = analyser
        self.analysis_funcs = analysis_funcs or []
        self.latest_zip_no = None

    def setup_camera(self, n_frames):
        self.cam.setup_acquisition("snap", nframes=n_frames)
        self.cam.start_acquisition()

    def do_analysis(self):
        for analysis_func in self.analysis_funcs:
            analysis_func(self.latest_zip_no)

    def run(self, n_repeats: int = 1, save: bool = True):
        self.latest_zip_no = self.analyser.get_next_zipno()
        self.cam.setup_acquisition("snap", nframes=n_repeats * self.n_images)
        for i in tqdm(range(len(self.scan_vals)), desc="Experiment Scan"):
            scan_val = self.scan_vals[i]
            self.cam.start_acquisition()
            self.experiment_runner.scan_motmaster_paramter(
                self.script, self.scan_param, scan_val, interations=n_repeats, save=save
            )
            imgs = self.cam.acquire_n_frames(n_repeats * self.n_images, autosave=True)

        time.sleep(0.1)

        self.do_analysis()


class DummyExperimentRunner:
    def scan_motmaster_paramter(
        self, script, param_name, param_value, interations=1, save=True
    ):
        time.sleep(0.5)

def dummy_analysis(zip_no):
    tqdm.write(f"Analyzing data from zip number {zip_no}")
    
class DummyCamera:
    def setup_acquisition(self, mode, nframes):
        tqdm.write(f"Setting up camera in {mode} mode for {nframes} frames.")

    def start_acquisition(self):
        tqdm.write("Starting camera acquisition.")

    def acquire_n_frames(self, n_frames, autosave=True):
        # tqdm.write(f"Acquiring {n_frames} frames. Autosave={autosave}")
        return [f"Image_{i}" for i in range(n_frames)]
    
class DummyAnalyser:
    def get_next_zipno(self):
        return 42
    
if __name__ == "__main__":
    experiment_runner = DummyExperimentRunner()
    camera = DummyCamera()
    analyser = DummyAnalyser()
    experiment = Experiment(
        experiment_runner=experiment_runner,
        script="dummy_script.mmp",
        camera=camera,
        scan_param="param1",
        scan_vals=[1, 2, 3],
        analyser=analyser,
        n_images=2,
        analysis_funcs=[dummy_analysis],
    )
    experiment.run(n_repeats=2, save=False)