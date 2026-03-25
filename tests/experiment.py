from dataclasses import dataclass
import time
from typing import Callable, Optional
from pytweezer.imagemX2 import ImagEMX2Camera
from tqdm import tqdm
from pytweezer.analysis import TweezerExperimentAnalysis
import h5py
import json

PROPERTIES_FILE = "../properties.json"


class Experiment:

    def __init__(
        self,
        experiment_runner,
        script: str,
        camera: ImagEMX2Camera,
        analyser: TweezerExperimentAnalysis,
        n_images: int = 2,
        analysis_funcs: Optional[list[Callable]] = None,
    ):
        self.experiment_runner = experiment_runner
        self.script = script
        self.cam = camera
        self.n_images = n_images
        self.analyser = analyser
        self.analysis_funcs = analysis_funcs or []

    def setup_camera(self, n_frames):
        self.cam.setup_acquisition("snap", nframes=n_frames)
        self.cam.start_acquisition()

    def do_analysis(self, imgs):
        results = []
        for analysis_func in self.analysis_funcs:
            result = analysis_func(imgs)
            results.append(result)
        return results

    def run(self, scan_param, scan_vals, n_repeats: int = 1, save: bool = True):
        
        self.cam.setup_acquisition("snap", nframes=n_repeats * self.n_images)
        for i in tqdm(range(len(scan_vals)), desc="Experiment Scan"):
            scan_val = scan_vals[i]
            self.cam.start_acquisition()
            self.experiment_runner.scan_motmaster_paramter(
                self.script, scan_param, scan_val, interations=n_repeats, save=save
            )
            imgs = self.cam.acquire_n_frames(n_repeats * self.n_images, autosave=True)
            start_zip = self.inject()
            zips = [start_zip + j for j in range(len(scan_vals))]
            

        time.sleep(0.1)
        analysis_results = self.do_analysis(imgs)
        return Result(
            rid=get_rid(),
            scan_param=scan_param,
            scan_vals=scan_vals,
            images=imgs,
            analysis_results=analysis_results,
            zips=zips
        )
    
    def inject(self):
        """move images from the temp folder to the zip"""
        zip_no = self.analyser.get_next_zipno()
        if self.n_images == 1:
            self.analyser.tweezer_inject(zip_no)
        elif self.n_images == 2:
            self.analyser.tweezer_inject_double(zip_no)
        else:
            raise ValueError("We haven't written the injection code for more than 2 images per scan yet.")
        return zip_no
            
def get_rid():
    with open(PROPERTIES_FILE, "r") as f:
        properties = json.load(f)
    rid = properties["next_rid"]
    properties["next_rid"] += 1
    with open(PROPERTIES_FILE, "w") as f:
        json.dump(properties, f)
    return rid
    
class Result:
    # def __init__(self, rid):
    #     self.rid = rid
    #     self.scan_param = None
    #     self.scan_vals = []
    #     self.images = [] 
    #     self.analysis_results = []
    
    def __init__(
        self,
        rid,
        scan_param,
        scan_vals,
        images,
        analysis_results,
        zips,
        create_hdf5=True
    ):
        self.rid = rid
        self.scan_param = scan_param
        self.scan_vals = scan_vals
        self.images = images
        self.analysis_results = analysis_results
        self.zips = zips
        if create_hdf5:
            self.create_hdf5(f"experiment_result_{rid}.h5")
    
    
    def create_hdf5(self, filename):
        with h5py.File(filename, 'w') as f:
            f.attrs['rid'] = self.rid
            f.attrs['scan_param'] = self.scan_param
            f.create_dataset('scan_vals', data=self.scan_vals)
            f.create_dataset('images', data=self.images)
            f.create_dataset('zips', data=self.zips)
            # Assuming analysis_results is a list lists
            for i, result in enumerate(self.analysis_results):
                f.create_dataset(f'analysis_result_{i}', data=result, dtype=float)
        


class DummyExperimentRunner:
    def scan_motmaster_paramter(
        self, script, param_name, param_value, interations=1, save=True
    ):
        time.sleep(0.5)


def dummy_analysis(imgs):
    tqdm.write(f"Analyzing data")
    # return list of random numbers for testing
    return [[sum(img) for img in imgs]]


class DummyCamera:
    def setup_acquisition(self, mode, nframes):
        tqdm.write(f"Setting up camera in {mode} mode for {nframes} frames.")

    def start_acquisition(self):
        tqdm.write("Starting camera acquisition.")

    def acquire_n_frames(self, n_frames, autosave=True):
        # tqdm.write(f"Acquiring {n_frames} frames. Autosave={autosave}")
        #return random data for testing
        return [[i + j for j in range(5)] for i in range(n_frames)]


class DummyAnalyser:
    def get_next_zipno(self):
        return 42
    
    def tweezer_inject(self, zip_no):
        tqdm.write(f"Injecting data from zip number {zip_no} into analysis pipeline.") 
        
    def tweezer_inject_double(self, zip_no):
        tqdm.write(f"Injecting data from zip number {zip_no} (double) into analysis pipeline.")


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
