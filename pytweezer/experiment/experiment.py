import importlib
import inspect
import time
from typing import  Optional, Union
import json
import pathlib
import time
import json
import sys

import numpy as np
from pytweezer.configuration.device_db import device_db

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System.Collections.Generic import Dictionary
    from System import String, Object
    from System import Activator
except Exception as e:
    print(f"Error: {e} encountered, probably no pythonnet")

# config dir is relative to the root of the repository, which is added to the path when pytweezer is installed, so should be findable from anywhere in the code using a relative path from the root. If this becomes an issue we can add some code to find the config dir based on the location of this file.
CONFIG_DIR =  "pytweezer/configuration"
PROPERTIES_FILE = CONFIG_DIR + "/properties.json"
DEFAULTS_FILE = CONFIG_DIR + "/defaults.json"
EXPERIMENTS_FILE = CONFIG_DIR + "/experiments.json"


class ExperimentHandler:

    def __init__(self):
        self.motmaster_interface = MotMasterInterface()
        
    def set_scan_parameters(self, scan_param, scan_vals):
        self.scan_param = scan_param
        self.scan_vals = scan_vals
        
    def set_scan_linear(self, scan_param, start, stop, n_points, randomize=False):
        scan_vals = np.linspace(start, stop, n_points)
        if randomize:
            np.random.shuffle(scan_vals)
        self.set_scan_parameters(scan_param, scan_vals)
        
        
    def set_scan_centered(self, scan_param, center, span, n_points, randomize=False):
        start = center - span/2
        stop = center + span/2
        self.set_scan_linear(scan_param, start, stop, n_points, randomize=randomize)
    

    def run_experiment(self, experiment):
        with open(EXPERIMENTS_FILE, "r") as f:
            experiments_config = json.load(f)
        experiment_config = experiments_config[experiment]
        experiment_module = importlib.import_module(experiment_config["module"])
        experiment_class = getattr(experiment_module, experiment_config["class"])
        experiment_instance: Experiment = experiment_class(self.motmaster_interface)
        experiment_instance.build()
        for scan_val in self.scan_vals:
            # set scan val here
            experiment_instance.pre_run()
            experiment_instance.run()
            experiment_instance.post_run()
        experiment_instance.cleanup()
        
    def check_param_in_device_db(self, experiment, param):
        # check if the parameter is in the motmaster dictionary
        for device in experiment.devices.keys():
            if param in experiment.devices[device]["parameters"]:
                return device
        return False
    
    def set_scan_value(self, experiment, param, value):
        if device_name := self.check_param_in_device_db(experiment, param):
            device = experiment.devices[device_name]
            setattr(device, param, value)
        elif param in self.motmaster_interface.get_params():
            self.motmaster_interface.parameter_dictionary[param] = value
        else:
            raise ValueError(f"Parameter {param} not found in device database or MotMaster parameters")


class MotMasterInterface:
    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        with open(PROPERTIES_FILE, "r") as f:
            self.config = json.load(f)
        self.root = pathlib.Path(self.config["script_root_path"])
        self.interval = interval
        self.motmaster = None
        self.script = None

    def _add_ref(self, path: str) -> None:
        _path = pathlib.Path(path)
        sys.path.append(_path.parent)
        clr.AddReference(path)
        return None

    def connect(self) -> None:
        for path in self.config["dll_paths"].values():
            clr.AddReference(path)
        for key, path_info in self.config.items():
            if key == "dll_paths":
                for path in path_info.values():
                    self._add_ref(path)
            elif key == "motmaster":
                self._add_ref(path_info["exe_path"])
                try:
                    import MOTMaster

                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller, path_info["remote_path"]
                    )
                    print("Connected to MotMaster.")
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHadwareControl

                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHadwareControl.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")

    def disconnect(self) -> None:
        self.stage.close()
        return None

    def set_motmaster_experiment(
        self,
        script: str,
    ):
        self.script = script
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            print(f"MotMaster script set to {script}.")
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def set_motmaster_dictionary(self):
        self.parameter_dictionary = Dictionary[String, Object]()
        with open(DEFAULTS_FILE, "r") as f:
            default_parameters = json.load(f)
        for key, value in default_parameters.items():
            self.parameter_dictionary[key] = value

    def start_motmaster_experiment(
        self,
    ):
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        try:
            self.motmaster.Go(self.parameter_dictionary)
            time.sleep(self.interval)
        except Exception as e:
            print(f"Error starting MotMaster experiment {self.script}: {e}")
        return None
    
    def get_params(self):
        return dict(self.motmaster.GetParameters())


class Experiment:
    motmaster_script: Optional[str] = None

    def __init__(self, motmaster_interface: MotMasterInterface):
        self._device_db = device_db
        print(f"Loaded device database: {self._device_db}")
        self.devices = {}
        self.device_params = []
        self._motmaster_interface = motmaster_interface
        self._motmaster_interface.set_motmaster_experiment(self.motmaster_script)
        self._motmaster_interface.set_motmaster_dictionary()

    def build(self):
        """
        This method should be used to set up the experiment, e.g. by calling setattr_device and defining any other necessary attributes that will be constant throughout the experiment.
        """
        pass

    def pre_run(self):
        """
        This method is called before each run, e.g. for reinitialising devices.
        """
        pass

    def run(self):
        """
        This method will usually only need to trigger the motmaster pattern.
        """
        self._motmaster_interface.start_motmaster_experiment()

    def post_run(self):
        """
        This method is called after each run e.g. for getting camera images.
        """
        pass

    def cleanup(self):
        """
        This method is called at the end of the measurment, after all scan points and repetitions are complete, e.g. for closing devices.
        """
        for device in self.devices.values():
            try:
                device.close()
            except AttributeError:
                pass

    def setattr_device(self, device_name, *args, **kwargs):
        try:
            module = importlib.import_module(self._device_db[device_name]["module"])
        except Exception as e:
            print("Error while importing module: {0}".format(e), "error")
            raise e
        # loop through the module and try to find the class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name == self._device_db[device_name]["class"]:
                device_class = obj
                break
        else:
            raise ValueError(
                f"Device class {self._device_db[device_name]['class']} not found in module {self._device_db[device_name]['module']}"
            )
        device_instance = device_class(*args, **kwargs)
        setattr(self, device_name, device_instance)
        self.devices[device_name] = device_instance
        return device_instance


# class Experiment:

#     def __init__(
#         self,
#         experiment_runner,
#         script: str,
#         camera: ImagEMX2Camera,
#         analyser: TweezerExperimentAnalysis,
#         n_images: int = 2,
#         analysis_funcs: Optional[list[Callable]] = None,
#     ):
#         self.experiment_runner = experiment_runner
#         self.script = script
#         self.cam = camera
#         self.n_images = n_images
#         self.analyser = analyser
#         self.analysis_funcs = analysis_funcs or []

#     def setup_camera(self, n_frames):
#         self.cam.setup_acquisition("snap", nframes=n_frames)
#         self.cam.start_acquisition()

#     def do_analysis(self, imgs):
#         results = []
#         for analysis_func in self.analysis_funcs:
#             result = analysis_func(imgs)
#             results.append(result)
#         return results

#     def run(self, scan_param, scan_vals, n_repeats: int = 1, save: bool = True):

#         self.cam.setup_acquisition("snap", nframes=n_repeats * self.n_images)
#         for i in tqdm(range(len(scan_vals)), desc="Experiment Scan"):
#             scan_val = scan_vals[i]
#             self.cam.start_acquisition()
#             self.experiment_runner.scan_motmaster_paramter(
#                 self.script, scan_param, scan_val, interations=n_repeats, save=save
#             )
#             imgs = self.cam.acquire_n_frames(n_repeats * self.n_images, autosave=True)
#             start_zip = self.inject()
#             zips = [start_zip + j for j in range(len(scan_vals))]


#         time.sleep(0.1)
#         analysis_results = self.do_analysis(imgs)
#         return Result(
#             rid=get_rid(),
#             scan_param=scan_param,
#             scan_vals=scan_vals,
#             images=imgs,
#             analysis_results=analysis_results,
#             zips=zips
#         )

#     def inject(self):
#         """move images from the temp folder to the zip"""
#         zip_no = self.analyser.get_next_zipno()
#         if self.n_images == 1:
#             self.analyser.tweezer_inject(zip_no)
#         elif self.n_images == 2:
#             self.analyser.tweezer_inject_double(zip_no)
#         else:
#             raise ValueError("We haven't written the injection code for more than 2 images per scan yet.")
#         return zip_no


def get_rid():
    with open(PROPERTIES_FILE, "r") as f:
        properties = json.load(f)
    rid = properties["next_rid"]
    properties["next_rid"] += 1
    with open(PROPERTIES_FILE, "w") as f:
        json.dump(properties, f)
    return rid
