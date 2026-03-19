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
from configuration.device_db import device_db

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
        experiment_instance._start_build()
        for scan_val in self.scan_vals:
            # set scan val here
            experiment_instance.pre_run()
            experiment_instance.run()
            experiment_instance.post_run()
        experiment_instance.cleanup()
        

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

    def __init__(self, props, motmaster_interface: Optional[MotMasterInterface] = None):
        self._device_db = device_db
        print(f"Loaded device database: {self._device_db}")
        self.devices = {}
        self.arguments = {}
        self._props = props
        self._motmaster_interface = motmaster_interface
        if self.motmaster_script is not None and self._motmaster_interface is None:
            raise ValueError("MotMaster interface is required when a script is specified.")
        if self._motmaster_interface is not None:
            self._motmaster_interface.set_motmaster_experiment(self.motmaster_script)
            self._motmaster_interface.set_motmaster_dictionary()
            self.mm_params = {}
            
    def _start_build(self):
        if self.motmaster_script is not None:
            for param_name in self._motmaster_interface.parameter_dictionary.keys():
                self.setattr_mm_param(param_name)
                
    def _start_run(self, task=None):
        """
        start a single run
        """
        # change to __enter__ ???
        global globaltime
        global starttime
        global _experiment
        _experiment = self
        self._name = self.name
        self._starttime = time.time()
        run = self._props.get('_totalrun', 0)
        self._props.set('_totalrun', run + 1)
        if task is None:
            task = self._props.get('/Experiments/_task')
        globaltime = 0
        starttime = time.time()
        self._publish_startvalues(task)
        self.pre_run()
        self.run()
        self.post_run()
        self._endtime = time.time()
        self._publish_endvalues()
        
    def _publish_startvalues(self, task):
        """send basic information about experiment started to data channel
        """
        if not hasattr(self, 'exp_name'):
            self.exp_name = self._name
        info = {'_starttime': self._starttime, '_run': self._run, '_repetition': self._repetition, '_task': task,
                '_name': self._name, '_exp_name': self.exp_name, "mm_script": self.motmaster_script}
        arguments = dict((arg.name, arg.value) for arg in self._arguments.values())
        mm_params = dict((param.name, param.value) for param in self.mm_params.values())
        info.update(arguments)
        info.update(mm_params)
        self._dataq.send(info, channel='.start')

    def _publish_endvalues(self):
        info = {'_endtime': self._endtime}
        self._dataq.send(info, channel='.end')

            

    def build(self):
        """
        This method should be used to set up the experiment, e.g. by calling setattr_device and defining any other necessary attributes that will be constant throughout the experiment.
        """
        pass

    def pre_run(self):
        """
        This method is called before each run, e.g. for reinitialising devices.
        """
        if self.motmaster_script is not None:
            for param_name, param in self.mm_params.items():
                self._motmaster_interface.parameter_dictionary[param_name] = param.value

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
    
              
    def setattr_mm_param(self, param_name):
        """Create an attribute for a MotMaster parameter, which can be modified by the user in the GUI."""
        if self._motmaster_interface is None:
            raise ValueError("MotMaster interface is not set.")
        param_type = type(self._motmaster_interface.parameter_dictionary[param_name])
        if param_type == int:
            param = NumberValue(name=param_name, ndecimals=0, step=1, value=self._motmaster_interface.parameter_dictionary[param_name], minval=-1E6, maxval=1E6)
        elif param_type == float:
            param = NumberValue(name=param_name, ndecimals=3, step=0.001, value=self._motmaster_interface.parameter_dictionary[param_name], minval=-1E6, maxval=1E6)
        elif param_type == bool:
            param = BoolValue(name=param_name, value=self._motmaster_interface.parameter_dictionary[param_name])
        elif param_type == str:
            param = StringCombo(name=param_name, stringlist=[self._motmaster_interface.parameter_dictionary[param_name]])
        else:            
            raise ValueError(f"Unsupported parameter type {param_type} for parameter {param_name}")
        setattr(self, param_name, param)
        self.mm_params[param_name] = param
        
    def setattr_argument(self, name, arg):
        """
        Create a user-controllable attribute
        The user will have an element in the GUI he can modify

        Args:
            name: (str)
                Name of the attribute. You can address the attribute via ``self.myname``
            argtype: (class)
                Currently only NumberValue is available

        Example::

            self.setattr_argument("Nr_Ablation_Pulses", NumberValue(ndecimals=0, step=1,value=20))
        """
        setattr(self, name, arg)
        self.arguments[name] = arg



class NumberValue:
    """ floating point argument for experiment window

        Args:

            name: (str)
                name of the argument
            ndecimals: (int)
                number of decimals e.g. 2 means 0.01
            step:   (float)
                step size of the spin box
            value:  (float)
                default value
            minval: (float)
                minimum value
            maxval: (float)
                maximum value
            tooltip: (str)
                tooltip

    """
    __argtype__ = 'NumberValue'

    def __init__(self, name='No name', ndecimals=0, step=1, value=42, minval=0, maxval=1E6, **kwargs):
        self.__dict__.update(kwargs)
        self.name = name
        self.step = step
        self.ndecimals = ndecimals
        self.value = value
        self.minval = minval
        self.maxval = maxval


class StringCombo:
    """String choice argument for expeirment window. """
    __argtype__ = 'StringCombo'

    def __init__(self, name='Noname', stringlist=['']):
        '''
        stringlist: list of strings from which the user can choose
        name:       name of the argument
        '''
        self.name = name
        self.stringlist = stringlist
        self.value = stringlist[0]


class BoolValue:
    """Boolean argument for experiment window. """
    __argtype__ = 'BoolValue'

    def __init__(self, name='Noname', value=False):
        '''
        name: str
            name of the argument
        value: bool
            value of the argument
        '''
        self.name = name
        self.value = value


def get_rid():
    with open(PROPERTIES_FILE, "r") as f:
        properties = json.load(f)
    rid = properties["next_rid"]
    properties["next_rid"] += 1
    with open(PROPERTIES_FILE, "w") as f:
        json.dump(properties, f)
    return rid
