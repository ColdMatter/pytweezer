import importlib
import inspect
import time
from typing import Optional, Union
import json
import pathlib
import time
import json
import sys

import numpy as np
from configuration.device_db import device_db
from pytweezer.servers import DataClient, Properties
from pytweezer.experiment.motmaster_client import MotMasterClient

# config dir is relative to the root of the repository, which is added to the path when pytweezer is installed, so should be findable from anywhere in the code using a relative path from the root. If this becomes an issue we can add some code to find the config dir based on the location of this file.
CONFIG_DIR = "pytweezer/configuration"
PROPERTIES_FILE = CONFIG_DIR + "/properties.json"
DEFAULTS_FILE = CONFIG_DIR + "/defaults.json"
EXPERIMENTS_FILE = CONFIG_DIR + "/experiments.json"

def get_experiment(filepath, name):
    """Load an experiment class from a given file path. The file should contain exactly one subclass of Experiment, which will be returned."""
    try:
        module_name = f"_experiment_{name}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {filepath}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        candidates = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Experiment)
            and obj is not Experiment
            and obj.__module__ == module.__name__
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one Experiment subclass in {filepath}, found {len(candidates)}"
            )

        experiment_cls = candidates[0]
        return experiment_cls
    except Exception as e:
        print(f"Error loading experiment from {filepath}: {e}")
        raise e

class Experiment:
    motmaster_script: Optional[str] = None
    _gui_columns = 4

    def __init__(self, props, motmaster_client: Optional[MotMasterClient] = None):
        self.name = self.__class__.__name__
        self._props = Properties("Experiments/" + self.name)
        self._dataq = DataClient("Experiment")
        self._device_db = device_db
        print(f"Loaded device database: {self._device_db}")
        self.devices = {}
        self._arguments = []
        self._motmaster_client = motmaster_client
        self._task = 0
        self._run = 0
        self._rep = 0
        self._experiment_params: dict = {}
        if self.motmaster_script is not None and self._motmaster_client is None:
            self._motmaster_client = MotMasterClient()
        if self._motmaster_client is not None:
            self._motmaster_client.set_script(self.motmaster_script)
            self.mm_params = []

    def _start_build(self):
        self.build()

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
        run = self._props.get("_totalrun", 0)
        self._props.set("_totalrun", run + 1)
        if task is None:
            task = self._props.get("/Experiments/_task")
        globaltime = 0
        starttime = time.time()
        self._publish_startvalues(task)
        self.pre_run()
        self.run()
        self.post_run()
        self._endtime = time.time()
        self._publish_endvalues()

    def _publish_startvalues(self, task):
        """send basic information about experiment started to data channel"""
        if not hasattr(self, "exp_name"):
            self.exp_name = self._name
        info = {
            "_starttime": self._starttime,
            "_run": self._run,
            "_repetition": self._repetition,
            "_task": task,
            "_name": self._name,
            "_exp_name": self.exp_name,
            "mm_script": self.motmaster_script,
        }
        arguments = dict((arg.name, self.__dict__[arg.name]) for arg in self._arguments)
        print(arguments)
        mm_params = dict((param.name, self.__dict__[param.name]) for param in self.mm_params)
        info.update(arguments)
        info.update(mm_params)
        self._dataq.send(info, channel=".start")

    def _publish_endvalues(self):
        info = {"_endtime": self._endtime}
        self._dataq.send(info, channel=".end")

    def build(self):
        """
        This method should be used to set up the experiment, e.g. by calling setattr_device and defining any other necessary attributes that will be constant throughout the experiment.
        """
        if self.motmaster_script is not None:
            response = self._motmaster_client.get_params()
            if response.get("ok"):
                params = response.get("params", {})
                for param_name in params.keys():
                    self.setattr_mm_param(param_name)

    def pre_run(self):
        """
        This method is called before each run, e.g. for reinitialising devices.
        """
        if self.motmaster_script is not None:
            self._experiment_params = {}
            for param in self.mm_params:
                name = param.name
                self._experiment_params[name] = getattr(self, name)

    def run(self):
        """
        This method will usually only need to trigger the motmaster pattern.
        """
        if self._motmaster_client:
            if self._experiment_params:
                self._motmaster_client.start_experiment(parameters=self._experiment_params)
            else:
                self._motmaster_client.start_experiment()

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
        if self._motmaster_client is None:
            raise ValueError("MotMaster client is not set.")
        response = self._motmaster_client.get_params()
        if not response.get("ok"):
            raise ValueError(f"Failed to get MotMaster parameters: {response}")
        param_dict = response.get("params", {})
        if param_name not in param_dict:
            raise ValueError(f"Parameter '{param_name}' not found in MotMaster parameters")
        param_value = param_dict[param_name]
        param_type = type(param_value)
        print(param_type)
        if param_type == int:
            param = NumberValue(
                name=param_name,
                ndecimals=0,
                step=1,
                value=param_value,
                minval=-1e6,
                maxval=1e6,
            )
        elif param_type == float:
            param = NumberValue(
                name=param_name,
                ndecimals=3,
                step=0.001,
                value=param_value,
                minval=-1e6,
                maxval=1e6,
            )
        elif param_type == bool:
            param = BoolValue(
                name=param_name,
                value=param_value,
            )
        elif param_type == str:
            param = StringCombo(
                name=param_name,
                stringlist=[param_value],
            )
        else:
            raise ValueError(
                f"Unsupported parameter type {param_type} for parameter {param_name}"
            )
        setattr(self, param_name, param.value)
        self.mm_params.append(param)

    def setattr_argument(self, name, argtype, *args, **kwargs):
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
        arg_inst = argtype(name, *args, **kwargs)
        setattr(self, name, arg_inst.value)
        self._arguments.append(arg_inst)
        


class NumberValue:
    """floating point argument for experiment window

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

    __argtype__ = "NumberValue"

    def __init__(
        self,
        name="No name",
        ndecimals=0,
        step=1,
        value=42,
        minval=0,
        maxval=1e6,
        **kwargs,
    ):
        self.__dict__.update(kwargs)
        self.name = name
        self.step = step
        self.ndecimals = ndecimals
        self.value = value
        self.minval = minval
        self.maxval = maxval


class StringCombo:
    """String choice argument for expeirment window."""

    __argtype__ = "StringCombo"

    def __init__(self, name="Noname", stringlist=[""]):
        """
        stringlist: list of strings from which the user can choose
        name:       name of the argument
        """
        self.name = name
        self.stringlist = stringlist
        self.value = stringlist[0]


class BoolValue:
    """Boolean argument for experiment window."""

    __argtype__ = "BoolValue"

    def __init__(self, name="Noname", value=False):
        """
        name: str
            name of the argument
        value: bool
            value of the argument
        """
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
