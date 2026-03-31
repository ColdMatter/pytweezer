import importlib
import inspect
import time
from typing import Optional, Union
import json
import pathlib
import time
import json
import sys
import subprocess

import numpy as np
import zmq
from configuration.device_db import device_db
from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import DataClient, Properties
from pytweezer.experiment.motmaster_client import MotMasterClient
from pytweezer.servers.clients import ImageClient

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
    _save_results_h5 = True
    _results_h5_dir = ""

    def __init__(self, props, motmaster_client: Optional[MotMasterClient] = None, context: Optional[zmq.Context] = None):
        self.name = self.__class__.__name__
        self._props = Properties("Experiments/" + self.name)
        self._dataq = DataClient("Experiment/" + self.name)
        self._imgq = ImageClient("Experiment/" + self.name)
        self._device_db = device_db
        self.devices = {}
        self.args: dict[str, Union["NumberValue", "StringCombo", "BoolValue"]] = {}
        self.mm_args: dict[str, Union["NumberValue", "StringCombo", "BoolValue"]] = {}
        self._motmaster_client = motmaster_client
        self._task = 0
        self._run = 0
        self._rep = 0
        self._experiment_params: dict = {}
        if self.motmaster_script is not None and self._motmaster_client is None:
            self._motmaster_client = MotMasterClient(context=context)
        if self._motmaster_client is not None:
            self._motmaster_client.set_script(self.motmaster_script)
        self.result_channels: dict[str, Union[ResultChannel, ImageResultChannel]] = {}
        self._last_start_info: dict = {}

    @property
    def _arguments(self):
        """Ordered experiment arguments for GUI compatibility."""
        return list(self.args.values())

    @property
    def mm_params(self):
        """Ordered MotMaster arguments for GUI compatibility."""
        return list(self.mm_args.values())

    def _resolve_argument_container(self, name: str, mm: Optional[bool] = None):
        if mm is True:
            return self.mm_args
        if mm is False:
            return self.args

        # Auto-resolve by name when caller doesn't specify argument domain.
        if name in self.args:
            return self.args
        if name in self.mm_args:
            return self.mm_args
        raise KeyError(f"Unknown argument '{name}'")

    def set_argument_value(self, name: str, value, mm: Optional[bool] = None):
        container = self._resolve_argument_container(name, mm)
        arg = container[name]
        arg.value = self._coerce_argument_value(arg, value)

    def get_argument_value(self, name: str, mm: Optional[bool] = None):
        container = self._resolve_argument_container(name, mm)
        return container[name].get()

    @staticmethod
    def _coerce_argument_value(arg, value):
        if isinstance(arg, BoolValue):
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, str):
                txt = value.strip().lower()
                if txt in ("1", "true", "t", "yes", "y", "on"):
                    return True
                if txt in ("0", "false", "f", "no", "n", "off"):
                    return False
            try:
                return bool(int(float(value)))
            except (TypeError, ValueError):
                pass
            raise ValueError(
                f"Argument '{arg.name}' could not coerce value {value!r} to bool"
            )

        if isinstance(arg, StringCombo):
            coerced = str(value)
            if coerced not in arg.stringlist:
                raise ValueError(
                    f"Argument '{arg.name}' value '{coerced}' not in allowed options {arg.stringlist}"
                )
            return coerced

        if isinstance(arg, NumberValue):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Argument '{arg.name}' could not coerce value {value!r} to number"
                )
            if arg.ndecimals == 0:
                return int(round(numeric_value))
            return numeric_value

        raise TypeError(
            f"Unsupported argument object type for '{getattr(arg, 'name', '?')}': {type(arg).__name__}"
        )

    def _start_build(self):
        self.build()

    def _start_run(self, task=None):
        """
        start a single run
        """
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
        arguments = {name: arg.get() for name, arg in self.args.items()}
        mm_params = {name: arg.get() for name, arg in self.mm_args.items()}
        info["arguments"] = arguments
        info["mm_params"] = mm_params
        self._last_start_info = dict(info)
        self._dataq.send(info, channel=".start")

    def _publish_endvalues(self):
        info = {"_endtime": self._endtime}
        result_payloads = {}
        for name, channel in self.result_channels.items():
            payload = channel.push()
            if payload is not None:
                result_payloads[name] = payload
        self._dataq.send(info, channel=".end")

        if self._save_results_h5:
            self._save_measurement_h5(start_info=self._last_start_info, end_info=info, results=result_payloads)

    def _get_results_h5_dir(self) -> pathlib.Path:
        if isinstance(self._results_h5_dir, str) and self._results_h5_dir.strip():
            out_dir = pathlib.Path(self._results_h5_dir)
        else:
            base = self._props.get('/Servers/Imagepath', str(pathlib.Path.cwd() / 'Data'))
            out_dir = pathlib.Path(base) / 'measurements_h5'
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _sanitize_h5_key(key: str) -> str:
        return str(key).replace('/', '_').replace(' ', '_')

    def _write_h5_value(self, group, key: str, value):
        key = self._sanitize_h5_key(key)

        if isinstance(value, dict):
            sub = group.create_group(key)
            for k, v in value.items():
                self._write_h5_value(sub, str(k), v)
            return

        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
            return

        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.dtype == object:
                group.create_dataset(key, data=np.asarray([str(v) for v in value], dtype='S'))
            else:
                group.create_dataset(key, data=arr)
            return

        if isinstance(value, (np.integer, int, np.floating, float, np.bool_, bool)):
            group.attrs[key] = value
            return

        if isinstance(value, str):
            group.attrs[key] = value
            return

        if value is None:
            group.attrs[key] = 'None'
            return

        group.attrs[key] = str(value)

    def _save_measurement_h5(self, start_info: dict, end_info: dict, results: dict):
        try:
            import h5py
        except Exception as error:
            print_error(f"Could not import h5py for measurement save: {error}", 'error')
            return

        run = int(start_info.get('_run', self._run))
        rep = int(start_info.get('_repetition', getattr(self, '_repetition', self._rep)))
        task = int(start_info.get('_task', 0))
        filename = f"{self.name}_task{task}.h5"
        path = self._get_results_h5_dir() / filename

        try:
            with h5py.File(path, 'a') as f:
                f.attrs['experiment_name'] = self.name
                f.attrs['task'] = task

                base_group_name = f"run_{run:06d}_rep_{rep:04d}"
                group_name = base_group_name
                if group_name in f:
                    i = 1
                    while f"{base_group_name}_dup{i}" in f:
                        i += 1
                    group_name = f"{base_group_name}_dup{i}"

                g_run = f.create_group(group_name)

                g_start = g_run.create_group('start')
                for k, v in start_info.items():
                    self._write_h5_value(g_start, k, v)

                g_end = g_run.create_group('end')
                for k, v in end_info.items():
                    self._write_h5_value(g_end, k, v)

                g_res = g_run.create_group('results')
                for name, payload in results.items():
                    self._write_h5_value(g_res, name, payload)

                g_git = g_run.create_group('git')
                for k, v in self._get_git_repo_info().items():
                    self._write_h5_value(g_git, k, v)

            print_error(f"Saved measurement run to HDF5: {path} [{group_name}]", 'success')
        except Exception as error:
            print_error(f"Failed to save measurement HDF5 at {path}: {error}", 'error')

    @staticmethod
    def _run_git(repo_root: pathlib.Path, args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return ""
            return completed.stdout.strip()
        except Exception:
            return ""

    def _get_git_repo_info(self) -> dict:
        info = {
            "available": False,
            "repo_root": "",
            "branch": "",
            "commit": "",
            "commit_short": "",
            "dirty": False,
            "remote_origin": "",
        }

        try:
            start_path = pathlib.Path(__file__).resolve().parent
            top = self._run_git(start_path, ["rev-parse", "--show-toplevel"])
            if not top:
                return info

            repo_root = pathlib.Path(top)
            info["available"] = True
            info["repo_root"] = str(repo_root)
            info["branch"] = self._run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
            info["commit"] = self._run_git(repo_root, ["rev-parse", "HEAD"])
            info["commit_short"] = self._run_git(repo_root, ["rev-parse", "--short", "HEAD"])
            info["remote_origin"] = self._run_git(repo_root, ["config", "--get", "remote.origin.url"])

            status = self._run_git(repo_root, ["status", "--porcelain"])
            info["dirty"] = bool(status)
        except Exception:
            return info

        return info

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
            for name, param in self.mm_args.items():
                self._experiment_params[name] = param.get()

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
        """Register a MotMaster parameter as an argument object attribute."""
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
        self.mm_args[param_name] = param
        setattr(self, param_name, param)

    def setattr_argument(self, name, argtype, *args, **kwargs):
        """
        Create a user-controllable argument object attribute.
        The user will have an element in the GUI they can modify.

        Args:
            name: (str)
                Name of the argument. You can address the argument object via ``self.myname``
            argtype: (class)
                Currently only NumberValue is available

        Example::

            self.setattr_argument("Nr_Ablation_Pulses", NumberValue, ndecimals=0, step=1, value=20)
            self.Nr_Ablation_Pulses.get()
        """
        arg_inst = argtype(name, *args, **kwargs)
        self.args[name] = arg_inst
        setattr(self, name, arg_inst)
        
    def setattr_result(self, name, result_type: Optional[type] = None):
        """Set a result value to be published at the end of the run."""
        if result_type is None:
            result_type = ResultChannel
        channel = result_type(name, client=self._dataq if result_type is ResultChannel else self._imgq)
        setattr(self, name, channel)
        self.result_channels[name] = channel
        

class ResultChannel:
    """Descriptor for experiment result values that should be published to the data channel at the end of each run."""
    def __init__(self, name: str, client: DataClient):
        self.name = name
        self.client = client
        self.data = None
        
    def set(self, value):
        self.data = value

    def push(self):
        if self.data is None:
            print_error(f"Result '{self.name}' has no data set. Use set_data() to set the value before pushing.")
            return None
        payload = {self.name: self.data}
        self.client.send(payload, channel=".result")
        return payload
        
class ImageResultChannel(ResultChannel):
    """Descriptor for experiment result values that should be published to the image channel at the end of each run."""
    
    
    def push(self):
        if self.data is None:
            print_error(f"Image result '{self.name}' has no image data set.")
            return None
        self.client: ImageClient
        self.client.send(self.data, header={"name": self.name}, channel=".result")
        return {self.name: self.data}

        
        
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

    def get(self):
        return self.value


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

    def get(self):
        return self.value


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

    def get(self):
        return self.value


def get_rid():
    with open(PROPERTIES_FILE, "r") as f:
        properties = json.load(f)
    rid = properties["next_rid"]
    properties["next_rid"] += 1
    with open(PROPERTIES_FILE, "w") as f:
        json.dump(properties, f)
    return rid
