import argparse
import time
from typing import Any, Optional, Union
import json
import pathlib
import subprocess
import sys
from typing import Any, Callable, Dict, List, Tuple, Union

import pythonnet
import numbers
from sipyco.pc_rpc import simple_server_loop
from pytweezer.servers.configreader import ConfigReader

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System.Collections.Generic import Dictionary  # type: ignore
    from System import String, Object  # type: ignore
    from System import Activator   # type: ignore
    from System import Int32  # type: ignore
except Exception as e:
    print(f"Error: {e} encountered, probably no pythonnet")

# config directory local to the package.
CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configuration"

class MotMasterInterface:
    def __init__(self, config_file: str, interval: Union[int, float] = 0.1) -> None:
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.script_root = pathlib.Path(self.config["script_root_path"])
        self.interval = interval
        self.motmaster = None
        self.hardware_controller = None
        self.script = None
        self.script_path = None

    def _add_ref(self, path: str) -> None:
        _path = pathlib.Path(path)
        sys.path.append(_path.parent)
        clr.AddReference(path)
        return None

    def _is_process_running(self, process_name: str) -> bool:
        try:
            if sys.platform.startswith("win"):
                result = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                output = result.stdout.lower()
                return (
                    result.returncode == 0
                    and process_name.lower() in output
                    and "no tasks are running" not in output
                )

            result = subprocess.run(
                ["pgrep", "-f", process_name],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def _start_process(self, exe_path: str) -> None:
        subprocess.Popen(
            [exe_path],
            cwd=str(pathlib.Path(exe_path).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _ensure_motmaster_running(
        self,
        exe_path: str,
        startup_timeout: float = 15.0,
        poll_interval: float = 0.5,
    ) -> None:
        process_name = pathlib.Path(exe_path).name
        if self._is_process_running(process_name):
            return None

        print(f"MOTMaster process '{process_name}' not found. Starting it now...")
        self._start_process(exe_path)

        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._is_process_running(process_name):
                print(f"MOTMaster process '{process_name}' is running.")
                return None
            time.sleep(poll_interval)

        raise RuntimeError(
            f"Timed out waiting for process '{process_name}' to start from '{exe_path}'."
        )

    def connect(
        self
    ) -> None:
        for path in self.config["dll_paths"].values():
            clr.AddReference(path)
        for key, path_info in self.config.items():
            if key == "dll_paths":
                for path in path_info.values():
                    self._add_ref(path)
            elif key == "motmaster":
                self._ensure_motmaster_running(path_info["exe_path"])
                self._add_ref(path_info["exe_path"])
                try:
                    import MOTMaster
                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "cafbec_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import CaFBECHadwareController
                    self.hardware_controller = Activator.GetObject(
                        CaFBECHadwareController.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHardwareControl
                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHardwareControl.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "transfer_cavity_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import TransferCavityLock2012
                    self.transfer_cavity_lock = Activator.GetObject(
                        TransferCavityLock2012.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "wavemeter_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import WavemeterLock
                    self.wavemeter_lock = Activator.GetObject(
                        WavemeterLock.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "picomotor":
                if "connect" in path_info:
                    if path_info["connect"]:
                        self.picomotor_default_axis = None
                        self.picomotor_default_speed = None
                        self.picomotor_default_acceleration = None
                        self.picomotor_default_steps = None
                        self.picomotor_default_max_steps = None
                        self.picomotor_steps_moved = 0
                        try:
                            from pylablib.devices import Newport
                            n = Newport.get_usb_devices_number_picomotor()
                            if n == 1:
                                self.stage = Newport.Picomotor8742()
                                if "motor" in path_info["defaults"]:
                                    self.picomotor_default_motor = \
                                        path_info["defaults"]["motor"]
                                if "speed" in path_info["defaults"]:
                                    self.picomotor_default_speed = \
                                        path_info["defaults"]["speed"]
                                if "aceeleration" in path_info["defaults"]:
                                    self.picomotor_default_acceleration = \
                                        path_info["defaults"]["acceleration"]
                                if "steps" in path_info["defaults"]:
                                    self.picomotor_default_steps = \
                                        path_info["defaults"]["steps"]
                                if "max_steps" in path_info["defaults"]:
                                    self.picomotor_default_max_steps = \
                                        path_info["defaults"]["max_steps"]
                            elif n == 0:
                                print("No PicoMotor device detected!")
                            else:
                                print("Too many PicoMotor device detected!")
                        except Exception as e:
                            print(f"Error: {e} encountered")
        print(self.hardware_controller)
        self._ensure_motmaster_running
        return None

    def disconnect(self) -> None:
        self.motmaster = None
        self.hardware_controller = None
        if hasattr(self, "stage"):
            self.stage.close()
        return None

    def set_motmaster_experiment(
        self,
        script: str,
    ):
        self.script = script
        self.script_path = str(self.script_root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(self.script_path)
            print(f"MotMaster script set to {script}.")
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def get_motmaster_dictionary(self):
        self.parameter_dictionary = self.motmaster.GetParameters()
        # self.parameter_dictionary = Dictionary[String, Object]()
        # with open(DEFAULTS_FILE, "r") as f:
        #     default_parameters = json.load(f)
        # for key, value in default_parameters.items():
        #     self.parameter_dictionary[key] = value

    def python_to_cs_dict(self, parameters: dict):
        pars_csdict = Dictionary[String, Object]()
        for key, value in parameters.items():
            if isinstance(value, numbers.Integral):
                value = Int32(value)
            pars_csdict[key] = value
        return pars_csdict


    def start_motmaster_experiment(
        self, parameters: Optional[dict] = None
    ):
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        try:
            if parameters is not None:
                pars_csdict = self.python_to_cs_dict(parameters)
                self.motmaster.Go(pars_csdict)
            else:
                self.motmaster.Go()
            time.sleep(self.interval)
        except Exception as e:
            print(f"Error starting MotMaster experiment {self.script}: {e}")
        return None

    def get_params(self):
        return dict(self.motmaster.GetParameters())

    def get_params_csdict(self):
        return self.motmaster.GetParameters()

    def set_run_until_stopped(self, value: bool):
        self.motmaster.SetRunUntilStopped(value)
    
    def set_iterations(self, iterations: int):
        self.motmaster.SetIterations(iterations)

    def set_save_toggle(self, save: bool):
        self.motmaster.SaveToggle(save)

    def set_trigger_mode(self, value: bool):
        self.motmaster.SetTriggered(value)
        
    def save_pattern_info(self, save_folder, file_tag, task_nr):
        """Save pattern information to files. calls saveToFiles from MMDataIOHelper"""
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        self.motmaster.ioHelper.saveToFiles(file_tag, save_folder, task_nr, self.script_path)



    def _move_picomotor_with_default_settings(
        self
    ) -> None:
        if (self.picomotor_default_speed is not None) and \
                (self.picomotor_default_acceleration is not None):
            self.stage.setup_velocity(
                speed=self.picomotor_default_speed,
                accel=self.picomotor_default_acceleration
            )
        if (self.picomotor_default_axis is not None) and \
                (self.picomotor_default_steps is not None):
            motor: str = self.picomotor_default_motor
            self.stage.move_by(
                axis=self.config["picomotor"]["motor_to_axis"][motor],
                steps=self.picomotor_default_steps
            )
            self.stage.wait_move()
            self.picomotor_steps_moved += abs(self.picomotor_default_steps)
            if self.picomotor_steps_moved >= \
                    self.picomotor_default_max_steps:
                self.picomotor_default_steps *= -1.0
        return None

    def get_laser_set_points_tcl(
        self
    ) -> Dict[str, Dict[str, float]]:
        lasers = {}
        for laser, cavity in self.config["lasers"].items():
            voltage = self.transfer_cavity_lock.GetLaserVoltage(
                cavity, laser
            )
            set_point = self.transfer_cavity_lock.GetLaserSetpoint(
                cavity, laser
            )
            lasers[laser] = {"voltage": voltage, "set_point": set_point}
            print(
                f"{laser}: voltage = {voltage}, set_point = {set_point}"
            )
        return lasers

    def get_laser_set_points_wml(
        self
    ) -> Dict[str, Dict[str, float]]:
        lasers = {}
        for laser, _ in self.config["lasers"].items():
            set_point = self.wavemeter_lock.getSlaveFrequency(
                laser
            )
            lasers[laser] = {"set_point": set_point}
            print(
                f"{laser}: set_point = {set_point}"
            )
        return lasers

    def get_laser_frequencies_actual(
        self
    ) -> Dict[str, Dict[str, float]]:
        lasers = {}
        for laser, _ in self.config["lasers"].items():
            channel = int(self.wavemeter_lock.getChannelNum(laser))
            frequency = float(self.wavemeter_lock.getFrequency(channel))
            lasers[laser] = {"frequency": frequency}
            #print(f"{laser}: frequency = {frequency} THz")
        return lasers



    def scan_tcl_laser_set_points(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_value: Union[int, float] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        cavity = self.config["lasers"][laser]
        current_set_point = self.transfer_cavity_lock.GetLaserSetpoint(
            cavity, laser
        )
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            if move_yag_spot:
                self._move_picomotor_with_default_settings()
            if (motmaster_parameter and motmaster_value) is not None:
                _dictionary[motmaster_parameter] = motmaster_value
            while current_set_point > values[0]:
                current_set_point -= 0.001
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.001
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.transfer_cavity_lock.SetLaserSetpoint(
                    cavity, laser, values[i]
                )
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_value: Union[int, float] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            if move_yag_spot:
                self._move_picomotor_with_default_settings()
            if (motmaster_parameter and motmaster_value) is not None:
                _dictionary[motmaster_parameter] = motmaster_value
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                self.motmaster.Go(_dictionary)
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points_with_motmaster_values(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_values: List[Union[int, float]] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                if (motmaster_parameter and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        _dictionary[motmaster_parameter] = motmaster_values[k]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(values[i])
                            results.append(result)
                else:
                    self.motmaster.Go(_dictionary)
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(values[i])
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

    def scan_wm_laser_set_points_with_motmaster_multiple_parameters(
        self,
        script: str,
        laser: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None,
        motmaster_parameters: List[str] = None,
        motmaster_values: List[Tuple[Union[int, float]]] = None
    ) -> List[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(
            laser
        ))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(
                    laser, current_set_point
                )
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(
                    laser, values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                if (motmaster_parameters and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        motmaster_value: Tuple = motmaster_values[k]
                        for t, parameter in enumerate(motmaster_parameters):
                            _dictionary[parameter] = motmaster_value[t]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(values[i])
                            results.append(result)
                else:
                    self.motmaster.Go(_dictionary)
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(values[i])
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results



    def scan_microwave_amplitude(
        self,
        script: str,
        synthesizer: str = "Gigatronics Synthesizer 2",
        values: List[float] = [],
        move_yag_spot: bool = False
    ) -> None:
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetAmplitude(
                    values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_microwave_frequency(
        self,
        script: str,
        synthesizer: str = "Gigatronics Synthesizer 2",
        values: List[float] = [],
        move_yag_spot: bool = False
    ) -> None:
        path = str(self.root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetFrequency(
                    values[i]
                )
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")
        return None

    def scan_picomotor_steps(
        self,
        script: str,
        motor: str,
        interval_steps: int,
        total_steps: int,
        speed: int = None,
        accel: int = None,
        callback: Callable = None,
        motmaster_parameter: str = None,
        motmaster_values: List[Tuple[Union[int, float]]] = None
    ) -> List[Any]:
        path = str(self.root.joinpath(f"{script}.cs"))
        _dictionary = Dictionary[String, Object]()
        try:
            self.motmaster.SetScriptPath(path)
            results = []
            if (speed and accel) is not None:
                self.stage.setup_velocity(speed=speed, accel=accel)
            axis: int = self.config["picomotor"]["motor_to_axis"][motor]
            n_steps: int = int(total_steps/abs(interval_steps))
            for step_index in track(range(n_steps)):
                self.stage.move_by(axis=axis, steps=interval_steps)
                self.stage.wait_move()
                if (motmaster_parameter and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        _dictionary[motmaster_parameter] = motmaster_values[k]
                        self.motmaster.Go(_dictionary)
                        time.sleep(self.interval)
                        if callback is not None:
                            result = callback(step_index)
                            results.append(result)
                else:
                    self.motmaster.Go()
                    time.sleep(self.interval)
                    if callback is not None:
                        result = callback(step_index)
                        results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results
