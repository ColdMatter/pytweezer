"""Driver for a MOTMaster experiment sequencer.

MOTMaster is a .NET application reached over .NET remoting through pythonnet, so
this module talks to it via `clr` rather than a Python instrument library. A
`MotMasterInterface` is constructed with the name of a JSON config file in the
package `configuration/` dir naming the MOTMaster executable, its remoting URL,
the `.cs` script root, and the DLLs to load; constructing one starts
`MOTMaster.exe` if it isn't already running and connects, so a constructed
interface is a connected one.
"""

import json
import numbers
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rich.progress import track

from pytweezer.servers.device_client import get_device

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


def _is_process_running(process_name: str) -> bool:
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


def _start_process(exe_path: str) -> None:
    subprocess.Popen(
        [exe_path],
        cwd=str(pathlib.Path(exe_path).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_motmaster_running(
    config_file: str,
    startup_timeout: float = 15.0,
    poll_interval: float = 0.5,
) -> None:
    """Start the MOTMaster application named by ``config_file`` unless it is
    already running, and block until it appears in the process table."""
    with open(config_file, "r") as f:
        config = json.load(f)
    exe_path = config["motmaster"]["exe_path"]
    process_name = pathlib.Path(exe_path).name
    if _is_process_running(process_name):
        return None

    print(f"MOTMaster process '{process_name}' not found. Starting it now...")
    _start_process(exe_path)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _is_process_running(process_name):
            print(f"MOTMaster process '{process_name}' is running.")
            return None
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Timed out waiting for process '{process_name}' to start from '{exe_path}'."
    )


class MotMasterInterface:
    def __init__(self, config_file: str, interval: Union[int, float] = 0.1) -> None:
        # ``config_file`` names a JSON file in the package ``configuration/`` dir; an
        # absolute path is taken as-is. Bringing the device up — making sure the
        # MOTMaster application is running, then connecting — happens here, so a
        # constructed interface is a connected one (as with the other drivers).
        config_path = CONFIG_DIR / config_file
        _ensure_motmaster_running(str(config_path))
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.script_root = pathlib.Path(self.config["script_root_path"])
        self.interval = interval
        self.motmaster = None
        self.hardware_controller = None
        self.script = None
        self.script_path = None
        self.connect()
        if self.motmaster is None:
            raise RuntimeError("Failed to connect to MotMaster.")

    def _add_ref(self, path: str) -> None:
        _path = pathlib.Path(path)
        sys.path.append(_path.parent)
        clr.AddReference(path)
        return None


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
                self._add_ref(path_info["exe_path"])
                try:
                    import MOTMaster # type: ignore
                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "cafbec_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import CaFBECHadwareController  # type: ignore
                    self.hardware_controller = Activator.GetObject(
                        CaFBECHadwareController.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHardwareControl  # type: ignore
                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHardwareControl.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "transfer_cavity_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import TransferCavityLock2012  # type: ignore
                    self.transfer_cavity_lock = Activator.GetObject(
                        TransferCavityLock2012.Controller,
                        path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "wavemeter_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import WavemeterLock  # type: ignore
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



    def scan_motmaster_parameter(
        self,
        script: str,
        parameter: str,
        values: List[Union[int, float]],
        move_yag_spot: bool = False,
        callback: Callable = None
    ) -> List[Any]:
        """Run ``script`` once per entry in ``values``, with the MOTMaster
        parameter ``parameter`` set to that entry. ``callback`` is called with
        the value after each shot and its return appended to the results."""
        path = str(self.script_root.joinpath(f"{script}.cs"))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go(
                    self.python_to_cs_dict({parameter: values[i]})
                )
                time.sleep(self.interval)
                if callback is not None:
                    result = callback(values[i])
                    results.append(result)
        except Exception as e:
            print(f"Error: {e} encountered")
        return results

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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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
        path = str(self.script_root.joinpath(f"{script}.cs"))
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

    def auto_mot(
        self,
        scan_ranges: Dict[str, Sequence[float]],
        camera: Union[str, Any],
        shots_per_point: int = 1,
        script: str = "AMOTBasic",
        field_parameter: str = "MOTCoilsCurrentValue",
        bg_field_value: float = 0.0,
        mot_field_value: float = 1.0,
        display_results: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Tune the MOT by scanning wavemeter-lock set points laser by laser.

        A background is taken first with the MOT coils at ``bg_field_value``, then
        each laser named in ``scan_ranges`` is stepped over its set points (coils at
        ``mot_field_value``) while ``shots_per_point`` fluorescence frames are read
        off ``camera``. Background-subtracted counts pick the set point with the most
        atoms, and the laser is left parked there before moving on to the next one —
        so lasers are optimised in the order ``scan_ranges`` gives.

        ``scan_ranges`` maps laser name (a key of the config's ``lasers`` block) to
        the slave frequencies to try, in THz. ``camera`` is a device name resolved with
        :func:`~pytweezer.servers.device_client.get_device`, or an already-connected
        camera (a client, or the object itself for a camera in this process); a
        client opened here is closed again on the way out. **Configure the camera
        before calling** — exposure, ROI, external trigger, and a
        ``setup_acquisition`` buffer holding at least the whole scan. ``auto_mot``
        arms it once and then reads consecutive frames, so the camera must capture
        exactly ``shots_per_point`` frames per MOTMaster shot.

        Returns, per laser, the scan range, mean and standard-error counts at each
        point, the chosen set point and the set point the laser started from; pass
        that dict to :func:`plot_auto_mot_results` to see the scans.
        """
        client = get_device(camera) if isinstance(camera, str) else camera
        try:
            results = self._auto_mot(
                scan_ranges, client, int(shots_per_point), script,
                field_parameter, bg_field_value, mot_field_value
            )
        finally:
            try:
                client.stop_acquisition()
            finally:
                if isinstance(camera, str):
                    client.close_rpc()
        if display_results:
            plot_auto_mot_results(results)
        return results

    def _auto_mot(
        self,
        scan_ranges: Dict[str, Sequence[float]],
        camera: Any,
        shots_per_point: int,
        script: str,
        field_parameter: str,
        bg_field_value: float,
        mot_field_value: float
    ) -> Dict[str, Dict[str, Any]]:
        """The body of :meth:`auto_mot`, with the camera already connected."""
        # Frames are numbered from the arming below and read in consecutive
        # windows, so every shot's frames are the ones that shot produced.
        frames_read = 0

        def grab(value: float) -> Tuple[float, np.ndarray]:
            nonlocal frames_read
            images = camera.acquire_n_frames(shots_per_point, start_frame=frames_read)
            frames_read += shots_per_point
            return value, np.asarray(images, dtype=float)

        camera.start_acquisition()
        background = self.scan_motmaster_parameter(
            script,
            field_parameter,
            [bg_field_value],
            False,
            grab
        )
        if not background or background[0][1].size == 0:
            raise RuntimeError(
                f"The camera returned no frames for the background shot — check "
                f"that {script} triggers it {shots_per_point} time(s) per run."
            )
        images_bg = np.mean(background[0][1], axis=0)
        time.sleep(1)

        results: Dict[str, Dict[str, Any]] = {}
        for laser, scan_range in scan_ranges.items():
            if laser not in self.config["lasers"]:
                raise KeyError(
                    f"'{laser}' is not a laser in this MOTMaster config; known "
                    f"lasers are {sorted(self.config['lasers'])}."
                )
            scan_range = [float(value) for value in scan_range]
            # Where the laser sat before this scan: the reference the results are
            # plotted as a detuning from.
            start_set_point = float(self.wavemeter_lock.getSlaveFrequency(laser))
            shots = self.scan_wm_laser_set_points(
                script,
                laser,
                scan_range,
                False,
                grab,
                field_parameter,
                mot_field_value
            )
            if len(shots) != len(scan_range):
                raise RuntimeError(
                    f"Scan of {laser} returned {len(shots)} of "
                    f"{len(scan_range)} points; the scan did not complete."
                )
            numbers, errors = [], []
            for _, images in shots:
                counts = np.sum(images - images_bg, axis=(1, 2))
                numbers.append(float(np.mean(counts)))
                errors.append(float(np.std(counts) / np.sqrt(len(counts))))
            set_point = scan_range[int(np.argmax(numbers))]
            # Leave the laser on its best set point before optimising the next.
            self.scan_wm_laser_set_points(
                script,
                laser,
                [set_point],
                False,
                grab,
                field_parameter,
                mot_field_value
            )
            results[laser] = {
                "scan_range": scan_range,
                "numbers": numbers,
                "errors": errors,
                "set_point": set_point,
                "start_set_point": start_set_point,
            }
        return results


def plot_auto_mot_results(results: Dict[str, Dict[str, Any]]):
    """Plot the scans returned by :meth:`MotMasterInterface.auto_mot`: one panel per
    laser, normalised number against detuning from the set point the laser started
    on, with an arrow marking the set point chosen. Returns the ``(figure, axes)``
    pair."""
    import matplotlib.pyplot as plt

    n_cols = min(3, max(1, len(results)))
    n_rows = max(1, -(-len(results) // n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True,
        squeeze=False
    )
    flat_axes = axes.flatten()
    for ax, (laser, scan) in zip(flat_axes, results.items()):
        numbers = np.array(scan["numbers"])
        errors = np.array(scan["errors"])
        scale = np.max(numbers) if np.max(numbers) > 0 else 1.0
        # THz set points shown as MHz detunings from where the laser started.
        start = scan["start_set_point"]
        detunings = (np.array(scan["scan_range"]) - start) * 1e6
        ax.errorbar(detunings, numbers / scale, yerr=errors / scale, fmt="ok")
        ax.arrow(
            (scan["set_point"] - start) * 1e6, 1.15, 0, -0.1,
            head_width=2.0, head_length=0.03, width=1.0, fc="r", ec="r"
        )
        ax.set_title(f"{laser} set @ {scan['set_point']} THz")
        ax.set_xlabel("Detuning to previous set point [MHz]")
        ax.set_ylim((0, 1.2))
    for ax in flat_axes[len(results):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("Norm. number")
    fig.tight_layout()
    return fig, axes