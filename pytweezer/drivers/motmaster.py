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
from collections.abc import Callable, Sequence
from typing import Any

from rich.progress import track

from pytweezer.servers.simulated_device import simulate

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System import (  # type: ignore
        Activator,  # type: ignore
        Int32,  # type: ignore
        Object,
        String,
    )
    from System.Collections.Generic import Dictionary  # type: ignore
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
        return

    print(f"MOTMaster process '{process_name}' not found. Starting it now...")
    _start_process(exe_path)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _is_process_running(process_name):
            print(f"MOTMaster process '{process_name}' is running.")
            return
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Timed out waiting for process '{process_name}' to start from '{exe_path}'."
    )


_referenced: set[str] = set()


def _add_reference(path: str) -> None:
    # Once per process: a poller that rebuilds its proxy must not grow sys.path.
    if path in _referenced:
        return
    sys.path.append(pathlib.Path(path).parent)
    clr.AddReference(path)
    _referenced.add(path)


def motmaster_controller(config_file: str) -> Any:
    """The remoting proxy for the MOTMaster ``Controller`` named by ``config_file``.

    ``config_file`` is resolved against ``CONFIG_DIR`` as `MotMasterInterface`
    does. The references are added once per process. Building the proxy does not start MOTMaster or contact it; the first
    call through the proxy does.

    Args:
        config_file: JSON config naming the ``"motmaster"`` entry and ``dll_paths``.

    Returns:
        The ``MOTMaster.Controller`` proxy.
    """
    with open(CONFIG_DIR / config_file, "r") as f:
        config = json.load(f)
    for path in config["dll_paths"].values():
        _add_reference(path)
    motmaster = config["motmaster"]
    _add_reference(motmaster["exe_path"])
    import MOTMaster  # type: ignore

    return Activator.GetObject(MOTMaster.Controller, motmaster["remote_path"])


class MotMasterInterface:
    def __init__(self, config_file: str, interval: float = 0.1) -> None:
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
        self.linked = False
        self.connect()
        if self.motmaster is None:
            raise RuntimeError("Failed to connect to MotMaster.")

    def _add_ref(self, path: str) -> None:
        _add_reference(path)

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
                    import MOTMaster  # type: ignore

                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "cafbec_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import CaFBECHadwareController  # type: ignore

                    self.hardware_controller = Activator.GetObject(
                        CaFBECHadwareController.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHardwareControl  # type: ignore

                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHardwareControl.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "transfer_cavity_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import TransferCavityLock2012  # type: ignore

                    self.transfer_cavity_lock = Activator.GetObject(
                        TransferCavityLock2012.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "wavemeter_lock":
                self._add_ref(path_info["exe_path"])
                try:
                    import WavemeterLock  # type: ignore

                    self.wavemeter_lock = Activator.GetObject(
                        WavemeterLock.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "picomotor" and path_info.get("connect"):
                self.picomotor_default_motor = None
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
                            self.picomotor_default_motor = path_info["defaults"][
                                "motor"
                            ]
                        if "speed" in path_info["defaults"]:
                            self.picomotor_default_speed = path_info["defaults"][
                                "speed"
                            ]
                        if "acceleration" in path_info["defaults"]:
                            self.picomotor_default_acceleration = path_info["defaults"][
                                "acceleration"
                            ]
                        if "steps" in path_info["defaults"]:
                            self.picomotor_default_steps = path_info["defaults"][
                                "steps"
                            ]
                        if "max_steps" in path_info["defaults"]:
                            self.picomotor_default_max_steps = path_info["defaults"][
                                "max_steps"
                            ]
                    elif n == 0:
                        print("No PicoMotor device detected!")
                    else:
                        print("Too many PicoMotor device detected!")
                except Exception as e:
                    print(f"Error: {e} encountered")
        print(self.hardware_controller)

    def disconnect(self) -> None:
        self.motmaster = None
        self.hardware_controller = None
        if hasattr(self, "stage"):
            self.stage.close()

    def set_motmaster_experiment(
        self,
        script: str,
    ):
        script_path = str(self.script_root.joinpath(f"{script}.cs"))
        self.motmaster.SetScriptPath(script_path)
        self.script = script
        self.script_path = script_path

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

    def set_linked(self, value: bool) -> None:
        """Turn MOTMaster's linked mode on or off.

        Linked mode lets one script on this MOTMaster (the primary) drive a second
        MOTMaster too, which it claims over .NET remoting and fires by TTL. Only a
        MOTMaster whose Hardware class declares a ``MOTMasterLinkConfig`` can be a
        primary; turning it on anywhere else raises. While linked, every ``Go``
        drives both machines, and any failure on either raises here rather than
        ending in a MessageBox.

        Turning it on when already linked keeps the existing claim. Turning it off
        does nothing on a MOTMaster that is not linked, including a build that
        predates linked mode.
        """
        if not value and not hasattr(self.motmaster, "SetLinked"):
            self.linked = False
            return
        if not (value and self.motmaster.IsLinked):
            self.motmaster.SetLinked(value)
        self.linked = value

    def is_linked(self) -> bool:
        """Whether MOTMaster itself reports linked mode on (it drops it if the
        secondary takes back control)."""
        return bool(self.motmaster.IsLinked)

    def start_motmaster_experiment(self, parameters: dict | None = None):
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        # MOTMaster turns linked mode off by itself when the secondary takes back
        # control; a linked script must then not run on the primary alone.
        if self.linked and not self.is_linked():
            self.linked = False
            raise RuntimeError(
                "MOTMaster is no longer linked (the secondary has probably taken "
                "back control); the linked script was not run. Call "
                "set_linked(True) to claim the secondary again."
            )
        if parameters is not None:
            self.motmaster.Go(self.python_to_cs_dict(parameters))
        else:
            self.motmaster.Go()
        time.sleep(self.interval)

    def get_params(self):
        return dict(self.motmaster.GetParameters())

    def get_params_csdict(self):
        return self.motmaster.GetParameters()

    def script_parameters(self, script: str) -> dict[str, Any]:
        """Return ``{name: default}`` for ``script``, leaving the selected script unchanged.

        Must not run during a sequence: it briefly changes MOTMaster's script path.
        """
        previous = self.script
        self.set_motmaster_experiment(script)
        try:
            return self.get_params()
        finally:
            if previous is not None:
                self.set_motmaster_experiment(previous)
            else:
                self.script = self.script_path = None

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
        self.motmaster.ioHelper.saveToFiles(
            file_tag, save_folder, task_nr, self.script_path
        )

    def _move_picomotor_with_default_settings(self) -> None:
        if (self.picomotor_default_speed is not None) and (
            self.picomotor_default_acceleration is not None
        ):
            self.stage.setup_velocity(
                speed=self.picomotor_default_speed,
                accel=self.picomotor_default_acceleration,
            )
        if (self.picomotor_default_motor is not None) and (
            self.picomotor_default_steps is not None
        ):
            motor: str = self.picomotor_default_motor
            self.stage.move_by(
                axis=self.config["picomotor"]["motor_to_axis"][motor],
                steps=self.picomotor_default_steps,
            )
            self.stage.wait_move()
            self.picomotor_steps_moved += abs(self.picomotor_default_steps)
            if self.picomotor_steps_moved >= self.picomotor_default_max_steps:
                self.picomotor_default_steps = -self.picomotor_default_steps
                self.picomotor_steps_moved = 0

    def get_laser_set_points_tcl(self) -> dict[str, dict[str, float]]:
        lasers = {}
        for laser, cavity in self.config["lasers"].items():
            voltage = self.transfer_cavity_lock.GetLaserVoltage(cavity, laser)
            set_point = self.transfer_cavity_lock.GetLaserSetpoint(cavity, laser)
            lasers[laser] = {"voltage": voltage, "set_point": set_point}
            print(f"{laser}: voltage = {voltage}, set_point = {set_point}")
        return lasers

    def get_laser_set_points_wml(self) -> dict[str, dict[str, float]]:
        lasers = {}
        for laser in self.config["lasers"]:
            set_point = self.wavemeter_lock.getSlaveFrequency(laser)
            lasers[laser] = {"set_point": set_point}
            print(f"{laser}: set_point = {set_point}")
        return lasers

    def laser_set_points_wml(self) -> dict[str, float]:
        """``{laser: wavemeter-lock set point in THz}`` for every configured laser."""
        return {
            laser: float(self.wavemeter_lock.getSlaveFrequency(laser))
            for laser in self.config["lasers"]
        }

    def set_laser_set_point_wml(self, laser: str, value: float) -> None:
        self.wavemeter_lock.setSlaveFrequency(laser, float(value))

    def ramp_laser_set_point_wml(
        self, laser: str, target: float, step: float = 1e-5, pause_s: float = 0.05
    ) -> None:
        """Walk ``laser``'s set point to ``target`` in ``step`` THz so the lock follows."""
        start = float(self.wavemeter_lock.getSlaveFrequency(laser))
        direction = 1.0 if target > start else -1.0
        for i in range(1, int(abs(target - start) // step) + 1):
            self.wavemeter_lock.setSlaveFrequency(laser, start + direction * i * step)
            time.sleep(pause_s)
        self.wavemeter_lock.setSlaveFrequency(laser, float(target))

    def get_laser_frequencies_actual(self) -> dict[str, dict[str, float]]:
        lasers = {}
        for laser in self.config["lasers"]:
            channel = int(self.wavemeter_lock.getChannelNum(laser))
            frequency = float(self.wavemeter_lock.getFrequency(channel))
            lasers[laser] = {"frequency": frequency}
            # print(f"{laser}: frequency = {frequency} THz")
        return lasers

    def scan_motmaster_parameter(
        self,
        script: str,
        parameter: str,
        values: list[int | float],
        move_yag_spot: bool = False,
        callback: Callable | None = None,
    ) -> list[Any]:
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
                self.motmaster.Go(self.python_to_cs_dict({parameter: values[i]}))
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
        values: list[int | float],
        move_yag_spot: bool = False,
        callback: Callable | None = None,
        motmaster_parameter: str | None = None,
        motmaster_value: float | None = None,
    ) -> list[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.script_root.joinpath(f"{script}.cs"))
        cavity = self.config["lasers"][laser]
        current_set_point = self.transfer_cavity_lock.GetLaserSetpoint(cavity, laser)
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
                self.transfer_cavity_lock.SetLaserSetpoint(cavity, laser, values[i])
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
        values: list[int | float],
        move_yag_spot: bool = False,
        callback: Callable | None = None,
        motmaster_parameter: str | None = None,
        motmaster_value: float | None = None,
    ) -> list[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.script_root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(laser))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            if move_yag_spot:
                self._move_picomotor_with_default_settings()
            if (motmaster_parameter and motmaster_value) is not None:
                _dictionary[motmaster_parameter] = motmaster_value
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(laser, values[i])
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
        values: list[int | float],
        move_yag_spot: bool = False,
        callback: Callable | None = None,
        motmaster_parameter: str | None = None,
        motmaster_values: list[int | float] | None = None,
    ) -> list[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.script_root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(laser))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(laser, values[i])
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
        values: list[int | float],
        move_yag_spot: bool = False,
        callback: Callable | None = None,
        motmaster_parameters: list[str] | None = None,
        motmaster_values: list[tuple[int | float]] | None = None,
    ) -> list[Any]:
        _dictionary = Dictionary[String, Object]()
        path = str(self.script_root.joinpath(f"{script}.cs"))
        current_set_point = float(self.wavemeter_lock.getSlaveFrequency(laser))
        results = []
        try:
            self.motmaster.SetScriptPath(path)
            while current_set_point > values[0]:
                current_set_point -= 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.05)
            while current_set_point < values[0]:
                current_set_point += 0.00001
                self.wavemeter_lock.setSlaveFrequency(laser, current_set_point)
                time.sleep(0.1)
            for i in track(range(len(values))):
                self.wavemeter_lock.setSlaveFrequency(laser, values[i])
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                if (motmaster_parameters and motmaster_values) is not None:
                    for k in range(len(motmaster_values)):
                        motmaster_value: tuple = motmaster_values[k]
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
        values: Sequence[float] = (),
        move_yag_spot: bool = False,
    ) -> None:
        path = str(self.script_root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetAmplitude(values[i])
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")

    def scan_microwave_frequency(
        self,
        script: str,
        synthesizer: str = "Gigatronics Synthesizer 2",
        values: Sequence[float] = (),
        move_yag_spot: bool = False,
    ) -> None:
        path = str(self.script_root.joinpath(f"{script}.cs"))
        try:
            self.motmaster.SetScriptPath(path)
            for i in track(range(len(values))):
                self.hardware_controller.tabs[synthesizer].SetFrequency(values[i])
                if move_yag_spot:
                    self._move_picomotor_with_default_settings()
                self.motmaster.Go()
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error: {e} encountered")

    def scan_picomotor_steps(
        self,
        script: str,
        motor: str,
        interval_steps: int,
        total_steps: int,
        speed: int | None = None,
        accel: int | None = None,
        callback: Callable | None = None,
        motmaster_parameter: str | None = None,
        motmaster_values: list[tuple[int | float]] | None = None,
    ) -> list[Any]:
        path = str(self.script_root.joinpath(f"{script}.cs"))
        _dictionary = Dictionary[String, Object]()
        try:
            self.motmaster.SetScriptPath(path)
            results = []
            if (speed and accel) is not None:
                self.stage.setup_velocity(speed=speed, accel=accel)
            axis: int = self.config["picomotor"]["motor_to_axis"][motor]
            n_steps: int = int(total_steps / abs(interval_steps))
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


SIMULATED_SCRIPT_PARAMETERS: dict[str, dict[str, Any]] = {
    "RbTweezerBasic": {"tDelay1": 5, "tPulse": 20e-6, "coil_current": 1.5, "nShots": 1},
    "CaFTweezerLoad": {"tLoad": 100, "bTop": 0.5, "tHold": 10e-3},
    "LinkedScopeTest": {
        "PatternLength": 20000,
        "PulseStart": 1000,
        "PulseDuration": 1000,
        "SecondaryTriggerDelay": 0,
    },
    "RbTweezerBasic2026_2": {
        **dict.fromkeys(
            [
                "tMOTLoadRepVCO",
                "tMOTLoadCoolVCO",
                "tMOTLoadRepVVA",
                "tMOTLoadCoolVVA",
                "tMolCoolVCO1",
                "tMolRepVCO1",
                "tMolCoolVVA1",
                "tMolRepVVA1",
                "tMolCoolVCO2",
                "tMolRepVCO2",
                "tMolCoolVVA2",
                "tMolRepVVA2",
                "tImgCoolVCO1",
                "tImgCoolVVA1",
                "tImgCoolVCO2",
                "tImgCoolVVA2",
                "tMOTShimXccValue",
                "tMOTShimYccValue",
                "tMOTShimZccValue",
                "tMolassesShimXccValue",
                "tMolassesShimYccValue",
                "tMolassesShimZccValue",
                "tTweezerSetVVA0",
                "tTweezerSetVVA1",
            ],
            0.0,
        ),
        "BackgroundDrop": 0,
        "tDelay2": 100,
        "tDropDuration": 4,
    },
}
_SIMULATED_FALLBACK = {"tDelay1": 5, "tPulse": 20e-6, "coil_current": 1.5}
SIMULATED_LASER_SET_POINTS = {
    "v0": 494.431874,
    "BX": 564.582340,
    "v1": 476.958908,
    "v2": 477.299380,
    "v3": 477.627302,
    "TCool": 494.431874,
}


@simulate(MotMasterInterface)
class SimulatedMotMasterInterface:
    """MOTMaster stand-in with canned script parameters that rejects what the real one would.

    Scripts missing from :data:`SIMULATED_SCRIPT_PARAMETERS` get a fallback parameter set.
    """

    #: Seconds a simulated run takes, so simulated cameras catch some frames.
    sim_run_s = 0.5

    def __init__(self, *args, **kwargs):
        self.script = None
        self.linked = False
        self.set_points = dict(SIMULATED_LASER_SET_POINTS)

    def set_linked(self, value: bool) -> None:
        self.linked = value

    def is_linked(self) -> bool:
        return self.linked

    def laser_set_points_wml(self) -> dict[str, float]:
        return dict(self.set_points)

    def set_laser_set_point_wml(self, laser: str, value: float) -> None:
        self.set_points[laser] = float(value)

    def ramp_laser_set_point_wml(
        self, laser: str, target: float, step: float = 1e-5, pause_s: float = 0.05
    ) -> None:
        self.set_points[laser] = float(target)

    def script_parameters(self, script: str) -> dict[str, Any]:
        return dict(SIMULATED_SCRIPT_PARAMETERS.get(script, _SIMULATED_FALLBACK))

    def set_motmaster_experiment(self, script: str):
        self.script = script

    def get_params(self) -> dict[str, Any]:
        if self.script is None:
            raise ValueError("MotMaster script not set")
        return self.script_parameters(self.script)

    def start_motmaster_experiment(self, parameters: dict | None = None):
        known = self.get_params()
        for name, value in (parameters or {}).items():
            if name not in known:
                raise KeyError(f"script {self.script!r} has no parameter {name!r}")
            if isinstance(known[name], int) and type(value) is not int:
                raise TypeError(
                    f"{name!r} is an Int32 in script {self.script!r}; got {value!r}"
                )
        time.sleep(self.sim_run_s)
