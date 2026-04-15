import argparse
import time
from typing import Any, Optional, Union
import json
import pathlib
import subprocess
import sys

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


def _resolve_config_file(config_file: Optional[str]) -> pathlib.Path:
    candidate = config_file or "rb_mm_config.json"
    path = pathlib.Path(candidate)
    if path.is_absolute():
        return path
    return CONFIG_DIR / path

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

    def connect(self) -> None:
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
                    import MOTMaster  # type: ignore

                    self.motmaster = Activator.GetObject(
                        MOTMaster.Controller, path_info["remote_path"]
                    )
                    print("Connected to MotMaster.")
                except Exception as e:
                    print(f"Error: {e} encountered")
            elif key == "caf_hardware_controller":
                self._add_ref(path_info["exe_path"])
                try:
                    import MoleculeMOTHadwareControl  # type: ignore

                    self.hardware_controller = Activator.GetObject(
                        MoleculeMOTHadwareControl.Controller, path_info["remote_path"]
                    )
                except Exception as e:
                    print(f"Error: {e} encountered")

    def disconnect(self) -> None:
        self.motmaster = None
        self.hardware_controller = None
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


    
class  DummyMotMasterInterface(MotMasterInterface):
    
    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        self.interval = interval
        self.motmaster = None
        self.hardware_controller = None
        self.script = None
        self.script_path = None
        self.parameter_dictionary = {}
    
    def connect(self) -> None:
        print("DummyMotMasterInterface: connect called.")
        self.motmaster = None
        self.script = None
        return None
    
    def set_motmaster_experiment(
        self,
        script: str,
    ):
        print(f"DummyMotMasterInterface: set_motmaster_experiment called with script '{script}'")
        self.script = script
        return None
    
    def set_motmaster_dictionary(self):
        print("DummyMotMasterInterface: set_motmaster_dictionary called")
        self.parameter_dictionary = {}
        return None
    
    def start_motmaster_experiment(
        self,
        parameters: Optional[dict] = None,
    ):
        print(f"DummyMotMasterInterface: start_motmaster_experiment called with script '{self.script}'")
        return None
    
    def get_params(self):
        return {"param1": 1, "param2": 2.0}

    def set_run_until_stopped(self, value: bool):
        print(f"DummyMotMasterInterface: set_run_until_stopped({value})")

    def set_iterations(self, iterations: int):
        print(f"DummyMotMasterInterface: set_iterations({iterations})")

    def set_save_toggle(self, save: bool):
        print(f"DummyMotMasterInterface: set_save_toggle({save})")

    def set_trigger_mode(self, value: bool):
        print(f"DummyMotMasterInterface: set_trigger_mode({value})")
    
    def disconnect(self) -> None:
        print("DummyMotMasterInterface: disconnect called")
        return None

    def save_pattern_info(self, save_folder, file_tag, task_nr):
        print(
            "DummyMotMasterInterface: save_pattern_info called "
            f"(save_folder={save_folder}, file_tag={file_tag}, task_nr={task_nr})"
        )
        return None


class MotMasterRPCService:
    def __init__(self, interface: MotMasterInterface) -> None:
        self.interface = interface

    def ping(self) -> dict[str, Any]:
        return {"ok": True, "status": "alive"}

    def get_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "script": self.interface.script,
            "connected": self.interface.motmaster is not None
            or isinstance(self.interface, DummyMotMasterInterface),
        }

    def set_script(self, script: str) -> dict[str, Any]:
        if not script:
            raise ValueError("'script' must be a non-empty string")
        self.interface.set_motmaster_experiment(script)
        return {"ok": True, "script": script}

    def get_params(self) -> dict[str, Any]:
        return {"ok": True, "params": self.interface.get_params()}

    def start_experiment(self, parameters: Optional[dict] = None) -> dict[str, Any]:
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a dictionary when provided")
        self.interface.start_motmaster_experiment(parameters=parameters)
        return {
            "ok": True,
            "script": self.interface.script,
            "completed": True,
            "has_parameters": parameters is not None,
        }

    def set_run_until_stopped(self, value: bool) -> dict[str, Any]:
        if not isinstance(value, bool):
            raise ValueError("'value' must be a boolean")
        self.interface.set_run_until_stopped(value)
        return {"ok": True, "value": value}

    def set_iterations(self, iterations: int) -> dict[str, Any]:
        if not isinstance(iterations, int):
            raise ValueError("'iterations' must be an integer")
        self.interface.set_iterations(iterations)
        return {"ok": True, "iterations": iterations}

    def set_save_toggle(self, save: bool) -> dict[str, Any]:
        if not isinstance(save, bool):
            raise ValueError("'save' must be a boolean")
        self.interface.set_save_toggle(save)
        return {"ok": True, "save": save}

    def set_trigger_mode(self, value: bool) -> dict[str, Any]:
        if not isinstance(value, bool):
            raise ValueError("'value' must be a boolean")
        self.interface.set_trigger_mode(value)
        return {"ok": True, "value": value}

    def save_pattern_info(
        self,
        save_folder: str,
        file_tag: str,
        task_nr: int,
    ) -> dict[str, Any]:
        if not isinstance(save_folder, str) or not save_folder:
            raise ValueError("'save_folder' must be a non-empty string")
        if not isinstance(file_tag, str) or not file_tag:
            raise ValueError("'file_tag' must be a non-empty string")
        if not isinstance(task_nr, int):
            raise ValueError("'task_nr' must be an integer")

        self.interface.save_pattern_info(save_folder, file_tag, task_nr)
        return {
            "ok": True,
            "save_folder": save_folder,
            "file_tag": file_tag,
            "task_nr": task_nr,
        }

    def call_interface(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        if not isinstance(method, str) or not method:
            raise ValueError("'method' must be a non-empty string")
        if method.startswith("_"):
            raise ValueError("Calling private methods is not allowed")
        target = getattr(self.interface, method, None)
        if target is None or not callable(target):
            raise ValueError(f"Interface method '{method}' not found")
        result = target(*args, **kwargs)
        return {"ok": True, "method": method, "result": result}

    def shutdown(self) -> dict[str, Any]:
        # simple_server_loop owns the serve loop; stop this process with ProcessManager.
        return {
            "ok": True,
            "message": "Shutdown not handled in-process for sipyco server; stop process externally.",
        }


def run_motmaster_rpc_server(
    host: str = "0.0.0.0",
    port: int = 5557,
    interval: Union[int, float] = 0.1,
    simulate: bool = False,
    config_file: Optional[str] = None,
) -> None:
    resolved_config_file = _resolve_config_file(config_file)

    if simulate:
        interface = DummyMotMasterInterface(interval=interval)
    else:
        interface = MotMasterInterface(
            config_file=str(resolved_config_file),
            interval=interval,
        )
    interface.connect()
    if (not simulate) and interface.motmaster is None:
        raise RuntimeError("Failed to connect to MotMaster.")

    rpc_service = MotMasterRPCService(interface=interface)
    print(f"MotMaster sipyco RPC server listening on {host}:{port}")

    try:
        simple_server_loop(
            {"motmaster": rpc_service},
            host=host,
            port=int(port),
            description="MotMaster sipyco RPC server",
        )
    finally:
        try:
            interface.disconnect()
        except Exception:
            pass


def _load_runtime_options_from_config(process_token: Optional[str]) -> dict:
    defaults = {
        "host": "0.0.0.0",
        "port": 5557,
        "simulate": False,
        "interval": 0.1,
        "config_file": "rb_mm_config.json",
    }

    try:
        conf = ConfigReader.getConfiguration()
        servers = conf.get("Servers", {})

        if isinstance(process_token, str) and process_token.startswith("Servers/"):
            process_name = process_token.split("/", 1)[1]
            entry = servers.get(process_name, {})
            if isinstance(entry, dict):
                defaults.update(
                    {
                        "host": entry.get("host", defaults["host"]),
                        "port": entry.get("port", defaults["port"]),
                        "simulate": entry.get("simulate", defaults["simulate"]),
                        "interval": entry.get("interval", defaults["interval"]),
                        "config_file": entry.get("config_file", defaults["config_file"]),
                    }
                )
            return defaults

        for _name, entry in servers.items():
            if not isinstance(entry, dict):
                continue
            script = str(entry.get("script", ""))
            if script.endswith("motmaster_server.py"):
                defaults.update(
                    {
                        "host": entry.get("host", defaults["host"]),
                        "port": entry.get("port", defaults["port"]),
                        "simulate": entry.get("simulate", defaults["simulate"]),
                        "interval": entry.get("interval", defaults["interval"]),
                        "config_file": entry.get("config_file", defaults["config_file"]),
                    }
                )
                break
    except Exception as error:
        print(f"Warning: failed to read runtime options from config: {error}")

    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotMaster sipyco RPC server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host interface to bind (use 0.0.0.0 for remote clients)",
    )
    parser.add_argument("--port", type=int, default=None, help="TCP port to bind")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run with dummy MotMaster interface for testing",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Delay after starting MotMaster experiment (seconds)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="MotMaster config file (absolute path or relative to pytweezer/configuration)",
    )
    parser.add_argument(
        "process_token",
        nargs="?",
        default=None,
        help="Optional process token passed by ProcessManager, e.g. Servers/Dummy MotMaster Server",
    )
    args, _unknown = parser.parse_known_args()

    options = _load_runtime_options_from_config(args.process_token)
    host = args.host if args.host is not None else options["host"]
    port = args.port if args.port is not None else int(options["port"])
    interval = args.interval if args.interval is not None else float(options["interval"])
    simulate = args.simulate or bool(options["simulate"])
    config_file = args.config_file if args.config_file is not None else options["config_file"]

    run_motmaster_rpc_server(
        host=host,
        port=port,
        interval=interval,
        simulate=simulate,
        config_file=config_file,
    )


if __name__ == "__main__":
    main()
    
