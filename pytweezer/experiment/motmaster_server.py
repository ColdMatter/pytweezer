import argparse
import time
from typing import Optional, Union
import json
import pathlib
import subprocess
import sys

from sipyco.pc_rpc import simple_server_loop
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers.simulated_device import simulate
from pytweezer.experiment.motmaster_interface import MotMasterInterface
# from pycaf.experiment import Experiment

# config directory local to the package.
CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configuration"

#: Methods where returning bare None would be a poor fake (e.g. scan_* methods
#: that real callers iterate over, or dict-returning parameter lookups).
_MOTMASTER_SIMULATE_DEFAULTS = {
    "get_motmaster_dictionary": dict,
    "get_params_csdict": dict,
    "get_laser_set_points_tcl": dict,
    "get_laser_set_points_wml": dict,
    "get_laser_frequencies_actual": dict,
    "python_to_cs_dict": dict,
    "scan_tcl_laser_set_points": list,
    "scan_wm_laser_set_points": list,
    "scan_wm_laser_set_points_with_motmaster_values": list,
    "scan_wm_laser_set_points_with_motmaster_multiple_parameters": list,
    "scan_picomotor_steps": list,
}


@simulate(MotMasterInterface, defaults=_MOTMASTER_SIMULATE_DEFAULTS)
class SimulatedMotMasterInterface:
    """Simulated MotMaster backend. Any MotMasterInterface method not defined
    here is auto-stubbed by @simulate so this class can never silently drift
    out of sync with it (unlike its predecessor, this class does not subclass
    MotMasterInterface, so an un-overridden method never falls through to
    real hardware-touching code).
    """

    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        self.interval = interval
        self.motmaster = None
        self.script = None

    def connect(self) -> None:
        print("SimulatedMotMasterInterface: connect called.")
        self.motmaster = None
        self.script = None
        return None

    def set_motmaster_experiment(
        self,
        script: str,
    ):
        print(
            f"SimulatedMotMasterInterface: set_motmaster_experiment called with script '{script}'"
        )
        self.script = script
        return None

    def start_motmaster_experiment(
        self,
        parameters: Optional[dict] = None,
    ):
        print(
            f"SimulatedMotMasterInterface: start_motmaster_experiment called with script '{self.script}'"
        )
        return None

    def get_params(self):
        return {"param1": 1, "param2": 2.0}

    def disconnect(self) -> None:
        print("SimulatedMotMasterInterface: disconnect called")
        return None

    def save_pattern_info(self, save_folder, file_tag, task_nr):
        print(
            "SimulatedMotMasterInterface: save_pattern_info called "
            f"(save_folder={save_folder}, file_tag={file_tag}, task_nr={task_nr})"
        )
        return None


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


def run_motmaster_command_server(
    host: str = "0.0.0.0",
    port: int = 5557,
    interval: Union[int, float] = 0.1,
    simulate: bool = False,
    config_file: Optional[str] = None,
) -> None:
    _ensure_motmaster_running(config_file)
    if simulate:
        interface = SimulatedMotMasterInterface(interval=interval)
    else:
        interface = MotMasterInterface(config_file, interval=interval)
    interface.connect()
    if (not simulate) and interface.motmaster is None:
        raise RuntimeError("Failed to connect to MotMaster.")

    try:
        simple_server_loop(
            {"motmaster": interface},
            host=host,
            port=port,
            description="MotMaster command server",
        )
    finally:
        try:
            interface.disconnect()
        except Exception:
            # Connection teardown should not mask shutdown.
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotMaster sipyco RPC server")
    parser.add_argument(
        "--name", nargs="?", default=None, help="name of this program instance"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (use 0.0.0.0 for remote clients)",
    )
    parser.add_argument("--port", type=int, default=8888, help="TCP port to bind")
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

    args, _unknown = parser.parse_known_args()

    from pytweezer.configuration.config import CONFIG
    from pytweezer.servers import tweezerpath
    
    if args.name is not None:
        config_dict = CONFIG["Devices"][args.name]
        host = config_dict["host"]
        port = config_dict["port"]
        simulate = config_dict["simulate"]
        interval = config_dict["interval"]
    else:
        host = args.host
        port = args.port
        simulate = args.simulate
        interval = args.interval
    config_file = (
        tweezerpath + "/pytweezer/configuration/" + config_dict.get("config_file")
    )

    run_motmaster_command_server(
        host=host,
        port=port,
        interval=interval,
        simulate=simulate,
        config_file=config_file,
    )


if __name__ == "__main__":
    main()
