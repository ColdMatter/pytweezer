import argparse
import time
from typing import Optional, Union
import json
import pathlib
import sys
import threading

import numpy as np
import zmq

# NOTE: these imports will only work with the pythonnet package
try:
    import clr
    from System.Collections.Generic import Dictionary
    from System import String, Object
    from System import Activator
except Exception as e:
    print(f"Error: {e} encountered, probably no pythonnet")

# config dir is relative to the root of the repository, which is added to the path when pytweezer is installed, so should be findable from anywhere in the code using a relative path from the root. If this becomes an issue we can add some code to find the config dir based on the location of this file.
CONFIG_DIR = "pytweezer/configuration"
PROPERTIES_FILE = CONFIG_DIR + "/configuration.json"
DEFAULTS_FILE = CONFIG_DIR + "/defaults.json"
EXPERIMENTS_FILE = CONFIG_DIR + "/experiments.json"


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

    def get_motmaster_dictionary(self):
        self.parameter_dictionary = self.motmaster.GetParameters()
        # self.parameter_dictionary = Dictionary[String, Object]()
        # with open(DEFAULTS_FILE, "r") as f:
        #     default_parameters = json.load(f)
        # for key, value in default_parameters.items():
        #     self.parameter_dictionary[key] = value

    def set_motmaster_dictionary(self):
        d = self.get_params()
        pars = Dictionary[String, Object]()
        for key, value in d.items():
            pars[key] = value
        self.parameter_dictionary = pars
        return None

    def start_motmaster_experiment(
        self,
    ):
        if self.script is None:
            raise ValueError(
                "MotMaster script not set. Please call set_motmaster_experiment first."
            )
        try:

            self.motmaster.Go()
            time.sleep(self.interval)
        except Exception as e:
            print(f"Error starting MotMaster experiment {self.script}: {e}")
        return None

    def get_params(self):
        return dict(self.motmaster.GetParameters())
    
class  DummyMotMasterInterface(MotMasterInterface):
    
    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        pass
    
    def connect(self) -> None:
        print("DummyMotMasterInterface: connect called, but no connection made.")
        self.motmaster = None
        self.script = None
        return None
    
    def set_motmaster_experiment(
        self,
        script: str,
    ):
        print(f"DummyMotMasterInterface: set_motmaster_experiment called with script '{script}', but no experiment set.")
        self.script = script
        return None
    
    def set_motmaster_dictionary(self):
        print("DummyMotMasterInterface: set_motmaster_dictionary called, but no dictionary set.")
        self.parameter_dictionary = {}
        return None
    
    def start_motmaster_experiment(
        self,
    ):
        print(f"DummyMotMasterInterface: start_motmaster_experiment called with script '{self.script}', but no experiment started.")
        return None
    
    def get_params(self):
        print("DummyMotMasterInterface: get_params called, but no parameters to return.")
        return {}
    
    def disconnect(self) -> None:
        print("DummyMotMasterInterface: disconnect called, but no connection to close.")
        return None


class MotMasterCommandServer:
    def __init__(
        self,
        interface: MotMasterInterface,
        host: str = "localhost",
        port: int = 5557,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self.interface = interface
        self.host = host
        self.port = port
        self.address = f"tcp://{host}:{port}"
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self._running = False
        self._experiment_thread: Optional[threading.Thread] = None

    def _run_experiment(self) -> None:
        self.interface.start_motmaster_experiment()

    def _handle_request(self, request: dict) -> dict:
        command = request.get("command")

        if command == "ping":
            return {"ok": True, "command": command, "status": "alive"}

        if command == "get_status":
            experiment_running = bool(
                self._experiment_thread and self._experiment_thread.is_alive()
            )
            return {
                "ok": True,
                "command": command,
                "server_running": self._running,
                "experiment_running": experiment_running,
                "script": self.interface.script,
            }

        if command == "set_script":
            script = request.get("script")
            if not script:
                raise ValueError("'script' is required for set_script")
            self.interface.set_motmaster_experiment(script)
            return {"ok": True, "command": command, "script": script}

        if command == "get_params":
            return {"ok": True, "command": command, "params": self.interface.get_params()}

        if command == "start_experiment":
            if self._experiment_thread and self._experiment_thread.is_alive():
                return {
                    "ok": False,
                    "command": command,
                    "error": "Experiment already running",
                    "script": self.interface.script,
                }

            self._experiment_thread = threading.Thread(
                target=self._run_experiment,
                name="motmaster-start-experiment",
                daemon=True,
            )
            self._experiment_thread.start()
            return {
                "ok": True,
                "command": command,
                "script": self.interface.script,
                "started": True,
            }

        if command == "shutdown":
            self._running = False
            return {"ok": True, "command": command}

        raise ValueError(f"Unknown command '{command}'")

    def serve_forever(self) -> None:
        self.socket.bind(self.address)
        self._running = True
        print(f"MotMaster command server listening on {self.address}")
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while self._running:
                # Use a polling timeout so KeyboardInterrupt is handled promptly.
                events = dict(poller.poll(timeout=2000))
                if self.socket not in events:
                    continue

                request = self.socket.recv_json()
                try:
                    if not isinstance(request, dict):
                        raise ValueError("Request must be a JSON object")
                    response = self._handle_request(request)
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                self.socket.send_json(response)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping server.")
            self._running = False
        finally:
            self.socket.close()
            try:
                self.interface.disconnect()
            except Exception:
                # Connection teardown should not mask shutdown.
                pass


def run_motmaster_command_server(
    host: str = "0.0.0.0", port: int = 5557, interval: Union[int, float] = 0.1, simulate: bool = False
) -> None:
    if simulate:
        interface = DummyMotMasterInterface(interval=interval)
    else:
        interface = MotMasterInterface(interval=interval)
    interface.connect()
    if (not simulate) and interface.motmaster is None:
        raise RuntimeError("Failed to connect to MotMaster.")

    server = MotMasterCommandServer(interface=interface, host=host, port=port)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotMaster ZeroMQ command server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (use 0.0.0.0 for remote clients)",
    )
    parser.add_argument("--port", type=int, default=5557, help="TCP port to bind")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run with dummy MotMaster interface for testing",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Delay after starting MotMaster experiment (seconds)",
    )
    args = parser.parse_args()

    run_motmaster_command_server(
        host=args.host,
        port=args.port,
        interval=args.interval,
        simulate=args.simulate,
    )


if __name__ == "__main__":
    main()
    
