import argparse
import time
from typing import Callable, Optional, Tuple, Union
import json
import pathlib
import subprocess
import sys

import numpy as np
import zmq
import pythonnet
import numbers
from pytweezer.servers.configreader import ConfigReader
from pytweezer.experiment.motmaster_interface import MotMasterInterface


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

    
class  DummyMotMasterInterface(MotMasterInterface):
    
    def __init__(self, interval: Union[int, float] = 0.1) -> None:
        pass
    
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
    
    def disconnect(self) -> None:
        print("DummyMotMasterInterface: disconnect called")
        return None

    def save_pattern_info(self, save_folder, file_tag, task_nr):
        print(
            "DummyMotMasterInterface: save_pattern_info called "
            f"(save_folder={save_folder}, file_tag={file_tag}, task_nr={task_nr})"
        )
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

    def _invoke_interface_method(
        self,
        method_name: str,
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ):
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("'method' must be a non-empty string")
        if method_name.startswith("_"):
            raise ValueError("Calling private methods is not allowed")

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        if not isinstance(args, list):
            raise ValueError("'args' must be a list when provided")
        if not isinstance(kwargs, dict):
            raise ValueError("'kwargs' must be a dictionary when provided")

        method = getattr(self.interface, method_name, None)
        if method is None or not callable(method):
            raise ValueError(f"Interface method '{method_name}' not found")

        return method(*args, **kwargs)

    def _handle_request(self, request: dict) -> Tuple[dict, Optional[Callable[[], None]]]:
        command = request.get("command")

        if command == "ping":
            return {"ok": True, "command": command, "status": "alive"}, None

        if command == "get_status":
            return {
                "ok": True,
                "command": command,
                "server_running": self._running,
                "script": self.interface.script,
            }, None

        if command == "call_interface":
            method_name = request.get("method")
            args = request.get("args")
            kwargs = request.get("kwargs")
            wait_for_result = bool(request.get("wait_for_result", False))

            if wait_for_result:
                result = self._invoke_interface_method(
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs,
                )
                return {
                    "ok": True,
                    "command": command,
                    "method": method_name,
                    "result": result,
                    "state": "completed",
                }, None

            response = {
                "ok": True,
                "command": command,
                "method": method_name,
                "state": "accepted",
            }
            deferred = lambda: self._invoke_interface_method(
                method_name=method_name,
                args=args,
                kwargs=kwargs,
            )
            return response, deferred

        if command == "shutdown":
            self._running = False
            return {"ok": True, "command": command}, None

        if isinstance(command, str) and not command.startswith("_"):
            payload = dict(request)
            payload.pop("command", None)
            args = payload.pop("args", [])
            kwargs = payload.pop("kwargs", {})
            wait_for_result = bool(payload.pop("wait_for_result", False))
            if payload:
                kwargs = {**payload, **kwargs}

            if wait_for_result:
                result = self._invoke_interface_method(
                    method_name=command,
                    args=args,
                    kwargs=kwargs,
                )
                return {
                    "ok": True,
                    "command": "interface_method",
                    "method": command,
                    "result": result,
                    "state": "completed",
                }, None

            response = {
                "ok": True,
                "command": "interface_method",
                "method": command,
                "state": "accepted",
            }
            deferred = lambda: self._invoke_interface_method(
                method_name=command,
                args=args,
                kwargs=kwargs,
            )
            return response, deferred

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
                    response, deferred = self._handle_request(request)
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                    deferred = None
                self.socket.send_json(response)

                # Execute deferred work only after replying so clients are non-blocking.
                if deferred is not None:
                    try:
                        deferred()
                    except Exception as error:
                        print(f"Deferred interface call failed: {error}")
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
    host: str = "0.0.0.0", port: int = 5557, interval: Union[int, float] = 0.1, simulate: bool = False, config_file: Optional[str] = None
) -> None:
    if simulate:
        interface = DummyMotMasterInterface(interval=interval)
    else:
        interface = MotMasterInterface(config_file, interval=interval)
    interface.connect()
    if (not simulate) and interface.motmaster is None:
        raise RuntimeError("Failed to connect to MotMaster.")


    server = MotMasterCommandServer(interface=interface, host=host, port=port)
    server.serve_forever()



def main() -> None:
    parser = argparse.ArgumentParser(description="Run MotMaster ZeroMQ command server")
    parser.add_argument('--name', nargs='?', default=None, help='name of this program instance')
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

    args, _unknown = parser.parse_known_args()

    from pytweezer.configuration.config import CONFIG
    from pytweezer.servers import tweezerpath
    config_dict = CONFIG["Servers"][f"{args.name} MotMaster Server"]
    host = args.host or config_dict.get("host")
    port = args.port or config_dict.get("port")
    simulate = args.simulate or config_dict.get("simulate", False)
    interval = args.interval or config_dict.get("interval", 0.1)
    config_file = tweezerpath + "/pytweezer/configuration/" + config_dict.get("config_file")

    run_motmaster_command_server(
        host=host,
        port=port,
        interval=interval,
        simulate=simulate,
        config_file=config_file,
    )


if __name__ == "__main__":
    main()
    
