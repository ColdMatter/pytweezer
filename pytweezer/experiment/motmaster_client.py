import argparse
import json
from typing import Any, Optional

from sipyco.pc_rpc import Client as RPCClient

from pytweezer.servers.configreader import ConfigReader


class MotMasterClient:
    def __init__(
        self,
        host: str = "10.59.3.2",
        port: int = 5557,
        timeout_ms: int = 5000,
        context: Any = None,
        server_name: str = "Rb MotMaster Server",
        target_name: str = "motmaster",
    ) -> None:
        del context
        timeout_s = timeout_ms / 1000.0

        try:
            cr = ConfigReader.getConfiguration()
            server_conf = cr.get("Servers", {}).get(server_name, {})
            self.host = server_conf.get("host", host)
            self.port = int(server_conf.get("port", port))
        except Exception:
            self.host = host
            self.port = port
            print(
                "Warning: Could not load configuration. "
                "Using explicit host and port for MotMasterClient."
            )

        self._rpc = RPCClient(
            host=self.host,
            port=self.port,
            target=target_name,
            timeout=timeout_s,
        )

def probe_server(host: str = "127.0.0.1", port: int = 5557, timeout_ms: int = 1200) -> bool:
    try:
        with MotMasterClient(host=host, port=port, timeout_ms=timeout_ms) as client:
            response = client.ping()
            return isinstance(response, dict) and response.get("ok") is True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MotMaster sipyco RPC test client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5557, help="Server TCP port")
    parser.add_argument(
        "--script",
        default="test_script",
        help="Script name for set_script command",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Send shutdown command after test sequence",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Only verify server reachability using ping",
    )
    parser.add_argument(
        "--run-until-stopped",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set run-until-stopped mode (0 or 1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Set iteration count before starting experiment",
    )
    parser.add_argument(
        "--save-toggle",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set save toggle (0 or 1)",
    )
    parser.add_argument(
        "--trigger-mode",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set trigger mode (0 or 1)",
    )
    parser.add_argument(
        "--save-folder",
        default=None,
        help="Folder path for save_pattern_info",
    )
    parser.add_argument(
        "--file-tag",
        default=None,
        help="File tag for save_pattern_info",
    )
    parser.add_argument(
        "--task-nr",
        type=int,
        default=None,
        help="Task number for save_pattern_info",
    )
    args = parser.parse_args()

    try:
        with MotMasterClient(host=args.host, port=args.port) as client:
            if args.test_connection:
                response = client.ping()
                print(json.dumps(response, default=str))
                return

            responses = [
                ("ping", client.ping()),
                ("get_status", client.get_status()),
                ("set_script", client.set_script(args.script)),
            ]

            if args.run_until_stopped is not None:
                responses.append(
                    (
                        "set_run_until_stopped",
                        client.set_run_until_stopped(bool(args.run_until_stopped)),
                    )
                )

            if args.iterations is not None:
                responses.append(("set_iterations", client.set_iterations(args.iterations)))

            if args.save_toggle is not None:
                responses.append(
                    ("set_save_toggle", client.set_save_toggle(bool(args.save_toggle)))
                )

            if args.trigger_mode is not None:
                responses.append(
                    ("set_trigger_mode", client.set_trigger_mode(bool(args.trigger_mode)))
                )

            if (
                args.save_folder is not None
                and args.file_tag is not None
                and args.task_nr is not None
            ):
                responses.append(
                    (
                        "save_pattern_info",
                        client.save_pattern_info(
                            args.save_folder,
                            args.file_tag,
                            args.task_nr,
                        ),
                    )
                )

            responses.extend([
                ("get_params", client.get_params()),
                ("start_experiment", client.start_experiment()),
                ("get_status", client.get_status()),
            ])

            if args.shutdown:
                responses.append(("shutdown", client.shutdown_server()))

            for command_name, response in responses:
                print(f"command:  {command_name}")
                print(f"response: {json.dumps(response, default=str)}")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting client.")


if __name__ == "__main__":
    main()
