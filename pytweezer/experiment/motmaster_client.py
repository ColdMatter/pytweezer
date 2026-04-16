import argparse
import json
from typing import Any, Optional
from pytweezer.servers.configreader import ConfigReader
import zmq


class MotMasterClient:
	def __init__(
		self,
		host: str = "10.59.3.2",
		port: int = 5557,
		timeout_ms: int = 5000,
		context: zmq.Context | None = None,
	) -> None:
		try:
			cr = ConfigReader.getConfiguration()
			self.host = cr["Servers"]["MotMaster Server"].get("host", host)
			self.port = cr["Servers"]["MotMaster Server"].get("port", port)
		except ModuleNotFoundError:
			self.host = host
			self.port = port
			print(
				"Warning: Could not load configuration. Using default host and port for MotMasterClient."
			)
			print(f"Host: {self.host}, Port: {self.port}")
		self.address = f"tcp://{self.host}:{self.port}"
		self.timeout_ms = timeout_ms
		self.context = context or zmq.Context.instance()

	def _create_socket(self, timeout_ms: int) -> zmq.Socket:
		socket = self.context.socket(zmq.REQ)
		socket.setsockopt(zmq.LINGER, 0)
		socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
		socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
		socket.connect(self.address)
		return socket

	def close(self) -> None:
		return None

	def __enter__(self) -> "MotMasterClient":
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def __getattr__(self, name: str):
		if name.startswith("_"):
			raise AttributeError(name)

		def remote_method(*args, **kwargs):
			return self.call_interface(name, *args, **kwargs)

		return remote_method

	def send_command(
		self,
		payload: dict[str, Any],
		*,
		timeout_ms: int | None = None,
		retries: int = 0,
	) -> dict[str, Any]:
		effective_timeout = self.timeout_ms if timeout_ms is None else timeout_ms
		for attempt in range(retries + 1):
			socket = self._create_socket(effective_timeout)
			try:
				socket.send_json(payload)
				return socket.recv_json()
			except zmq.error.Again as error:
				if attempt == retries:
					raise TimeoutError(
						f"Timed out waiting for response to command '{payload.get('command')}' "
						f"after {effective_timeout} ms"
					) from error
			finally:
				socket.close()
		raise RuntimeError("Unreachable state in send_command")

	def call_interface(
		self,
		method: str,
		*args,
		timeout_ms: int | None = None,
		retries: int = 0,
		**kwargs,
	) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"command": "call_interface",
			"method": method,
			"args": list(args),
			"kwargs": kwargs,
		}
		return self.send_command(payload, timeout_ms=timeout_ms, retries=retries)

	def call_method_by_command(
		self,
		method: str,
		*args,
		timeout_ms: int | None = None,
		retries: int = 0,
		**kwargs,
	) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"command": method,
			"args": list(args),
			"kwargs": kwargs,
		}
		return self.send_command(payload, timeout_ms=timeout_ms, retries=retries)

	def ping(self) -> dict[str, Any]:
		return self.send_command({"command": "ping"})

	def get_status(self) -> dict[str, Any]:
		return self.send_command({"command": "get_status"})

	def set_script(self, script: str, timeout_ms: int = 10000) -> dict[str, Any]:
		return self.send_command(
			{"command": "set_script", "script": script},
			timeout_ms=timeout_ms,
		)

	def get_params(self) -> dict[str, Any]:
		return self.send_command({"command": "get_params"})

	def start_experiment(
		self,
		parameters: Optional[dict] = None,
		timeout_ms: int = -1,
	) -> dict[str, Any]:
		payload = {"command": "start_experiment"}
		if parameters is not None:
			payload["parameters"] = parameters
		return self.send_command(payload, timeout_ms=timeout_ms)

	def shutdown_server(self) -> dict[str, Any]:
		return self.send_command({"command": "shutdown"})

	def set_run_until_stopped(self, value: bool) -> dict[str, Any]:
		return self.send_command(
			{"command": "set_run_until_stopped", "value": value}
		)

	def set_iterations(self, iterations: int) -> dict[str, Any]:
		return self.send_command(
			{"command": "set_iterations", "iterations": iterations}
		)

	def set_save_toggle(self, save: bool) -> dict[str, Any]:
		return self.send_command({"command": "set_save_toggle", "save": save})

	def set_trigger_mode(self, value: bool) -> dict[str, Any]:
		return self.send_command({"command": "set_trigger_mode", "value": value})

	def save_pattern_info(
		self,
		save_folder: str,
		file_tag: str,
		task_nr: int,
	) -> dict[str, Any]:
		return self.send_command(
			{
				"command": "save_pattern_info",
				"save_folder": save_folder,
				"file_tag": file_tag,
				"task_nr": task_nr,
			}
		)


def probe_server(host: str = "127.0.0.1", port: int = 5557, timeout_ms: int = 1200) -> bool:

	try:
		with MotMasterClient(host=host, port=port, timeout_ms=timeout_ms) as client:
			response = client.ping()
			return isinstance(response, dict) and response.get("ok") is True
	except Exception:
		return False


def main() -> None:
	parser = argparse.ArgumentParser(description="Simple MotMaster ZeroMQ test client")
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
	except zmq.error.Again:
		print(
			json.dumps(
				{
					"ok": False,
					"error": "Timed out waiting for server response",
					"host": args.host,
					"port": args.port,
				}
			)
		)
	except KeyboardInterrupt:
		print("Keyboard interrupt received. Exiting client.")


if __name__ == "__main__":
    main()