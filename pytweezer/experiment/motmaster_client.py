import argparse
import json
from typing import Any, Optional

import zmq


class MotMasterClient:
	def __init__(
		self,
		host: str = "127.0.0.1",
		port: int = 5557,
		timeout_ms: int = 1200,
		context: zmq.Context | None = None,
	) -> None:
		self.host = host
		self.port = port
		self.address = f"tcp://{host}:{port}"
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

	def start_experiment(self, parameters: Optional[dict] = None) -> dict[str, Any]:
		payload = {"command": "start_experiment"}
		if parameters is not None:
			payload["parameters"] = parameters
		return self.send_command(payload)

	def shutdown_server(self) -> dict[str, Any]:
		return self.send_command({"command": "shutdown"})


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
				("get_params", client.get_params()),
				("start_experiment", client.start_experiment()),
				("get_status", client.get_status()),
			]

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