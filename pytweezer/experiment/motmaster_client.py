import argparse
import json
import shlex
import subprocess
import time
from typing import Any
import zmq


def send_command(socket: zmq.Socket, payload: dict[str, Any]) -> dict[str, Any]:
	socket.send_json(payload)
	return socket.recv_json()


def probe_server(address: str, timeout_ms: int = 1200) -> bool:
	context = zmq.Context.instance()
	socket = context.socket(zmq.REQ)
	socket.setsockopt(zmq.LINGER, 0)
	socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
	socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
	socket.connect(address)

	try:
		socket.send_json({"command": "get_params"})
		response = socket.recv_json()
		return isinstance(response, dict)
	except Exception:
		return False
	finally:
		socket.close()



def wait_for_server(address: str, timeout_s: float = 15.0, step_s: float = 0.5) -> bool:
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		if probe_server(address):
			return True
		time.sleep(step_s)
	return False


def main() -> None:
	print("running")
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
		"--autostart",
		action="store_true",
		help="Auto-start server if it is not reachable",
	)
	parser.add_argument(
		"--server-command",
		default=None,
		help="Command to launch the server process",
	)
	parser.add_argument(
		"--server-workdir",
		default=None,
		help="Working directory when launching server locally",
	)
	args = parser.parse_args()

	address = f"tcp://{args.host}:{args.port}"
	server_command = args.server_command
	context = zmq.Context.instance()
	socket = context.socket(zmq.REQ)
	socket.setsockopt(zmq.LINGER, 0)
	socket.setsockopt(zmq.RCVTIMEO, 1200)
	socket.setsockopt(zmq.SNDTIMEO, 1200)
	socket.connect(address)

	try:
		commands = [
			{"command": "set_script", "script": args.script},
			{"command": "get_params"},
			{"command": "start_experiment"},
		]

		if args.shutdown:
			commands.append({"command": "shutdown"})

		for command in commands:
			try:
				response = send_command(socket, command)
			except zmq.error.Again:
				response = {
					"ok": False,
					"error": "Timed out waiting for server response",
				}
			print(f"request:  {json.dumps(command)}")
			print(f"response: {json.dumps(response, default=str)}")
	except KeyboardInterrupt:
		print("Keyboard interrupt received. Exiting client.")
	finally:
		socket.close()


if __name__ == "__main__":
    main()