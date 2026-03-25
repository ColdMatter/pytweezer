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


def launch_server_local(command: str, workdir: str | None = None) -> None:
	subprocess.Popen(
		shlex.split(command),
		cwd=workdir,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)


def launch_server_ssh(
	ssh_host: str,
	ssh_user: str,
	ssh_port: int,
	server_command: str,
	remote_workdir: str | None = None,
	ssh_key_file: str | None = None,
	ssh_password: str | None = None,
) -> None:
	try:
		import paramiko
	except ImportError as error:
		raise RuntimeError(
			"paramiko is required for --ssh-host autostart"
		) from error

	client = paramiko.SSHClient()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(
		hostname=ssh_host,
		username=ssh_user,
		port=ssh_port,
		key_filename=ssh_key_file,
		password=ssh_password,
		timeout=10,
	)

	if remote_workdir:
		command = f"cd {shlex.quote(remote_workdir)} && {server_command}"
	else:
		command = server_command

	detached = f"nohup {command} >/tmp/motmaster_server.log 2>&1 &"
	client.exec_command(detached)
	client.close()


def wait_for_server(address: str, timeout_s: float = 15.0, step_s: float = 0.5) -> bool:
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		if probe_server(address):
			return True
		time.sleep(step_s)
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
	parser.add_argument("--ssh-host", default=None, help="SSH host for remote autostart")
	parser.add_argument("--ssh-user", default=None, help="SSH username for remote autostart")
	parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
	parser.add_argument("--ssh-key-file", default=None, help="SSH private key path")
	parser.add_argument("--ssh-password", default=None, help="SSH password (optional)")
	parser.add_argument(
		"--remote-workdir",
		default=None,
		help="Remote directory to cd into before starting server",
	)
	args = parser.parse_args()

	address = f"tcp://{args.host}:{args.port}"
	if not probe_server(address):
		if not args.autostart:
			raise RuntimeError(
				f"Server is not reachable at {address}. Use --autostart to launch it."
			)

		server_command = args.server_command or (
			f"python -m pytweezer.experiment.motmaster_server --host {args.host} --port {args.port}"
		)

		if args.ssh_host:
			if not args.ssh_user:
				raise ValueError("--ssh-user is required when --ssh-host is provided")
			launch_server_ssh(
				ssh_host=args.ssh_host,
				ssh_user=args.ssh_user,
				ssh_port=args.ssh_port,
				server_command=server_command,
				remote_workdir=args.remote_workdir,
				ssh_key_file=args.ssh_key_file,
				ssh_password=args.ssh_password,
			)
		else:
			launch_server_local(server_command, workdir=args.server_workdir)

		if not wait_for_server(address):
			raise RuntimeError(f"Server failed to start within timeout at {address}")

	context = zmq.Context.instance()
	socket = context.socket(zmq.REQ)
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
			response = send_command(socket, command)
			print(f"request:  {json.dumps(command)}")
			print(f"response: {json.dumps(response, default=str)}")
	finally:
		socket.close()


if __name__ == "__main__":
    main()