import argparse
import os
import signal
import subprocess
import time
from typing import Any

import zmq

from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import Properties, tweezerpath, zmqcontext
from pytweezer.servers.configreader import ConfigReader


class AnalysisManagerService:
    """Standalone analysis process manager controlled over TCP (REQ/REP)."""

    def __init__(self, server_name: str = "Analysis Manager") -> None:
        self.server_name = server_name
        self.conf = ConfigReader.getConfiguration()
        self.server_conf = self.conf.get("Servers", {}).get(server_name, {})
        self.rep_endpoint = self.server_conf.get("rep", "tcp://127.0.0.1:3111")

        # Keep legacy property namespace so existing analysis configs still work.
        self.props = Properties("Analysis")
        self.analysisdir = self.props.get("analysisdir", tweezerpath + "/pytweezer/analysis/")

        self.context = zmqcontext
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        if self.rep_endpoint.startswith("tcp://"):
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            if hasattr(zmq, "TCP_KEEPALIVE_IDLE"):
                self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            if hasattr(zmq, "TCP_KEEPALIVE_INTVL"):
                self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
            if hasattr(zmq, "TCP_KEEPALIVE_CNT"):
                self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
        self.socket.bind(self.rep_endpoint)

        self.processes: dict[str, subprocess.Popen] = {}
        self._running = True

    def _analysis_token(self, category: str, name: str) -> str:
        return f"Analysis/{category}/{name}"

    def _script_path(self, script: str) -> str:
        return os.path.join(self.analysisdir, script)

    def _key(self, category: str, name: str) -> str:
        return f"{category}/{name}"

    def _list_filters(self) -> dict[str, dict[str, Any]]:
        filters: dict[str, dict[str, Any]] = {}
        for category in ("Image", "Data"):
            cat_dict = self.props.get(category, {})
            if not isinstance(cat_dict, dict):
                continue
            for name, cfg in cat_dict.items():
                if not isinstance(cfg, dict):
                    continue
                key = self._key(category, name)
                filters[key] = {
                    "name": name,
                    "category": category,
                    "script": cfg.get("script", ""),
                    "active": bool(cfg.get("active", False)),
                    "streams": cfg.get(f"{category.lower()}streams", ["nostream"]),
                }
        return filters

    def _refresh_running(self) -> dict[str, bool]:
        running: dict[str, bool] = {}
        dead_keys = []
        for key, process in self.processes.items():
            alive = process is not None and process.poll() is None
            running[key] = bool(alive)
            if not alive:
                dead_keys.append(key)
        for key in dead_keys:
            self.processes.pop(key, None)
        return running

    def _ensure_stopped(self, category: str, name: str) -> None:
        key = self._key(category, name)
        process = self.processes.get(key)
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except Exception:
            process.kill()
        self.processes.pop(key, None)

    def _start_process(self, category: str, name: str) -> dict[str, Any]:
        filters = self._list_filters()
        key = self._key(category, name)
        if key not in filters:
            raise ValueError(f"Unknown analysis filter: {key}")

        script = filters[key]["script"]
        if not script:
            raise ValueError(f"Filter '{key}' has no script configured")

        script_path = self._script_path(script)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Analysis script not found: {script_path}")

        self._ensure_stopped(category, name)
        token = self._analysis_token(category, name)
        process = subprocess.Popen(["python3", script_path, token])
        self.processes[key] = process
        self.props.set(f"{category}/{name}/active", True)

        return {"ok": True, "key": key, "pid": process.pid}

    def _stop_process(self, category: str, name: str) -> dict[str, Any]:
        key = self._key(category, name)
        self._ensure_stopped(category, name)
        self.props.set(f"{category}/{name}/active", False)
        return {"ok": True, "key": key}

    def _set_active(self, category: str, name: str, active: bool) -> dict[str, Any]:
        if active:
            return self._start_process(category, name)
        return self._stop_process(category, name)

    def _add_filter(self, category: str, name: str, script: str, streams: list[str]) -> dict[str, Any]:
        cat_dict = self.props.get(category, {})
        if not isinstance(cat_dict, dict):
            cat_dict = {}

        cat_dict[name] = {
            "script": script,
            "active": False,
            f"{category.lower()}streams": streams,
        }
        self.props.set(category, cat_dict)
        return {"ok": True, "key": self._key(category, name)}

    def _delete_filter(self, category: str, name: str) -> dict[str, Any]:
        self._ensure_stopped(category, name)

        cat_dict = self.props.get(category, {})
        if isinstance(cat_dict, dict) and name in cat_dict:
            del cat_dict[name]
            self.props.set(category, cat_dict)
        return {"ok": True, "key": self._key(category, name)}

    def _snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "analysisdir": self.analysisdir,
            "filters": self._list_filters(),
            "running": self._refresh_running(),
        }

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")

        if command == "ping":
            return {"ok": True, "status": "alive"}
        if command == "snapshot":
            return self._snapshot()
        if command == "set_active":
            return self._set_active(
                category=request.get("category"),
                name=request.get("name"),
                active=bool(request.get("active", False)),
            )
        if command == "start":
            return self._start_process(category=request.get("category"), name=request.get("name"))
        if command == "stop":
            return self._stop_process(category=request.get("category"), name=request.get("name"))
        if command == "add_filter":
            return self._add_filter(
                category=request.get("category"),
                name=request.get("name"),
                script=request.get("script"),
                streams=request.get("streams", []),
            )
        if command == "delete_filter":
            return self._delete_filter(category=request.get("category"), name=request.get("name"))
        if command == "shutdown":
            self._running = False
            return {"ok": True}

        raise ValueError(f"Unknown command '{command}'")

    def _stop_all(self) -> None:
        keys = list(self.processes.keys())
        for key in keys:
            category, name = key.split("/", 1)
            try:
                self._stop_process(category, name)
            except Exception:
                pass

    def serve_forever(self) -> None:
        print_error(
            f"Analysis manager service listening on {self.rep_endpoint}",
            "info",
        )
        while self._running:
            try:
                request = self.socket.recv_json()
            except Exception as error:
                print_error(f"analysis_manager recv error: {error}", "warning")
                continue

            try:
                response = self._handle_request(request)
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            self.socket.send_json(response)

        self._stop_all()
        self.socket.close()


def _resolve_server_name(token: str, conf: dict) -> str:
    if isinstance(token, str) and token.startswith("Servers/"):
        return token.split("/", 1)[1]
    if isinstance(token, str) and token in conf.get("Servers", {}):
        return token
    return "Analysis Manager"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        nargs="?",
        default="Servers/Analysis Manager",
        help="name of this program instance",
    )
    args, _unknown = parser.parse_known_args()

    conf = ConfigReader.getConfiguration()
    server_name = _resolve_server_name(args.name, conf)

    service = AnalysisManagerService(server_name=server_name)

    def _shutdown(_signo, _frame):
        service._running = False
        # Unblock recv by connecting briefly and sending shutdown.
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(service.rep_endpoint)
            sock.send_json({"command": "shutdown"})
            sock.recv_json()
        except Exception:
            pass
        finally:
            sock.close()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    service.serve_forever()


if __name__ == "__main__":
    main()
