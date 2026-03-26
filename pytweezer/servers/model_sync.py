import argparse
import copy
import threading
from typing import Any, Optional

import zmq
from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, Qt

from pytweezer.GUI.models import PrepModel, ScheduleModel


class ModelSyncServer:
    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        command_port: int = 6010,
        publish_port: int = 6011,
        models: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.bind_host = bind_host
        self.command_port = command_port
        self.publish_port = publish_port
        self.context = zmq.Context.instance()
        self.command_socket = self.context.socket(zmq.REP)
        self.publish_socket = self.context.socket(zmq.PUB)
        self._running = False
        self._lock = threading.Lock()
        self.models = models or {}

    def _validate_model(self, model_name: str) -> dict[str, Any]:
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'")
        return self.models[model_name]

    def _publish_snapshot(self, model_name: str) -> None:
        model_entry = self.models[model_name]
        snapshot = copy.deepcopy(model_entry["data"])
        self.publish_socket.send_pyobj(
            {
                "type": "snapshot",
                "model": model_name,
                "kind": model_entry["kind"],
                "data": snapshot,
            }
        )

    def _get_snapshot(self, model_name: str) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        return {
            "ok": True,
            "model": model_name,
            "kind": model_entry["kind"],
            "data": copy.deepcopy(model_entry["data"]),
        }

    def _set_data(self, model_name: str, row_or_key: Any, field: str, value: Any) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        if model_entry["kind"] == "dict":
            model_entry["data"][row_or_key][field] = value
        elif model_entry["kind"] == "list":
            model_entry["data"][row_or_key][field] = value
        else:
            raise ValueError(f"Unsupported model kind '{model_entry['kind']}'")

        self._publish_snapshot(model_name)
        return {"ok": True}

    def _set_item(self, model_name: str, key: Any, value: Any) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        if model_entry["kind"] != "dict":
            raise ValueError("set_item only valid for dict-backed model")
        model_entry["data"][key] = value
        self._publish_snapshot(model_name)
        return {"ok": True}

    def _del_item(self, model_name: str, key: Any) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        if model_entry["kind"] == "dict":
            del model_entry["data"][key]
        elif model_entry["kind"] == "list":
            del model_entry["data"][key]
        else:
            raise ValueError(f"Unsupported model kind '{model_entry['kind']}'")
        self._publish_snapshot(model_name)
        return {"ok": True}

    def _list_append(self, model_name: str, value: Any) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        if model_entry["kind"] != "list":
            raise ValueError("append only valid for list-backed model")
        model_entry["data"].append(value)
        self._publish_snapshot(model_name)
        return {"ok": True}

    def _list_insert(self, model_name: str, index: int, value: Any) -> dict[str, Any]:
        model_entry = self._validate_model(model_name)
        if model_entry["kind"] != "list":
            raise ValueError("insert only valid for list-backed model")
        model_entry["data"].insert(index, value)
        self._publish_snapshot(model_name)
        return {"ok": True}

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        model_name = request.get("model")

        if command == "ping":
            return {"ok": True, "status": "alive"}

        if command == "snapshot":
            return self._get_snapshot(model_name)

        if command == "set_data":
            return self._set_data(
                model_name=model_name,
                row_or_key=request.get("row_or_key"),
                field=request.get("field"),
                value=request.get("value"),
            )

        if command == "set_item":
            return self._set_item(model_name, request.get("key"), request.get("value"))

        if command == "del_item":
            return self._del_item(model_name, request.get("key"))

        if command == "list_append":
            return self._list_append(model_name, request.get("value"))

        if command == "list_insert":
            return self._list_insert(model_name, request.get("index"), request.get("value"))

        if command == "shutdown":
            self._running = False
            return {"ok": True}

        raise ValueError(f"Unknown command '{command}'")

    def serve_forever(self) -> None:
        self.command_socket.bind(f"tcp://{self.bind_host}:{self.command_port}")
        self.publish_socket.bind(f"tcp://{self.bind_host}:{self.publish_port}")
        self._running = True
        print(
            f"Model sync server listening: command=tcp://{self.bind_host}:{self.command_port}, "
            f"publish=tcp://{self.bind_host}:{self.publish_port}"
        )

        while self._running:
            request = self.command_socket.recv_pyobj()
            try:
                with self._lock:
                    response = self._handle(request)
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            self.command_socket.send_pyobj(response)

        self.command_socket.close()
        self.publish_socket.close()


class _ModelSyncClient(QtCore.QObject):
    snapshot_received = QtCore.pyqtSignal(object)

    def __init__(
        self,
        model_name: str,
        host: str,
        command_port: int,
        publish_port: int,
        timeout_ms: int = 2000,
        poll_interval_ms: int = 50,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.host = host
        self.command_port = command_port
        self.publish_port = publish_port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 0)
        self.sub_socket.connect(f"tcp://{host}:{publish_port}")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_updates)
        self.timer.start(poll_interval_ms)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(f"tcp://{self.host}:{self.command_port}")
        try:
            socket.send_pyobj(payload)
            return socket.recv_pyobj()
        finally:
            socket.close()

    def get_snapshot(self) -> dict[str, Any]:
        return self._request({"command": "snapshot", "model": self.model_name})

    def apply_operation(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["model"] = self.model_name
        return self._request(payload)

    def _poll_updates(self) -> None:
        while True:
            try:
                message = self.sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                return

            if (
                isinstance(message, dict)
                and message.get("type") == "snapshot"
                and message.get("model") == self.model_name
            ):
                self.snapshot_received.emit(message.get("data"))


class SyncedScheduleModel(ScheduleModel):
    def __init__(
        self,
        init_or_model_name: Any = None,
        host: str = "127.0.0.1",
        command_port: int = 6010,
        publish_port: int = 6011,
        timeout_ms: int = 2000,
    ) -> None:
        if isinstance(init_or_model_name, str) and init_or_model_name:
            model_name = init_or_model_name
        else:
            model_name = "schedule"

        self._sync = _ModelSyncClient(
            model_name=model_name,
            host=host,
            command_port=command_port,
            publish_port=publish_port,
            timeout_ms=timeout_ms,
        )
        snapshot = self._sync.get_snapshot()
        if not snapshot.get("ok"):
            raise RuntimeError(f"Failed to fetch snapshot: {snapshot}")
        super().__init__(snapshot.get("data", {}))
        self._sync.snapshot_received.connect(self._apply_snapshot)

    def _apply_snapshot(self, data: dict[Any, Any]) -> None:
        self.beginResetModel()
        self.backing_store = data
        self.row_to_key = sorted(
            self.backing_store.keys(),
            key=lambda k: self.sort_key(k, self.backing_store[k]),
        )
        self.endResetModel()

    def setData(self, index, value, role=Qt.DisplayRole):
        if value == "":
            return False
        if role == Qt.EditRole:
            k = self.row_to_key[index.row()]
            col = index.column()
            if col == 7:
                value = value.toString()
            if col in (4, 5, 6):
                value = int(value)
            response = self._sync.apply_operation(
                {
                    "command": "set_data",
                    "row_or_key": k,
                    "field": self.dataNames[col],
                    "value": value,
                }
            )
            return bool(response.get("ok"))
        return False

    def __setitem__(self, k, v):
        response = self._sync.apply_operation(
            {"command": "set_item", "key": k, "value": v}
        )
        if not response.get("ok"):
            raise RuntimeError(response)

    def __delitem__(self, k):
        response = self._sync.apply_operation({"command": "del_item", "key": k})
        if not response.get("ok"):
            raise RuntimeError(response)


class SyncedPrepModel(PrepModel):
    def __init__(
        self,
        init_or_model_name: Any = None,
        host: str = "127.0.0.1",
        command_port: int = 6010,
        publish_port: int = 6011,
        timeout_ms: int = 2000,
    ) -> None:
        if isinstance(init_or_model_name, str) and init_or_model_name:
            model_name = init_or_model_name
        else:
            model_name = "prep"

        self._sync = _ModelSyncClient(
            model_name=model_name,
            host=host,
            command_port=command_port,
            publish_port=publish_port,
            timeout_ms=timeout_ms,
        )
        snapshot = self._sync.get_snapshot()
        if not snapshot.get("ok"):
            raise RuntimeError(f"Failed to fetch snapshot: {snapshot}")
        super().__init__(snapshot.get("data", []))
        self._sync.snapshot_received.connect(self._apply_snapshot)

    def _apply_snapshot(self, data: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self.backing_store = data
        self.endResetModel()

    def setData(self, index, value, role=Qt.DisplayRole):
        if value == "":
            return False
        if role == Qt.EditRole:
            k = index.row()
            col = index.column()
            if col == 7:
                value = value.toString()
            if self.backing_store[k]["status"] != "Sleeping":
                if QDateTime.fromString(self.backing_store[k]["dueDateTime"]) <= QDateTime.currentDateTime():
                    status_value = "Queued"
                else:
                    status_value = "Waiting"
                self._sync.apply_operation(
                    {
                        "command": "set_data",
                        "row_or_key": k,
                        "field": "status",
                        "value": status_value,
                    }
                )

            response = self._sync.apply_operation(
                {
                    "command": "set_data",
                    "row_or_key": k,
                    "field": self.dataNames[col],
                    "value": value,
                }
            )
            return bool(response.get("ok"))
        return False

    def __delitem__(self, k):
        response = self._sync.apply_operation({"command": "del_item", "key": k})
        if not response.get("ok"):
            raise RuntimeError(response)

    def append(self, value: dict[str, Any]) -> None:
        response = self._sync.apply_operation({"command": "list_append", "value": value})
        if not response.get("ok"):
            raise RuntimeError(response)

    def insert(self, index: int, value: dict[str, Any]) -> None:
        response = self._sync.apply_operation(
            {"command": "list_insert", "index": index, "value": value}
        )
        if not response.get("ok"):
            raise RuntimeError(response)


def run_model_sync_server(
    schedule_data: Optional[dict[Any, dict[str, Any]]] = None,
    prep_data: Optional[list[dict[str, Any]]] = None,
    bind_host: str = "0.0.0.0",
    command_port: int = 6010,
    publish_port: int = 6011,
) -> None:
    server = ModelSyncServer(
        bind_host=bind_host,
        command_port=command_port,
        publish_port=publish_port,
        models={
            "schedule": {"kind": "dict", "data": schedule_data or {}},
            "prep": {"kind": "list", "data": prep_data or []},
        },
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run table-model sync server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--command-port", type=int, default=6010, help="Command TCP port")
    parser.add_argument("--publish-port", type=int, default=6011, help="Publish TCP port")
    args, _unknown = parser.parse_known_args()

    run_model_sync_server(
        bind_host=args.host,
        command_port=args.command_port,
        publish_port=args.publish_port,
    )


if __name__ == "__main__":
    main()
