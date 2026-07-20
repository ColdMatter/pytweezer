"""Device-server status service.

The server PC runs :class:`DeviceStatusServer`, which periodically probes every
device RPC server listed in ``CONFIG["Devices"]`` — regardless of which PC it
runs on — and publishes an aggregate up/down snapshot over a ZMQ PUB socket.
Client GUIs subscribe with :class:`DeviceStatusClient` and display the result,
so any machine can see the global device picture (not just its own devices).

This is a pull model: only the server PC probes, and it needs no cooperation
from the device servers or the client GUIs beyond the config they already share.
"""

import argparse
import signal
import time

import zmq
from PyQt5 import QtCore

from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers.reachability import is_reachable

from pytweezer.logging_utils import get_logger

logger = get_logger("Device Status")

SERVER_NAME = "Device Status"
DEFAULT_POLL_INTERVAL = 2.0
PROBE_TIMEOUT = 0.3


def _server_conf():
    conf = ConfigReader.getConfiguration()
    return conf["Servers"][SERVER_NAME]


class DeviceStatusServer:
    """Polls every ``CONFIG["Devices"]`` server and publishes up/down snapshots."""

    def __init__(self, host=None, pub_port=None, poll_interval=None, probe_timeout=PROBE_TIMEOUT):
        conf = ConfigReader.getConfiguration()
        server_conf = conf["Servers"][SERVER_NAME]
        self.host = host or server_conf["host"]
        self.pub_port = int(pub_port or server_conf["pub_port"])
        self.poll_interval = float(
            poll_interval
            if poll_interval is not None
            else server_conf.get("poll_interval", DEFAULT_POLL_INTERVAL)
        )
        self.probe_timeout = probe_timeout
        self.devices = conf.get("Devices", {})

        self.context = zmq.Context.instance()
        self.pub_socket = self.context.socket(zmq.PUB)
        self._running = False

    def build_snapshot(self):
        """Probe every device once and return the publishable snapshot dict."""
        now = time.time()
        devices = {}
        for name, params in self.devices.items():
            host = params.get("host")
            port = params.get("port")
            if not params.get("active", True):
                state, last_seen = "disabled", None
            elif host is None or port is None:
                state, last_seen = "down", None
            elif is_reachable(host, port, self.probe_timeout):
                state, last_seen = "up", now
            else:
                state, last_seen = "down", None
            devices[name] = {
                "state": state,
                "host": host,
                "port": port,
                "last_seen": last_seen,
            }
        return {"type": "device_status", "timestamp": now, "devices": devices}

    def serve_forever(self):
        self.pub_socket.bind(f"tcp://{self.host}:{self.pub_port}")
        self._running = True
        logger.info(
            "Device Status server publishing on tcp://%s:%s (%d devices, %.1fs interval)",
            self.host,
            self.pub_port,
            len(self.devices),
            self.poll_interval,
        )

        def _stop(_signo, _frame):
            self._running = False

        try:
            signal.signal(signal.SIGTERM, _stop)
        except (ValueError, OSError):
            # Not on the main thread (e.g. under test) -- rely on stop()/KeyboardInterrupt.
            pass

        try:
            while self._running:
                snapshot = self.build_snapshot()
                self.pub_socket.send_json(snapshot)
                # Sleep in slices so SIGTERM/stop() takes effect promptly.
                waited = 0.0
                while self._running and waited < self.poll_interval:
                    time.sleep(0.1)
                    waited += 0.1
        except KeyboardInterrupt:
            logger.info("Device Status server interrupted, shutting down.")
        finally:
            self._running = False
            self.pub_socket.close(linger=0)

    def stop(self):
        self._running = False


class DeviceStatusClient(QtCore.QObject):
    """Subscribes to the device-status feed and emits the latest device dict.

    Mirrors ``model_sync._ModelSyncClient``: a non-blocking QTimer drains the SUB
    socket and re-emits the newest snapshot's ``devices`` payload.
    """

    status_received = QtCore.pyqtSignal(object)

    def __init__(self, host=None, pub_port=None, poll_interval_ms=200, parent=None):
        super().__init__(parent)
        server_conf = _server_conf()
        self.host = host or server_conf["host"]
        self.pub_port = int(pub_port or server_conf["pub_port"])

        self.context = zmq.Context.instance()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 0)
        self.sub_socket.connect(f"tcp://{self.host}:{self.pub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_updates)
        self.timer.start(poll_interval_ms)

    def _poll_updates(self):
        latest = None
        while True:
            try:
                message = self.sub_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if isinstance(message, dict) and message.get("type") == "device_status":
                latest = message.get("devices", {})
        if latest is not None:
            self.status_received.emit(latest)

    def close(self):
        self.timer.stop()
        self.sub_socket.close(linger=0)


def main():
    parser = argparse.ArgumentParser(description="Run the device-status publisher")
    parser.add_argument("name", nargs="?", default=None, help="process-manager label (ignored)")
    parser.add_argument("--host", default=None, help="override PUB bind host")
    parser.add_argument("--pub-port", type=int, default=None, help="override PUB bind port")
    parser.add_argument("--poll-interval", type=float, default=None, help="seconds between probes")
    args, _unknown = parser.parse_known_args()

    server = DeviceStatusServer(
        host=args.host,
        pub_port=args.pub_port,
        poll_interval=args.poll_interval,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
