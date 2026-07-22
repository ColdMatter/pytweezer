"""Device-server status service.

The server PC runs :class:`DeviceStatusServer`, which periodically probes every
device RPC server listed in ``CONFIG["Devices"]`` — regardless of which PC it
runs on — and publishes an aggregate up/down snapshot over a ZMQ PUB socket.
Client GUIs subscribe with :class:`DeviceStatusClient` and display the result,
so any machine can see the global device picture (not just its own devices).

This is a pull model: only the server PC probes, and it needs no cooperation
from the device servers or the client GUIs beyond the config they already share.

A **composite** device serves several sub-devices from one port, and any of them
may be missing while the rest run (see ``device_server._make_composite``), so a
TCP probe of that port is not enough to tell whether a given sub-device is up.
Composites therefore get a second, RPC-level probe: a sipyco handshake returns
the target names the server is actually serving, and each sub-device is reported
individually under its own name — flat in the same ``devices`` dict as top-level
devices, since device names are unique across the whole category. Sub-device
entries carry ``"parent"``; a composite's entry carries ``"children"``.
"""

import argparse
import signal
import time

import zmq
from PyQt6 import QtCore
from sipyco.pc_rpc import Client as RPCClient

from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers.device_server import (
    composite_target_name,
    coordinator_target_name,
)
from pytweezer.servers.reachability import is_reachable

from pytweezer.logging_utils import get_logger

logger = get_logger("Device Status")

SERVER_NAME = "Device Status"
DEFAULT_POLL_INTERVAL = 2.0
PROBE_TIMEOUT = 0.3

# Timeout for the sipyco handshake that lists a composite's targets. Longer than
# PROBE_TIMEOUT because it costs a connect *plus* a round trip, and a server busy
# in a synchronous RPC call answers it late rather than never.
TARGET_PROBE_TIMEOUT = 1.0


def server_targets(host, port, timeout=TARGET_PROBE_TIMEOUT):
    """Return the set of RPC target names a sipyco server serves, or ``None``.

    ``None`` means the question could not be answered — the server is unreachable,
    too busy to complete the handshake, or not a sipyco server at all — as opposed
    to an empty set, which means it is serving nothing.

    Connects with ``target_name=None`` so no target is selected: the handshake
    alone carries the target list, and no RPC method is ever invoked.
    """
    client = None
    try:
        client = RPCClient(host, port, target_name=None, timeout=timeout)
        target_names, _description = client.get_rpc_id()
        return set(target_names)
    except Exception:
        logger.debug("Could not list RPC targets at %s:%s", host, port, exc_info=True)
        return None
    finally:
        if client is not None:
            try:
                client.close_rpc()
            except Exception:
                pass


def _server_conf():
    conf = ConfigReader.getConfiguration()
    return conf["Servers"][SERVER_NAME]


def _target_state(targets, target_name):
    """State of one sub-device given its composite's served target names.

    ``targets is None`` means the target list is unknown, which is reported as
    ``"unknown"`` rather than guessed at. A target missing from a server we did
    reach is a device that failed to start while its rig came up.
    """
    if targets is None:
        return "unknown"
    return "up" if target_name in targets else "failed"


class DeviceStatusServer:
    """Polls every ``CONFIG["Devices"]`` server and publishes up/down snapshots.

    Composites are probed twice: a TCP reachability check for the process, then an
    RPC handshake for the targets it serves, so each sub-device gets its own entry.
    """

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
        self.target_timeout = TARGET_PROBE_TIMEOUT
        self.devices = conf.get("Devices", {})
        self._last_targets = {}

        self.context = zmq.Context.instance()
        self.pub_socket = self.context.socket(zmq.PUB)
        self._running = False

    def build_snapshot(self):
        """Probe every device once and return the publishable snapshot dict.

        A composite contributes one entry per sub-device as well as its own, all
        keyed by device name in the same flat dict.
        """
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
            entry = {
                "state": state,
                "host": host,
                "port": port,
                "last_seen": last_seen,
            }

            sub_confs = params.get("devices")
            if not sub_confs:
                devices[name] = entry
                continue

            targets = self._composite_targets(name, host, port) if state == "up" else None
            entry["children"] = list(sub_confs)
            if state == "up" and params.get("coordinator"):
                # The composite's own name addresses its coordinator, which stands
                # down when a sub-device is missing: the process runs, but calling
                # the rig as a whole does not work.
                if targets is None:
                    entry["state"] = "unknown"
                elif coordinator_target_name(params) not in targets:
                    entry["state"] = "degraded"
            devices[name] = entry

            for sub_name in sub_confs:
                sub_state = (
                    _target_state(targets, composite_target_name(sub_name))
                    if state == "up"
                    else state
                )
                devices[sub_name] = {
                    "state": sub_state,
                    "host": host,
                    "port": port,
                    "last_seen": now if sub_state == "up" else None,
                    "parent": name,
                }
        return {"type": "device_status", "timestamp": now, "devices": devices}

    def _composite_targets(self, name, host, port):
        """Targets ``name``'s server currently serves, or ``None`` if unknown.

        A sipyco server runs synchronous RPC methods inline on its single event
        loop, so during a long call (a camera grab) the OS still accepts the TCP
        connection but the handshake goes unanswered. Reusing the last answer keeps
        a busy rig from reporting all of its sub-devices as failed; only a server
        that has never completed a handshake yields ``None``.
        """
        targets = server_targets(host, port, self.target_timeout)
        if targets is None:
            return self._last_targets.get(name)
        self._last_targets[name] = targets
        return targets

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
