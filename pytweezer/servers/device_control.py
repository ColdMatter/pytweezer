"""Start/stop requests for devices on other PCs.

A device's process belongs to the GUI on its config ``host``, so another PC
cannot spawn or kill it directly. Instead it publishes a request on the
Commandhub; every GUI's :class:`DeviceControlChannel` receives it, and only the
one that owns the device acts, exactly as if its own toggle had been clicked.
The request is fire-and-forget: the requester learns the outcome from the
Device Status feed, like any other state change.
"""

from socket import gethostname

from PyQt6 import QtCore

from pytweezer.logging_utils import get_logger
from pytweezer.servers.clients import CommandClient

logger = get_logger("device_control")

CHANNEL = "Device control"
ACTIONS = ("start", "stop")


class DeviceControlChannel(QtCore.QObject):
    """Sends start/stop requests and emits the ones addressed to any device.

    A non-blocking QTimer drains the Commandhub subscription, so a receiver
    needs nothing but the Qt event loop.
    """

    requested = QtCore.pyqtSignal(str, str, str)
    """Emitted with ``(action, device, requester_host)`` for each request."""

    def __init__(self, poll_interval_ms=200, parent=None):
        super().__init__(parent)
        self._client = CommandClient(CHANNEL, recvtimeout=0, subscribe=CHANNEL)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(poll_interval_ms)

    def request(self, action, device):
        if action not in ACTIONS:
            raise ValueError(f"unknown device action {action!r}")
        logger.info("Requesting %s of device %s", action, device)
        self._client.send(
            cmd=action,
            data={"action": action, "device": device, "from": gethostname()},
        )

    def _poll(self):
        while True:
            message = self._client.recv(flags=0)
            if message is None:
                return
            data = message[1] if len(message) > 1 else {}
            action, device = data.get("action"), data.get("device")
            if action in ACTIONS and isinstance(device, str):
                self.requested.emit(action, device, str(data.get("from", "?")))

    def close(self):
        self.timer.stop()
        self._client.pub_socket.close(linger=0)
        self._client.sub_socket.close(linger=0)
