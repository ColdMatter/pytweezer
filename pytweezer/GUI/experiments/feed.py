"""Qt-side subscriber to the Experiment Manager's PUB feed."""

import time

import zmq
from PyQt6 import QtCore

from pytweezer.experiment.client import pub_endpoint

#: A manager publishes at least every 2 s; silence for longer means it is down.
STALE_AFTER_S = 5.0


class ExperimentFeed(QtCore.QObject):
    """Drains the manager's PUB socket on a timer and re-emits its messages.

    ``queue_changed`` carries only the newest queue snapshot per poll;
    ``point_received`` carries every per-point message, for live plots.
    """

    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, endpoint=None, poll_interval_ms=200, parent=None):
        super().__init__(parent)
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint or pub_endpoint())
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._last_message = 0.0
        self.connected = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(poll_interval_ms)

    def poll(self):
        latest = None
        while True:
            try:
                message = self._socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self._last_message = time.monotonic()
            kind = message.get("type") if isinstance(message, dict) else None
            if kind == "experiment_queue":
                latest = message
            elif kind == "experiment_point":
                self.point_received.emit(message)
        self._set_connected(time.monotonic() - self._last_message < STALE_AFTER_S)
        if latest is not None:
            self.queue_changed.emit(latest)

    def _set_connected(self, connected):
        if connected != self.connected:
            self.connected = connected
            self.connection_changed.emit(connected)

    def close(self):
        self._timer.stop()
        self._socket.close(linger=0)
