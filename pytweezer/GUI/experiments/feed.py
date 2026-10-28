"""Qt-side live copy of the Experiment Manager's published state."""

import copy
import time

from PyQt6 import QtCore

from pytweezer.experiment.client import NOTIFIER_NAME, sync_address
from pytweezer.servers.sync import SyncMirror

#: The manager bumps ``alive`` every 2 s; silence for longer means it is hung.
STALE_AFTER_S = 5.0


class ExperimentFeed(QtCore.QObject):
    """Mirrors the manager's ``"experiment"`` state and signals what changed.

    A timer drains the mirror's modifications on the Qt thread.
    ``queue_changed`` carries the queue snapshot (without the points) when any
    of it changes, and always after a (re)connect; ``point_received`` carries
    each point the running task measures, as ``{"rid", "index", "values",
    "scalars", "t_start", "t_end"}``. Points measured before this feed connected
    are in :meth:`points`.

    Args:
        address: ``(host, port)`` of the manager's state; CONFIG by default.
        poll_interval_ms: How often to drain the mirror.
    """

    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, address=None, poll_interval_ms=200, parent=None):
        super().__init__(parent)
        host, port = address or sync_address()
        self._mirror = SyncMirror(host, port, NOTIFIER_NAME, keep_mods=True)
        self._points_rid = None
        self._last_heard = 0.0
        self.connected = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(poll_interval_ms)

    def poll(self):
        queue_changed = False
        rows = []
        for mod in self._mirror.drain():
            self._last_heard = time.monotonic()
            if mod["action"] == "init":
                queue_changed = True
                self._points_rid = mod["struct"]["points"]["rid"]
            elif mod["path"] == ["points", "rows"] and mod["action"] == "append":
                rows.append({"rid": self._points_rid, **mod["x"]})
            elif mod["path"] == [] and mod["key"] == "points":
                self._points_rid = mod["value"]["rid"]
            elif not (mod["path"] == [] and mod["key"] == "alive"):
                queue_changed = True
        self._set_connected(
            self._mirror.connected
            and time.monotonic() - self._last_heard < STALE_AFTER_S
        )
        if queue_changed:
            self.queue_changed.emit(self.snapshot())
        for row in rows:
            self.point_received.emit(row)

    def snapshot(self):
        """The latest queue snapshot, or ``{}`` before the first connection."""
        return self._mirror.read(
            lambda state: {
                key: copy.deepcopy(value)
                for key, value in (state or {}).items()
                if key != "points"
            }
        )

    def points(self, rid):
        """Every point held for task ``rid``: the running or last-run task only."""
        return self._mirror.read(
            lambda state: (
                copy.deepcopy(state["points"]["rows"])
                if state and state["points"]["rid"] == rid
                else []
            )
        )

    def _set_connected(self, connected):
        if connected != self.connected:
            self.connected = connected
            self.connection_changed.emit(connected)

    def close(self):
        self._timer.stop()
        self._mirror.close()
