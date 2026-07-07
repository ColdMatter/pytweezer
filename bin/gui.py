#!/usr/bin/python3
"""
Role-based tabbed GUIs for pytweezer.

Two windows share a single tabbed shell (:class:`TabbedGUI`) and differ only in
which panels they expose:

* ``pytweezer-server`` (:func:`server_main`) runs on the server PC and gives full
  start/stop control of the server processes: Servers | Devices | Streams | Applets.
* ``pytweezer-client`` (:func:`client_main`) runs on client PCs: Server Status |
  Devices | Applets | Streams. The server tab is view-only because the servers
  run on a remote machine.

Each panel is reused from its original module (``bin.process_manager``,
``pytweezer.GUI.applet_launcher``, ``pytweezer.GUI.streammonitor``). The new
components are :class:`ServerStatusPanel` (client-side reachability probe of the
hub/logger servers) and :class:`DeviceStatusPanel` (the server-published,
cross-PC device-server status feed).
"""

import logging
import os
import signal
import sys
import time

from PyQt5.QtCore import QSettings, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from bin.process_manager import ServerManager, DeviceManager
from pytweezer.GUI.applet_launcher import AppletLauncher
from pytweezer.GUI.streammonitor import make_stream_monitor
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers.device_status import DeviceStatusClient
from pytweezer.servers.reachability import is_reachable

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer GUI")


# --------------------------------------------------------------------------- #
# View-only server status panel (client GUI)
# --------------------------------------------------------------------------- #

def _status_port(params):
    """Return the TCP port to probe for a server, or ``None`` if it has none.

    Hubs expose ``pub_port``/``sub_port``; managers/loggers expose ``port``.
    Pure subscribers (Datalogger, Imagelogger) bind no port and are unprobeable.
    """
    return params.get("port") or params.get("pub_port")


class _ProbeWorker(QThread):
    """Background reachability poller.

    Emits ``{server_name: True|False|None}`` every ``interval`` seconds, where
    ``None`` means the server has no probeable port. Runs off the GUI thread so
    connect timeouts to a down host never freeze the UI.
    """

    results_ready = pyqtSignal(dict)

    def __init__(self, targets, interval=3.0, timeout=0.3, parent=None):
        super().__init__(parent)
        self._targets = targets  # name -> (host, port|None)
        self._interval = interval
        self._timeout = timeout
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            results = {}
            for name, (host, port) in self._targets.items():
                if port is None or host is None:
                    results[name] = None
                else:
                    results[name] = is_reachable(host, port, self._timeout)
            self.results_ready.emit(results)
            # Sleep in small slices so stop() takes effect promptly.
            waited = 0.0
            while self._running and waited < self._interval:
                time.sleep(0.1)
                waited += 0.1


class ServerStatusPanel(QWidget):
    """Read-only grid of server reachability indicators for the client GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        conf = ConfigReader.getConfiguration()
        servers = conf.get("Servers", {})

        layout = QGridLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Server</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Address</b>"), 0, 2)

        self._indicators = {}
        targets = {}
        for row, (name, params) in enumerate(sorted(servers.items()), start=1):
            host = params.get("host")
            port = _status_port(params)
            targets[name] = (host, port)

            dot = QLabel("●")  # ●
            self._set_indicator(dot, None)
            addr = f"{host}:{port}" if port else f"{host}"

            layout.addWidget(dot, row, 0)
            layout.addWidget(QLabel(name), row, 1)
            layout.addWidget(QLabel(addr), row, 2)
            self._indicators[name] = dot

        layout.setRowStretch(len(servers) + 1, 1)
        layout.setColumnStretch(3, 1)
        self.setLayout(layout)

        self._worker = _ProbeWorker(targets)
        self._worker.results_ready.connect(self._apply_results)
        self._worker.start()

    def _apply_results(self, results):
        for name, status in results.items():
            dot = self._indicators.get(name)
            if dot is not None:
                self._set_indicator(dot, status)

    @staticmethod
    def _set_indicator(dot, status):
        if status is True:
            color, tip = "#2ecc40", "reachable"
        elif status is False:
            color, tip = "#ff4136", "unreachable"
        else:
            color, tip = "#aaaaaa", "n/a (no probeable port)"
        dot.setStyleSheet(f"color: {color}; font-size: 16px;")
        dot.setToolTip(tip)

    def closeEvent(self, event):
        try:
            self._worker.stop()
            self._worker.wait(1500)
        except Exception:
            logger.exception("Failed to stop server-status probe thread")
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Cross-PC device-server status panel (published by the server PC)
# --------------------------------------------------------------------------- #

class DeviceStatusPanel(QWidget):
    """Displays the cross-PC device-server status published by the server PC.

    Read-only: rows are driven entirely by the snapshots from
    :class:`DeviceStatusClient`, so this shows *all* device servers across every
    PC (unlike the host-filtered ``DeviceManager`` control tab, which only shows
    the local machine's devices).
    """

    _STATE_STYLE = {
        "up": ("#2ecc40", "reachable"),
        "down": ("#ff4136", "unreachable"),
        "disabled": ("#aaaaaa", "disabled (inactive in config)"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QGridLayout()
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addWidget(QLabel("<b>Device</b>"), 0, 1)
        self._layout.addWidget(QLabel("<b>Address</b>"), 0, 2)
        self._layout.addWidget(QLabel("<b>State</b>"), 0, 3)
        self._layout.addWidget(QLabel("<b>Last seen</b>"), 0, 4)
        self._layout.setColumnStretch(5, 1)
        self.setLayout(self._layout)

        self._hint = QLabel("waiting for device-status server…")
        self._layout.addWidget(self._hint, 1, 1, 1, 4)
        self._rows = {}

        try:
            self._client = DeviceStatusClient()
            self._client.status_received.connect(self._apply_snapshot)
        except Exception:
            logger.exception("Could not start device-status subscriber")
            self._client = None
            self._hint.setText("device-status server unavailable")

    def _apply_snapshot(self, devices):
        if self._hint is not None:
            self._hint.setParent(None)
            self._hint = None
        now = time.time()
        for name in sorted(devices):
            info = devices[name]
            self._ensure_row(name)
            dot, addr_label, state_label, age_label = self._rows[name]
            host, port = info.get("host"), info.get("port")
            state = info.get("state", "down")
            color, tip = self._STATE_STYLE.get(state, ("#aaaaaa", state))
            dot.setStyleSheet(f"color: {color}; font-size: 16px;")
            dot.setToolTip(tip)
            addr_label.setText(f"{host}:{port}" if port else str(host))
            state_label.setText(state)
            last_seen = info.get("last_seen")
            age_label.setText(f"{now - last_seen:.0f}s ago" if last_seen else "—")

    def _ensure_row(self, name):
        if name in self._rows:
            return
        row = len(self._rows) + 1
        dot = QLabel("●")
        addr_label, state_label, age_label = QLabel(), QLabel(), QLabel()
        self._layout.addWidget(dot, row, 0)
        self._layout.addWidget(QLabel(name), row, 1)
        self._layout.addWidget(addr_label, row, 2)
        self._layout.addWidget(state_label, row, 3)
        self._layout.addWidget(age_label, row, 4)
        self._rows[name] = (dot, addr_label, state_label, age_label)

    def closeEvent(self, event):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            logger.exception("Failed to stop device-status subscriber")
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Shared tabbed window shell
# --------------------------------------------------------------------------- #

class TabbedGUI(QMainWindow):
    """A titled main window hosting the given panels as tabs.

    ``tabs`` is a list of ``(label, widget)``. Grid/tree panels are wrapped in a
    scroll area so they stay usable when embedded; an already-tabbed panel (the
    stream monitor) is added as-is.

    Deliberately a plain ``QMainWindow`` (not ``BMainWindow``): the shell has no
    need for a ``Properties`` connection, and avoiding it keeps startup fast and
    independent of the Propertyhub/Propertylogger being reachable. Geometry is
    persisted with ``QSettings``, mirroring the ``BWidget`` convention.
    """

    def __init__(self, name, tabs, parent=None):
        super().__init__(parent)
        self._name = name
        self.setWindowTitle(name)

        geometry = QSettings("pytweezer", name).value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        self._panels = []
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        for label, widget in tabs:
            self._panels.append(widget)
            self._tabs.addTab(self._wrap(widget), label)

    @staticmethod
    def _wrap(widget):
        # Nested tab widgets (the stream monitor) manage their own layout.
        if isinstance(widget, QTabWidget):
            return widget
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def closeEvent(self, event):
        settings = QSettings("pytweezer", self._name)
        settings.setValue("geometry", self.saveGeometry())
        # Force a synchronous flush: _run() calls os._exit(0) right after the
        # event loop ends, which would otherwise skip QSettings' deferred write.
        settings.sync()
        # Qt only delivers closeEvent to top-level widgets, so explicitly close
        # each embedded panel to trigger its subprocess/thread teardown.
        for panel in self._panels:
            try:
                panel.close()
            except Exception:
                logger.exception("Error closing panel %r", panel)
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# GUI builders and entry points
# --------------------------------------------------------------------------- #

def build_server_gui():
    return TabbedGUI(
        "Server GUI",
        [
            ("Servers", ServerManager("server")),
            ("Devices", DeviceManager("device")),
            ("Device Status", DeviceStatusPanel()),
            ("Streams", make_stream_monitor("StreamMonitor")),
            ("Applets", AppletLauncher("Applet Launcher")),
        ],
    )


def build_client_gui():
    return TabbedGUI(
        "Client GUI",
        [
            ("Server Status", ServerStatusPanel()),
            ("Device Status", DeviceStatusPanel()),
            ("Devices", DeviceManager("device")),
            ("Applets", AppletLauncher("Applet Launcher")),
            ("Streams", make_stream_monitor("StreamMonitor")),
        ],
    )


def _run(build):
    app = QApplication(sys.argv)
    win = build()
    win.show()

    def on_exit(_signo, _frame):
        logger.info("Received SIGTERM, closing GUI")
        win.close()
        app.quit()

    signal.signal(signal.SIGTERM, on_exit)
    app.exec_()

    # The Qt event loop has ended: the window closed and every panel's closeEvent
    # ran, terminating the child subprocesses they manage. Some embedded panels
    # (via Properties) start *non-daemon* ZMQ ``event_monitor`` threads that loop
    # forever, which would otherwise hang interpreter shutdown. Cleanup is done,
    # so flush logs and exit hard rather than wait on those threads.
    logging.shutdown()
    os._exit(0)


def server_main():
    _run(build_server_gui)


def client_main():
    _run(build_client_gui)


if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "server"
    if role == "client":
        client_main()
    else:
        server_main()
