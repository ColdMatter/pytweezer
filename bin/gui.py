#!/usr/bin/python3
"""
Role-based tabbed GUIs for pytweezer.

Two windows share a single tabbed shell (:class:`TabbedGUI`) and differ only in
which panels they expose:

* ``pytweezer-server`` (:func:`server_main`) runs on the server PC and gives full
  start/stop control of the server processes: Servers | Devices | Loggers |
  Streams | Applets.
* ``pytweezer-client`` (:func:`client_main`) runs on client PCs: Server Status |
  Devices | Applets | Streams. The Servers tab is view-only because the
  servers run on a remote machine.

Each panel is reused from its original module (``bin.managed_panel``,
``pytweezer.GUI.applet_launcher``, ``pytweezer.GUI.streammonitor``). The
Servers, Loggers and Devices tabs (:class:`~bin.managed_panel.ControlPanel` /
:class:`~bin.managed_panel.DevicesPanel`) show live status and a single
Start/Stop toggle in the same row, rather than splitting control and status
into separate tabs.
"""

import logging
import os
import signal
import sys

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QTabWidget

from bin.managed_panel import ControlPanel, DevicesPanel
from pytweezer.GUI.applet_launcher import AppletLauncher
from pytweezer.GUI.streammonitor import make_stream_monitor
from pytweezer.GUI.theme import DARK_STYLESHEET

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer GUI")


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
            ("Servers", ControlPanel("server", "Servers", controllable=True)),
            ("Devices", DevicesPanel("device")),
            ("Loggers", ControlPanel("logger", "Loggers", controllable=True)),
            ("Streams", make_stream_monitor("StreamMonitor")),
            ("Applets", AppletLauncher("Applet Launcher")),
        ],
    )


def build_client_gui():
    return TabbedGUI(
        "Client GUI",
        [
            ("Server Status", ControlPanel("server status", "Servers", controllable=False)),
            ("Devices", DevicesPanel("device")),
            ("Applets", AppletLauncher("Applet Launcher")),
            ("Streams", make_stream_monitor("StreamMonitor")),
        ],
    )


def _run(build):
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
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
