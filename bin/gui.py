#!/usr/bin/python3
"""
Role-based tabbed GUIs for pytweezer.

Two windows share a single tabbed shell (:class:`TabbedGUI`) and differ only in
which panels they expose:

* ``pytweezer-server`` (:func:`server_main`) runs on the server PC and gives full
  start/stop control of the server processes: Servers | Devices | Loggers |
  Streams | Applets | Analysis | Properties.
* ``pytweezer-client`` (:func:`client_main`) runs on client PCs: Server Status |
  Devices | Applets | Analysis | Properties | Streams. The Servers tab is
  view-only because the servers run on a remote machine.

Each panel is reused from its original module (``bin.managed_panel``,
``pytweezer.GUI.applet_launcher``, ``pytweezer.GUI.analysismanager``,
``pytweezer.GUI.property_editor``, ``pytweezer.GUI.streammonitor``). The
Servers, Loggers and Devices tabs (:class:`~bin.managed_panel.ControlPanel` /
:class:`~bin.managed_panel.DevicesPanel`) show live status and a single
Start/Stop toggle in the same row, rather than splitting control and status
into separate tabs.

Each panel lives in a dock widget, so any tab can be dragged out into its own
window, re-docked, or rearranged; the layout is remembered between sessions.
"""

import logging
import os
import signal
import sys

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from bin.managed_panel import ControlPanel, DevicesPanel
from pytweezer.GUI.applet_launcher import AppletLauncher
from pytweezer.GUI.streammonitor import make_stream_monitor
from pytweezer.GUI.theme import DARK_STYLESHEET

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer GUI")


def _safe_panel(label, factory):
    """Build a panel, substituting a placeholder if construction raises.

    Several panels connect to the Propertyhub / Analysis Manager on
    construction and raise if those aren't reachable. Isolating each build
    keeps one unreachable service from aborting the whole window.
    """
    try:
        return factory()
    except Exception:
        logger.exception("Failed to build %r panel", label)
        placeholder = QLabel(f"{label} unavailable — see logs.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        return placeholder


# --------------------------------------------------------------------------- #
# Shared tabbed window shell
# --------------------------------------------------------------------------- #

class TabbedGUI(QMainWindow):
    """A titled main window hosting each panel in a dock widget.

    ``tabs`` is a list of ``(label, widget)``. Panels appear as tabs by default
    (their docks are tabified into a single area), but each can be dragged out
    into its own floating window, re-docked, or rearranged — QMainWindow's dock
    system provides that for free. Grid/tree panels are wrapped in a scroll area
    so they stay usable when embedded; an already-tabbed panel (the stream
    monitor) is added as-is.

    Deliberately a plain ``QMainWindow`` (not ``BMainWindow``): the shell has no
    need for a ``Properties`` connection, and avoiding it keeps startup fast and
    independent of the Propertyhub/Propertylogger being reachable. Window
    geometry and the dock layout are persisted with ``QSettings``, mirroring the
    ``BWidget`` convention.
    """

    def __init__(self, name, tabs, parent=None):
        super().__init__(parent)
        self._name = name
        self.setWindowTitle(name)
        self.setDockNestingEnabled(True)
        # Tabbed docks default to a bottom tab bar; put it on top to match the
        # conventional tab-widget look.
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        # QMainWindow always reserves a central area; a zero-size placeholder
        # lets the tabified docks fill the window like a plain tab widget.
        central = QWidget()
        central.setMaximumSize(0, 0)
        self.setCentralWidget(central)

        self._panels = []
        self._docks = []
        previous = None
        for label, widget in tabs:
            self._panels.append(widget)
            dock = QDockWidget(label, self)
            # Stable objectName is required for saveState()/restoreState().
            dock.setObjectName(f"dock:{label}")
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            # Movable + floatable, but not closable: hiding a panel with no easy
            # way back would just be a footgun.
            dock.setFeatures(
                QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
            )
            dock.setWidget(self._wrap(widget))
            self.addDockWidget(Qt.TopDockWidgetArea, dock)
            if previous is not None:
                self.tabifyDockWidget(previous, dock)
            previous = dock
            self._docks.append(dock)
        if self._docks:
            self._docks[0].raise_()

        settings = QSettings("pytweezer", name)
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        dockstate = settings.value("dockstate")
        if dockstate is not None:
            self.restoreState(dockstate)

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
        # saveState() captures the dock layout, including any floated-out panels.
        settings.setValue("dockstate", self.saveState())
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
    # Imported lazily (inside the builder, after the QApplication exists) so
    # property_editor's module-level QApplication reuse doesn't fight _run().
    from pytweezer.GUI.property_editor import PropEdit
    from pytweezer.GUI.analysismanager import AnalysisManager

    return TabbedGUI(
        "Server GUI",
        [
            ("Servers", _safe_panel("Servers", lambda: ControlPanel("server", "Servers", controllable=True))),
            ("Devices", _safe_panel("Devices", lambda: DevicesPanel("device"))),
            ("Loggers", _safe_panel("Loggers", lambda: ControlPanel("logger", "Loggers", controllable=True))),
            ("Streams", _safe_panel("Streams", lambda: make_stream_monitor("StreamMonitor"))),
            ("Applets", _safe_panel("Applets", lambda: AppletLauncher("Applet Launcher"))),
            ("Analysis", _safe_panel("Analysis", lambda: AnalysisManager("Analysis Manager UI"))),
            ("Properties", _safe_panel("Properties", lambda: PropEdit("/"))),
        ],
    )


def build_client_gui():
    from pytweezer.GUI.property_editor import PropEdit
    from pytweezer.GUI.analysismanager import AnalysisManager

    return TabbedGUI(
        "Client GUI",
        [
            ("Server Status", _safe_panel("Server Status", lambda: ControlPanel("server status", "Servers", controllable=False))),
            ("Devices", _safe_panel("Devices", lambda: DevicesPanel("device"))),
            ("Applets", _safe_panel("Applets", lambda: AppletLauncher("Applet Launcher"))),
            ("Analysis", _safe_panel("Analysis", lambda: AnalysisManager("Analysis Manager UI"))),
            ("Properties", _safe_panel("Properties", lambda: PropEdit("/"))),
            ("Streams", _safe_panel("Streams", lambda: make_stream_monitor("StreamMonitor"))),
        ],
    )


def _run(build):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    win = build()
    win.show()

    def on_exit(signo, _frame):
        # Registering SIGINT here too matters: without it, Python's default
        # handler raises a bare KeyboardInterrupt wherever the interpreter next
        # checks for it (typically deep inside some unrelated QTimer callback's
        # blocking zmq poll() call), which just gets dumped as a traceback by
        # Qt's default unhandled-exception hook instead of closing the window.
        logger.info("Received %s, closing GUI", signal.Signals(signo).name)
        win.close()
        app.quit()

    signal.signal(signal.SIGTERM, on_exit)
    signal.signal(signal.SIGINT, on_exit)
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
