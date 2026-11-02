"""One pytweezer GUI per PC.

Two GUIs on one PC would both start that PC's devices (and the server GUI its
shared servers), so a second launch is refused before it builds anything. The
running GUI listens on :data:`GUI_LOCK_PORT` on localhost: the OS frees the
port however the GUI exits, and the port is the same for every user and every
clone of the repository on the machine. A second launch connects to it, learns
which GUI is running, and asks it to bring its window to the front.
"""

import getpass
import json
import os
import sys
import time

from PyQt6 import QtCore, QtNetwork

#: Just below the range ``get_next_port()`` hands out, so no CONFIG port takes it.
GUI_LOCK_PORT = 7277
_REPLY_TIMEOUT_MS = 2000


class GuiAlreadyRunning(RuntimeError):
    """Another GUI holds this PC; ``owner`` describes it, or is ``None`` if unknown."""

    def __init__(self, message: str, owner: dict | None):
        super().__init__(message)
        self.owner = owner


class GuiLock(QtCore.QObject):
    """This PC's GUI lock, held while the GUI runs; tells later launches who holds it."""

    raise_requested = QtCore.pyqtSignal()

    def __init__(self, server: QtNetwork.QTcpServer, owner: dict):
        super().__init__()
        self._server = server
        self._owner = json.dumps(owner).encode() + b"\n"
        server.newConnection.connect(self._answer)

    def _answer(self):
        while (connection := self._server.nextPendingConnection()) is not None:
            connection.write(self._owner)
            connection.readyRead.connect(lambda c=connection: self._read(c))
            connection.disconnected.connect(connection.deleteLater)

    def _read(self, connection):
        while connection.canReadLine():
            if bytes(connection.readLine()).strip() == b"raise":
                self.raise_requested.emit()

    def close(self):
        self._server.close()


def claim_gui(role: str, port: int = GUI_LOCK_PORT) -> GuiLock:
    """Take this PC's GUI lock for the ``role`` ("server" or "client") GUI.

    Raises:
        GuiAlreadyRunning: another GUI runs here (it is asked to come to the
            front), or another program holds the port.
    """
    server = QtNetwork.QTcpServer()
    if server.listen(
        QtNetwork.QHostAddress(QtNetwork.QHostAddress.SpecialAddress.LocalHost), port
    ):
        owner = {
            "role": role,
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "since": time.strftime("%H:%M on %d %b"),
        }
        return GuiLock(server, owner)
    owner = _ask_running_gui_to_raise(port)
    if owner is None:
        raise GuiAlreadyRunning(
            f"Another program on this PC is using port {port}, which pytweezer "
            "needs to keep to one GUI per PC. Close that program, then start "
            "pytweezer again.",
            None,
        )
    raise GuiAlreadyRunning(
        f"The pytweezer {owner['role']} GUI is already running on this PC "
        f"(started by {owner['user']} at {owner['since']}, process {owner['pid']}); "
        "its window has been brought to the front. Use that one, or close it "
        "before starting another.",
        owner,
    )


def _ask_running_gui_to_raise(port: int) -> dict | None:
    """The running GUI's description, after asking it to raise; ``None`` if it isn't one."""
    connection = QtNetwork.QTcpSocket()
    connection.connectToHost("127.0.0.1", port)
    try:
        if not connection.waitForConnected(_REPLY_TIMEOUT_MS):
            return None
        while not connection.canReadLine():
            if not connection.waitForReadyRead(_REPLY_TIMEOUT_MS):
                return None
        try:
            owner = json.loads(bytes(connection.readLine()))
        except ValueError:
            return None
        if (
            not isinstance(owner, dict)
            or not {"role", "user", "pid", "since"} <= owner.keys()
        ):
            return None
        if sys.platform == "win32":
            # Windows lets a background process take the foreground only when
            # the foreground process (this launch) allows it.
            import ctypes

            ctypes.windll.user32.AllowSetForegroundWindow(int(owner["pid"]))
        connection.write(b"raise\n")
        connection.waitForBytesWritten(_REPLY_TIMEOUT_MS)
        return owner
    finally:
        connection.disconnectFromHost()
        connection.close()
