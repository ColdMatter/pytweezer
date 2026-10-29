"""Calls to one ``NICards`` device, off the GUI thread."""

import queue
import threading
from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.ni_cards")

_DEVICE_TIMEOUT_S = 5
#: How sipyco reports a server that is gone: refused, reset or timed out
#: (``OSError``), or closed mid-reply (``EOFError``).
_CONNECTION_ERRORS = (OSError, EOFError)


def _connect_to(name: str) -> Any:
    from pytweezer.configuration import config

    if config.SIMULATING:
        from pytweezer.experiment.simulation import SimulatedDevices

        return SimulatedDevices().get(name)
    from pytweezer.servers.device_client import get_device

    return get_device(name, timeout=_DEVICE_TIMEOUT_S)


class CardsClient(QtCore.QObject):
    """Queues calls to the device named ``name`` and reports results as signals.

    Failing to connect, or losing the connection mid-call, drops the connection,
    discards the calls still queued (so no write lands late) and reports
    ``connection_changed(False)``; that is reported after every failed attempt,
    and the next call reconnects. Any other error is the driver refusing the call
    and is reported by ``failed(method, message)``.
    """

    layout_received = QtCore.pyqtSignal(dict)
    state_received = QtCore.pyqtSignal(dict)
    values_received = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str, str)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, name: str, connect: Callable[[], Any] | None = None):
        super().__init__()
        self.name = name
        self._connect = connect or (lambda: _connect_to(name))
        self._calls: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def fetch_layout(self) -> None:
        self._calls.put(("layout", (), self.layout_received))

    def fetch_state(self) -> None:
        self._calls.put(("state", (), self.state_received))

    def write_do(self, physical: str, high: bool) -> None:
        self._calls.put(("write_do", (physical, high), None))
        self.fetch_state()

    def write_ao(self, physical: str, volts: float) -> None:
        self._calls.put(("write_ao", (physical, volts), None))
        self.fetch_state()

    def read(self, physicals: list[str]) -> None:
        self._calls.put(("read", (list(physicals),), self.values_received))

    def close(self) -> None:
        self._calls.put(None)

    def _run(self) -> None:
        device = None
        connected = None
        while (call := self._calls.get()) is not None:
            method, args, signal = call
            try:
                if device is None:
                    device = self._connect()
                result = getattr(device, method)(*args)
            except Exception as error:
                if device is not None and not isinstance(error, _CONNECTION_ERRORS):
                    self.failed.emit(method, str(error) or type(error).__name__)
                    continue
                if connected is not False:
                    logger.warning("%s: %s failed: %r", self.name, method, error)
                self._drop(device)
                device = None
                connected = False
                self._discard_queued()
                self.connection_changed.emit(False)
                continue
            if connected is not True:
                connected = True
                self.connection_changed.emit(True)
            if signal is not None:
                signal.emit(result)
        self._drop(device)

    def _discard_queued(self) -> None:
        while True:
            try:
                call = self._calls.get_nowait()
            except queue.Empty:
                return
            if call is None:
                self._calls.put(None)
                return

    @staticmethod
    def _drop(device) -> None:
        close = getattr(device, "close_rpc", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
