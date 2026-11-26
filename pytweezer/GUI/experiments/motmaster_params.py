"""Search-and-add boxes for MOTMaster script parameters, and the scripts on offer, in the argument editor."""

import threading
from collections.abc import Callable, Collection
from typing import Any

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCompleter,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from pytweezer.GUI.components import set_state

_SEPARATOR = " · "
_DEVICE_TIMEOUT_S = 10


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class DeviceParameterSource:
    """Reads a script's parameters, and the scripts on offer, from a MOTMaster device (or the simulated one)."""

    def __init__(self, simulated: Callable[[], bool] = lambda: False) -> None:
        self._simulated = simulated
        self._simulated_devices = None

    def __call__(self, device: str, script: str) -> dict[str, Any]:
        return self._ask(device, lambda client: client.script_parameters(script))

    def scripts(self, device: str) -> list[str]:
        return self._ask(device, lambda client: client.script_names())

    def _ask(self, device: str, call: Callable[[Any], Any]) -> Any:
        if self._simulated():
            if self._simulated_devices is None:
                from pytweezer.experiment.simulation import SimulatedDevices

                self._simulated_devices = SimulatedDevices()
            return call(self._simulated_devices.get(device))
        from pytweezer.servers.device_client import get_device

        client = get_device(device, timeout=_DEVICE_TIMEOUT_S)
        try:
            return call(client)
        finally:
            client.close_rpc()


class DeviceFetcher(QtCore.QObject):
    """Calls ``source(*key)`` off the GUI thread and caches each answer by ``key``."""

    fetched = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(object, str)
    _finished = QtCore.pyqtSignal(object, object, object)

    def __init__(self, source: Callable[..., Any], *, threaded: bool = True):
        super().__init__()
        self.source = source
        self.threaded = threaded
        self.cache: dict[tuple, Any] = {}
        self._pending: set[tuple] = set()
        self._finished.connect(self._on_finished)

    def request(self, *key: str) -> None:
        if key in self.cache:
            self.fetched.emit(key, self.cache[key])
        elif key not in self._pending:
            self._pending.add(key)
            if self.threaded:
                threading.Thread(target=self._run, args=(key,), daemon=True).start()
            else:
                self._run(key)

    def forget(self, *key: str) -> None:
        self.cache.pop(key, None)

    def _run(self, key: tuple) -> None:
        try:
            self._finished.emit(key, self.source(*key), None)
        except Exception as error:
            self._finished.emit(key, None, str(error) or type(error).__name__)

    def _on_finished(self, key, parameters, error) -> None:
        self._pending.discard(key)
        if error is None:
            self.cache[key] = parameters
            self.fetched.emit(key, parameters)
        else:
            self.failed.emit(key, error)


class MotMasterBox(QGroupBox):
    """The experiment's sequencer: a search field and the rows chosen from it.

    ``declared`` names script parameters the experiment already sets through an
    argument, and the experiment's argument names, which a script parameter
    cannot take; neither is offered. A ``linked`` sequencer's script also drives
    its secondary MOTMaster, so its parameters include the secondary's.
    """

    parameter_chosen = QtCore.pyqtSignal(str)

    def __init__(
        self,
        device: str,
        script: str,
        parent=None,
        *,
        declared: Collection[str] = (),
        linked: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("EditorGroup")
        self.device, self.script, self.linked = device, script, linked
        self._show_title()
        self.declared = frozenset(declared)
        self.defaults: dict[str, int | float] = {}
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search for a script parameter to add")
        self.search.setFixedWidth(360)
        self.search.setVisible(False)
        self.search.returnPressed.connect(lambda: self.choose(self.search.text()))
        self.retry = QPushButton("Retry")
        self.retry.setToolTip("Ask the MOTMaster for the script's parameters again")
        self.retry.setVisible(False)
        self.completer = QCompleter([], self.search)
        self.completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.completer.activated.connect(self.choose)
        self.search.setCompleter(self.completer)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setColumnStretch(4, 1)
        top = QHBoxLayout()
        top.addWidget(self.search)
        top.addWidget(self.retry)
        top.addWidget(self.status, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(self.grid)
        self.set_loading()

    def set_script(self, script: str) -> None:
        """Run ``script`` instead; its parameters are unknown until :meth:`set_parameters`."""
        self.script = script
        self.defaults = {}
        self._show_title()
        self.set_loading()

    def set_linked(self, linked: bool) -> None:
        self.linked = linked
        self._show_title()

    def _show_title(self) -> None:
        mode = ", linked" if self.linked else ""
        self.setTitle(f"MOTMaster ({self.script}{mode})")
        self.setToolTip(
            f"{self.device} runs {self.script} in linked mode: one script drives "
            "both MOTMasters, so the secondary's parameters are set here too"
            if self.linked
            else ""
        )

    def set_loading(self) -> None:
        self.search.setVisible(False)
        self.retry.setVisible(False)
        self._show_status("Loading the script's parameters…")

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        self.defaults = {name: v for name, v in parameters.items() if is_number(v)}
        labels = [
            f"{name}{_SEPARATOR}{'int' if isinstance(v, int) else 'float'}{_SEPARATOR}{v}"
            for name, v in self.defaults.items()
            if name not in self.declared
        ]
        self.completer.setModel(QtCore.QStringListModel(labels, self.completer))
        self.search.setVisible(True)
        self.retry.setVisible(False)
        self._show_status(
            f"{len(labels)} other script parameters; any not added keep the "
            "script's values"
        )

    def set_error(self, error: str) -> None:
        self.search.setVisible(False)
        self.retry.setVisible(True)
        self._show_status(f"Could not read the script's parameters: {error}", True)

    def _show_status(self, text: str, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setObjectName("StatusLabel" if error else "")
        self.status.setProperty("role", "" if error else "regionHint")
        set_state(self.status, "crashed" if error else "")

    def choose(self, text: str) -> None:
        name = text.split(_SEPARATOR)[0].strip()
        if name in self.defaults and name not in self.declared:
            self.search.clear()
            self.parameter_chosen.emit(name)
