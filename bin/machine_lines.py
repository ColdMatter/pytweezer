"""The Devices tab's machine lines: pytweezer's state on each lab PC, with Start/Stop.

Every check and action goes over SSH through :mod:`pytweezer.machines` and can
take seconds, so :class:`MachineWorker` runs them off the GUI thread and reports
by signal.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt6 import QtCore
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton

from pytweezer import machines
from pytweezer.GUI.theme import apply_dot_style, apply_label_style
from pytweezer.logging_utils import get_logger

logger = get_logger("machine_lines")

POLL_INTERVAL_S = 15.0

#: How long "Starting…"/"Stopping…" holds against polls that disagree.
_PENDING_TIMEOUT_S = 60.0

#: (``STATE_STYLE`` key for the colour, text) for an SSH check that did not get through.
_NOT_REACHED = {
    "unreachable": ("down", "unreachable"),
    "no key": ("degraded", "no SSH key"),
    "host key changed": ("failed", "host key changed"),
    "no answer": ("degraded", "no answer"),
}
_PENDING = {"start": ("starting", "Starting…"), "stop": ("stopping", "Stopping…")}


class MachineWorker(QThread):
    """Polls every machine's GUI lock and runs Start/Stop, all off the GUI thread.

    Polls every ``interval_s`` and again right after a Start/Stop is asked for and
    after one succeeds. A Start/Stop runs on its own thread, since a stop can take
    over 30 s and must not hold up the polls.
    """

    states_ready = pyqtSignal(dict)  # machine name -> GuiState
    action_done = pyqtSignal(object)  # machines.Result

    def __init__(self, machine_list, interval_s=POLL_INTERVAL_S, parent=None):
        super().__init__(parent)
        self._machines = list(machine_list)
        self._interval_s = interval_s
        self._running = True
        self._wake = threading.Event()

    def act(self, action, machine):
        """Run ``machines.start`` or ``machines.stop`` on ``machine``."""
        self._wake.set()
        threading.Thread(
            target=self._act,
            args=(action, machine),
            name=f"machine-{action}-{machine.name}",
            daemon=True,
        ).start()

    def _act(self, action, machine):
        try:
            result = getattr(machines, action)(machine)
        except Exception as error:
            logger.exception("Could not %s pytweezer on %s", action, machine.name)
            result = machines.Result(machine.name, False, str(error))
        self.action_done.emit(result)
        if result.ok:
            self._wake.set()

    def stop(self):
        self._running = False
        self._wake.set()

    def run(self):
        with ThreadPoolExecutor(max_workers=max(len(self._machines), 1)) as pool:
            while self._running:
                self._wake.clear()
                states = dict(pool.map(self._check, self._machines))
                states.pop(None, None)
                if self._running:
                    self.states_ready.emit(states)
                self._wake.wait(self._interval_s)

    @staticmethod
    def _check(machine):
        try:
            return machine.name, machines.gui_state(machine)
        except Exception:
            logger.exception("Could not check pytweezer on %s", machine.name)
            return None, None


class MachineLine(QFrame):
    """The header of one machine's device rows: its name, address, pytweezer state and Start/Stop.

    The PC this GUI runs on gets no button. ``requested`` carries ``"start"`` or
    ``"stop"`` once the operator has asked (and, for Stop, confirmed).
    """

    requested = pyqtSignal(str)

    def __init__(self, machine, own_pc, simulating, control_width, parent=None):
        super().__init__(parent)
        self.machine = machine
        self.name = machine.name
        self._simulating = simulating
        self._state = None
        self._pending = None  # "start"/"stop" until a poll agrees or it times out
        self._pending_since = 0.0
        self._requested = None  # the last action asked for
        self._failure = None  # (action, detail) until the next poll

        self.setObjectName("GroupHeader")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 4)
        layout.setSpacing(10)

        name_label = QLabel(machine.name)
        name_label.setProperty("role", "regionTitle")
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)

        address = f"{machine.address} · this PC" if own_pc else machine.address
        address_label = QLabel(address)
        address_label.setProperty("role", "regionHint")
        address_label.setMinimumWidth(170)
        layout.addWidget(address_label)
        layout.addStretch(1)

        self.dot = QLabel("●")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)
        self.stateLabel = QLabel()
        self.stateLabel.setObjectName("StatusLabel")
        self.stateLabel.setMinimumWidth(80)
        layout.addWidget(self.stateLabel)

        self.button = None
        if own_pc:
            placeholder = QLabel()
            placeholder.setFixedWidth(control_width)
            layout.addWidget(placeholder)
        else:
            self.button = QPushButton("Start")
            self.button.setObjectName("ToggleButton")
            self.button.setFixedWidth(control_width)
            self.button.clicked.connect(self._clicked)
            layout.addWidget(self.button)
        self._render()

    def apply(self, state):
        """Show a polled ``machines.GuiState``."""
        self._state = state
        self._failure = None
        if self._pending is not None and (
            self._poll_agrees(state)
            or time.monotonic() - self._pending_since > _PENDING_TIMEOUT_S
        ):
            self._pending = None
        self._render()

    def action_finished(self, result):
        """Show a Start/Stop that failed; a successful one waits for the poll to agree."""
        if result.ok:
            return
        logger.warning(
            "Could not %s pytweezer on %s: %s",
            self._requested,
            self.name,
            result.detail,
        )
        self._failure = (self._requested, result.detail)
        self._pending = None
        self._render()

    def _poll_agrees(self, state):
        running = state.reach == "ok" and state.owner is not None
        stopped = state.reach == "ok" and state.owner is None
        return running if self._pending == "start" else stopped

    def _running(self):
        return self._state is not None and self._state.owner is not None

    def _clicked(self):
        action = "stop" if self._running() else "start"
        if action == "stop":
            answer = QMessageBox.question(
                self,
                "Stop pytweezer",
                f"Stop pytweezer on {self.name}? Its GUI and devices will close.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._pending = self._requested = action
        self._pending_since = time.monotonic()
        self._failure = None
        self._render()
        self.requested.emit(action)

    def _render(self):
        tooltip = ""
        if self._pending is not None:
            style, text = _PENDING[self._pending]
        elif self._failure is not None:
            action, tooltip = self._failure
            style, text = "failed", f"{action} failed"
        elif self._state is None:
            style, text = "unknown", "checking…"
        elif self._state.reach != "ok":
            style, text = _NOT_REACHED.get(
                self._state.reach, ("down", self._state.reach)
            )
        elif self._state.owner is not None:
            owner = self._state.owner
            style = "running"
            text = f"{owner['role']} · {owner['user']} since {owner['since']}"
        else:
            style, text = "stopped", "not running"
        apply_dot_style(self.dot, style)
        apply_label_style(self.stateLabel, style, text)
        self.stateLabel.setToolTip(tooltip)
        self._render_button()

    def _render_button(self):
        if self.button is None:
            return
        stop = self._running()
        self.button.setText("Stop" if stop else "Start")
        self.button.setProperty("kind", "stop" if stop else "start")
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)
        blocked = self._blocked_because()
        self.button.setEnabled(blocked is None)
        if blocked is not None:
            self.button.setToolTip(blocked)
        elif stop:
            self.button.setToolTip(
                f"Close pytweezer's GUI on {self.name}, which stops its devices"
            )
        else:
            self.button.setToolTip(
                f"Run {self.name}'s launch task, which opens pytweezer there"
            )

    def _blocked_because(self):
        name = self.name
        if self._simulating:
            return "Disabled in a simulated session"
        if self.machine.os != "windows" or self.machine.launcher is None:
            return (
                "Start and Stop need a Windows PC with a launcher in MACHINES; "
                f"{name} is not one"
            )
        if self._pending is not None:
            return f"Waiting for pytweezer on {name} to {self._pending}"
        if self._state is None:
            return f"Checking pytweezer on {name}"
        return {
            "ok": None,
            "no key": f"Run pytweezer-machines setup {name} on this PC first",
            "host key changed": (
                f"This PC no longer trusts {name}'s SSH host key; "
                "check it before reconnecting"
            ),
            "unreachable": f"{name} does not answer over SSH",
            "no answer": (
                f"Something holds pytweezer's lock on {name} but does not answer"
            ),
        }.get(self._state.reach, f"{name}: {self._state.reach}")
