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

#: Reaches that only an operator can fix, so the PC is checked this seldom.
_NO_ACCESS = ("no key", "host key changed")
_NO_ACCESS_INTERVAL_S = 300.0

#: How long after a Start/Stop has reported "Starting…"/"Stopping…" holds against polls that disagree.
_PENDING_TIMEOUT_S = 60.0

#: Widest a failure's reason gets on the line before it is elided (in full in its tooltip).
_STATE_TEXT_MAX_PX = 420

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
    after one succeeds. A machine whose SSH key or host key was refused is checked
    only every five minutes, or at the next Start/Stop asked for on any machine.
    A Start/Stop runs on its own thread, since a stop can take over 30 s and must
    not hold up the polls.
    """

    #: The polled machines' states (machine name -> GuiState), and the
    #: ``time.monotonic()`` at which the poll started.
    states_ready = pyqtSignal(dict, float)
    action_done = pyqtSignal(object)  # machines.Result

    def __init__(self, machine_list, interval_s=POLL_INTERVAL_S, parent=None):
        super().__init__(parent)
        self._machines = list(machine_list)
        self._interval_s = interval_s
        self._running = True
        self._wake = threading.Event()
        self._asked_at = float("-inf")  # the latest Start/Stop asked for

    def act(self, action, machine):
        """Run ``machines.start`` or ``machines.stop`` on ``machine``."""
        self._asked_at = time.monotonic()
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
        last_checks = {}  # machine name -> (when the check started, reach)
        with ThreadPoolExecutor(max_workers=max(len(self._machines), 1)) as pool:
            while self._running:
                self._wake.clear()
                polled_at = time.monotonic()
                due = [
                    machine
                    for machine in self._machines
                    if self._due(last_checks.get(machine.name), polled_at)
                ]
                states = dict(pool.map(self._check, due))
                states.pop(None, None)
                for name, state in states.items():
                    last_checks[name] = (polled_at, state.reach)
                if self._running and states:
                    self.states_ready.emit(states, polled_at)
                self._wake.wait(self._interval_s)

    def _due(self, last_check, now):
        if last_check is None:
            return True
        checked_at, reach = last_check
        return (
            reach not in _NO_ACCESS
            or self._asked_at > checked_at
            or now - checked_at >= _NO_ACCESS_INTERVAL_S
        )

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
        # "start"/"stop" from the click until the action has reported and a poll
        # agrees: a stop's cleanup runs on after the lock is free, and starting a
        # GUI then would have its devices killed.
        self._pending = None
        self._acting = False  # the action has not reported yet
        self._reported_at = 0.0
        self._requested = None  # the last action asked for
        # (action, detail) until the next click, or until a poll started after
        # the failure was reported shows a state other than ``_state_at_failure``.
        self._failure = None
        self._state_at_failure = None

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

    def apply(self, state, polled_at):
        """Show a polled ``machines.GuiState``, from a poll started at ``polled_at``."""
        self._state = state
        if (
            self._failure is not None
            and polled_at > self._reported_at
            and state != self._state_at_failure
        ):
            self._failure = None
        if (
            self._pending is not None
            and not self._acting
            and (
                self._poll_agrees(state)
                or time.monotonic() - self._reported_at > _PENDING_TIMEOUT_S
            )
        ):
            self._pending = None
        self._render()

    def action_finished(self, result):
        """Take a Start/Stop's report: a failure shows at once, a success waits for a poll to agree."""
        self._acting = False
        self._reported_at = time.monotonic()
        if result.ok:
            return
        logger.warning(
            "Could not %s pytweezer on %s: %s",
            self._requested,
            self.name,
            result.detail,
        )
        self._failure = (self._requested, result.detail)
        self._state_at_failure = self._state
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
            answer = QMessageBox.question(self, "Stop pytweezer", self._stop_question())
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._pending = self._requested = action
        self._acting = True
        self._failure = None
        self._render()
        self.requested.emit(action)

    def _stop_question(self):
        if self.machine.launcher == "server":
            return (
                f"Stop pytweezer on {self.name}? Its GUI and devices will close, "
                "and so will the shared servers (hubs, loggers, Experiment "
                "Manager) that every PC relies on."
            )
        return f"Stop pytweezer on {self.name}? Its GUI and devices will close."

    def _render(self):
        tooltip = ""
        if self._pending is not None:
            style, text = _PENDING[self._pending]
        elif self._failure is not None:
            action, tooltip = self._failure
            style, text = "failed", f"{action} failed: {tooltip}"
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
        if self._failure is not None:
            self.stateLabel.ensurePolished()
            self.stateLabel.setText(
                self.stateLabel.fontMetrics().elidedText(
                    text, QtCore.Qt.TextElideMode.ElideRight, _STATE_TEXT_MAX_PX
                )
            )
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
