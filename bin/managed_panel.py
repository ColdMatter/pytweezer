"""Combined process-control + live-status panel.

Each row shows a traffic-light status (dot + text label + address) alongside a
single Start/Stop toggle, when this PC is allowed to manage that process. This
replaces the previous four-way split (``ServerManager``/``DeviceManager``
control grids in ``bin/process_manager.py`` plus the read-only
``ServerStatusPanel``/``DeviceStatusPanel`` feeds that used to live in
``bin/gui.py``) with one row type reused across the Servers, Loggers and
Devices tabs, so control and status are never in a separate tab from each
other.

A row's status can be driven two ways, but both display the *same* vocabulary
(see :data:`pytweezer.GUI.theme.STATE_STYLE`) so "Running"/"Stopped" reads
identically regardless of source:

* Internally, via :meth:`ManagedRow.enable_self_polling` — the row's own
  ``subprocess.poll()`` result. Used by :class:`ControlPanel` (Servers,
  Loggers): whichever PC controls the process owns its subprocess, so this
  is the authoritative and consistent signal, and it lets the toggle reflect
  the true child state (running / stopped / crashed).
* Externally, via :meth:`ManagedRow.apply_status` — used by
  :class:`DevicesPanel` (fed by the cross-PC
  :class:`~pytweezer.servers.device_status.DeviceStatusClient`) and by
  :class:`ControlPanel` in its view-only client mode (fed by
  :class:`_ProbeWorker`, a local TCP reachability check).
"""

import subprocess
import time
from socket import gethostname

from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.GUI.theme import apply_dot_style, apply_label_style
from pytweezer.configuration.config import HOSTS
from pytweezer.servers import tweezerpath
from pytweezer.servers.configreader import ConfigReader, DEVICE_SERVER_SCRIPT
from pytweezer.servers.device_status import DeviceStatusClient
from pytweezer.servers.reachability import is_reachable

from pytweezer.logging_utils import get_logger

logger = get_logger("managed_panel")

# States that mean "the process is fully up".
RUNNING_STATES = ("running", "up")

# States where the toggle should offer "Stop" (i.e. clicking it terminates):
# a running process, or one still coming up.
ACTIVE_STATES = RUNNING_STATES + ("starting",)

# How long a feed-driven row stays "Starting" after a start request before it
# accepts a "down"/"stopped" reading as a genuine failed start. Covers the gap
# between the device subprocess spawning and its RPC server becoming reachable
# (camera/hardware init can take several seconds).
_STARTING_GRACE_S = 30.0

# Column heading for the name column, per config category.
_ITEM_LABEL = {"Servers": "Server", "Loggers": "Logger", "Devices": "Device"}


def _status_port(params):
    """Return the TCP port to probe for a server, or ``None`` if it has none.

    Hubs expose ``pub_port``/``sub_port``; managers/loggers expose ``port``.
    Pure subscribers (Datalogger, Imagelogger) bind no port and are unprobeable.
    """
    return params.get("port") or params.get("pub_port")


class _ProbeWorker(QThread):
    """Background TCP reachability poller for servers with a probeable port.

    Emits ``{server_name: True|False}`` every ``interval`` seconds. Runs off
    the GUI thread so a connect timeout to a down host never freezes the UI.
    """

    results_ready = pyqtSignal(dict)

    def __init__(self, targets, interval=3.0, timeout=0.3, parent=None):
        super().__init__(parent)
        self._targets = targets  # name -> (host, port)
        self._interval = interval
        self._timeout = timeout
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            results = {
                name: is_reachable(host, port, self._timeout)
                for name, (host, port) in self._targets.items()
            }
            self.results_ready.emit(results)
            # Sleep in small slices so stop() takes effect promptly.
            waited = 0.0
            while self._running and waited < self._interval:
                time.sleep(0.1)
                waited += 0.1


class ManagedRow(QFrame):
    """One config entry: status indicator + optional Start/Stop toggle."""

    def __init__(self, name, script=None, active=False, tooltip=None,
                 controllable=True, show_last_seen=False, parent=None):
        super().__init__(parent)
        self.process = None
        self.script = script
        self.name = name
        self.controllable = controllable
        self._self_polling = False
        self._starting_since = None  # monotonic time of last start, or None
        self.timer = None

        self.setObjectName("ProcessTile")
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        name_label = QLabel(name)
        if tooltip:
            name_label.setToolTip(tooltip)
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)

        self.addressLabel = QLabel("—")
        self.addressLabel.setMinimumWidth(170)
        layout.addWidget(self.addressLabel)

        self.lastSeenLabel = None
        if show_last_seen:
            self.lastSeenLabel = QLabel("—")
            self.lastSeenLabel.setMinimumWidth(80)
            layout.addWidget(self.lastSeenLabel)

        layout.addStretch(1)

        # Status (dot + label) sits on the right, grouped with the Control
        # toggle so the readout and the action that changes it are together.
        self.dot = QLabel("●")
        self.dot.setObjectName("StatusDot")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)

        self.stateLabel = QLabel()
        self.stateLabel.setObjectName("StatusLabel")
        self.stateLabel.setMinimumWidth(80)
        layout.addWidget(self.stateLabel)

        self.toggleButton = None
        if controllable:
            toggle = QPushButton("Start")
            toggle.setObjectName("ToggleButton")
            toggle.setFixedWidth(72)
            toggle.clicked.connect(self._toggle)
            layout.addWidget(toggle)
            self.toggleButton = toggle

        self.setLayout(layout)
        self._set_state("stopped" if controllable else "unknown")

        if controllable and active:
            self.startProcess()

    def enable_self_polling(self):
        """Track state via this row's own ``subprocess.poll()`` result.

        Used for controllable server/logger rows: the PC that runs the GUI owns
        the subprocess, so its liveness is the authoritative status and drives
        the toggle button directly.
        """
        self._self_polling = True
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._poll_self)
        self.timer.start()
        self._poll_self()

    def _poll_self(self):
        if self.process is not None:
            self._set_state("running" if self.process.poll() is None else "crashed")
        else:
            self._set_state("stopped")

    def apply_status(self, state, address=None, last_seen=None):
        """Externally driven status update (device feed / reachability probe)."""
        # While waiting for a just-started service to come up, keep showing
        # "Starting" instead of the feed's transient "down" — otherwise a start
        # reads as stopped → running → stopped → running. A real "up" clears the
        # window; exceeding the grace period accepts the feed as a failed start.
        if self._starting_since is not None:
            if state in RUNNING_STATES:
                self._starting_since = None
            elif self.process is not None and self.process.poll() is not None:
                # We own this subprocess and it has already exited — the start
                # failed, so surface it now instead of waiting out the grace.
                self._starting_since = None
                self.process = None
                state = "crashed"
            elif time.monotonic() - self._starting_since < _STARTING_GRACE_S:
                state = "starting"
            else:
                self._starting_since = None
        self._set_state(state)
        if address is not None:
            self.addressLabel.setText(address)
        if self.lastSeenLabel is not None:
            self.lastSeenLabel.setText(last_seen if last_seen is not None else "—")

    def _toggle(self):
        state = self.property("state")
        if state in ACTIVE_STATES:
            self.terminateProcess()
        else:
            self.startProcess()

    def startProcess(self):
        if not self.controllable or not self.script:
            return
        self.terminateProcess()
        logger.info(f"Starting process {self.name} with script {self.script}")
        self.process = subprocess.Popen(['python3', self.script, self.name])
        if self._self_polling:
            # The subprocess *is* the service, so liveness is the truth.
            self._poll_self()
        else:
            # Feed-driven (device): the subprocess has spawned but its RPC
            # server isn't reachable yet — show "Starting" until the feed
            # confirms it (see apply_status).
            self._starting_since = time.monotonic()
            self._set_state("starting")

    def terminateProcess(self):
        self._starting_since = None
        if self.process is not None:
            logger.info(f"Stopping process {self.name}")
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except Exception:
                logger.error(f"process termination failed killing process {self.name}")
                self.process.kill()
            if self.process.poll() is not None:
                self.process = None
        self._poll_self() if self._self_polling else self._set_state("stopped")

    def killProcess(self):
        if self.process is not None:
            logger.info(f"Killing process {self.name}")
            self.process.kill()
            if self.process.poll() is not None:
                self.process = None
        if self._self_polling:
            self._poll_self()

    def _set_state(self, state):
        try:
            apply_dot_style(self.dot, state)
            apply_label_style(self.stateLabel, state)
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)
            self._update_toggle(state)
        except RuntimeError:
            # QWidget may already be deleted during teardown.
            pass

    def _update_toggle(self, state):
        if self.toggleButton is None:
            return
        active = state in ACTIVE_STATES
        self.toggleButton.setText("Stop" if active else "Start")
        self.toggleButton.setProperty("kind", "stop" if active else "start")
        self.toggleButton.style().unpolish(self.toggleButton)
        self.toggleButton.style().polish(self.toggleButton)


def _header_row(left_columns, controllable):
    row = QHBoxLayout()
    row.setContentsMargins(10, 0, 10, 0)
    row.setSpacing(10)
    for title, width in left_columns:
        label = QLabel(title)
        label.setProperty("role", "heading")
        label.setMinimumWidth(width)
        row.addWidget(label)
    row.addStretch(1)
    # "Status" heading spans the row's dot (14) + spacing (10) + label (80) so
    # it lines up above the right-hand status readout, next to "Control".
    status = QLabel("Status")
    status.setProperty("role", "heading")
    status.setMinimumWidth(14 + 10 + 80)
    row.addWidget(status)
    if controllable:
        control = QLabel("Control")
        control.setProperty("role", "heading")
        control.setFixedWidth(72)
        row.addWidget(control)
    return row


class ControlPanel(BWidget):
    """Start/Stop controls + live status for a controllable config category.

    Serves both the Servers and Loggers tabs (``category`` selects which). When
    ``controllable`` (the server PC managing its own processes), every row
    self-polls its subprocess, giving a consistent Running/Stopped/Crashed
    state and a working toggle. When not controllable (the client GUI viewing
    servers that run on a different PC), rows are view-only and their status
    comes from a TCP reachability probe of any server that exposes a port.
    """

    def __init__(self, name, category, controllable=True, parent=None):
        super().__init__(name, parent=parent, create_props=False)
        self.category = category
        self._rows = {}
        self._worker = None

        outer = QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)
        self.setLayout(outer)

        item_label = _ITEM_LABEL.get(category, "Name")
        columns = [(item_label, 150), ("Address", 170)]
        outer.addLayout(_header_row(columns, controllable))

        conf = ConfigReader.getConfiguration()
        entries = conf.get(category, {})
        probe_targets = {}
        for pname, params in sorted(entries.items()):
            host = params.get("host")
            port = _status_port(params)
            row = ManagedRow(
                pname,
                script=tweezerpath + "/bin/" + params["script"],
                active=params.get("active", False),
                tooltip=params.get("tooltip"),
                controllable=controllable,
                show_last_seen=False,
            )
            row.addressLabel.setText(f"{host}:{port}" if port else (str(host) if host else "—"))
            if controllable:
                # Authoritative: this PC owns the subprocess.
                row.enable_self_polling()
            elif host is not None and port is not None:
                probe_targets[pname] = (host, port)
            else:
                row.addressLabel.setToolTip("no probeable port")
            self._rows[pname] = row
            outer.addWidget(row)
        outer.addStretch(1)

        if probe_targets:
            self._worker = _ProbeWorker(probe_targets)
            self._worker.results_ready.connect(self._apply_probe_results)
            self._worker.start()

    def _apply_probe_results(self, results):
        for pname, reachable in results.items():
            row = self._rows.get(pname)
            if row is not None:
                row.apply_status("up" if reachable else "down")

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(1500)
        for row in self._rows.values():
            row.terminateProcess()
        logger.info("Terminated all %s processes.", self.category)
        super().closeEvent(event)


class DevicesPanel(BWidget):
    """Devices tab: Start/Stop toggle (for local devices) + cross-PC status.

    Replaces the previous split Devices (host-filtered control grid) /
    Device Status (cross-PC read-only feed) tabs with one row per device:
    every device is listed, but the toggle only appears for devices whose
    config ``host`` matches this machine — status still updates for every row
    via the server-published :class:`DeviceStatusClient` feed.
    """

    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent, create_props=False)
        self.host_name = gethostname()
        self.host_addr = HOSTS.get(self.host_name, None)
        if self.host_addr is None:
            self.host_addr = "127.0.0.1"
            logger.warning(
                f"Host {self.host_name} not found in config. Defaulting to localhost ({self.host_addr})."
            )

        self._rows = {}

        outer = QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)
        self.setLayout(outer)

        columns = [("Device", 150), ("Address", 170), ("Last seen", 80)]
        outer.addLayout(_header_row(columns, controllable=True))

        conf = ConfigReader.getConfiguration()
        devices = conf.get("Devices", {})
        for pname, params in sorted(devices.items()):
            local = self.check_host(params.get("host"))
            row = ManagedRow(
                pname,
                script=(
                    tweezerpath + "/bin/" + DEVICE_SERVER_SCRIPT
                    if local
                    else None
                ),
                active=params.get("active", False) if local else False,
                tooltip=params.get("tooltip"),
                controllable=local,
                show_last_seen=True,
            )
            self._rows[pname] = row
            outer.addWidget(row)
        outer.addStretch(1)

        self._client = None
        try:
            self._client = DeviceStatusClient()
            self._client.status_received.connect(self._apply_snapshot)
        except Exception:
            logger.exception("Could not start device-status subscriber")

    def check_host(self, host_addr):
        """Whitespace-/case-insensitive match against this machine's host."""
        if host_addr is None:
            return False
        return host_addr.strip().lower() == self.host_addr.strip().lower()

    def _apply_snapshot(self, devices):
        now = time.time()
        for pname, info in devices.items():
            row = self._rows.get(pname)
            if row is None:
                continue
            host, port = info.get("host"), info.get("port")
            state = info.get("state", "down")
            address = f"{host}:{port}" if port else str(host)
            last_seen = info.get("last_seen")
            age = f"{now - last_seen:.0f}s ago" if last_seen else "—"
            row.apply_status(state, address=address, last_seen=age)

    def closeEvent(self, event):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            logger.exception("Failed to stop device-status subscriber")
        for row in self._rows.values():
            row.terminateProcess()
        logger.info("Terminated all Devices processes.")
        super().closeEvent(event)
