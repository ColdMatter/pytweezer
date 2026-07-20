from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
import subprocess

from pytweezer.servers import icon_path
from pytweezer.GUI.theme import apply_dot_style, apply_label_style
from pytweezer.logging_utils import get_logger
logger = get_logger("process_manager")


class ProcessTile(QFrame):
    """Shared process tile with start/stop controls and status polling.

    State is shown redundantly (colour dot + text label + a coloured left
    border on the tile itself) so it never depends on colour perception alone.
    """

    def __init__(self, script='', name='', active=False, category='', parent=None, tooltip=None):
        self.process = None
        self.script = script
        self.processname = name
        self.category = category
        super().__init__(parent)

        self.setObjectName("ProcessTile")
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.dot = QLabel("●")
        self.dot.setObjectName("StatusDot")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)

        self.stateLabel = QLabel()
        self.stateLabel.setObjectName("StatusLabel")
        self.stateLabel.setMinimumWidth(64)
        layout.addWidget(self.stateLabel)

        start_button = QPushButton(name)
        if tooltip is not None:
            start_button.setToolTip(tooltip)
        start_button.setObjectName("TileNameButton")
        self.startButton = start_button
        start_button.clicked.connect(self.startProcess)
        layout.addWidget(start_button, 1)

        kill_button = QPushButton('')
        kill_button.setObjectName("KillButton")
        kill_button.setFixedSize(24, 24)
        kill_button.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))
        kill_button.setToolTip("Terminate")
        kill_button.clicked.connect(self.terminateProcess)
        layout.addWidget(kill_button)

        self.setLayout(layout)

        self._set_state("stopped")

        if active:
            self.startProcess()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.updateStatus)
        self.timer.start()

    def __del__(self):
        self.killProcess()

    def startProcess(self):
        self.terminateProcess()
        logger.info(f"Starting process {self.processname} with script {self.script}")
        self.process = subprocess.Popen(['python3', self.script, self.processname])

    def terminateProcess(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except Exception:
                logger.error(f"process termination failed killing process {self.processname}")
                self.process.kill()
            if self.process.poll() is not None:
                self._set_state("stopped")
                self.process = None

    def killProcess(self):
        if self.process is not None:
            logger.info(f"Killing process {self.processname}")
            self.process.kill()
            if self.process.poll() is not None:
                self._set_state("stopped")
                self.process = None

    def updateStatus(self):
        if self.process is not None:
            if self.process.poll() is None:
                self._set_state("running")
            else:
                self._set_state("crashed")
        else:
            self._set_state("stopped")

    def _set_state(self, state):
        try:
            apply_dot_style(self.dot, state)
            apply_label_style(self.stateLabel, state)
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)
        except RuntimeError:
            # QWidget may already be deleted during teardown.
            pass
