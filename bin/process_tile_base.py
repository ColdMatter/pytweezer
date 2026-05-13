from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton
import subprocess

from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import icon_path
from pytweezer.logging_utils import get_logger
logger = get_logger("pytweezer.GUI.process_tile_base")

class ProcessTile(QFrame):
    """Shared process tile with start/stop controls and status polling."""

    def __init__(self, script='', name='', active=False, category='', parent=None, tooltip=None):
        self.process = None
        self.script = script
        self.processname = name
        self.category = category
        super().__init__(parent)

        self.setStyleSheet(
            "QFrame {background-color: rgb(210,230,240); color: blue; margin: 1px; border: 2px solid rgb(220, 240, 255);}"
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        start_button = QPushButton(name)
        if tooltip is not None:
            start_button.setToolTip(tooltip)
        start_button.setStyleSheet(
            "QPushButton {color: blue; background-color: rgb(0, 255, 127);}"
        )
        self.startButton = start_button
        start_button.clicked.connect(self.startProcess)
        layout.addWidget(start_button)

        kill_button = QPushButton('')
        kill_button.setMaximumWidth(20)
        kill_button.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))
        kill_button.clicked.connect(self.terminateProcess)
        layout.addWidget(kill_button)

        self.setLayout(layout)
        self.setAttribute(QtCore.Qt.WA_StyledBackground)

        if active:
            self.startProcess()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.updateStatus)
        self.timer.start()

    def __del__(self):
        print('__del__', self.processname)
        self.killProcess()


    def startProcess(self):
        self.terminateProcess()
        # print_error(
        #     f"startProcess(): Starting {self.processname}.",
        #     'bold',
        # )
        logger.info(f"Starting process {self.processname} with script {self.script}")
        self.process = subprocess.Popen(['python3', self.script, self.category + self.processname])

    def terminateProcess(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except Exception as error:
                # print_error(
                #     f"process termination failed killing process {self.processname}",
                #     'error',
                # )
                logger.error(f"process termination failed killing process {self.processname}")
                # logger.error(error)
                self.process.kill()
            if self.process.poll() is not None:
                self._set_start_button_style("color: gray")
                self.process = None

    def killProcess(self):
        if self.process is not None:
            logger.info(f"Killing process {self.processname}")
            self.process.kill()
            if self.process.poll() is not None:
                self._set_start_button_style("color: gray")
                self.process = None

    def updateStatus(self):
        if self.process is not None:
            if self.process.poll() is None:
                self._set_start_button_style("color: blue")
            else:
                self._set_start_button_style("color: red")
        else:
            self._set_start_button_style("color: gray")

    def _set_start_button_style(self, style):
        try:
            self.startButton.setStyleSheet(style)
        except RuntimeError:
            # QWidget may already be deleted during teardown.
            pass
