#!/usr/bin/python3
"""
Controller process for pytweezer.
Launches and manages all server processes (Model Sync, MotMaster, Hubs, Loggers, etc.)
Runs as a background daemon that keeps servers alive.
"""

import sys
import signal
import subprocess
import time
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath, icon_path
from pytweezer.analysis.print_messages import print_error


class SingleProcess(QFrame):
    def __init__(self, script='', name='', active=False, category='', parent=None, tooltip=None):
        self.process = None
        self.script = script
        self.processname = name
        self.category = category
        super().__init__(parent)

        self.setStyleSheet("SingleProcess {background-color: rgb(210,230,240);color:blue; margin:7px; border:7px solid rgb(220, 240, 255); } QPushButton {background-color: rgb(210,230,240);color:#000000; margin:1px; border:0px solid rgb(20, 240, 255);} ")
        self.setStyleSheet("SingleProcess {background-color: rgb(210,230,240);color:blue; margin:1px; border:2px solid rgb(220, 240, 255); } ")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        startButton = QPushButton(name)
        if tooltip is not None:
            startButton.setToolTip(tooltip)
        startButton.setStyleSheet("QPushButton {"
                                  "color: blue;"
                                  "background-color: rgb(0, 255, 127);"
                                  "}")
        self.startButton = startButton
        startButton.clicked.connect(self.startProcess)
        layout.addWidget(startButton)
        killButton = QPushButton('')
        killButton.setMaximumWidth(20)
        killButton.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))
        killButton.clicked.connect(self.terminateProcess)
        layout.addWidget(killButton)
        self.setLayout(layout)
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        if active:
            self.startProcess()
        ## Process monitoring
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.updateStatus)
        self.timer.start()

    def __del__(self):
        print('__del__', self.processname)
        self.killProcess()

    def startProcess(self):
        self.terminateProcess()
        print_error('controller.py - startProcess(): Starting {0}.'.format(self.processname), 'bold')
        self.process = subprocess.Popen(['python3', self.script, self.category + self.processname])

    def terminateProcess(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except Exception as e:
                print_error('process termination failed killing process {0}'.format(self.processname), 'error')
                print(e)
                self.process.kill()
            if self.process.poll() == -15:
                self.startButton.setStyleSheet("color: gray")
                self.process = None

    def killProcess(self):
        if self.process is not None:
            print('killing', self.processname)
            self.process.kill()
            if self.process.poll() == -15:
                self.startButton.setStyleSheet("color: gray")
                self.process = None

    def updateStatus(self):
        if self.process is not None:
            if self.process.poll() == None:
                self.startButton.setStyleSheet("color: blue")
            else:
                self.startButton.setStyleSheet("color: red")
        else:
            self.startButton.setStyleSheet("color: gray")


class Controller(BWidget):
    def __init__(self, parent=None):
        super().__init__('Controller', parent, create_props=False)
        self.setStyleSheet("Controller {background-color: rgb(195,205,230);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} ")
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        
        conf = ConfigReader.getConfiguration()
        self.processlist = []
        
        # Only show Servers category
        if 'Servers' in conf:
            line = 1
            layout.addWidget(QLabel('Servers'), line, 0)
            for name, param in sorted(conf['Servers'].items())[::-1]:
                tooltip = None
                if 'tooltip' in param:
                    tooltip = param['tooltip']
                process = SingleProcess(tweezerpath + '/bin/' + param['script'], name,
                                        param['active'], 'Servers/', tooltip=tooltip)
                line = line + 1
                layout.addWidget(process, line, 0)
                self.processlist.append(process)
        
        self.setLayout(layout)

    def closeEvent(self, event):
        '''on shutdown terminate all server processes first'''
        for p in self.processlist:
            p.terminateProcess()
        print_error('controller.py: Terminated all server processes.', 'info')
        event.accept()

    def __del__(self):
        print('deleting controller')


def main():
    app = QApplication(sys.argv)
    Win = Controller()
    Win.show()

    def on_exit(_signo, _stack_frame):
        print('closing controller')
        Win.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)
    app.exec_()


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        main()
    else:
        print('Running in interactive mode. Controller won\'t be started.')
