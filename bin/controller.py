#!/usr/bin/python3
"""
Controller process for pytweezer.
Launches and manages all server processes (Model Sync, MotMaster, Hubs, Loggers, etc.)
Runs as a background daemon that keeps servers alive.
"""

import sys
import signal
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath
from pytweezer.analysis.print_messages import print_error
from bin.process_tile_base import BaseProcessTile


class ControllerProcessTile(BaseProcessTile):
    LOG_SOURCE = 'controller.py'


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
                process = ControllerProcessTile(tweezerpath + '/bin/' + param['script'], name,
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
