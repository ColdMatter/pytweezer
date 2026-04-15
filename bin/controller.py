#!/usr/bin/python3
"""
Controller process for pytweezer.
Launches and manages all server processes (Model Sync, MotMaster, Hubs, Loggers, etc.)
Runs as a background daemon that keeps servers alive.
"""

import sys
import signal
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout
import pytweezer
from pytweezer.GUI.pytweezerQt import BWidget
import pytweezer.configuration
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath
from pytweezer.analysis.print_messages import print_error
from bin.process_tile_base import BaseProcessTile

from pytweezer.configuration.config import HOSTS

class ControllerProcessTile(BaseProcessTile):
    LOG_SOURCE = "controller.py"


class Controller(BWidget):
    def __init__(self, host_name, parent=None):
        super().__init__("Controller", parent, create_props=False)
        self.setStyleSheet(
            "Controller {background-color: rgb(195,205,230);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} "
        )
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        conf = ConfigReader.getConfiguration()
        self.processlist = []
        
        host_addr = HOSTS.get(host_name, None)
        if host_addr is None:
            print_error(f"Unknown host name '{host_name}' in configuration. No processes will be launched.", "error")
            return

        # Only show Servers category
        if "Servers" in conf:
            line = 1
            layout.addWidget(QLabel("Servers"), line, 0)
            for name, param in sorted(conf["Servers"].items())[::-1]:
                if param["host"] == host_addr:
                    tooltip = None
                    if "tooltip" in param:
                        tooltip = param["tooltip"]
                    process = ControllerProcessTile(
                        tweezerpath + "/bin/" + param["script"],
                        name,
                        param["active"],
                        "Servers/",
                        tooltip=tooltip,
                    )
                    line = line + 1
                    layout.addWidget(process, line, 0)
                    self.processlist.append(process)
        if "Devices" in conf:
            line = 1
            layout.addWidget(QLabel("Devices"), line, 1)
            for name, param in sorted(conf["Devices"].items())[::-1]:
                if param["host"] == host_addr:
                    tooltip = None
                    if "tooltip" in param:
                        tooltip = param["tooltip"]
                    process = ControllerProcessTile(
                        tweezerpath + "/bin/" + param["script"],
                        name,
                        param["active"],
                        "Devices/",
                        tooltip=tooltip,
                    )
                    line = line + 1
                    layout.addWidget(process, line, 1)
                    self.processlist.append(process)
        self.setLayout(layout)

    def closeEvent(self, event):
        """on shutdown terminate all server processes first"""
        for p in self.processlist:
            p.terminateProcess()
        print_error("controller.py: Terminated all server processes.", "info")
        event.accept()

    def __del__(self):
        print("deleting controller")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: controller.py <host_name>")
        sys.exit(1)
    host_name = args[0]
    app = QApplication(sys.argv)
    Win = Controller(host_name)
    Win.show()

    def on_exit(_signo, _stack_frame):
        print("closing controller")
        Win.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)
    app.exec_()


if __name__ == "__main__":
    if sys.flags.interactive != 1:
        main()
    else:
        print("Running in interactive mode. Controller won't be started.")
