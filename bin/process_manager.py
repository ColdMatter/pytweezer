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
from pytweezer.configuration.config import HOSTS
from bin.process_tile_base import ProcessTile



class ProcessManager(BWidget):
    categories = []

    def __init__(self, name):
        super().__init__(name, create_props=False)
        self.setStyleSheet(
            "Controller {background-color: rgb(195,205,230);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} "
        )
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.setLayout(layout)

        self.processlist = []


        self.add_processes()

    def check_host(self, params):
        return True

    def add_processes(self):
        conf = ConfigReader.getConfiguration()
        layout = self.layout()
        for i, category in enumerate(self.categories):
            line = 1
            layout.addWidget(QLabel(category), line, i)
            for name, params in sorted(conf[category].items())[::-1]:
                if self.check_host(params):
                    tooltip = None
                    if "tooltip" in params:
                        tooltip = params["tooltip"]
                    process = ProcessTile(
                        tweezerpath + "/bin/" + params["script"],
                        name,
                        params["active"],
                        category + "/",
                        tooltip=tooltip,
                    )
                    self.processlist.append(process)
                    line += 1
                    layout.addWidget(process, line, i)

    def closeEvent(self, event):
        """on shutdown terminate all server processes first"""
        for p in self.processlist:
            p.terminateProcess()
        print_error("controller.py: Terminated all server processes.", "info")
        event.accept()

    def __del__(self):
        print("deleting controller")
        
class Controller(ProcessManager):
    categories = ["Servers", "Devices"]
    
    def __init__(self, host_name):
        super().__init__("Controller")
        self.host_addr = HOSTS.get(host_name, None)
        if self.host_addr is None:
            print_error(
                f"Unknown host name '{host_name}' in configuration. No processes will be launched.",
                "error",
            )
            return


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
        print(f"Closing controller")
        Win.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_exit)
    app.exec_()


if __name__ == "__main__":
    if sys.flags.interactive != 1:
        main()
    else:
        print("Running in interactive mode. Controller won't be started.")
