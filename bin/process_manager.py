#!/usr/bin/python3
"""
Controller process for pytweezer.
Launches and manages all server processes (Model Sync, MotMaster, Hubs, Loggers, etc.)
Runs as a background daemon that keeps servers alive.
"""

import sys
import signal
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout
import zmq
from socket import gethostname
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath
from pytweezer.analysis.print_messages import print_error
from pytweezer.configuration.config import HOSTS
from bin.process_tile_base import ProcessTile



class ProcessManager(BWidget):
    categories = []

    def __init__(self, name) -> None:
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

    def check_host(self, host_addr):
        return True

    def add_processes(self):
        conf = ConfigReader.getConfiguration()
        layout = self.layout()
        for i, category in enumerate(self.categories):
            line = 1
            layout.addWidget(QLabel(category), line, i)
            for name, params in sorted(conf[category].items())[::-1]:
                host_addr = params.get("host")
                if self.check_host(host_addr):
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
        
class ServerManager(ProcessManager):
    categories = ["Servers"]
    
class DeviceManager(ProcessManager):
    categories = ["Devices"]
    
    def __init__(self, name) -> None:
        self.host_name = gethostname()
        self.host_addr = HOSTS.get(self.host_name, None)
        if self.host_addr is None:
            self.host_addr = "127.0.0.1"
            print_error(f"Host {self.host_name} not found in config. Defaulting to localhost ({self.host_addr}).", "warning")
        super().__init__(name)
        
    def check_host(self, host_addr):
        return host_addr == self.host_addr 
    
class Dashboard(ProcessManager):
    categories = ["GUI", "Viewer"]


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: process_manager.py <manager_type>")
        sys.exit(1)

    manager_type = args[0]
    app = QApplication(sys.argv)
    if manager_type == "server":
        Win = ServerManager(manager_type)
    elif manager_type == "device":
        Win = DeviceManager(manager_type)
    elif manager_type == "dashboard":
        Win = Dashboard(manager_type)
    else:
        print("Invalid manager type. Use 'server', 'device', or 'dashboard'.")
        sys.exit(1)
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
