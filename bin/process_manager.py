#!/usr/bin/python3
"""
Controller process for pytweezer.
Launches and manages all server processes (Model Sync, MotMaster, Hubs, Loggers, etc.)
Runs as a background daemon that keeps servers alive.
"""

from PyQt5.QtWidgets import QLabel, QGridLayout
from socket import gethostname
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath
from pytweezer.analysis.print_messages import print_error
from pytweezer.configuration.config import HOSTS
from bin.process_tile_base import ProcessTile

from pytweezer.logging_utils import get_logger
logger = get_logger("Process Manager")

class ProcessManager(BWidget):
    categories = []

    def __init__(self, name) -> None:
        super().__init__(name, create_props=False)
        layout = QGridLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
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
            heading = QLabel(category.upper())
            heading.setProperty("role", "heading")
            layout.addWidget(heading, line, i)
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
        """on shutdown terminate all managed processes first"""
        for p in self.processlist:
            p.terminateProcess()
        logger.info("Terminated all %s processes.", "/".join(self.categories) or "managed")
        event.accept()

    def __del__(self):
        logger.info("Deleting controller")
        
class ServerManager(ProcessManager):
    categories = ["Servers"]

class LoggerManager(ProcessManager):
    # Loggers own their own devices and run on the server PC alongside InfluxDB,
    # so — like ServerManager — no host filtering.
    categories = ["Loggers"]

class DeviceManager(ProcessManager):
    categories = ["Devices"]
    
    def __init__(self, name) -> None:
        self.host_name = gethostname()
        self.host_addr = HOSTS.get(self.host_name, None)
        if self.host_addr is None:
            self.host_addr = "127.0.0.1"
            # print_error(f"Host {self.host_name} not found in config. Defaulting to localhost ({self.host_addr}).", "warning")
            logger.warning(f"Host {self.host_name} not found in config. Defaulting to localhost ({self.host_addr}).")
        super().__init__(name)
        
    def check_host(self, host_addr):
        """
        make the comparison case-insensitive and ignore whitespace, to be more robust to formatting issues in the config file
        """
        if host_addr is None:
            return False
        host_addr = host_addr.strip().lower()
        return host_addr == self.host_addr.strip().lower()
    
# ``ServerManager`` and ``DeviceManager`` are embedded as tabs by ``bin/gui.py``
# (the ``pytweezer-server`` / ``pytweezer-client`` entry points). This module no
# longer has its own CLI dispatch; keep the classes importable as reusable panels.
