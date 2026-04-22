"""
Readin configuration file and startup values for Properties




"""

import os
import json

tweezerpath = os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + "/../..")
configpath = os.path.realpath(
    os.path.dirname(os.path.abspath(__file__)) + "/../../configuration/"
)
icon_path = tweezerpath + "/pytweezer/GUI/icons/"
configfilename = configpath + "/config.py"
propertyfilename = configpath + "/properties/properties.json"


def Config():
    """return configuration dictionary"""
    # with open(configfilename) as inputfile:
    #     configuration = json.load(inputfile)
    # return configuration
    from pytweezer.configuration.config import CONFIG
    return CONFIG


def DashboardConfig():
    """return configuration dictionary for GUI/Viewer launchers"""

    from pytweezer.configuration.config import get_dashboard_config
    return get_dashboard_config()


def ControllerConfig():
    """return configuration dictionary for controller launchers"""

    from pytweezer.configuration.config import get_controller_config
    return get_controller_config()


def MachineName():
    """return the active machine name"""

    from pytweezer.configuration.config import get_machine_name
    return get_machine_name()


def MachineProfile():
    """return the active machine profile"""

    from pytweezer.configuration.config import get_machine_profile
    return get_machine_profile()


class ConfigReader:
    """read the static configuration file"""

    @staticmethod
    def getConfiguration():
        """get current confg file contents

        Returns: dict

            configuration dictionary
        """

        return Config()

    @staticmethod
    def getDashboardConfiguration():
        return DashboardConfig()

    @staticmethod
    def getControllerConfiguration():
        return ControllerConfig()

    @staticmethod
    def getMachineName():
        return MachineName()

    @staticmethod
    def getMachineProfile():
        return MachineProfile()


def Properties():
    """return dictionary of startup properties"""
    try:
        with open(propertyfilename) as inputfile:
            prop = json.load(inputfile)
        return prop
    except:
        return {}


if __name__ == "__main__":
    pass
