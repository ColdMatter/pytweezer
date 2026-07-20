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
    def script_for(category, params):
        """Return the launcher script for one config entry, relative to ``bin/``.

        Entries may omit ``"script"`` when every entry in their category runs the
        same launcher (``CONFIG["Devices"]`` all run ``device_server.py``, selected
        by their ``"driver"`` key). Returns ``None`` if the entry has no script and
        its category has no default, meaning it is not a launchable process.
        """
        from pytweezer.configuration.config import DEFAULT_SCRIPTS

        return params.get("script") or DEFAULT_SCRIPTS.get(category)

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
