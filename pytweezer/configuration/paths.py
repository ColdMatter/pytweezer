"""Filesystem layout for the pytweezer checkout.

Single source of truth for where the repo root, the root-level ``configuration/``
data directory (JSON files — **not** the Python ``CONFIG`` dict, which lives in
:mod:`pytweezer.configuration.config`), and the GUI icons are, plus a loader for
the Properties startup state.
"""

import json
import os

_here = os.path.dirname(os.path.abspath(__file__))

#: Repo root: the directory containing ``pytweezer/`` and ``bin/``.
tweezerpath = os.path.realpath(_here + "/../..")

#: Root-level ``configuration/`` dir holding JSON data files (properties.json,
#: defaults.json, ...). Distinct from the ``CONFIG`` dict in
#: ``pytweezer.configuration.config``.
configpath = os.path.realpath(_here + "/../../configuration/")

icon_path = tweezerpath + "/pytweezer/GUI/icons/"
#: The scripts the Analysis Manager lists and runs.
live_analyses_path = tweezerpath + "/pytweezer/analyses/live/"
#: This PC's Properties tree, saved by the Properties server; not tracked by git.
propertyfilename = configpath + "/properties/properties.json"
#: The committed tree a PC starts from until it has saved its own.
propertydefaultsfilename = configpath + "/properties/properties.default.json"

#: Device and logger config entries omit ``"script"``: every device runs
#: device_server.py and every logger logger_server.py, differentiated by their
#: ``"class"`` key. Paths are relative to ``bin/`` (callers prefix
#: ``tweezerpath + "/bin/"``), matching how ManagedRow/process_cleanup spawn them.
DEVICE_SERVER_SCRIPT = "../pytweezer/servers/device_server.py"
LOGGER_SERVER_SCRIPT = "../pytweezer/servers/logger_server.py"
#: The launcher each category's entries run when they name no ``"script"``.
CATEGORY_SCRIPTS = {"Devices": DEVICE_SERVER_SCRIPT, "Loggers": LOGGER_SERVER_SCRIPT}


def read_json_tree(path: str) -> dict:
    """Return the JSON object in ``path``, or ``{}`` if it is missing or unreadable."""
    try:
        with open(path) as inputfile:
            return json.load(inputfile)
    except (OSError, ValueError):
        return {}


def load_properties(
    path: str = propertyfilename, default_path: str = propertydefaultsfilename
) -> dict:
    """Return the Properties tree saved in ``path``, else the one in ``default_path``."""
    return read_json_tree(path) or read_json_tree(default_path)
