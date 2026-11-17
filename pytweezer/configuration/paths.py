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
#: This PC's Properties tree, saved by the Properties server; not tracked by git.
propertyfilename = configpath + "/properties/properties.json"
#: The committed tree a PC starts from until it has saved its own.
propertydefaultsfilename = configpath + "/properties/properties.default.json"

#: Device config entries omit ``"script"``: they all run device_server.py,
#: differentiated by their ``"class"`` key. This is the one launcher not named
#: per-entry in CONFIG. Path is relative to ``bin/`` (callers prefix
#: ``tweezerpath + "/bin/"``), matching how ManagedRow/process_cleanup spawn it.
DEVICE_SERVER_SCRIPT = "../pytweezer/servers/device_server.py"


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
