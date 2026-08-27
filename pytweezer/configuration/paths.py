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
propertyfilename = configpath + "/properties/properties.json"

#: Device config entries omit ``"script"``: they all run device_server.py,
#: differentiated by their ``"class"`` key. This is the one launcher not named
#: per-entry in CONFIG. Path is relative to ``bin/`` (callers prefix
#: ``tweezerpath + "/bin/"``), matching how ManagedRow/process_cleanup spawn it.
DEVICE_SERVER_SCRIPT = "../pytweezer/servers/device_server.py"


def load_properties():
    """Return the Properties startup dict, or ``{}`` if the file is missing or unreadable."""
    try:
        with open(propertyfilename) as inputfile:
            return json.load(inputfile)
    except (OSError, ValueError):
        return {}
