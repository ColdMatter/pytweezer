"""The Analysis Manager lists and runs the scripts in ``pytweezer/analyses/live``."""

import os

import zmq

from pytweezer.configuration.paths import (
    live_analyses_path,
    load_properties,
    propertydefaultsfilename,
)
from pytweezer.GUI.analysismanager import AddFilterSheet
from pytweezer.servers import analysis_manager


def test_the_manager_lists_the_live_analyses():
    scripts = AddFilterSheet._scan_scripts(live_analyses_path)

    assert "tweezer_loading.py" in scripts["Image"]
    assert "cloud_fit.py" in scripts["Image"]
    assert "gaussian_fit.py" in scripts["Data"]


def test_every_default_filter_names_a_live_script():
    analysis = load_properties(propertydefaultsfilename)["Analysis"]

    for category in ("Image", "Data"):
        for filter_name, settings in analysis[category].items():
            script = os.path.join(live_analyses_path, settings["script"])
            assert os.path.isfile(script), f"{category}/{filter_name}: {script}"


def test_the_server_ignores_a_saved_script_folder(monkeypatch):
    class SavedProperties:
        def __init__(self, *_):
            pass

        def get(self, key, default=None):
            return "C:\\old\\pytweezer/analysis/" if key == "analysisdir" else default

    monkeypatch.setattr(analysis_manager, "Properties", SavedProperties)
    monkeypatch.setattr(zmq.Socket, "bind", lambda self, endpoint: None)

    service = analysis_manager.AnalysisManagerService()
    service.socket.close()

    assert service.analysisdir == live_analyses_path
