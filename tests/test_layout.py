"""The package layout: what the framework imports, and which analyses the Results tab offers."""

import os
import pkgutil
import subprocess
import sys

import pytest

import pytweezer.analyses.default
import pytweezer.analysis


def _modules(package):
    return [
        module.name
        for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
    ]


def _imports_qt(module: str) -> bool:
    code = f"import sys, {module}; print('PyQt6' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ | {"MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "True"


def test_files_recorded_before_the_move_keep_their_analyses():
    from pytweezer.GUI.default_analyses import ANALYSES

    assert set(ANALYSES) == {
        "tweezer_histogram",
        "tweezer_photon_histogram",
        "cloud_fit",
        "cloud_temperature",
    }


@pytest.mark.parametrize("module", _modules(pytweezer.analysis))
def test_framework_analysis_imports_no_qt(module):
    assert not _imports_qt(module)


@pytest.mark.parametrize("module", _modules(pytweezer.analyses.default))
def test_default_analysis_calculations_import_no_qt(module):
    assert not _imports_qt(module)
