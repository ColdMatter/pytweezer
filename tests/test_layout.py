"""The package layout: what the framework imports, and which analyses the Results tab offers."""

import os
import pkgutil
import subprocess
import sys

import pytweezer.analyses.default
import pytweezer.analysis


def _modules(package):
    return [
        module.name
        for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
    ]


def _modules_importing_qt(package) -> list[str]:
    """Import each module of ``package`` in its own fresh interpreter, all at once."""
    processes = {
        module: subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import sys, {module}; print('PyQt6' in sys.modules)",
            ],
            env=os.environ | {"MPLBACKEND": "Agg"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for module in _modules(package)
    }
    importing_qt = []
    for module, process in processes.items():
        stdout, stderr = process.communicate()
        assert process.returncode == 0, f"{module} failed to import:\n{stderr}"
        if stdout.strip() == "True":
            importing_qt.append(module)
    return importing_qt


def test_files_recorded_before_the_move_keep_their_analyses():
    from pytweezer.GUI.default_analyses import ANALYSES

    assert {
        "tweezer_histogram",
        "tweezer_photon_histogram",
        "cloud_fit",
        "cloud_temperature",
    } <= set(ANALYSES)


def test_framework_analysis_imports_no_qt():
    assert _modules_importing_qt(pytweezer.analysis) == []


def test_default_analysis_calculations_import_no_qt():
    assert _modules_importing_qt(pytweezer.analyses.default) == []
