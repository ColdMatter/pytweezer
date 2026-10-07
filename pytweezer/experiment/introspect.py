"""Describe the experiments in a module as JSON, without running them.

``python -m pytweezer.experiment.introspect pytweezer.experiments.demo``

Run as a subprocess by the Experiment Manager, so importing a broken or slow
experiment module can never affect the manager or a GUI. Always prints one
JSON object; an import failure is reported in its ``error`` field.
"""

import importlib
import json
import sys
import traceback
from typing import Any

from pytweezer.experiment.experiment import Experiment


def runnable_experiments(module: Any) -> list[type[Experiment]]:
    """Experiment subclasses defined in ``module`` that implement ``run_point``."""
    return [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, Experiment)
        and obj.__module__ == module.__name__
        and obj.run_point is not Experiment.run_point
    ]


def _device_warnings(experiment_cls: type[Experiment]) -> list[str]:
    from pytweezer.servers.device_server import resolve_address

    warnings = []
    for attribute, device in experiment_cls.devices().items():
        try:
            resolve_address(device.device_name)
        except Exception as error:
            warnings.append(
                f"{experiment_cls.__name__}.{attribute}: "
                f"device {device.device_name!r} not found ({error})"
            )
    return warnings


def describe(module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module": module_name,
        "classes": [],
        "warnings": [],
        "error": None,
    }
    try:
        module = importlib.import_module(module_name)
        for experiment_cls in runnable_experiments(module):
            result["classes"].append(experiment_cls.schema())
            result["warnings"] += _device_warnings(experiment_cls)
    except BaseException:
        result["error"] = traceback.format_exc()
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(describe(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
