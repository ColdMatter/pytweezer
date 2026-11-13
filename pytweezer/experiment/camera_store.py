"""Keep a camera's settings as saved parameters, so the lab can see what they were set to."""

from __future__ import annotations

from typing import Any

from pytweezer.drivers.camera_catalogue import camera_driver, camera_key
from pytweezer.experiment.client import set_saved_parameters

_KIND_TYPES = {"integer": int, "number": float, "bool": bool, "choice": str}


def store_camera_settings(camera: str, values: dict[str, Any], reason: str) -> None:
    """Store ``values`` (SI, by control name) as the saved parameters of ``camera``.

    Each is saved as ``<camera_key>_<control name>`` in the group ``camera``,
    created if new, and coerced to its control's type so a key never changes
    type.

    Args:
        camera: CONFIG name of the camera.
        values: Control values by control name.
        reason: Recorded in the saved-parameter history.

    Raises:
        KeyError: If a name is not one of the camera's controls.
        ManagerUnavailable: If the Experiment Manager cannot be reached.
        ManagerError: If the Experiment Manager refuses the values.
    """
    controls = {control.name: control for control in camera_driver(camera).CONTROLS}
    prefix = camera_key(camera)
    set_saved_parameters(
        {
            f"{prefix}_{name}": _KIND_TYPES[controls[name].kind](value)
            for name, value in values.items()
        },
        reason,
        create=True,
        group=camera,
        units={f"{prefix}_{name}": controls[name].si_unit for name in values},
    )
