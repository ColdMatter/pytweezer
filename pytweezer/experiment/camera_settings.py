"""Experiments that set a camera up and leave it that way.

::

    class RbHamCamSettings(CameraSettings):
        camera = Device("Rb HamCam")

The form is built from the declared camera's driver: one entry per control in
its ``CONTROLS``. Each setting is sent only when its box is ticked, or, for a
choice, when it is not "unchanged"; the camera keeps everything else. The
settings last until something changes them, and an experiment that uses a
camera sets what it needs in its own ``prepare`` on every task, so these are
for the settings it leaves alone (readout speed, for example) and for a camera
used live or from a notebook.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pytweezer.drivers.camera_base import ROI, Control
from pytweezer.drivers.camera_catalogue import camera_driver
from pytweezer.experiment.arguments import (
    Argument,
    Bool,
    Choice,
    Device,
    Integer,
    Number,
)
from pytweezer.experiment.camera_store import store_camera_settings
from pytweezer.experiment.client import ManagerError, ManagerUnavailable
from pytweezer.experiment.experiment import Experiment
from pytweezer.logging_utils import get_logger

logger = get_logger(__name__)

UNCHANGED = "unchanged"

_ROI_NAMES = {control.name for control in ROI}


def _form_entries(control: Control, default: Any) -> dict[str, Argument]:
    """The arguments that set ``control``: a tick box and a value, or one choice."""
    if control.kind == "choice":
        chosen = UNCHANGED if default is None else default
        return {
            control.name: Choice(
                [UNCHANGED, *control.choices], chosen, group=control.section
            )
        }
    if control.kind == "bool":
        chosen = UNCHANGED if default is None else ("on" if default else "off")
        return {
            control.name: Choice(
                [UNCHANGED, "on", "off"], chosen, group=control.section
            )
        }
    limits = {"min": control.minimum, "max": control.maximum}
    group = control.section
    value = control.default if default is None else default
    if control.kind == "integer":
        argument = Integer(
            int(value),
            unit=control.unit,
            group=group,
            **{k: None if v is None else int(v) for k, v in limits.items()},
        )
    else:
        argument = Number(
            value, unit=control.unit, scale=control.scale, group=group, **limits
        )
    tick = "set_roi" if control.name in _ROI_NAMES else f"set_{control.name}"
    return {tick: Bool(False, group=group), control.name: argument}


class CameraSettings(Experiment):
    """Send the declared ``camera`` the settings chosen in the form.

    A subclass declares ``camera = Device("<CONFIG name>")`` and gets its form
    from the controls that camera's driver declares. ``control_defaults`` gives
    a control's form default, in SI units, where it should differ from the
    driver's. The camera applies the settings in a safe order and reads every
    control back; those sent are saved as the camera's saved parameters, and the
    driver's ``settings()`` is recorded as ``camera_settings``. The ROI is in
    unbinned sensor pixels; the origin the camera took is recorded as
    ``roi_origin``.
    """

    #: Form defaults by control name, in SI units; a choice or bool takes its value.
    control_defaults: ClassVar[dict[str, Any]] = {}
    #: The controls of the declared camera's driver.
    camera_controls: ClassVar[tuple[Control, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        device = vars(cls).get("camera")
        if not isinstance(device, Device):
            return
        cls.camera_controls = camera_driver(device.device_name).CONTROLS
        for control in cls.camera_controls:
            entries = _form_entries(control, cls.control_defaults.get(control.name))
            for name, argument in entries.items():
                if name not in vars(cls):
                    setattr(cls, name, argument)
                    argument.__set_name__(cls, name)

    def chosen_values(self) -> dict[str, Any]:
        """The controls the form asks to change, with their values in SI units."""
        values = {}
        for control in self.camera_controls:
            name = control.name
            if control.kind == "choice":
                chosen = getattr(self, name) != UNCHANGED
                value = getattr(self, name)
            elif control.kind == "bool":
                chosen = getattr(self, name) != UNCHANGED
                value = getattr(self, name) == "on"
            else:
                tick = "set_roi" if name in _ROI_NAMES else f"set_{name}"
                chosen = getattr(self, tick)
                value = getattr(self, name)
            if chosen:
                values[name] = value
        return values

    def run_point(self) -> None:
        values = self.chosen_values()
        result = self.camera.apply_controls(values)
        taken = {control["name"]: control["value"] for control in result}
        self.record("camera_settings", self.camera.settings())
        if _ROI_NAMES & values.keys():
            self.record("roi_origin", [taken["roi_x0"], taken["roi_y0"]], unit="px")
        if values:
            self._store({name: taken[name] for name in values})

    def _store(self, taken: dict[str, Any]) -> None:
        try:
            store_camera_settings(
                type(self).devices()["camera"].device_name,
                taken,
                f"{type(self).__name__}, rid {self.rid}",
            )
        except (ManagerUnavailable, ManagerError):
            logger.warning("Could not store the camera settings", exc_info=True)
