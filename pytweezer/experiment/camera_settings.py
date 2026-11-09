"""Experiments that set a camera up and leave it that way.

::

    class RbHamCamSettings(ImagEMSettings):
        camera = Device("Rb HamCam")

Each setting is sent only when its box is ticked, or, for a choice, when it is
not "unchanged"; the camera keeps everything else. The settings last until
something changes them, and an experiment that uses a camera sets what it needs
in its own ``prepare`` on every task, so these are for the settings it leaves
alone (readout speed, for example) and for a camera used live or from a
notebook.
"""

from __future__ import annotations

from typing import Any

from pytweezer.experiment.arguments import Bool, Choice, Integer, Number
from pytweezer.experiment.experiment import Experiment

UNCHANGED = "unchanged"


class CameraSettings(Experiment):
    """Send the declared ``camera`` the settings chosen in the form.

    A subclass declares ``camera = Device("<CONFIG name>")`` and sends its
    driver's own settings from :meth:`configure`. The camera is stopped while it
    is changed and put back in live mode afterwards if it was live. Settings go
    in the order binning, ROI (``set_binning`` can reset it), exposure time,
    :meth:`configure`, gain. The ROI is in unbinned sensor pixels; the origin
    the camera took is recorded as ``roi_origin`` and the driver's
    ``settings()`` as ``camera_settings``.
    """

    set_exposure = Bool(False, group="Exposure")
    exposure = Number(
        10e-3, unit="ms", scale=1e-3, min=1e-6, ndecimals=3, group="Exposure"
    )
    set_gain = Bool(False, group="Gain")
    gain = Number(0.0, min=0, group="Gain", tooltip="In the camera's own unit")
    set_binning = Bool(False, group="ROI")
    binning = Integer(1, min=1, group="ROI", tooltip="Pixels per side")
    set_roi = Bool(False, group="ROI")
    roi_x0 = Integer(0, min=0, unit="px", group="ROI")
    roi_width = Integer(1024, min=1, unit="px", group="ROI")
    roi_y0 = Integer(0, min=0, unit="px", group="ROI")
    roi_height = Integer(1024, min=1, unit="px", group="ROI")

    def configure(self, camera: Any) -> None:
        """Send driver-specific settings; called after the exposure time, before the gain."""

    def run_point(self) -> None:
        camera = self.camera
        was_live = camera.is_live()
        camera.stop_live()
        if self.set_binning:
            camera.set_binning(self.binning)
        if self.set_roi:
            origin = camera.set_roi(
                self.roi_x0, self.roi_width, self.roi_y0, self.roi_height
            )
            self.record("roi_origin", list(origin), unit="px")
        if self.set_exposure:
            camera.set_exposure_time(self.exposure)
        self.configure(camera)
        if self.set_gain:
            camera.set_gain(self.gain)
        self.record("camera_settings", camera.settings())
        if was_live:
            camera.start_live()


class HamamatsuSettings(CameraSettings):
    """A Hamamatsu DCAM camera: adds the exposure mode."""

    exposure_mode = Choice(
        [UNCHANGED, "internal", "edge", "level"],
        group="Exposure",
        tooltip="internal: free-running; edge: each trigger starts an exposure "
        "of the set time; level: the trigger pulse's width sets the exposure",
    )

    def configure(self, camera: Any) -> None:
        if self.exposure_mode != UNCHANGED:
            camera.set_exposure_mode(self.exposure_mode)


class ImagEMSettings(HamamatsuSettings):
    """The ImagEM X2: the gain is the sensitivity, which needs EM gain on."""

    gain = Integer(
        600,
        min=0,
        group="Gain",
        tooltip="Sensitivity: the EM gain level once EM gain is on",
    )
    em_gain = Choice([UNCHANGED, "on", "off"], group="Gain")
    direct_em_gain = Choice([UNCHANGED, "on", "off"], group="Gain")

    def configure(self, camera: Any) -> None:
        super().configure(camera)
        if self.em_gain != UNCHANGED:
            camera.enable_em_gain(self.em_gain == "on")
        if self.direct_em_gain != UNCHANGED:
            camera.enable_direct_em_gain(self.direct_em_gain == "on")


class ORCASettings(HamamatsuSettings):
    """The ORCA-R2: the gain is the contrast-gain index; adds the readout speed."""

    gain = Integer(0, min=0, group="Gain", tooltip="Contrast-gain index")
    roi_width = Integer(1344, min=1, unit="px", group="ROI")
    set_readout_speed = Bool(False, group="Readout")
    readout_speed = Integer(
        1, min=1, group="Readout", tooltip="DCAM readout-speed index; 1 is slowest"
    )

    def configure(self, camera: Any) -> None:
        super().configure(camera)
        if self.set_readout_speed:
            camera.set_readout_speed(self.readout_speed)


class ThorCamSettings(CameraSettings):
    """A Thorlabs camera: the gain is in dB; adds the trigger mode and polarity."""

    trigger_mode = Choice(
        [UNCHANGED, "int", "ext", "bulb"],
        group="Exposure",
        tooltip="int: free-running; ext: each trigger starts an exposure of the "
        "set time; bulb: the trigger pulse's width sets the exposure",
    )
    trigger_polarity = Choice(
        [UNCHANGED, "rise", "fall"],
        group="Exposure",
        tooltip="rise: active high; fall: active low",
    )
    gain = Number(0.0, unit="dB", min=0, group="Gain")

    def configure(self, camera: Any) -> None:
        if self.trigger_mode != UNCHANGED:
            camera.set_trigger_source(self.trigger_mode)
        if self.trigger_polarity != UNCHANGED:
            camera.set_trigger_polarity(self.trigger_polarity)
