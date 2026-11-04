"""Put a camera into free-running acquisition on its image stream."""

from __future__ import annotations

from pytweezer.drivers.camera_base import Camera
from pytweezer.experiment import Bool, Choice, Device, Experiment, Integer, Number
from pytweezer.logging_utils import get_logger
from pytweezer.servers import device_server

logger = get_logger(__name__)


def camera_names() -> list[str]:
    """Every active CONFIG device whose driver is a :class:`Camera`."""
    names = []
    for address in device_server.device_index().values():
        path = address.conf.get("class")
        if not path or address.owner_conf.get("active", True) is False:
            continue
        try:
            driver = device_server._load(path)
        except Exception:
            logger.warning(
                "Cannot import the driver of %r", address.name, exc_info=True
            )
            continue
        if isinstance(driver, type) and issubclass(driver, Camera):
            names.append(address.name)
    return sorted(names)


CAMERAS = camera_names()


class ContinuousAcquisition(Experiment):
    """Leave a camera acquiring continuously and broadcasting every frame.

    The trigger is the camera's internal one, so frames arrive at the rate the
    exposure and ROI allow. The camera keeps acquiring after the task ends.

    The task reserves every camera in the drop-down, because reservations are
    fixed per class and do not follow the choice.
    """

    camera_name = Choice(CAMERAS, tooltip="Camera to set running")
    exposure = Number(
        10e-3, unit="ms", scale=1e-3, min=1e-6, ndecimals=3, group="Settings"
    )
    buffer_frames = Integer(
        1000,
        min=1,
        group="Settings",
        tooltip="Ring buffer length; frames older than this are overwritten",
    )
    broadcast = Bool(
        True, group="Settings", tooltip="Send every frame to the camera's image stream"
    )
    set_binning = Bool(
        False, group="Settings", tooltip="Otherwise the camera keeps its binning"
    )
    binning = Integer(1, min=1, group="Settings", tooltip="Pixels per side")
    set_gain = Bool(
        False, group="Settings", tooltip="Otherwise the camera keeps its gain"
    )
    gain = Number(
        0.0,
        min=0,
        group="Settings",
        tooltip="In the camera's own unit: dB (ThorCams), contrast gain (ORCA), "
        "sensitivity (HamCam, once EM gain is enabled)",
    )
    set_roi = Bool(False, group="ROI", tooltip="Otherwise the camera keeps its ROI")
    roi_x0 = Integer(0, min=0, unit="px", group="ROI")
    roi_width = Integer(1024, min=1, unit="px", group="ROI")
    roi_y0 = Integer(0, min=0, unit="px", group="ROI")
    roi_height = Integer(1024, min=1, unit="px", group="ROI")

    @classmethod
    def devices(cls) -> dict[str, Device]:
        return {name: Device(name) for name in CAMERAS}

    def run_point(self):
        camera = self.device(self.camera_name)
        camera.stop_live()
        camera.set_trigger_source("int")
        camera.set_exposure_time(self.exposure)
        if self.set_gain:
            camera.set_gain(self.gain)
        if self.set_binning:
            camera.set_binning(self.binning)
        if self.set_roi:
            x0, y0 = camera.set_roi(
                self.roi_x0, self.roi_width, self.roi_y0, self.roi_height
            )
            self.record("roi_origin", [x0, y0], unit="px")
        if self.broadcast:
            camera.start_live(self.buffer_frames)
        else:
            camera.arm(self.buffer_frames)
        self.record("camera_settings", camera.settings())
