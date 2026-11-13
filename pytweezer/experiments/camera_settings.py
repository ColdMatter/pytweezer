"""Set up one camera and leave it set up: one experiment per camera."""

from typing import Any, ClassVar

from pytweezer.experiment import Device
from pytweezer.experiment.camera_settings import CameraSettings


class RbHamCamSettings(CameraSettings):
    """Exposure mode, EM gain, sensitivity, exposure time and ROI of the Rb HamCam."""

    camera = Device("Rb HamCam")
    control_defaults: ClassVar[dict[str, Any]] = {
        "roi_x0": 110,
        "roi_width": 256,
        "roi_y0": 220,
        "roi_height": 256,
    }


class CaFOrcaSettings(CameraSettings):
    """Exposure mode, readout speed, contrast gain, exposure time and ROI of the ORCA."""

    camera = Device("ORCA")


class RbThorCamSettings(CameraSettings):
    """Trigger, gain, exposure time and ROI of the Rb ThorCam."""

    camera = Device("Rb ThorCam")


class TweezerMonitorThorCamSettings(CameraSettings):
    """Trigger, gain, exposure time and ROI of the Tweezer Monitor ThorCam."""

    camera = Device("Tweezer Monitor ThorCam")


class CaFThorCamSettings(CameraSettings):
    """Trigger, gain, exposure time and ROI of the CaF ThorCam."""

    camera = Device("CaF ThorCam")
    control_defaults: ClassVar[dict[str, Any]] = {"gain": 12.0, "binning": 16}
