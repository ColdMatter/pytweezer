"""Set up one camera and leave it set up: one experiment per camera."""

from typing import Any, ClassVar

from pytweezer.experiment import Device
from pytweezer.experiment.camera_settings import CameraSettings


class RbHamCamSettings(CameraSettings):
    """Settings of the Rb HamCam, from its driver's controls."""

    camera = Device("Rb HamCam")
    control_defaults: ClassVar[dict[str, Any]] = {
        "gain": 600,
        "roi_x0": 110,
        "roi_width": 256,
        "roi_y0": 220,
        "roi_height": 256,
    }


class CaFOrcaSettings(CameraSettings):
    """Settings of the ORCA, from its driver's controls."""

    camera = Device("ORCA")
    control_defaults: ClassVar[dict[str, Any]] = {"roi_width": 1344}


class RbThorCamSettings(CameraSettings):
    """Settings of the Rb ThorCam, from its driver's controls."""

    camera = Device("Rb ThorCam")


class TweezerMonitorThorCamSettings(CameraSettings):
    """Settings of the Tweezer Monitor ThorCam, from its driver's controls."""

    camera = Device("Tweezer Monitor ThorCam")


class CaFThorCamSettings(CameraSettings):
    """Settings of the CaF ThorCam, from its driver's controls."""

    camera = Device("CaF ThorCam")
    control_defaults: ClassVar[dict[str, Any]] = {"gain": 12.0, "binning": 16}
