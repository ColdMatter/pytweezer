"""Set up one camera and leave it set up: one experiment per camera."""

from pytweezer.experiment import Device, Integer, Number
from pytweezer.experiment.camera_settings import (
    ImagEMSettings,
    ORCASettings,
    ThorCamSettings,
)


class RbHamCamSettings(ImagEMSettings):
    """Exposure mode, EM gain, sensitivity, exposure time and ROI of the Rb HamCam."""

    camera = Device("Rb HamCam")
    roi_x0 = Integer(110, min=0, unit="px", group="ROI")
    roi_width = Integer(256, min=1, unit="px", group="ROI")
    roi_y0 = Integer(220, min=0, unit="px", group="ROI")
    roi_height = Integer(256, min=1, unit="px", group="ROI")


class CaFOrcaSettings(ORCASettings):
    """Exposure mode, readout speed, contrast gain, exposure time and ROI of the ORCA."""

    camera = Device("ORCA")


class RbThorCamSettings(ThorCamSettings):
    """Trigger, gain, exposure time and ROI of the Rb ThorCam."""

    camera = Device("Rb ThorCam")


class TweezerMonitorThorCamSettings(ThorCamSettings):
    """Trigger, gain, exposure time and ROI of the Tweezer Monitor ThorCam."""

    camera = Device("Tweezer Monitor ThorCam")


class CaFThorCamSettings(ThorCamSettings):
    """Trigger, gain, exposure time and ROI of the CaF ThorCam."""

    camera = Device("CaF ThorCam")
    gain = Number(12.0, unit="dB", min=0, group="Gain")
    binning = Integer(16, min=1, group="ROI", tooltip="Pixels per side")
