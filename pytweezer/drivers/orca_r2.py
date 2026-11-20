from pytweezer.drivers.camera_base import (
    BINNING,
    EXPOSURE,
    FRAME_RATE,
    ROI,
    TRIGGER_POLARITY,
    Control,
    gain,
    simulated_camera_for,
    trigger_source,
)
from pytweezer.drivers.hamamatsu import EXPOSURE_MODE, HamamatsuCamera


class ORCAR2Camera(HamamatsuCamera):
    """dcam shim for the Hamamatsu ORCA-R2 (an interline CCD), adding its
    readout-speed and contrast-gain controls. Arguments are those of
    :class:`HamamatsuCamera`; with no ``serial_number`` the only connected
    ORCA-R2 is opened.
    """

    image_prefix = "ORCA"
    model_prefix = "C10600"
    default_timeout = 5 * 60.0

    CONTROLS = (
        EXPOSURE,
        FRAME_RATE,
        EXPOSURE_MODE,
        trigger_source(("int", "ext", "software")),
        TRIGGER_POLARITY,
        gain("Contrast gain", "integer", minimum=0),
        Control("readout_speed", "Readout speed", "Readout", "integer", 1, minimum=1),
        BINNING,
        *ROI,
    )

    def _read_control_values(self) -> dict:
        return {
            **super()._read_control_values(),
            "gain": int(self._read_attribute("contrast_gain", default=0)),
            "readout_speed": int(self._read_attribute("readout_speed", default=1)),
        }

    def set_readout_speed(self, speed: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("readout_speed", int(speed))
            self._keep(readout_speed=int(self._read_attribute("readout_speed")))

    def set_gain(self, gain: float):
        """Set the contrast gain (an integer index)."""
        with self._lock:
            self._backend.set_attribute_value("contrast_gain", int(gain))
            self._keep(gain=int(self._read_attribute("contrast_gain")))


SimulatedORCAR2Camera = simulated_camera_for(ORCAR2Camera)
