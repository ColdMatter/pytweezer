from typing import Any

from pytweezer.drivers.camera_base import (
    BINNING,
    EXPOSURE,
    FRAME_RATE,
    ROI,
    TRIGGER_POLARITY,
    Control,
    simulated_camera_for,
    trigger_source,
)
from pytweezer.drivers.hamamatsu import EXPOSURE_MODE, HamamatsuCamera

#: DCAM ``trigger_global_exposure`` values.
_DELAYED, _GLOBAL_RESET = 3, 5
#: DCAM ``defect_correct_mode`` values.
_CORRECTION_OFF, _CORRECTION_ON = 1, 2


class ORCAFusionCamera(HamamatsuCamera):
    """dcam shim for the Hamamatsu ORCA-Fusion BT (a rolling-shutter sCMOS),
    adding its readout-speed, global-reset and hot-pixel-correction controls. It
    has no gain setting.

    Args:
        global_reset: Start in global reset (:meth:`set_global_reset`).

    The other arguments are those of :class:`HamamatsuCamera`; with no
    ``serial_number`` the only connected ORCA-Fusion is opened.
    """

    image_prefix = "ORCAFusion"
    model_prefix = "C15440"
    default_timeout = 5 * 60.0

    CONTROLS = (
        EXPOSURE,
        FRAME_RATE,
        EXPOSURE_MODE,
        trigger_source(("int", "ext", "software")),
        TRIGGER_POLARITY,
        Control("global_reset", "Global reset", "Exposure & trigger", "bool", True),
        Control(
            "readout_speed",
            "Readout speed (1 = quietest)",
            "Readout",
            "integer",
            1,
            minimum=1,
        ),
        Control(
            "hot_pixel_correction", "Hot-pixel correction", "Readout", "bool", True
        ),
        BINNING,
        *ROI,
    )

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float | None = None,
        stream_name: str | None = None,
        serial_number: str | None = None,
        external_exposure: bool = False,
        global_reset: bool = True,
        object_pixel_m: float | None = None,
        temperature_measurement: str | None = None,
        temperature_interval_s: float = 30.0,
        max_temperature_c: float | None = None,
    ):
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            serial_number=serial_number,
            external_exposure=external_exposure,
            object_pixel_m=object_pixel_m,
            temperature_measurement=temperature_measurement,
            temperature_interval_s=temperature_interval_s,
            max_temperature_c=max_temperature_c,
        )
        self.set_global_reset(global_reset)

    def _read_control_values(self) -> dict:
        values = {
            **super()._read_control_values(),
            "global_reset": int(self._read_attribute("trigger_global_exposure"))
            == _GLOBAL_RESET,
            "readout_speed": int(self._read_attribute("readout_speed")),
        }
        correction = self._read_attribute("defect_correct_mode", default=None)
        if correction is not None:
            values["hot_pixel_correction"] = int(correction) == _CORRECTION_ON
        return values

    def _control_limits(self) -> dict[str, dict[str, Any] | None]:
        limits = super()._control_limits()
        speed = self._backend.get_attribute("readout_speed", error_on_missing=False)
        if speed is not None:
            limits["readout_speed"] = {"minimum": speed.min, "maximum": speed.max}
        if (
            self._backend.get_attribute("defect_correct_mode", error_on_missing=False)
            is None
        ):
            limits["hot_pixel_correction"] = None
        return limits

    def set_global_reset(self, enable: bool = True):
        """Global reset starts every row exposing together on the trigger, but rows
        still end in turn as they are read out, so the light must be off by then
        for an even exposure. Off, each row starts a line time after the last."""
        with self._lock, self._backend.pausing_acquisition():
            self._backend.set_attribute_value(
                "trigger_global_exposure", _GLOBAL_RESET if enable else _DELAYED
            )
            self._keep(
                global_reset=int(self._read_attribute("trigger_global_exposure"))
                == _GLOBAL_RESET
            )

    def set_readout_speed(self, speed: int):
        """1 is the slowest, lowest-noise readout; higher is faster and noisier."""
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("readout_speed", int(speed))
            self._keep(readout_speed=int(self._read_attribute("readout_speed")))

    def set_hot_pixel_correction(self, enable: bool = True):
        """When on, the camera corrects its factory-mapped hot pixels from their neighbours."""
        with self._lock, self._backend.pausing_acquisition():
            self._backend.set_attribute_value(
                "defect_correct_mode", _CORRECTION_ON if enable else _CORRECTION_OFF
            )
            self._keep(
                hot_pixel_correction=int(self._read_attribute("defect_correct_mode"))
                == _CORRECTION_ON
            )


SimulatedORCAFusionCamera = simulated_camera_for(ORCAFusionCamera)
