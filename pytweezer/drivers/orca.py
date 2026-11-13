from pytweezer.drivers.camera_base import (
    Camera,
    simulated_camera_for,
)
from pytweezer.drivers.hamamatsu import open_dcam_camera, set_dcam_exposure_mode
from pytweezer.drivers.temperature_monitor import TemperatureMonitor


class ORCACamera(Camera):
    """dcam shim: translates the generic :class:`Camera` interface to the
    Hamamatsu ORCA-R2's pylablib DCAM driver, plus its binning, readout-speed
    and contrast-gain controls.

    It also runs the ORCA-Fusion BT, opened by serial number; that camera has no
    contrast gain, so :meth:`set_gain` fails on it.

    Args:
        serial_number: DCAM serial number of the camera to open (e.g. ``"280187"``).
            If None, the only connected ORCA-R2 is opened.
        external_exposure: Start in :meth:`set_external_exposure_mode`; otherwise
            the camera keeps DCAM's own trigger mode (internal after power-up).
        object_pixel_m: Metres per unbinned sensor pixel at the atoms.
        temperature_measurement: Database measurement the sensor temperature is
            written to every ``temperature_interval_s``; None to not monitor it.
        max_temperature_c: Sensor temperature above which the monitor warns.
    """

    image_prefix = "ORCA"
    model_prefix = "C10600"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5 * 60.0,
        stream_name: str | None = None,
        serial_number: str | None = None,
        external_exposure: bool = False,
        object_pixel_m: float | None = None,
        temperature_measurement: str | None = None,
        temperature_interval_s: float = 30.0,
        max_temperature_c: float | None = None,
    ):
        self.serial_number = serial_number
        self._temperature_monitor = None
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            object_pixel_m=object_pixel_m,
        )
        if external_exposure:
            self.set_external_exposure_mode()
        if temperature_measurement is not None:
            self._temperature_monitor = TemperatureMonitor(
                self.get_temperature,
                temperature_measurement,
                temperature_interval_s,
                max_temperature_c,
            )

    # ------------------------------------------------------------------ #
    # Camera hooks
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        try:
            camera = open_dcam_camera(self.serial_number, self.model_prefix)
        except Exception as exc:
            self._backend = None
            raise RuntimeError("Could not connect to ORCA camera") from exc
        self._backend = camera

    def _disconnect(self) -> None:
        if self._temperature_monitor is not None:
            self._temperature_monitor.stop()
        if self._backend is not None:
            self._backend.close()

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        *_, hbin, vbin = self._backend.get_roi()
        hstart, _, vstart, _, hbin, _ = self._backend.set_roi(
            x0, x0 + width, y0, y0 + height, hbin, vbin
        )
        self.roi_origin = (int(hstart), int(vstart))
        self.binning = int(hbin)
        return self.roi_origin

    def set_trigger_source(self, source: str):
        self._backend.set_trigger_mode(source)

    def set_exposure_time(self, exposure: float):
        self._backend.set_exposure(exposure)

    def setup_acquisition(self, nframes: int, mode: str = "sequence"):
        self._backend.setup_acquisition(mode, int(nframes))

    def start_acquisition(self):
        self._backend.start_acquisition()

    def stop_acquisition(self):
        self._backend.stop_acquisition()

    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images

    # ------------------------------------------------------------------ #
    # ORCA-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    def set_binning(self, binning: int):
        # Through set_roi: DCAM rejects a bare BINNING write while the frame
        # buffer is attached (DCAMERR_NOTSTABLE), and pylablib's set_roi frees it.
        hstart, hend, vstart, vend, *_ = self._backend.get_roi()
        hstart, _, vstart, _, hbin, _ = self._backend.set_roi(
            hstart, hend, vstart, vend, int(binning), int(binning)
        )
        self.roi_origin = (int(hstart), int(vstart))
        self.binning = int(hbin)

    def set_readout_speed(self, speed: int):
        self._backend.set_attribute_value("readout_speed", int(speed))

    def set_gain(self, gain: float):
        """Set the contrast gain (an integer index)."""
        self._backend.set_attribute_value("contrast_gain", int(gain))

    def set_exposure_mode(self, mode: str):
        """``"internal"``, ``"edge"`` (a trigger starts an exposure of the set
        time) or ``"level"`` (the trigger pulse's width sets it)."""
        set_dcam_exposure_mode(self._backend, mode)

    def set_external_exposure_mode(self):
        self.set_exposure_mode("level")

    def get_temperature(self) -> float:
        """Sensor temperature in °C."""
        with self._lock:
            return float(self._backend.get_attribute_value("sensor_temperature"))

    def trigger(self):
        """Send a software trigger to the camera."""
        self._backend.send_software_trigger()


SimulatedORCACamera = simulated_camera_for(ORCACamera)
