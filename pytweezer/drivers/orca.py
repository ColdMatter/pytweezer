from pytweezer.drivers.camera_base import (
    Camera,
    simulated_camera_for,
)
from pytweezer.drivers.hamamatsu import open_dcam_camera


class ORCACamera(Camera):
    """dcam shim: translates the generic :class:`Camera` interface to the
    Hamamatsu ORCA-R2's pylablib DCAM driver, plus its binning, readout-speed
    and contrast-gain controls.

    Args:
        serial_number: DCAM serial number of the camera to open (e.g. ``"280187"``).
            If None, the only connected ORCA-R2 is opened.
        external_exposure: Start in :meth:`set_external_exposure_mode`; otherwise
            the camera keeps DCAM's own trigger mode (internal after power-up).
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
    ):
        self.serial_number = serial_number
        super().__init__(image_dir=image_dir, timeout=timeout, stream_name=stream_name)
        if external_exposure:
            self.set_external_exposure_mode()

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
        if self._backend is not None:
            self._backend.close()

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        *_, hbin, vbin = self._backend.get_roi()
        hstart, _, vstart, *_ = self._backend.set_roi(
            x0, x0 + width, y0, y0 + height, hbin, vbin
        )
        self.roi_origin = (int(hstart), int(vstart))
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
        self._backend.set_attribute_value("binning", int(binning))

    def set_readout_speed(self, speed: int):
        self._backend.set_attribute_value("readout_speed", int(speed))

    def set_gain(self, gain: float):
        """Set the contrast gain (an integer index)."""
        self._backend.set_attribute_value("contrast_gain", int(gain))

    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self._backend.set_attribute_value("trigger_active", 2)
        self._backend.setup_ext_trigger(invert=True)

    def trigger(self):
        """Send a software trigger to the camera."""
        self._backend.send_software_trigger()


SimulatedORCACamera = simulated_camera_for(ORCACamera)
