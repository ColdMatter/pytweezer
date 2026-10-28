from pytweezer.drivers.camera_base import (
    Camera,
    requires_camera,
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
    """

    image_prefix = "ORCA"
    model_prefix = "C10600"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5 * 60.0,
        stream_name: str | None = None,
        serial_number: str | None = None,
    ):
        self.serial_number = serial_number
        super().__init__(image_dir=image_dir, timeout=timeout, stream_name=stream_name)

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

    @requires_camera
    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self._backend.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    @requires_camera
    def set_trigger_source(self, source: str):
        self._backend.set_trigger_mode(source)

    @requires_camera
    def set_exposure_time(self, exposure: float):
        self._backend.set_exposure(exposure)

    @requires_camera
    def setup_acquisition(self, acq_mode: str, nframes: int):
        self._backend.setup_acquisition(acq_mode, int(nframes))

    @requires_camera
    def start_acquisition(self):
        self._backend.start_acquisition()

    @requires_camera
    def stop_acquisition(self):
        self._backend.stop_acquisition()

    @requires_camera
    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images

    # ------------------------------------------------------------------ #
    # ORCA-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    @requires_camera
    def set_binning(self, binning: int):
        self._backend.set_attribute_value("binning", int(binning))

    @requires_camera
    def set_readout_speed(self, speed: int):
        self._backend.set_attribute_value("readout_speed", int(speed))

    @requires_camera
    def set_gain(self, gain: int):
        self._backend.set_attribute_value("contrast_gain", int(gain))

    @requires_camera
    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self._backend.set_attribute_value("trigger_active", 2)
        self._backend.setup_ext_trigger(invert=True)

    def trigger(self):
        """Send a software trigger to the camera."""
        self._backend.send_software_trigger()


SimulatedORCACamera = simulated_camera_for(ORCACamera)
