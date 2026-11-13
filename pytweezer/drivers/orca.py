from pytweezer.drivers.camera_base import (
    BINNING,
    EXPOSURE,
    ROI,
    TRIGGER_POLARITY,
    Camera,
    Control,
    gain,
    simulated_camera_for,
    trigger_source,
)
from pytweezer.drivers.hamamatsu import (
    dcam_roi_controls,
    dcam_trigger_controls,
    open_dcam_camera,
    read_dcam_controls,
    read_dcam_range_timed,
    set_dcam_exposure_mode,
    set_dcam_trigger_polarity,
)


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
    """

    image_prefix = "ORCA"
    model_prefix = "C10600"

    CONTROLS = (
        EXPOSURE,
        Control(
            "exposure_mode",
            "Exposure mode",
            "Exposure & trigger",
            "choice",
            "internal",
            choices=("internal", "edge", "level"),
        ),
        trigger_source(("int", "ext", "software")),
        TRIGGER_POLARITY,
        gain("Contrast gain", "integer", minimum=0),
        Control("readout_speed", "Readout speed", "Readout", "integer", 1, minimum=1),
        BINNING,
        *ROI,
    )

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5 * 60.0,
        stream_name: str | None = None,
        serial_number: str | None = None,
        external_exposure: bool = False,
        object_pixel_m: float | None = None,
    ):
        self.serial_number = serial_number
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            object_pixel_m=object_pixel_m,
        )
        if external_exposure:
            self.set_external_exposure_mode()
        self.read_controls()

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

    def _read_control_values(self) -> dict:
        return {
            **read_dcam_controls(self._backend),
            "gain": int(self._backend.get_attribute_value("contrast_gain", default=0)),
            "readout_speed": int(
                self._backend.get_attribute_value("readout_speed", default=1)
            ),
        }

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        with self._lock:
            *_, hbin, vbin = self._backend.get_roi()
            roi = self._backend.set_roi(x0, x0 + width, y0, y0 + height, hbin, vbin)
            return self._keep_roi(roi)

    def set_trigger_source(self, source: str):
        with self._lock:
            self._backend.set_trigger_mode(source)
            self._keep(**dcam_trigger_controls(self._backend))

    def set_trigger_polarity(self, polarity: str):
        with self._lock:
            set_dcam_trigger_polarity(self._backend, polarity)
            self._keep(**dcam_trigger_controls(self._backend))

    def set_exposure_time(self, exposure: float):
        with self._lock:
            self._keep(exposure=float(self._backend.set_exposure(exposure)))

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

    def _read_range_timed(self, start: int, stop: int):
        return read_dcam_range_timed(self._backend, start, stop)

    # ------------------------------------------------------------------ #
    # ORCA-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    def set_binning(self, binning: int):
        # Through set_roi: DCAM rejects a bare BINNING write while the frame
        # buffer is attached (DCAMERR_NOTSTABLE), and pylablib's set_roi frees it.
        with self._lock:
            hstart, hend, vstart, vend, *_ = self._backend.get_roi()
            roi = self._backend.set_roi(
                hstart, hend, vstart, vend, int(binning), int(binning)
            )
            self._keep_roi(roi)

    def _keep_roi(self, roi: tuple) -> tuple[int, int]:
        """Keep the ROI pylablib returned and the origin and binning frames are read with."""
        controls = dcam_roi_controls(roi)
        self.roi_origin = (controls["roi_x0"], controls["roi_y0"])
        self.binning = controls["binning"]
        self._keep(**controls)
        return self.roi_origin

    def set_readout_speed(self, speed: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("readout_speed", int(speed))
            self._keep(
                readout_speed=int(self._backend.get_attribute_value("readout_speed"))
            )

    def set_gain(self, gain: float):
        """Set the contrast gain (an integer index)."""
        with self._lock:
            self._backend.set_attribute_value("contrast_gain", int(gain))
            self._keep(gain=int(self._backend.get_attribute_value("contrast_gain")))

    def set_exposure_mode(self, mode: str):
        """``"internal"``, ``"edge"`` (a trigger starts an exposure of the set
        time) or ``"level"`` (the trigger pulse's width sets it)."""
        with self._lock:
            set_dcam_exposure_mode(self._backend, mode)
            self._keep(**dcam_trigger_controls(self._backend))

    def set_external_exposure_mode(self):
        self.set_exposure_mode("level")

    def trigger(self):
        """Send a software trigger to the camera."""
        self._backend.send_software_trigger()


SimulatedORCACamera = simulated_camera_for(ORCACamera)
