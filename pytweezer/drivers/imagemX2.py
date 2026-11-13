from typing import ClassVar

import numpy as np

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


class ImagEMX2Camera(Camera):
    """dcam shim: translates the generic :class:`Camera` interface to the
    Hamamatsu ImagEM X2's pylablib DCAM driver, plus the EM-gain/sensitivity
    controls specific to this camera.

    Args:
        serial_number: DCAM serial number of the camera to open (e.g. ``"000407"``).
            If None, the only connected ImagEM X2 is opened.
        external_exposure: Start in :meth:`set_external_exposure_mode`; otherwise
            the camera keeps DCAM's own trigger mode (internal after power-up).
        object_pixel_m: Metres per unbinned sensor pixel at the atoms.
    """

    image_prefix = "HamTweezer"
    model_prefix = "C9100"

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
        gain("Sensitivity", "integer", minimum=0),
        Control("em_gain", "EM gain", "Gain", "bool", True),
        Control("direct_em_gain", "Direct EM gain", "Gain", "bool", False),
        BINNING,
        *ROI,
    )
    CONTROL_SETTERS: ClassVar[dict[str, str]] = {
        "em_gain": "enable_em_gain",
        "direct_em_gain": "enable_direct_em_gain",
    }

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 20.0,
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
            raise RuntimeError("Could not connect to ImagEM X2 camera") from exc
        self._backend = camera

    def _disconnect(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def _read_control_values(self) -> dict:
        return {
            **read_dcam_controls(self._backend),
            "gain": int(self._backend.get_attribute_value("sensitivity")),
            "em_gain": int(self._backend.get_attribute_value("ccd_mode")) == 2,
            "direct_em_gain": int(
                self._backend.get_attribute_value("direct_em_gain_mode")
            )
            == 2,
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

    def set_gain(self, gain: float):
        """Set the sensitivity, the EM gain level once :meth:`enable_em_gain` is on."""
        self.set_sensitivity(int(gain))

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
    # ImagEM X2-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    def set_ccd_mode(self, mode: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("ccd_mode", int(mode))
            self._keep(em_gain=int(self._backend.get_attribute_value("ccd_mode")) == 2)

    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    def set_direct_em_gain_mode(self, mode: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("direct_em_gain_mode", int(mode))
            self._keep(
                direct_em_gain=int(
                    self._backend.get_attribute_value("direct_em_gain_mode")
                )
                == 2
            )

    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    def set_sensitivity(self, sensitivity: int):
        with self._lock:
            self._backend.set_attribute_value("sensitivity", int(sensitivity))
            self._keep(gain=int(self._backend.get_attribute_value("sensitivity")))

    def set_exposure_mode(self, mode: str):
        """``"internal"``, ``"edge"`` (a trigger starts an exposure of the set
        time) or ``"level"`` (the trigger pulse's width sets it)."""
        with self._lock:
            set_dcam_exposure_mode(self._backend, mode)
            self._keep(**dcam_trigger_controls(self._backend))

    def set_external_exposure_mode(self):
        self.set_exposure_mode("level")

    def get_conversion_factor(self) -> tuple[float, float]:
        """The camera's photoelectron conversion ``(coefficient, offset)`` at its current settings."""
        return (
            float(self._backend.get_attribute_value("CONVERSION FACTOR COEFF")),
            float(self._backend.get_attribute_value("CONVERSION FACTOR OFFSET")),
        )

    def get_all_attribute_values(self) -> dict:
        """Every readable DCAM attribute's current value, enums as their text labels."""
        values = self._backend.get_all_attribute_values(enum_as_str=True)
        return values.as_dict(style="flat")

    def trigger(self):
        """Send a software trigger to the camera."""
        self._backend.send_software_trigger()


#: One-size-fits-all simulated camera, with the ImagEM's EM-gain/sensitivity
#: setters auto-stubbed so the simulated surface matches the real driver.
SimulatedImagEMX2Camera = simulated_camera_for(
    ImagEMX2Camera, defaults={"get_conversion_factor": (0.0095, 1917.0)}
)


class SimulatedTweezerArrayCamera(SimulatedImagEMX2Camera):
    """Simulated tweezer camera imaging a stochastically loaded 13x13, 5 um array.

    Each acquisition alternates an atom image with an empty background image,
    so a two-image-per-shot sequence analyses like the real thing.
    """

    array_shape = (13, 13)
    site_spacing_px = 10.36  # 5 um at the HamCam's 2.07 px/um
    loading_probability = 0.55
    offset_counts = 1917.0
    atom_peak_counts = 600.0
    atom_sigma_px = 1.2
    _frame_index = 0

    def start_acquisition(self) -> None:
        super().start_acquisition()
        self._frame_index = 0

    def _generate_frame(self) -> np.ndarray:
        h, w = self._shape
        image = self._rng.normal(self.offset_counts, 30.0, size=(h, w))
        if self._frame_index % 2 == 0:
            rows, cols = self.array_shape
            y_sites = (np.arange(rows) - (rows - 1) / 2) * self.site_spacing_px + h / 2
            x_sites = (np.arange(cols) - (cols - 1) / 2) * self.site_spacing_px + w / 2
            y, x = np.mgrid[0:h, 0:w]
            for y_site in y_sites:
                for x_site in x_sites:
                    if self._rng.random() < self.loading_probability:
                        image += self.atom_peak_counts * np.exp(
                            -((x - x_site) ** 2 + (y - y_site) ** 2)
                            / (2 * self.atom_sigma_px**2)
                        )
        self._frame_index += 1
        return np.clip(image, 0.0, 65535.0).astype(np.uint16)
