import numpy as np

from pytweezer.drivers.camera_base import (
    Camera,
    simulated_camera_for,
)
from pytweezer.drivers.hamamatsu import open_dcam_camera

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/HamCamImages"
)


class ImagEMX2Camera(Camera):
    """dcam shim: translates the generic :class:`Camera` interface to the
    Hamamatsu ImagEM X2's pylablib DCAM driver, plus the EM-gain/sensitivity
    controls specific to this camera.

    Args:
        serial_number: DCAM serial number of the camera to open (e.g. ``"000407"``).
            If None, the only connected ImagEM X2 is opened.
    """

    image_prefix = "HamTweezer"
    model_prefix = "C9100"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 20.0,
        stream_name: str | None = None,
        serial_number: str | None = None,
    ):
        self.serial_number = serial_number
        super().__init__(
            image_dir=image_dir or IMAGE_DIRECTORY,
            timeout=timeout,
            stream_name=stream_name,
        )

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

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        hstart, _, vstart, *_ = self._backend.set_roi(x0, x0 + width, y0, y0 + height)
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
    # ImagEM X2-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    def set_ccd_mode(self, mode: int):
        self._backend.set_attribute_value("ccd_mode", int(mode))

    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    def set_direct_em_gain_mode(self, mode: int):
        self._backend.set_attribute_value("direct_em_gain_mode", int(mode))

    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    def set_sensitivity(self, sensitivity: int):
        self._backend.set_attribute_value("sensitivity", int(sensitivity))

    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self._backend.set_attribute_value("trigger_active", 2)
        self._backend.setup_ext_trigger(invert=True)

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
