from typing import ClassVar

import numpy as np

from pytweezer.drivers.camera_base import (
    BINNING,
    EXPOSURE,
    ROI,
    TRIGGER_POLARITY,
    Control,
    gain,
    simulated_camera_for,
    trigger_source,
)
from pytweezer.drivers.hamamatsu import EXPOSURE_MODE, HamamatsuCamera


class ImagEMX2Camera(HamamatsuCamera):
    """dcam shim for the Hamamatsu ImagEM X2, adding the EM-gain/sensitivity
    controls specific to this camera. Arguments are those of :class:`HamamatsuCamera`;
    with no ``serial_number`` the only connected ImagEM X2 is opened."""

    image_prefix = "HamTweezer"
    model_prefix = "C9100"
    default_timeout = 20.0

    CONTROLS = (
        EXPOSURE,
        EXPOSURE_MODE,
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

    def _read_control_values(self) -> dict:
        return {
            **super()._read_control_values(),
            "gain": int(self._read_attribute("sensitivity")),
            "em_gain": int(self._read_attribute("ccd_mode")) == 2,
            "direct_em_gain": int(self._read_attribute("direct_em_gain_mode")) == 2,
        }

    def set_gain(self, gain: float):
        """Set the sensitivity, the EM gain level once :meth:`enable_em_gain` is on."""
        self.set_sensitivity(int(gain))

    # ------------------------------------------------------------------ #
    # ImagEM X2-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    def set_ccd_mode(self, mode: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("ccd_mode", int(mode))
            self._keep(em_gain=int(self._read_attribute("ccd_mode")) == 2)

    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    def set_direct_em_gain_mode(self, mode: int):
        with self._lock, self._backend.pausing_acquisition(clear=True):
            self._backend.set_attribute_value("direct_em_gain_mode", int(mode))
            self._keep(
                direct_em_gain=int(self._read_attribute("direct_em_gain_mode")) == 2
            )

    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    def set_sensitivity(self, sensitivity: int):
        with self._lock:
            self._backend.set_attribute_value("sensitivity", int(sensitivity))
            self._keep(gain=int(self._read_attribute("sensitivity")))

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
