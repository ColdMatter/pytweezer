from typing import Any

import numpy as np
import pylablib as pll
import pylablib.devices.Thorlabs as thorcam
from pylablib.devices.Thorlabs.TLCamera import lib as tl_lib

from pytweezer.drivers.camera_base import (
    BINNING,
    EXPOSURE,
    FRAME_RATE,
    ROI,
    TRIGGER_POLARITY,
    Camera,
    gain,
    roi_controls,
    simulated_camera_for,
    trigger_source,
)

THORLABS_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"
pll.par["devices/dlls/thorlabs_tlcam"] = THORLABS_DLL_PATH

TRIGGER_SOURCES = ("int", "ext", "bulb", "software")
INTERNAL_TRIGGER_SOURCES = ("int", "software")


class _TLCamera(thorcam.ThorlabsTLCamera):
    """pylablib's Thorlabs camera, taking one frame per :meth:`send_software_trigger`
    while :attr:`software_triggered` is set.

    pylablib's ``"int"`` is the SDK's software-triggered mode started free-running;
    this starts it with one frame per trigger instead, including when pylablib
    restarts the acquisition after changing a setting.
    """

    software_triggered = False

    def start_acquisition(self, **kwargs):
        if self.software_triggered:
            kwargs.update(frames_per_trigger=1, auto_start=False)
        super().start_acquisition(**kwargs)


class ThorCam(Camera):
    """ThorLabs shim: translates the generic :class:`Camera` interface to the
    ThorLabs camera's pylablib driver, plus any camera-specific controls."""

    image_prefix = "ThorCam"

    CONTROLS = (
        EXPOSURE,
        FRAME_RATE,
        trigger_source(TRIGGER_SOURCES),
        TRIGGER_POLARITY,
        gain("Gain", "number", unit="dB", minimum=0),
        BINNING,
        *ROI,
    )

    #: Frequency of the camera's frame timestamp clock in Hz; ``None`` if it reports none.
    _timestamp_clock_hz: float | None = None

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        serial_number: str | None = "32148",
        trigger_mode: str | None = None,
        trigger_polarity: str = "rise",
        binning: int | None = None,
        gain_db: float | None = None,
        object_pixel_m: float | None = None,
        has_external_trigger: bool = True,
    ):
        """Open the camera and apply whichever settings are given.

        Args:
            serial_number: Camera to open; ``None`` opens the first one found.
            trigger_mode: ``"int"``, ``"ext"``, ``"bulb"`` (the trigger pulse sets
                each exposure) or ``"software"``; ``None`` keeps the trigger the
                camera starts on (see :class:`Camera`).
            trigger_polarity: ``"rise"`` (active high) or ``"fall"``; applied only
                on a camera with an external trigger.
            binning: Square binning, keeping the camera's ROI.
            gain_db: Analogue gain in dB.
            object_pixel_m: Metres per unbinned sensor pixel at the atoms.
            has_external_trigger: ``False`` for a camera without an external
                trigger input, which then offers only the internal and software
                triggers and starts on software triggers.
        """
        self.serial_number = serial_number
        self.has_external_trigger = has_external_trigger
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            object_pixel_m=object_pixel_m,
        )
        if trigger_mode is not None:
            self.set_trigger_source(trigger_mode)
        if self.has_external_trigger:
            self.set_trigger_polarity(trigger_polarity)
        if binning is not None:
            self.set_binning(binning)
        if gain_db is not None:
            self.set_gain(gain_db)
        self.read_controls()

    # ------------------------------------------------------------------ #
    # Camera hooks
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        try:
            serial = self.serial_number
            if serial is None:
                serial = thorcam.list_cameras_tlcam()[0]
            camera = _TLCamera(serial=serial)
            camera.open()
        except Exception as exc:
            self._backend = None
            raise RuntimeError("Could not connect to ThorLabs camera") from exc
        self._backend = camera
        self._timestamp_clock_hz = camera.get_timestamp_clock_frequency()

    def _disconnect(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def _read_control_values(self) -> dict:
        backend = self._backend
        values = {
            "exposure": float(backend.get_exposure()),
            "trigger_source": self._trigger_source(),
            "gain": float(backend.get_gain()),
            "frame_rate": self._frame_rate(),
            **roi_controls(backend.get_roi()),
        }
        if self.has_external_trigger:
            values["trigger_polarity"] = backend.get_ext_trigger_parameters()
        self.roi_origin = (values["roi_x0"], values["roi_y0"])
        self.binning = values["binning"]
        return values

    # pylablib itself pauses the acquisition for the trigger, ROI and binning
    # settings.

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        with self._lock:
            *_, hbin, vbin = self._backend.get_roi()
            roi = self._backend.set_roi(x0, x0 + width, y0, y0 + height, hbin, vbin)
            return self._keep_roi(roi)

    def set_trigger_source(self, source: str):
        with self._lock:
            software = source == "software"
            self._backend.software_triggered = software
            self._backend.set_trigger_mode("int" if software else source)
            self._keep(trigger_source=self._trigger_source())

    def _trigger_source(self) -> str:
        if self._backend.software_triggered:
            return "software"
        return self._backend.get_trigger_mode()

    def trigger(self):
        self._backend.send_software_trigger()

    def set_trigger_polarity(self, polarity: str):
        """``"rise"`` (active high) or ``"fall"`` (active low)."""
        with self._lock:
            self._keep(trigger_polarity=self._backend.setup_ext_trigger(polarity))

    def set_exposure_time(self, exposure: float):
        # pylablib does not pause the acquisition for this.
        with self._lock:
            self._keep(exposure=float(self._backend.set_exposure(exposure)))

    def set_binning(self, binning: int):
        """Bin ``binning`` x ``binning`` pixels together, keeping the ROI."""
        self._rebin_keeping_roi(binning)

    def _control_limits(self) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            horizontal, _ = self._backend.get_roi_limits()
            slowest, fastest = tl_lib.tl_camera_get_frame_rate_control_value_range(
                self._backend.handle
            )
        return {
            "binning": {"maximum": int(horizontal.maxbin)},
            "frame_rate": {"minimum": float(slowest), "maximum": float(fastest)},
            "trigger_source": {
                "choices": list(
                    TRIGGER_SOURCES
                    if self.has_external_trigger
                    else INTERNAL_TRIGGER_SOURCES
                )
            },
            "trigger_polarity": {} if self.has_external_trigger else None,
        }

    def _frame_rate(self) -> float:
        """The frame rate, from the camera's frame period."""
        return 1.0 / float(self._backend.get_frame_timings().frame_period)

    def set_frame_rate(self, rate: float):
        """Set the frame rate of the internal trigger, and of live mode's software
        triggers, clamped to what the camera allows."""
        with self._lock:
            self._backend.set_frame_period(1.0 / float(rate))
            self._keep(frame_rate=self._frame_rate())

    def set_gain(self, gain: float):
        """Set the analogue gain in dB, truncated to the range the camera allows.

        pylablib does not pause the acquisition for this.
        """
        with self._lock:
            self._keep(gain=float(self._backend.set_gain(float(gain))))

    def setup_acquisition(self, nframes: int, mode: str = "sequence"):
        self._backend.setup_acquisition(int(nframes))

    def start_acquisition(self):
        self._backend.start_acquisition()

    def stop_acquisition(self):
        self._backend.stop_acquisition()

    def settings(self) -> dict:
        backend = self._backend
        with self._lock:
            settings = {
                "serial_number": self.serial_number,
                "trigger_mode": self._trigger_source(),
                "binning": list(backend.get_roi()[4:6]),
                "gain_db": backend.get_gain(),
                "exposure_s": backend.get_exposure(),
                "roi": list(backend.get_roi()[:4]),
            }
            if self.has_external_trigger:
                settings["trigger_polarity"] = backend.get_ext_trigger_parameters()
            return settings

    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images

    def _read_range_timed(self, start: int, stop: int):
        try:
            images, infos = self._backend.read_multiple_images(
                (start, stop), peek=True, return_info=True
            )
        except TypeError:
            # pylablib cannot unpack the metadata of a frame that came with none.
            return self._read_range(start, stop), None
        if images is None:
            return np.asarray([]), []
        clock_hz = self._timestamp_clock_hz
        if not clock_hz or any(info.pixelclock is None for info in infos):
            return np.asarray(images), None
        return np.asarray(images), [info.pixelclock / clock_hz for info in infos]


#: One-size-fits-all simulated camera, with the ThorLabs camera's controls
#: auto-stubbed so the simulated surface matches the real driver.
SimulatedThorLabsCamera = simulated_camera_for(ThorCam)
