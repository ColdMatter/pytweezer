import pylablib as pll
import pylablib.devices.Thorlabs as thorcam

from pytweezer.drivers.camera_base import (
    Camera,
    simulated_camera_for,
)

THORLABS_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"
pll.par["devices/dlls/thorlabs_tlcam"] = THORLABS_DLL_PATH

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/ThorCamImages"
)


class ThorCam(Camera):
    """ThorLabs shim: translates the generic :class:`Camera` interface to the
    ThorLabs camera's pylablib driver, plus any camera-specific controls."""

    image_prefix = "ThorCam"

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
        buffer_frames: int = 1000,
        live: bool = False,
    ):
        """Open the camera and apply whichever settings are given.

        Args:
            serial_number: Camera to open; ``None`` opens the first one found.
            trigger_mode: pylablib's ``"int"``, ``"ext"`` or ``"bulb"`` (the trigger
                pulse sets each exposure); ``None`` leaves the camera's own.
            trigger_polarity: ``"rise"`` (active high) or ``"fall"``; applied only
                with a ``trigger_mode``.
            binning: Square binning over the full sensor.
            gain_db: Analogue gain in dB.
            buffer_frames: Ring-buffer size for :meth:`arm`.
            live: Arm now and broadcast every frame on the image stream.
        """
        self.serial_number = serial_number
        super().__init__(
            image_dir=image_dir or IMAGE_DIRECTORY,
            timeout=timeout,
            stream_name=stream_name,
        )
        if trigger_mode is not None:
            self._backend.set_trigger_mode(trigger_mode)
            self._backend.setup_ext_trigger(trigger_polarity)
        if binning is not None:
            self._backend.set_roi(hbin=int(binning), vbin=int(binning))
        if gain_db is not None:
            self._backend.set_gain(float(gain_db))
        if live:
            self.arm(buffer_frames)
            self.start_live()

    # ------------------------------------------------------------------ #
    # Camera hooks
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        try:
            serial = self.serial_number
            if serial is None:
                serial = thorcam.list_cameras_tlcam()[0]
            camera = thorcam.ThorlabsTLCamera(serial=serial)
            camera.open()
        except Exception as exc:
            self._backend = None
            raise RuntimeError("Could not connect to ThorLabs camera") from exc
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
        self._backend.setup_acquisition(int(nframes))

    def start_acquisition(self):
        self._backend.start_acquisition()

    def stop_acquisition(self):
        self._backend.stop_acquisition()

    def settings(self) -> dict:
        backend = self._backend
        return {
            "serial_number": self.serial_number,
            "trigger_mode": backend.get_trigger_mode(),
            "trigger_polarity": backend.get_ext_trigger_parameters(),
            "binning": list(backend.get_roi()[4:6]),
            "gain_db": backend.get_gain(),
            "exposure_s": backend.get_exposure(),
            "roi": list(backend.get_roi()[:4]),
        }

    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images


#: One-size-fits-all simulated camera, with the ThorLabs camera's controls
#: auto-stubbed so the simulated surface matches the real driver.
SimulatedThorLabsCamera = simulated_camera_for(ThorCam)
