import pylablib as pll
import pylablib.devices.Thorlabs as thorcam

from pytweezer.drivers.camera_base import (
    Camera,
    simulated_camera_for,
)

THORLABS_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"
pll.par["devices/dlls/thorlabs_tlcam"] = THORLABS_DLL_PATH


class ThorCam(Camera):
    """ThorLabs shim: translates the generic :class:`Camera` interface to the
    ThorLabs camera's pylablib driver, plus any camera-specific controls."""

    image_prefix = "ThorCam"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
    ):
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
        )

    # ------------------------------------------------------------------ #
    # Camera hooks
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        try:
            thorcam.list_cameras_tlcam()
            camera = thorcam.ThorlabsTLCamera(serial="38570")
            camera.open()
        except Exception as exc:
            self._backend = None
            raise RuntimeError("Could not connect to ThorLabs camera") from exc
        self._backend = camera

    def _disconnect(self) -> None:
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

    def set_trigger_polarity(self, polarity: str):
        """``"rise"`` (active high) or ``"fall"`` (active low)."""
        self._backend.setup_ext_trigger(polarity)

    def set_exposure_time(self, exposure: float):
        self._backend.set_exposure(exposure)

    def set_binning(self, binning: int):
        self._backend.set_roi(hbin=int(binning), vbin=int(binning))
        self.binning = int(binning)

    def set_gain(self, gain: float):
        """Set the analogue gain in dB."""
        self._backend.set_gain(float(gain))

    def setup_acquisition(self, nframes: int, mode: str = "sequence"):
        self._backend.setup_acquisition(int(nframes))

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


#: One-size-fits-all simulated camera, with the ThorLabs camera's controls
#: auto-stubbed so the simulated surface matches the real driver.
SimulatedThorLabsCamera = simulated_camera_for(ThorCam)
