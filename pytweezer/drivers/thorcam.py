import logging
import os
import time
from functools import wraps
from typing import Any, Optional
import numpy as np
import pylablib as pll
import pylablib.devices.Thorlabs as tlcam
import tifffile as tiff


THORCAM_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/ThorCamImages"
)

LOGGER = logging.getLogger(__name__)

pll.par["devices/dlls/thorlabs_tlcam"] = THORCAM_DLL_PATH

THORCAM_IDS = {
    "motcam": "32148",
    "tweezercam": "38570",
}

def requires_camera(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._require_camera()
        return func(self, *args, **kwargs)

    return wrapper

class ThorCamera:
    """Low-level wrapper around the Thorlabs TLCamera driver."""

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5*60.0,
        id: str = 'motcam'
    ):
        self.image_dir = image_dir or IMAGE_DIRECTORY
        self.timeout = timeout
        self.thorcamlist = tlcam.list_cameras_tlcam()
        self.tlcam: tlcam.ThorlabsTLCamera
        self.id = id
        self._open_camera()
        

    def _open_camera(self):
        try:
            self.tlcam = tlcam.ThorlabsTLCamera(serial=THORCAM_IDS[self.id])
            self.tlcam.open()
            print(f"Thorlabs TLCamera - {self.id} - initialised.")
        except Exception as exc:
            self.tlcam = None
            raise RuntimeError("Could not connect to Thorlabs TLCamera") from exc

    def _require_camera(self):
        if self.tlcam is None:
            raise RuntimeError("Thorlabs TLCamera connection has been relinquished")

    def close(self):
        try:
            if self.tlcam is not None:
                self.tlcam.close()
                print(f"Thorlabs TLCamera - {self.id} - closed.")
        except Exception:
            print(f"Failed to close Thorlabs TLCamera - {self.id} - cleanly")
        finally:
            self.tlcam = None

    def relinquish_camera(self):
        self.close()
        return {"ok": True, "relinquished": True}

    def reacquire_camera(self):
        if self.tlcam is None:
            self._open_camera()
        return {"ok": True, "relinquished": False}

    @requires_camera
    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self.tlcam.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    @requires_camera
    def set_trigger_source(self, source: str):
        self.tlcam.set_trigger_mode(source)

    @requires_camera
    def set_exposure_time(self, exposure: float):
        self.tlcam.set_exposure(exposure)

    @requires_camera
    def setup_acquisition(self, nframes: int):
        self.tlcam.setup_acquisition(int(nframes))

    @requires_camera
    def start_acquisition(self):
        self.tlcam.start_acquisition()

    @requires_camera
    def stop_acquisition(self):
        self.tlcam.stop_acquisition()

    @requires_camera
    def acquire_n_frames(self, nframes: int, start_frame: int = 0, autosave: bool = False, broadcast: bool = False) -> np.ndarray:
        self.tlcam.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self.tlcam.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        for i, image in enumerate(images):
            if autosave:
                ThorCamera.save_tiff(image=image, image_dir=self.image_dir)
            if broadcast:
                if self.image_client is None:
                    LOGGER.error("tried to broadcast image but no client exists")
                info = {
                    "timestamp": time.time(),
                    "index": start_frame + i,
                }
                self.image_client.send(image, info)
        return np.asarray(images)

    @requires_camera
    def acquire_single_frame(
        self,
        timeout=None,
        exp_info: dict[str, Any] | None = None,
        autosave: bool = False,
        broadcast: bool = False,
    ) -> np.ndarray:
        self.tlcam.wait_for_frame(n_frames=1, timeout=timeout)
        image, _info = self.tlcam.read_newest_image(return_info=True)
        image = np.asarray(image)
        if autosave:
            ThorCamera.save_tiff(image=image, image_dir=self.image_dir)
        info = {
            "timestamp": time.time(),
            "index": 0
        }
        if exp_info is not None:
            info.update(exp_info)
        if broadcast:
            if self.image_client is None:
                LOGGER.error("tried to broadcasr image but no client exists")
            self.image_client.send(image, info)
        return image

    @staticmethod
    def save_tiff(image: np.ndarray, image_dir: str | None = None, run_no: int = 0):
        image_dir = image_dir or IMAGE_DIRECTORY
        i = 1
        while os.path.exists(
            os.path.join(image_dir, f"HamTweezer{run_no:04d}_{i}.tif")
        ):
            i += 1
        filename = os.path.join(image_dir, f"HamTweezer{run_no:04d}_{i}.tif")
        tiff.imwrite(filename, image)

