import logging
import os
import time
from functools import wraps
from typing import Any, Optional
import numpy as np
import pylablib.devices.DCAM as dcam
import tifffile as tiff

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/HamCamImages"
)

LOGGER = logging.getLogger(__name__)

def requires_camera(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._require_camera()
        return func(self, *args, **kwargs)

    return wrapper


class ImagEMX2Camera:
    """Low-level wrapper around the Hamamatsu ImagEM X2 DCAM driver."""

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5*60.0,
    ):
        self.image_dir = image_dir or IMAGE_DIRECTORY
        self.timeout = timeout
        self.dcam: dcam.DCAMCamera
        self._open_camera()

    def _open_camera(self):
        try:
            self.dcam = dcam.DCAMCamera()
            self.dcam.open()
            print(f"ImagEM X2 camera initialised.")
        except Exception as exc:
            self.dcam = None
            raise RuntimeError("Could not connect to ImagEM X2 camera") from exc

    def _require_camera(self):
        if self.dcam is None:
            raise RuntimeError("ImagEM X2 camera connection has been relinquished")

    def close(self):
        try:
            if self.dcam is not None:
                self.dcam.close()
                print(f"ImagEM X2 camera closed.")
        except Exception:
            print("Failed to close ImagEM X2 camera cleanly.")
        finally:
            self.dcam = None

    def relinquish_camera(self):
        self.close()
        return {"ok": True, "relinquished": True}

    def reacquire_camera(self):
        if self.dcam is None:
            self._open_camera()
        return {"ok": True, "relinquished": False}

    @requires_camera
    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self.dcam.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    @requires_camera
    def set_ccd_mode(self, mode: int):
        self.dcam.set_attribute_value("ccd_mode", int(mode))

    @requires_camera
    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    @requires_camera
    def set_direct_em_gain_mode(self, mode: int):
        self.dcam.set_attribute_value("direct_em_gain_mode", int(mode))

    @requires_camera
    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    @requires_camera
    def set_sensitivity(self, sensitivity: int):
        self.dcam.set_attribute_value("sensitivity", int(sensitivity))

    @requires_camera
    def set_trigger_source(self, source: str):
        self.dcam.set_trigger_mode(source)

    @requires_camera
    def set_exposure_time(self, exposure: float):
        self.dcam.set_exposure(exposure)

    @requires_camera
    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self.dcam.set_attribute_value("trigger_active", 2)
        self.dcam.setup_ext_trigger(invert=True)

    @requires_camera
    def setup_acquisition(self, acq_mode: str, nframes: int):
        self.dcam.setup_acquisition(acq_mode, int(nframes))

    @requires_camera
    def start_acquisition(self):
        self.dcam.start_acquisition()

    @requires_camera
    def stop_acquisition(self):
        self.dcam.stop_acquisition()

    @requires_camera
    def acquire_n_frames(self, nframes: int, start_frame: int = 0, autosave: bool = False, broadcast: bool = False) -> np.ndarray:
        self.dcam.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self.dcam.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        for i, image in enumerate(images):
            if autosave:
                ImagEMX2Camera.save_tiff(image=image, image_dir=self.image_dir)
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
        self.dcam.wait_for_frame(nframes=1, timeout=timeout)
        image, _info = self.dcam.read_newest_image(return_info=True)
        image = np.asarray(image)
        if autosave:
            ImagEMX2Camera.save_tiff(image=image, image_dir=self.image_dir)
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

