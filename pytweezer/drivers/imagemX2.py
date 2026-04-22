import argparse
import logging
import os
import socket
import time
from functools import wraps
from tkinter import NO
from typing import Any, Optional
from unittest.mock import MagicMock

from imageio import config
import numpy as np
import pylablib.devices.DCAM as dcam
import tifffile as tiff
from sipyco.pc_rpc import Client as RPCClient, simple_server_loop

from pytweezer.servers import ImageClient
from pytweezer.servers.configreader import ConfigReader

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
        timeout: float = 5.0,
        stream_name: Optional[str] = None,
    ):
        self.image_dir = image_dir or IMAGE_DIRECTORY
        self.timeout = timeout
        self.dcam: dcam.DCAMCamera
        self._open_camera()

        self.stream_name = stream_name
        if stream_name:
            self.image_client = ImageClient(stream_name)

    def _open_camera(self):
        try:
            self.dcam = dcam.DCAMCamera()
            self.dcam.open()
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
        except Exception:
            LOGGER.exception("Failed to close ImagEM X2 camera cleanly")
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

    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    @requires_camera
    def set_direct_em_gain_mode(self, mode: int):
        self.dcam.set_attribute_value("direct_em_gain_mode", int(mode))

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
    def acquire_n_frames(self, nframes: int, start_frame: int = 0, timeout: float | None = None) -> np.ndarray:
        self.dcam.wait_for_frame(nframes=int(nframes), timeout=timeout or self.timeout)
        images, _infos = self.dcam.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return np.asarray(images)

    @requires_camera
    def acquire_single_frame(
        self,
        timeout=None,
        exp_info: dict[str, Any] | None = None,
        autosave: bool = False,
        broadcast: bool = False,
    ) -> np.ndarray:
        self.dcam.wait_for_frame(n_frames=1, timeout=timeout or self.timeout)
        image, _info = self.dcam.read_newest_image(return_info=True)
        image = np.asarray(image)
        if autosave:
            ImagEMX2Camera.save_tiff(image=image, image_dir=self.image_dir)
        info = {
            "timestamp": time.time(),
            "_imgresolution": [1, 1],
            "_offset": [0, 0],
        }
        if exp_info is not None:
            info.update(exp_info)
        if broadcast:
            if self.image_client is None:
                LOGGER.error("tried to broad image but no client exists")
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


class SimulatedImagEMX2Camera:
    """Drop-in simulated backend with the same control interface as ImagEMX2Camera."""

    def __init__(self, image_dir: str | None = None, timeout: float = 5.0):
        self.image_dir = image_dir or IMAGE_DIRECTORY
        self.timeout = timeout
        self._shape = (256, 256)
        self._nframes = 1
        self._running = False
        self._rng = np.random.default_rng()

    def close(self):
        self._running = False

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self._shape = (int(height), int(width))

    def set_ccd_mode(self, mode: int):
        return None

    def enable_em_gain(self, enable: bool = True):
        return None

    def set_direct_em_gain_mode(self, mode: int):
        return None

    def enable_direct_em_gain(self, enable: bool = True):
        return None

    def set_sensitivity(self, sensitivity: int):
        return None

    def set_trigger_source(self, source: str):
        return None

    def set_exposure_time(self, exposure: float):
        return None

    def set_external_exposure_mode(self):
        return None

    def setup_acquisition(self, acq_mode: str, nframes: int):
        self._nframes = max(1, int(nframes))

    def start_acquisition(self):
        self._running = True

    def stop_acquisition(self):
        self._running = False

    def _generate_frame(self) -> np.ndarray:
        h, w = self._shape
        image = self._rng.normal(loc=120.0, scale=9.0, size=(h, w)).astype(np.float32)
        y, x = np.mgrid[0:h, 0:w]
        for _ in range(12):
            cy = self._rng.integers(20, max(21, h - 20))
            cx = self._rng.integers(20, max(21, w - 20))
            amp = self._rng.uniform(80.0, 280.0)
            sigma = self._rng.uniform(1.3, 2.5)
            image += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
        return np.clip(image, 0.0, 65535.0).astype(np.uint16)

    def acquire_n_frames(self, nframes: int, start_frame: int = 0) -> np.ndarray:
        del start_frame
        if not self._running:
            raise RuntimeError("Simulated camera is not running")
        count = max(1, int(nframes))
        time.sleep(0.1 * count)
        return np.stack([self._generate_frame() for _ in range(count)], axis=0)

    @staticmethod
    def save_tiff(image: np.ndarray, image_dir: str | None = None, run_no: int = 0):
        ImagEMX2Camera.save_tiff(image=image, image_dir=image_dir, run_no=run_no)


class ImagEMX2CameraClient(RPCClient):
    """Experiment-side RPC client with direct access to camera methods."""

    def __init__(
        self,
        server_name: str = "ImagEM X2 Camera",
        host: str | None = None,
        port: int | None = None,
        target_name: Any = "camera",
        timeout: float | None = 5.0,
    ):

        conf = ConfigReader.getConfiguration()
        server_conf = conf.get("Devices", {}).get(server_name, {})

        self.host = host or server_conf.get("host", "127.0.0.1")
        self.port = int(port or server_conf.get("port", 3251))
        self.stream_name = server_conf.get("stream_name", "imagemx2")
        self.image_dir = IMAGE_DIRECTORY
        self.imstream = ImageClient(self.stream_name)
        super().__init__(
            host=self.host, port=self.port, target=target_name, timeout=timeout
        )

    def ping(self):
        return {"ok": True, "host": self.host, "port": self.port, "target": "camera"}


def run_server(
    host: str,
    port: int,
    stream_name: str = "imagemx2",
    image_dir: str | None = None,
    timeout: float = 5.0,
    simulate: bool = False,
):
    LOGGER.info(
        "Starting ImagEM X2 RPC server host=%s port=%s stream=%s simulate=%s",
        host,
        port,
        stream_name,
        simulate,
    )
    if simulate:
        LOGGER.warning("Running ImagEM X2 Camera server in SIMULATION MODE")
        camera = SimulatedImagEMX2Camera(image_dir=image_dir, timeout=timeout)
    else:
        camera = ImagEMX2Camera(
            stream_name=stream_name,
            image_dir=image_dir,
            timeout=timeout,
        )


    simple_server_loop(
        {"camera": camera},
        host=host,
        port=int(port),
        description="ImagEM X2 RPC server",
    )

def main():

    parser = argparse.ArgumentParser(description="Run ImagEM X2 sipyco RPC server")
    parser.add_argument("--host", default="127.0.0.1", help="RPC bind host")
    parser.add_argument("--port", type=int, default=3251, help="RPC bind port")
    parser.add_argument("--stream-name", default="imagemx2", help="Image stream name")
    parser.add_argument(
        "--image-dir", default=None, help="optional TIFF autosave directory"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="frame wait timeout in seconds"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="use simulated camera backend"
    )
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    args = parser.parse_args()
    
    config = ConfigReader.getConfiguration()
    server_conf = config.get("Devices", {}).get("ImagEM X2 Camera", {})
    simulate = server_conf.get("simulate", False)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_server(
        host=args.host,
        port=args.port,
        stream_name=args.stream_name,
        image_dir=args.image_dir,
        timeout=args.timeout,
        simulate=simulate,
    )


if __name__ == "__main__":
    main()
