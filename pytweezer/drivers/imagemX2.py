import argparse
import logging
import os
import time
from typing import Any

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


class ImagEMX2Camera:
    """Low-level wrapper around the Hamamatsu ImagEM X2 DCAM driver."""

    def __init__(self, image_dir: str | None = None, timeout: float = 5.0):
        try:
            self.dcam = dcam.DCAMCamera()
            self.dcam.open()
        except Exception as exc:
            raise RuntimeError("Could not connect to ImagEM X2 camera") from exc

        self.image_dir = image_dir or IMAGE_DIRECTORY
        self.timeout = timeout

    def close(self):
        try:
            self.dcam.close()
        except Exception:
            LOGGER.exception("Failed to close ImagEM X2 camera cleanly")

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self.dcam.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    def set_ccd_mode(self, mode: int):
        self.dcam.set_attribute_value("ccd_mode", int(mode))

    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    def set_direct_em_gain_mode(self, mode: int):
        self.dcam.set_attribute_value("direct_em_gain_mode", int(mode))

    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    def set_sensitivity(self, sensitivity: int):
        self.dcam.set_attribute_value("sensitivity", int(sensitivity))

    def set_trigger_source(self, source: str):
        self.dcam.set_trigger_mode(source)

    def set_exposure_time(self, exposure: float):
        self.dcam.set_exposure(exposure)

    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self.dcam.set_attribute_value("trigger_active", 2)
        self.dcam.setup_ext_trigger(invert=True)

    def setup_acquisition(self, acq_mode: str, nframes: int):
        self.dcam.setup_acquisition(acq_mode, int(nframes))

    def start_acquisition(self):
        self.dcam.start_acquisition()

    def stop_acquisition(self):
        self.dcam.stop_acquisition()

    def acquire_n_frames(self, nframes: int, start_frame: int = 0) -> np.ndarray:
        self.dcam.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self.dcam.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return np.asarray(images)

    @staticmethod
    def save_tiff(image: np.ndarray, image_dir: str | None = None, run_no: int = 0):
        image_dir = image_dir or IMAGE_DIRECTORY
        i = 1
        while os.path.exists(os.path.join(image_dir, f"HamTweezer{run_no:04d}_{i}.tif")):
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
        count = max(1, int(nframes))
        return np.asarray([self._generate_frame() for _ in range(count)])

    @staticmethod
    def save_tiff(image: np.ndarray, image_dir: str | None = None, run_no: int = 0):
        ImagEMX2Camera.save_tiff(image=image, image_dir=image_dir, run_no=run_no)


class ImagEMX2CameraClient:
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
        server_conf = conf.get("Servers", {}).get(server_name, {})

        self.host = host or server_conf.get("host", "127.0.0.1")
        self.port = int(port or server_conf.get("port", 3251))
        self.stream_name = server_conf.get("stream_name", "imagemx2")
        self.image_dir = IMAGE_DIRECTORY
        self.imstream = ImageClient(self.stream_name)
        self._rpc = RPCClient(self.host, self.port, target_name=target_name, timeout=timeout)

    def __getattr__(self, name):
        # Forward unknown methods to RPC target for direct driver access.
        return getattr(self._rpc, name)

    def close(self):
        try:
            self._rpc.close_rpc()
        except Exception:
            LOGGER.exception("Failed to close ImagEMX2 RPC client")

    def ping(self):
        return {
            "ok": True,
            "host": self.host,
            "port": self.port,
            "target": "camera",
        }

    def acquire_n_frames(
        self,
        nframes: int,
        exp_info: dict[str, Any] | None = None,
        start_frame: int = 0,
        autosave: bool = False,
        broadcast: bool = True,
    ) -> np.ndarray:
        images = np.asarray(self._rpc.acquire_n_frames(nframes=nframes, start_frame=start_frame))

        merged_info = exp_info or {"task": 0, "run": 0, "rep": 0}
        task = int(merged_info.get("task", 0))
        run = int(merged_info.get("run", 0))
        rep = int(merged_info.get("rep", 0))

        for index, image in enumerate(images):
            if autosave:
                ImagEMX2Camera.save_tiff(image=image, image_dir=self.image_dir, run_no=run)
            if broadcast:
                info = {
                    "timestamp": time.time(),
                    "task": task,
                    "run": run,
                    "rep": rep,
                    "_imgindex": index,
                    "_imageresolution": [1, 1],
                    "_offset": [0, 0],
                }
                self.imstream.send(image, info)

        return images


def run_server(
    host: str,
    port: int,
    stream_name: str = "imagemx2",
    image_dir: str | None = None,
    timeout: float = 5.0,
    simulate: bool = False,
):
    camera = SimulatedImagEMX2Camera(image_dir=image_dir, timeout=timeout) if simulate else ImagEMX2Camera(image_dir=image_dir, timeout=timeout)

    LOGGER.info(
        "Starting ImagEM X2 RPC server host=%s port=%s stream=%s simulate=%s",
        host,
        port,
        stream_name,
        simulate,
    )
    simple_server_loop({"camera": camera}, host=host, port=int(port), description="ImagEM X2 RPC server")


def run(name: str):
    # Backward-compatible entrypoint. Name is ignored here.
    del name
    conf = ConfigReader.getConfiguration()
    server_conf = conf.get("Servers", {}).get("ImagEM X2 Camera", {})
    run_server(
        host=server_conf.get("host", "127.0.0.1"),
        port=int(server_conf.get("port", 3251)),
        stream_name=server_conf.get("stream_name", "imagemx2"),
        timeout=float(server_conf.get("timeout", 5.0)),
        simulate=bool(server_conf.get("simulate", False)),
    )


def main():
    parser = argparse.ArgumentParser(description="Run ImagEM X2 sipyco RPC server")
    parser.add_argument("--host", default="127.0.0.1", help="RPC bind host")
    parser.add_argument("--port", type=int, default=3251, help="RPC bind port")
    parser.add_argument("--stream-name", default="imagemx2", help="Image stream name")
    parser.add_argument("--image-dir", default=None, help="optional TIFF autosave directory")
    parser.add_argument("--timeout", type=float, default=5.0, help="frame wait timeout in seconds")
    parser.add_argument("--simulate", action="store_true", help="use simulated camera backend")
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_server(
        host=args.host,
        port=args.port,
        stream_name=args.stream_name,
        image_dir=args.image_dir,
        timeout=args.timeout,
        simulate=args.simulate,
    )


if __name__ == "__main__":
    main()
