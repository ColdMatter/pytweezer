import argparse
import sys

import numpy as np
import pylablib.devices.DCAM as dcam
from sipyco.pc_rpc import Client as RPCClient
from sipyco.pc_rpc import simple_server_loop

from pytweezer.configuration.config import get_config
from pytweezer.drivers.camera_base import (
    Camera,
    requires_camera,
    simulated_camera_for,
)
from pytweezer.logging_utils import get_logger

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/HamCamImages"
)

LOGGER = get_logger("imagemx2")


class ImagEMX2Camera(Camera):
    """dcam shim: translates the generic :class:`Camera` interface to the
    Hamamatsu ImagEM X2's pylablib DCAM driver, plus the EM-gain/sensitivity
    controls specific to this camera."""

    image_prefix = "HamTweezer"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 20.0,
        stream_name: str | None = None,
    ):
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
            camera = dcam.DCAMCamera()
            camera.open()
        except Exception as exc:
            self._backend = None
            raise RuntimeError("Could not connect to ImagEM X2 camera") from exc
        self._backend = camera

    def _disconnect(self) -> None:
        if self._backend is not None:
            self._backend.close()

    @requires_camera
    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self._backend.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

    @requires_camera
    def set_trigger_source(self, source: str):
        self._backend.set_trigger_mode(source)

    @requires_camera
    def set_exposure_time(self, exposure: float):
        self._backend.set_exposure(exposure)

    @requires_camera
    def setup_acquisition(self, acq_mode: str, nframes: int):
        self._backend.setup_acquisition(acq_mode, int(nframes))

    @requires_camera
    def start_acquisition(self):
        self._backend.start_acquisition()

    @requires_camera
    def stop_acquisition(self):
        self._backend.stop_acquisition()

    @requires_camera
    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images

    # ------------------------------------------------------------------ #
    # ImagEM X2-specific controls (beyond the generic Camera interface)
    # ------------------------------------------------------------------ #

    @requires_camera
    def set_ccd_mode(self, mode: int):
        self._backend.set_attribute_value("ccd_mode", int(mode))

    @requires_camera
    def enable_em_gain(self, enable: bool = True):
        self.set_ccd_mode(2 if enable else 1)

    @requires_camera
    def set_direct_em_gain_mode(self, mode: int):
        self._backend.set_attribute_value("direct_em_gain_mode", int(mode))

    @requires_camera
    def enable_direct_em_gain(self, enable: bool = True):
        self.set_direct_em_gain_mode(2 if enable else 1)

    @requires_camera
    def set_sensitivity(self, sensitivity: int):
        self._backend.set_attribute_value("sensitivity", int(sensitivity))

    @requires_camera
    def set_external_exposure_mode(self):
        self.set_trigger_source("ext")
        self._backend.set_attribute_value("trigger_active", 2)
        self._backend.setup_ext_trigger(invert=True)

    @requires_camera
    def get_conversion_factor(self) -> tuple[float, float]:
        """The camera's photoelectron conversion ``(coefficient, offset)`` at its current settings."""
        return (
            float(self._backend.get_attribute_value("CONVERSION FACTOR COEFF")),
            float(self._backend.get_attribute_value("CONVERSION FACTOR OFFSET")),
        )

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


class ImagEMX2CameraClient(RPCClient):
    """Experiment-side RPC client with direct access to camera methods."""

    def __init__(
        self,
        server_name: str = "ImagEM X2 Camera",
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = 5.0,
    ):

        conf = get_config()
        server_conf = conf.get("Devices", {}).get(server_name, {})

        self.host = host or server_conf.get("host", "127.0.0.1")
        self.port = int(port or server_conf.get("port", 3251))
        super().__init__(host=self.host, port=self.port, timeout=timeout)

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
        camera = SimulatedImagEMX2Camera(
            stream_name=stream_name, image_dir=image_dir, timeout=timeout
        )
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
    parser = argparse.ArgumentParser(description="ImagEM X2 sipyco RPC server launcher")
    parser.add_argument(
        "name", help="process-manager label or explicit server name", default=None
    )
    parser.add_argument(
        "--stream-name",
        default="imagemx2",
        help="image stream name used by experiment-side clients",
    )
    parser.add_argument(
        "--image-dir", default=IMAGE_DIRECTORY, help="optional TIFF autosave directory"
    )
    parser.add_argument(
        "--timeout", type=float, default=5, help="frame wait timeout in seconds"
    )
    parser.add_argument("--host", default=None, help="override RPC bind host")
    parser.add_argument("--port", type=int, default=None, help="override RPC bind port")
    parser.add_argument(
        "--simulate", action="store_true", help="force synthetic camera backend"
    )
    args = parser.parse_args()

    conf = get_config()

    if args.name is not None:
        server_name = args.name
        server_conf = conf["Devices"][server_name]
        host = server_conf["host"]
        port = server_conf["port"]
        simulate = server_conf["simulate"]
        timeout = server_conf["timeout"]
        image_dir = server_conf["image_dir"]
        stream_name = server_conf["stream_name"]
    else:
        host = args.host
        port = args.port
        simulate = args.simulate
        timeout = args.timeout
        image_dir = args.image_dir
        stream_name = args.stream_name

    if host is None or port is None:
        print(
            "Error: RPC host and port must be specified either via command-line or configuration."
        )
        sys.exit(1)

    print(
        f"Starting ImagEM X2 Camera server with configuration:\n"
        f"  Host: {host}\n"
        f"  Port: {port}\n"
        f"  Stream Name: {stream_name}\n"
        f"  Image Directory: {image_dir}\n"
        f"  Timeout: {timeout} seconds\n"
        f"  Simulate: {simulate}"
    )

    run_server(
        host=host,
        port=port,
        stream_name=stream_name,
        image_dir=image_dir,
        timeout=timeout,
        simulate=simulate,
    )


if __name__ == "__main__":
    main()
