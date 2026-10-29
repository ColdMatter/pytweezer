import argparse
import sys

import pylablib as pll
import pylablib.devices.Thorlabs as thorcam
from sipyco.pc_rpc import Client as RPCClient
from sipyco.pc_rpc import simple_server_loop

from pytweezer.configuration.config import get_config
from pytweezer.drivers.camera_base import (
    Camera,
    requires_camera,
    simulated_camera_for,
)
from pytweezer.logging_utils import get_logger

THORLABS_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"
pll.par["devices/dlls/thorlabs_tlcam"] = THORLABS_DLL_PATH

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/ThorCamImages"
)

LOGGER = get_logger("thorcam")


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

    @requires_camera
    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self._backend.set_roi(x0, x0 + width, y0, y0 + height)

    @requires_camera
    def set_trigger_source(self, source: str):
        self._backend.set_trigger_mode(source)

    @requires_camera
    def set_exposure_time(self, exposure: float):
        self._backend.set_exposure(exposure)

    @requires_camera
    def setup_acquisition(self, nframes: int):
        self._backend.setup_acquisition(int(nframes))

    @requires_camera
    def arm(self, buffer_frames: int = 1000) -> None:
        with self._lock:
            if self._acquisition_in_progress():
                return
            self.setup_acquisition(buffer_frames)
            self.start_acquisition()

    @requires_camera
    def start_acquisition(self):
        self._backend.start_acquisition()

    @requires_camera
    def stop_acquisition(self):
        self._backend.stop_acquisition()

    @requires_camera
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

    @requires_camera
    def _read_frames(self, nframes: int, start_frame: int):
        self._backend.wait_for_frame(nframes=int(nframes), timeout=self.timeout)
        images, _infos = self._backend.read_multiple_images(
            (int(start_frame), int(start_frame) + int(nframes)), return_info=True
        )
        return images


#: One-size-fits-all simulated camera, with the ThorLabs camera's controls
#: auto-stubbed so the simulated surface matches the real driver.
SimulatedThorLabsCamera = simulated_camera_for(ThorCam)


class ThorCamClient(RPCClient):
    """Experiment-side RPC client with direct access to camera methods."""

    def __init__(
        self,
        server_name: str = "ThorLabs Camera",
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
    stream_name: str = "thorcam",
    image_dir: str | None = None,
    timeout: float = 5.0,
    simulate: bool = False,
):
    LOGGER.info(
        "Starting ThorLabs Camera RPC server host=%s port=%s stream=%s simulate=%s",
        host,
        port,
        stream_name,
        simulate,
    )
    if simulate:
        LOGGER.warning("Running ThorLabs Camera server in SIMULATION MODE")
        camera = SimulatedThorLabsCamera(
            stream_name=stream_name, image_dir=image_dir, timeout=timeout
        )
    else:
        camera = ThorCam(
            stream_name=stream_name,
            image_dir=image_dir,
            timeout=timeout,
        )

    simple_server_loop(
        {"camera": camera},
        host=host,
        port=int(port),
        description="ThorLabs Camera RPC server",
    )


def main():
    parser = argparse.ArgumentParser(
        description="ThorLabs Camera sipyco RPC server launcher"
    )
    parser.add_argument(
        "name", help="process-manager label or explicit server name", default=None
    )
    parser.add_argument(
        "--stream-name",
        default="thorcam",
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
        f"Starting ThorLabs Camera server with configuration:\n"
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
