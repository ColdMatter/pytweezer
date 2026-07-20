import argparse

import pylablib as pll
import pylablib.devices.Thorlabs as thorcam
from sipyco.pc_rpc import Client as RPCClient, simple_server_loop

from pytweezer.drivers.camera_base import (
    Camera,
    requires_camera,
    simulated_camera_for,
)
from pytweezer.servers.configreader import ConfigReader

from pytweezer.logging_utils import get_logger

THORLABS_DLL_PATH = "C:\\Program Files\\Thorlabs\\Scientific Imaging\\Scientific Camera Support\\Scientific Camera Interfaces\\SDK\\Python Toolkit\\dlls\\64_lib"
pll.par["devices/dlls/thorlabs_tlcam"] = THORLABS_DLL_PATH

IMAGE_DIRECTORY = (
    "C:/Users/CaFMOT/OneDrive - Imperial College London/caftweezers/ThorCamImages"
)

LOGGER = get_logger("thorcam")


class ThorLabsCamera(Camera):
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
            image_dir=image_dir or IMAGE_DIRECTORY,
            timeout=timeout,
            stream_name=stream_name,
        )

    # ------------------------------------------------------------------ #
    # Camera hooks
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        try:
            thorcamlist = thorcam.list_cameras_tlcam()
            camera = thorcam.ThorlabsTLCamera(serial=thorcamlist[0])
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
        self._backend.set_roi(x0, x0 + width - 1, y0, y0 + height - 1)

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


#: One-size-fits-all simulated camera, with the ThorLabs camera's controls
#: auto-stubbed so the simulated surface matches the real driver.
SimulatedThorLabsCamera = simulated_camera_for(ThorLabsCamera)


class ThorLabsCameraClient(RPCClient):
    """Experiment-side RPC client with direct access to camera methods."""

    def __init__(
        self,
        server_name: str = "ThorLabs Camera",
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = 5.0,
    ):

        conf = ConfigReader.getConfiguration()
        server_conf = conf.get("Devices", {}).get(server_name, {})

        self.host = host or server_conf.get("host", "127.0.0.1")
        self.port = int(port or server_conf.get("port", 3251))
        super().__init__(host=self.host, port=self.port, timeout=timeout)

    def ping(self):
        return {"ok": True, "host": self.host, "port": self.port, "target": "camera"}


def run_server(
    host: str,
    port: int,
    stream_name: str = "thorlabscamera",
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
        camera = ThorLabsCamera(
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
    parser = argparse.ArgumentParser(description="ThorLabs Camera sipyco RPC server launcher")
    parser.add_argument(
        "name",
        help="process-manager label or explicit server name",
        default=None
    )
    parser.add_argument(
        "--stream-name",
        default="thorlabscamera",
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

    conf = ConfigReader.getConfiguration()

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
        print("Error: RPC host and port must be specified either via command-line or configuration.")
        exit(1)

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
