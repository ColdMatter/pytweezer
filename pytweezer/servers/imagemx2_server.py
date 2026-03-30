import argparse
import time

import numpy as np

from pytweezer.drivers.imagemX2 import run_server
from pytweezer.servers import ImageClient
from pytweezer.servers.configreader import ConfigReader


class SyntheticImagEMX2Server:
    """Simulation backend that publishes synthetic frames on the same stream."""

    def __init__(self, stream_name: str = "imagemx2", frame_shape: tuple[int, int] = (256, 256)):
        self.stream_name = stream_name
        self.frame_shape = frame_shape
        self.imstream = ImageClient(stream_name)
        self.rng = np.random.default_rng()
        self.running = False

    def _generate_frame(self) -> np.ndarray:
        h, w = self.frame_shape
        image = self.rng.normal(loc=100.0, scale=8.0, size=(h, w)).astype(np.float32)

        # Add a few Gaussian-like synthetic tweezer spots.
        nspots = 12
        y, x = np.mgrid[0:h, 0:w]
        for _ in range(nspots):
            cy = self.rng.integers(20, h - 20)
            cx = self.rng.integers(20, w - 20)
            amp = self.rng.uniform(80.0, 250.0)
            sigma = self.rng.uniform(1.3, 2.5)
            image += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))

        return np.clip(image, 0.0, 65535.0).astype(np.uint16)

    def run(self):
        self.running = True
        index = 0
        while self.running:
            frame = self._generate_frame()
            info = {
                "timestamp": time.time(),
                "task": 0,
                "run": 0,
                "rep": 0,
                "_imgindex": index,
                "_imageresolution": [1, 1],
                "_offset": [0, 0],
                "simulated": True,
            }
            self.imstream.send(frame, info)
            index += 1
            time.sleep(0.2)


def _resolve_server_name(token: str, conf: dict) -> str:
    if isinstance(token, str) and token.startswith("Servers/"):
        return token.split("/", 1)[1]
    if isinstance(token, str) and token in conf.get("Servers", {}):
        return token
    return "ImagEM X2 Camera"


def main():
    parser = argparse.ArgumentParser(description="ImagEM X2 camera server launcher")
    parser.add_argument(
        "name",
        nargs="?",
        default="Servers/ImagEM X2 Camera",
        help="process-manager label or explicit server name",
    )
    parser.add_argument(
        "--stream-name",
        default=None,
        help="camera command/image stream name used by experiment-side clients",
    )
    parser.add_argument("--image-dir", default=None, help="optional TIFF autosave directory")
    parser.add_argument("--timeout", type=float, default=None, help="frame wait timeout in seconds")
    parser.add_argument("--simulate", action="store_true", help="force synthetic camera backend")
    args = parser.parse_args()

    conf = ConfigReader.getConfiguration()
    server_name = _resolve_server_name(args.name, conf)
    server_conf = conf.get("Servers", {}).get(server_name, {})

    stream_name = args.stream_name or server_conf.get("stream_name", "imagemx2")
    timeout = args.timeout if args.timeout is not None else float(server_conf.get("timeout", 5.0))
    simulate = bool(args.simulate or server_conf.get("simulate", False))

    if simulate:
        SyntheticImagEMX2Server(stream_name=stream_name).run()
    else:
        run_server(name=stream_name, image_dir=args.image_dir, timeout=timeout)


if __name__ == "__main__":
    main()
