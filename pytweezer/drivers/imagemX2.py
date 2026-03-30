import argparse
import logging
import os
import time
import uuid
from typing import Any

import numpy as np
import pylablib.devices.DCAM as dcam
import tifffile as tiff

from pytweezer.servers import CommandClient, DataClient, ImageClient

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


class ImagEMX2Server:
    """Long-lived server process that owns one persistent ImagEM X2 connection."""

    def __init__(
        self,
        stream_name: str,
        image_dir: str | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.005,
    ):
        self.stream_name = stream_name
        self.poll_interval = poll_interval
        self.running = False

        self.camera = ImagEMX2Camera(image_dir=image_dir, timeout=timeout)
        self.imstream = ImageClient(stream_name)
        self.cmdstream = CommandClient(stream_name)
        self.cmdstream.subscribe(stream_name)

        # Tracks current experiment indices to annotate images consistently.
        self.indexq = DataClient(f"{stream_name}.index")
        self.indexq.subscribe(["Experiment.start"])
        self.last_exp_info = {"task": 0, "run": 0, "rep": 0}

    def run(self):
        self.running = True
        LOGGER.info("ImagEMX2 server '%s' started", self.stream_name)
        try:
            while self.running:
                self._update_experiment_indices()
                self._handle_pending_commands()
                time.sleep(self.poll_interval)
        finally:
            self._shutdown()

    def stop(self):
        self.running = False

    def _shutdown(self):
        try:
            self.camera.stop_acquisition()
        except Exception:
            pass
        self.camera.close()
        LOGGER.info("ImagEMX2 server '%s' stopped", self.stream_name)

    def _update_experiment_indices(self):
        while self.indexq.has_new_data():
            recvmsg = self.indexq.recv()
            if recvmsg is None:
                continue

            if len(recvmsg) == 2:
                message, msg_dict = recvmsg
            elif len(recvmsg) == 3:
                message, msg_dict, _array = recvmsg
            else:
                continue

            if message != "Experiment.start":
                continue

            self.last_exp_info = {
                "task": msg_dict.get("_task", 0),
                "run": msg_dict.get("_run", 0),
                "rep": msg_dict.get("_repetition", 0),
            }

    def _handle_pending_commands(self):
        while self.cmdstream.has_new_data():
            msg = self.cmdstream.recv()
            if msg is None:
                return

            if len(msg) == 2:
                channel, payload = msg
            elif len(msg) == 3:
                channel, payload, _ = msg
            else:
                continue

            command = self._parse_command(channel)
            if command is None:
                continue

            try:
                self._execute_command(command, payload if isinstance(payload, dict) else {})
            except Exception:
                LOGGER.exception("ImagEMX2 command '%s' failed", command)

    def _parse_command(self, channel: str) -> str | None:
        prefix = f"{self.stream_name} "
        if not channel.startswith(prefix):
            return None
        return channel[len(prefix) :].strip()

    def _execute_command(self, command: str, payload: dict[str, Any]):
        if command == "shutdown":
            self.stop()
            return

        if command == "start":
            self.camera.start_acquisition()
            return

        if command == "stop":
            self.camera.stop_acquisition()
            return

        if command == "setup":
            acq_mode = payload.get("acq_mode", "sequence")
            nframes = int(payload.get("nframes", 1))
            self.camera.setup_acquisition(acq_mode=acq_mode, nframes=nframes)
            return

        if command == "set_roi":
            self.camera.set_roi(
                x0=int(payload["x0"]),
                width=int(payload["width"]),
                y0=int(payload["y0"]),
                height=int(payload["height"]),
            )
            return

        if command == "set_exposure":
            self.camera.set_exposure_time(float(payload["exposure"]))
            return

        if command == "capture":
            self._capture_and_publish(payload)
            return

        LOGGER.warning("Unknown ImagEMX2 command: %s", command)

    def _capture_and_publish(self, payload: dict[str, Any]):
        nframes = int(payload.get("nframes", 1))
        start_frame = int(payload.get("start_frame", 0))
        autosave = bool(payload.get("autosave", False))
        request_id = payload.get("request_id", "")

        exp_info = payload.get("exp_info") or self.last_exp_info
        task = int(exp_info.get("task", self.last_exp_info.get("task", 0)))
        run = int(exp_info.get("run", self.last_exp_info.get("run", 0)))
        rep = int(exp_info.get("rep", self.last_exp_info.get("rep", 0)))

        images = self.camera.acquire_n_frames(nframes=nframes, start_frame=start_frame)

        for index, image in enumerate(images):
            timestamp = time.time()
            info = {
                "timestamp": timestamp,
                "task": task,
                "run": run,
                "rep": rep,
                "_imgindex": index,
                "request_id": request_id,
                "_imageresolution": [1, 1],
                "_offset": [0, 0],
            }
            self.imstream.send(image, info)
            if autosave:
                self.camera.save_tiff(image, image_dir=self.camera.image_dir, run_no=run)


class ImagEMX2CameraClient:
    """Experiment-side proxy that requests captures from a persistent camera server."""

    def __init__(
        self,
        stream_name: str = "imagemx2",
        request_timeout: float = 10.0,
    ):
        self.stream_name = stream_name
        self.request_timeout = request_timeout
        self.cmdstream = CommandClient(stream_name)
        self.imstream = ImageClient(f"{stream_name}.client", recvtimeout=200)
        self.imstream.subscribe(stream_name)

    def setup_acquisition(self, acq_mode: str, nframes: int):
        self.cmdstream.send("setup", data={"acq_mode": acq_mode, "nframes": int(nframes)})

    def start_acquisition(self):
        self.cmdstream.send("start", data={})

    def stop_acquisition(self):
        self.cmdstream.send("stop", data={})

    def set_roi(self, x0: int, width: int, y0: int, height: int):
        self.cmdstream.send(
            "set_roi",
            data={"x0": int(x0), "width": int(width), "y0": int(y0), "height": int(height)},
        )

    def set_exposure_time(self, exposure: float):
        self.cmdstream.send("set_exposure", data={"exposure": float(exposure)})

    def acquire_n_frames(
        self,
        nframes: int,
        exp_info: dict[str, Any] | None = None,
        start_frame: int = 0,
        autosave: bool = False,
    ) -> list[np.ndarray]:
        request_id = uuid.uuid4().hex
        self.cmdstream.send(
            "capture",
            data={
                "nframes": int(nframes),
                "start_frame": int(start_frame),
                "autosave": bool(autosave),
                "exp_info": exp_info,
                "request_id": request_id,
            },
        )

        deadline = time.monotonic() + self.request_timeout
        images: list[np.ndarray] = []

        while len(images) < int(nframes):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {nframes} frames from stream '{self.stream_name}'"
                )

            msg = self.imstream.recv()
            if msg is None:
                continue
            if len(msg) != 3:
                continue

            _channel, header, image = msg
            if not isinstance(header, dict):
                continue
            if header.get("request_id") != request_id:
                continue
            images.append(image)

        return images

    def close(self):
        return None


def run_server(name: str, image_dir: str | None = None, timeout: float = 5.0):
    server = ImagEMX2Server(stream_name=name, image_dir=image_dir, timeout=timeout)
    server.run()


def run(name: str):
    """Backward-compatible entrypoint used by legacy launchers."""
    run_server(name=name)


def main():
    parser = argparse.ArgumentParser(description="Run persistent ImagEM X2 camera server")
    parser.add_argument("name", nargs="?", default="imagemx2", help="stream/server name")
    parser.add_argument("--image-dir", default=None, help="optional output directory for autosaved images")
    parser.add_argument("--timeout", type=float, default=5.0, help="frame wait timeout in seconds")
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    run_server(name=args.name, image_dir=args.image_dir, timeout=args.timeout)


if __name__ == "__main__":
    main()
