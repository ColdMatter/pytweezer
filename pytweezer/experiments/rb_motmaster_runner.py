"""Run any Rb MOTMaster script and record every frame it triggers on the chosen cameras."""

from typing import Any

import numpy as np

from pytweezer.drivers.motmaster import script_names
from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.experiment import Bool, Choice, Device, Integer, Number
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.servers.device_client import get_device_config

RB_SCRIPTS = script_names(get_device_config("Rb MotMaster")["config_file"])


class RbMotMasterRunner(MotMasterExperiment):
    """Run a script from the Rb script folder once per point, recording each camera's frames.

    Each ticked camera is put in external exposure and live mode, so every frame
    goes out on its own image stream as it arrives, and is left that way when
    the task ends. The frames that arrive during a run are recorded per point
    as ``<camera>_images``, ``(frames, H, W)``; the frame count must be the same
    at every point. ROI, binning and gain are left as the camera has them.
    """

    script = Choice(RB_SCRIPTS, scannable=False, tooltip="Script in the Rb folder")
    rb = MotMaster("Rb MotMaster", script_from="script")

    hamcam = Device("Rb HamCam")
    orca = Device("ORCA")
    use_hamcam = Bool(True, group="Cameras", scannable=False, tooltip="Rb HamCam")
    use_orca = Bool(False, group="Cameras", scannable=False, tooltip="ORCA")

    buffer_frames = Integer(
        1000,
        min=1,
        group="Frames",
        scannable=False,
        tooltip="Ring buffer length of each camera's live mode",
    )
    frame_settle = Number(
        0.3,
        unit="s",
        min=0,
        group="Frames",
        tooltip="How long every frame count must stay unchanged after the run "
        "before its frames are read",
    )
    frame_timeout = Number(
        5.0,
        unit="s",
        min=0,
        group="Frames",
        tooltip="Longest wait for the frame counts to settle",
    )

    def cameras(self) -> dict[str, Any]:
        """The ticked cameras by attribute name."""
        ticked = {"hamcam": self.use_hamcam, "orca": self.use_orca}
        return {name: getattr(self, name) for name, use in ticked.items() if use}

    def prepare(self) -> None:
        super().prepare()
        for camera in self.cameras().values():
            # DCAM refuses trigger settings while acquiring.
            camera.stop_live()
            camera.set_trigger_source("ext")
            camera.set_external_exposure_mode()
            camera.start_live(self.buffer_frames)
        for name, camera in self.cameras().items():
            self.record(f"{name}_settings", camera.settings())
        self.frames_per_run: dict[str, int] = {}

    def run_point(self) -> None:
        cameras = self.cameras()
        marks = {name: camera.frames_acquired() for name, camera in cameras.items()}
        self.run_sequences()
        settle_frame_counts(cameras, self.frame_settle, self.frame_timeout, 0.05)
        for name, camera in cameras.items():
            if camera.frames_acquired() < marks[name]:
                raise RuntimeError(
                    f"{name}'s acquisition restarted during the run; its frames are lost"
                )
            frames = np.asarray(camera.read_frames_from(marks[name]))
            if not len(frames):
                raise RuntimeError(f"{name} took no frames during {self.script}")
            expected = self.frames_per_run.setdefault(name, len(frames))
            if len(frames) != expected:
                raise RuntimeError(
                    f"{name} took {len(frames)} frames during this run but {expected} "
                    "during the first; every run must trigger it the same number of times"
                )
            self.record(f"{name}_images", frames)
