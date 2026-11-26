"""Run any script of one MOTMaster and record every frame it triggers on the chosen cameras.

A runner names its MOTMaster with ``script_from=``, its cameras in
:attr:`MotMasterRunner.CAMERAS` and a ``use_<prefix>`` tick box for each::

    class RbMotMasterRunner(MotMasterRunner):
        CAMERAS = {"hamcam": "Rb HamCam", "orca": "ORCA"}

        script = MotMasterScript("RbTweezerBasic2026_2")
        rb = MotMaster("Rb MotMaster", script_from="script")
        use_hamcam = Bool(True, group="Cameras", scannable=False)
        use_orca = Bool(False, group="Cameras", scannable=False)
"""

from typing import Any, ClassVar

import numpy as np

from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.experiment.arguments import Bool, Device, Integer, Number
from pytweezer.experiment.motmaster import MotMasterExperiment


class MotMasterRunner(MotMasterExperiment):
    """Run the chosen script once per point, recording each ticked camera's frames.

    Each ticked camera is put in external exposure and live mode, so every frame
    goes out on its own image stream as it arrives, tagged with the task's
    ``rid`` and its ``index`` in the run, and is left that way when the task
    ends. The frames that arrive during a run are recorded per point as
    ``<prefix>_images``, ``(frames, H, W)``; the frame count must be the same
    at every point. ROI, binning and gain are left as the camera has them.
    """

    #: Device name of each camera, by the prefix of its tick box and results.
    CAMERAS: ClassVar[dict[str, str]] = {}

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

    def prepare(self) -> None:
        super().prepare()
        self.cameras = {
            prefix: self.device(name)
            for prefix, name in self.CAMERAS.items()
            if getattr(self, f"use_{prefix}")
        }
        for camera in self.cameras.values():
            # DCAM refuses trigger settings while acquiring.
            camera.stop_live()
            camera.set_trigger_source("ext")
            camera.set_external_exposure_mode()
            camera.start_live(self.buffer_frames)
        for prefix, camera in self.cameras.items():
            self.record(f"{prefix}_settings", camera.settings())
        self.frames_per_run: dict[str, int] = {}

    def run_point(self) -> None:
        marks = {
            prefix: camera.start_shot({"rid": self.rid})
            for prefix, camera in self.cameras.items()
        }
        self.run_motmaster_sequence()
        settle_frame_counts(self.cameras, self.frame_settle, self.frame_timeout, 0.05)
        for prefix, camera in self.cameras.items():
            self.record(
                f"{prefix}_images", self.read_run_frames(prefix, camera, marks[prefix])
            )

    def finish(self) -> None:
        try:
            for camera in getattr(self, "cameras", {}).values():
                camera.start_live(self.buffer_frames)  # in case a failure stopped it
                camera.end_shots()
        finally:
            super().finish()

    def read_run_frames(self, prefix: str, camera, mark: int) -> np.ndarray:
        """``camera``'s frames since ``mark``, checked to be the same count as at the first point."""
        if camera.frames_acquired() < mark:
            raise RuntimeError(
                f"{prefix}'s acquisition restarted during the run; its frames are lost"
            )
        frames = np.asarray(camera.read_frames_from(mark))
        if not len(frames):
            raise RuntimeError(f"{prefix} took no frames during {self.script_name()}")
        expected = self.frames_per_run.setdefault(prefix, len(frames))
        if len(frames) != expected:
            raise RuntimeError(
                f"{prefix} took {len(frames)} frames during this run but {expected} "
                "during the first; every run must trigger it the same number of times"
            )
        return frames

    @classmethod
    def devices(cls) -> dict[str, Device]:
        # Reservations are fixed per class, so every camera is reserved; only
        # the ticked ones are connected to.
        return super().devices() | {
            prefix: Device(name) for prefix, name in cls.CAMERAS.items()
        }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        arguments = cls.arguments()
        for prefix in cls.CAMERAS:
            if not isinstance(arguments.get(f"use_{prefix}"), Bool):
                raise TypeError(
                    f"{cls.__name__}: camera {prefix!r} needs a Bool argument "
                    f"use_{prefix}"
                )
