"""CaF MOT fluorescence at chosen wavemeter-lock set points."""

import time
from typing import Any, ClassVar

import numpy as np

from pytweezer.drivers.shot_frames import settle_frame_counts
from pytweezer.experiment import Device, Number
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterNumber,
)
from pytweezer.logging_utils import get_logger

logger = get_logger(__name__)

#: The wavemeter-lock lasers :meth:`CaFFluorescence.measure` can set.
LASERS = ("v0", "BX", "v1", "v2", "v3", "TCool")


class CaFFluorescence(MotMasterExperiment):
    """Background-subtracted MOT fluorescence on the CaF ThorCam.

    ``prepare`` notes every laser's set point, recorded as the constant
    ``start_set_points`` (``{laser: THz}``), and takes a background shot with
    the MOT field at ``bg_field``. Each shot then runs the script's iterations
    at ``mot_field``; ``counts`` is the mean over its frames of each frame's
    background-subtracted sum, ``counts_error`` its standard error (NaN for a
    single frame). ``finish`` ramps every laser back to its starting set point,
    whatever happened.
    """

    #: Interval between frame-count polls while a shot's frames settle.
    frame_poll_s: ClassVar[float] = 0.05

    caf = MotMaster("CaF MotMaster", script="0. DDSFourSidebands5W_noDDS", iterations=3)
    camera = Device("CaF ThorCam")
    mot_field = MotMasterNumber(1.0, parameter="MOTCoilsCurrentValue")
    bg_field = Number(
        0.0, group="MOTMaster", tooltip="MOTCoilsCurrentValue for the background shot"
    )
    settle_s = Number(
        0.0,
        min=0,
        unit="s",
        group="Lasers",
        tooltip="Wait after moving any laser, before the next shot",
    )
    frame_settle_s = Number(
        0.3,
        min=0,
        unit="s",
        group="Camera",
        tooltip="How long the frame count must stay unchanged after a shot",
    )
    frame_timeout_s = Number(
        5.0, min=0, unit="s", group="Camera", tooltip="Longest wait for the frames"
    )

    def prepare(self) -> None:
        super().prepare()
        self.start_set_points = self.caf.laser_set_points_wml()
        self.record("start_set_points", self.start_set_points)
        background_frames = self.shot(**{"caf.MOTCoilsCurrentValue": self.bg_field})
        self.background = background_frames.mean(axis=0)

    def run_point(self) -> None:
        for name, value in self.measure().items():
            self.record(name, value)

    def finish(self) -> None:
        try:
            self.restore_lasers()
        finally:
            super().finish()

    def measure(
        self,
        *,
        v0: float | None = None,
        BX: float | None = None,
        v1: float | None = None,
        v2: float | None = None,
        v3: float | None = None,
        TCool: float | None = None,
    ) -> dict[str, float]:
        """Fluorescence counts with the given set points (THz); ``None`` leaves a laser alone."""
        self.set_lasers(v0=v0, BX=BX, v1=v1, v2=v2, v3=v3, TCool=TCool)
        frame_counts = (self.shot() - self.background).sum(axis=(1, 2))
        n_frames = len(frame_counts)
        return {
            "counts": float(np.mean(frame_counts)),
            "counts_error": float(np.std(frame_counts, ddof=1) / np.sqrt(n_frames))
            if n_frames > 1
            else float("nan"),
        }

    def set_lasers(self, **set_points: float | None) -> None:
        """Ramp each laser given a set point (THz) to it, then wait ``settle_s``.

        A ``None`` set point leaves that laser alone; if every one is ``None``
        there is no wait.
        """
        moved = False
        for laser, set_point in set_points.items():
            if set_point is None:
                continue
            self.caf.ramp_laser_set_point_wml(laser, float(set_point))
            moved = True
        if moved:
            time.sleep(self.settle_s)

    def restore_lasers(self) -> None:
        """Ramp every laser back to its set point from ``prepare``, trying them all.

        Raises:
            RuntimeError: Naming the lasers that could not be ramped back.
        """
        # Unset if prepare() failed before reading them, when nothing has moved yet.
        start_set_points = getattr(self, "start_set_points", {})
        errors: dict[str, Exception] = {}
        for laser, set_point in start_set_points.items():
            try:
                self.caf.ramp_laser_set_point_wml(laser, set_point)
            except Exception as error:
                logger.exception("Could not ramp %s back to %s THz", laser, set_point)
                errors[laser] = error
        if errors:
            stranded = {laser: start_set_points[laser] for laser in errors}
            raise RuntimeError(
                f"could not return lasers to their starting set points (THz) {stranded}"
            ) from next(iter(errors.values()))

    def shot(self, **overrides: Any) -> np.ndarray:
        """Run the script once and return the camera's frames from it, ``(n, H, W)``.

        Args:
            **overrides: Script parameters for this run, as for
                :meth:`run_sequences`.

        Raises:
            RuntimeError: If the camera caught no frames.
        """
        self.camera.arm()
        mark = self.camera.frames_acquired()
        self.run_sequences(**overrides)
        settle_frame_counts(
            {"camera": self.camera},
            self.frame_settle_s,
            self.frame_timeout_s,
            self.frame_poll_s,
        )
        frames = np.asarray(self.camera.read_frames_from(mark), dtype=float)
        if len(frames) == 0:
            raise RuntimeError(
                f"{type(self).camera.device_name!r} caught no frames from "
                f"{self.motmasters()['caf'].script!r}; is it armed on the "
                "MOTMaster's trigger?"
            )
        return frames
