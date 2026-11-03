"""CaF MOT fluorescence at chosen wavemeter-lock set points."""

import time

import numpy as np

from pytweezer.experiment import Number
from pytweezer.experiment.camera_experiment import CameraExperiment, CameraShots
from pytweezer.experiment.motmaster import MotMaster, MotMasterNumber
from pytweezer.logging_utils import get_logger

logger = get_logger(__name__)

#: The wavemeter-lock lasers :meth:`CaFFluorescence.measure` can set.
LASERS = ("v0", "BX", "v1", "v2", "v3", "TCool")


class CaFFluorescence(CameraExperiment):
    """Background-subtracted MOT fluorescence on the CaF ThorCam.

    ``prepare`` notes every laser's set point, recorded as the constant
    ``start_set_points`` (``{laser: THz}``), and takes a background shot with
    the MOT field at ``bg_field``. Each shot then runs the script's iterations
    at ``mot_field`` and reads the frames ``thorcam`` caught, streaming with
    the trigger, binning and gain its CONFIG entry sets; ``counts`` is the mean over its frames of each frame's
    background-subtracted sum, ``counts_error`` its standard error (NaN for a
    single frame). ``finish`` ramps every laser :meth:`set_lasers` moved back
    to its starting set point, whatever happened; a laser it never moved is
    left alone.
    """

    caf = MotMaster("CaF MotMaster", script="0. DDSFourSidebands5W_noDDS", iterations=3)
    thorcam = CameraShots(
        "CaF ThorCam", frames=3, stream=True, trigger=None, locked=("frames",)
    )
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

    def prepare(self) -> None:
        super().prepare()
        self._moved_lasers: set[str] = set()
        self.start_set_points = self.caf.laser_set_points_wml()
        self.record("start_set_points", self.start_set_points)
        background_frames = self.take_shot(
            **{"caf.MOTCoilsCurrentValue": self.bg_field}
        )["thorcam"]
        self.background = background_frames.astype(float).mean(axis=0)

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
        frames = self.take_shot()["thorcam"].astype(float)
        frame_counts = (frames - self.background).sum(axis=(1, 2))
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
            self._moved_lasers.add(laser)
            self.caf.ramp_laser_set_point_wml(laser, float(set_point))
            moved = True
        if moved:
            time.sleep(self.settle_s)

    def restore_lasers(self) -> None:
        """Ramp each laser :meth:`set_lasers` moved back to its set point from ``prepare``.

        Every such laser is tried even if one fails.

        Raises:
            RuntimeError: Naming the lasers that could not be ramped back.
        """
        # Unset if prepare() failed before setting them, when nothing has moved yet.
        moved = getattr(self, "_moved_lasers", set())
        start_set_points = {
            laser: set_point
            for laser, set_point in getattr(self, "start_set_points", {}).items()
            if laser in moved
        }
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
