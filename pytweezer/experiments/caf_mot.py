"""Fire the CaF MOT sequence and image the cloud at chosen wavemeter-lock set points."""

import time
from dataclasses import replace

import numpy as np

from pytweezer.experiment import Bool, Integer, Number
from pytweezer.experiment.camera_experiment import (
    CameraExperiment,
    CameraShots,
    ShotSettings,
)
from pytweezer.experiment.motmaster import MotMaster
from pytweezer.experiment.scan import Point, Scan
from pytweezer.logging_utils import get_logger

logger = get_logger(__name__)

#: The wavemeter-lock lasers :meth:`CaFMOT.measure` can set.
LASERS = ("v0", "BX", "v1", "v2", "v3", "TCool")


def frame_counts(frames: np.ndarray) -> dict[str, float]:
    """``counts``, the mean over ``frames`` of each frame's sum, and its standard error.

    The error is NaN for a single frame.
    """
    sums = frames.astype(float).sum(axis=(1, 2))
    return {
        "counts": float(np.mean(sums)),
        "counts_error": float(np.std(sums, ddof=1) / np.sqrt(len(sums)))
        if len(sums) > 1
        else float("nan"),
    }


class CaFMOT(CameraExperiment):
    """The CaF MOT on the CaF ThorCam, as the camera sees it: no background subtraction.

    ``prepare`` notes every laser's set point, recorded as the constant
    ``start_set_points`` (``{laser: THz}``). Each shot runs the script's
    ``iterations`` (an argument) at its own MOT field and reads one ``thorcam`` frame per
    iteration, with the trigger, binning and gain its CONFIG entry sets, which
    go out on the camera's own image stream. ``counts`` is the mean over its
    frames of each frame's sum, ``counts_error`` its standard error (NaN for a
    single frame). Each :meth:`measure` also records the Gaussian fit of the
    cloud in the mean frame as ``thorcam_*``
    (:meth:`~pytweezer.experiment.camera_experiment.CameraExperiment.record_cloud_fit`),
    from the same shot as ``counts``, so optimisers built on it record it too.
    ``finish`` ramps every laser :meth:`set_lasers` moved back to its starting
    set point, whatever happened; a laser it never moved is left alone.
    """

    cloud_species = "CaF"
    analyses = ("cloud_fit", "cloud_temperature")

    caf = MotMaster("CaF MotMaster", script="0. DDSFourSidebands5W_noDDS")
    thorcam = CameraShots("CaF ThorCam", trigger=None, locked=("frames",))
    iterations = Integer(
        1,
        min=1,
        group="MOTMaster",
        tooltip="Script iterations per shot, with one thorcam frame read for each",
    )
    save_mean_image = Bool(
        True,
        group="Camera",
        tooltip="record each point's mean frame, beside the fit to it",
    )
    settle_s = Number(
        0.0,
        min=0,
        unit="s",
        group="Lasers",
        tooltip="Wait after moving any laser, before the next shot",
    )

    def sequence_iterations(self, attribute: str) -> int:
        return self.iterations

    def shot_settings(self, name: str) -> ShotSettings:
        return replace(super().shot_settings(name), frames=self.iterations)

    def planned_points(self, scan: Scan) -> list[Point]:
        if any(axis.argument == "iterations" for axis in scan.axes):
            raise ValueError(
                "iterations changes the shape of the shot; set it for the whole task"
            )
        return super().planned_points(scan)

    def prepare(self) -> None:
        super().prepare()
        self._moved_lasers: set[str] = set()
        self.start_set_points = self.caf.laser_set_points_wml()
        self.record("start_set_points", self.start_set_points)

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
        """Fluorescence counts with the given set points (THz); ``None`` leaves a laser alone.

        Also records the shot's ``thorcam_*`` cloud fit, so it must run inside a point.
        """
        self.set_lasers(v0=v0, BX=BX, v1=v1, v2=v2, v3=v3, TCool=TCool)
        return self.analyse(self.take_shot()["thorcam"])

    def analyse(self, frames: np.ndarray) -> dict[str, float]:
        """Record the cloud fit of a shot's ``thorcam`` ``frames`` and return its counts."""
        self.record_cloud_fit("thorcam", frames)
        return frame_counts(frames)

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
