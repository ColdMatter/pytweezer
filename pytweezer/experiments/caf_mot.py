"""Fire the CaF MOT sequence and image the cloud at chosen wavemeter-lock set points."""

from __future__ import annotations

import time

import numpy as np

from pytweezer.analysis.cloud_fit import SPECIES_MASS_AMU, fit_frames, frame_counts
from pytweezer.drivers.orca import ORCACamera
from pytweezer.experiment import Bool, Device, Integer, Number
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_client import get_device_config

logger = get_logger(__name__)

#: The wavemeter-lock lasers :meth:`CaFMOT.measure` can set.
LASERS = ("v0", "BX", "v1", "v2", "v3", "TCool")


class CaFMOT(MotMasterExperiment):
    """The CaF MOT on the CaFMOT Cam (ORCA-Fusion), as the camera sees it: no background subtraction.

    ``prepare`` notes every laser's set point, recorded as the constant
    ``start_set_points`` (``{laser: THz}``), and puts the camera on its
    external trigger. Each shot runs the script's ``iterations`` and reads one
    frame per iteration. ``counts`` is the mean
    over the shot's frames of each frame's sum, ``counts_error`` its standard
    error (NaN for a single frame). Each :meth:`measure` also records the
    Gaussian fit of the cloud in the mean frame as ``cafmot_*``, so
    optimisers built on it record it too. ``finish`` ramps every laser
    :meth:`set_lasers` moved back to its starting set point, whatever
    happened; a laser it never moved is left alone.
    """

    analyses = ("cloud_fit", "cloud_temperature")

    caf = MotMaster("CaF MotMaster", script="0. MOT")
    cafmot_cam: ORCACamera = Device("CaFMOT Cam")

    iterations = Integer(
        1,
        min=1,
        group="MOTMaster",
        scannable=False,
        tooltip="Script iterations per shot, with one CaFMOT Cam frame read for each",
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

    #: Subtracted from every frame before it is summed or fitted.
    background: np.ndarray | None = None

    def prepare(self) -> None:
        super().prepare()
        self.caf.set_iterations(self.iterations)
        self._moved_lasers: set[str] = set()
        self.start_set_points = self.caf.laser_set_points_wml()
        self.record("start_set_points", self.start_set_points)
        self.record("cloud_mass_amu", SPECIES_MASS_AMU["CaF"], unit="u")
        self.cafmot_cam.stop_live()
        self.cafmot_cam.set_trigger_source("ext")
        self.cafmot_cam.start_live()

    def run_point(self) -> None:
        for name, value in self.measure().items():
            self.record(name, value)

    def finish(self) -> None:
        try:
            self.restore_lasers()
        finally:
            try:
                self.cafmot_cam.start_live()  # does nothing unless a failure left it stopped
                self.cafmot_cam.end_shots()
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

        Also records the shot's ``cafmot_*`` cloud fit, so it must run inside a point.
        """
        self.set_lasers(v0=v0, BX=BX, v1=v1, v2=v2, v3=v3, TCool=TCool)
        frames = self.take_shot()
        self.record_cloud_fit(frames)
        return frame_counts(frames, self.background)

    def take_shot(self, **overrides: float) -> np.ndarray:
        """Run the script once with ``overrides`` and return its ``(iterations, H, W)`` frames."""
        mark = self.cafmot_cam.start_shot({"rid": self.rid})
        self.run_motmaster_sequence(**overrides)
        return np.asarray(self.cafmot_cam.wait_frames(mark, self.iterations))

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

    def record_cloud_fit(self, frames: np.ndarray) -> None:
        """Fit the mean of ``frames`` less :attr:`background` and record the cloud as ``cafmot_*``."""
        pixel_m = get_device_config("CaFMOT Cam").get("object_pixel_m")
        # Read from the camera, which may have been set up outside this experiment.
        binning, (x0, y0) = self.cafmot_cam.frame_geometry()
        resolution = None if pixel_m is None else (pixel_m * binning,) * 2
        values, mean_image = fit_frames(
            frames, self.background, resolution, (x0 / binning, y0 / binning)
        )
        length = "px" if resolution is None else "m"
        for field, value in values.items():
            quantity = field.removesuffix("_error")
            unit = {"angle": "rad", "amplitude": "counts", "offset": "counts"}.get(
                quantity, "counts" if quantity == "integral" else length
            )
            self.record(f"cafmot_{field}", value, unit=unit)
        if self.save_mean_image:
            self.record("cafmot_mean_image", mean_image, unit="counts")
