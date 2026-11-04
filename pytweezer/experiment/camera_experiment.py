"""Experiments that read frames from declared cameras around MOTMaster runs.

::

    class PureRecipe(CameraExperiment):
        rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
        hamcam = CameraShots("Rb HamCam", frames=2, exposure=20e-3, roi=(0, 512, 0, 512))
        thorcam = CameraShots("Rb ThorCam", frames=1)
        orca = CameraShots("ORCA", frames=3, binning=2, gain=1)

Each :class:`CameraShots` is a device whose settings become form arguments
named ``<camera>.frames``, ``.exposure``, ``.binning``, ``.gain``,
``.roi_x0``, ``.roi_width``, ``.roi_y0`` and ``.roi_height``, defaulting to
the declaration (group "Camera: <camera>"). A setting declared ``None`` is
never sent to the camera and has no argument; one named in ``locked`` keeps
its declared value and has no argument either. The ROI is in unbinned sensor
pixels, so a camera binning ``b`` returns frames of
``(roi_height // b, roi_width // b)``. ``gain`` is in the camera's own unit:
dB on the ThorCams, the contrast-gain index on the ORCAs, and the sensitivity
on the HamCam, which takes effect only once EM gain is enabled.

Every camera runs in snap mode: it is set up for ``frames`` frames, started
before the sequences and read and stopped after them. A camera that was live
is taken out of live mode for the task and put back in :meth:`finish`.

``exposure``, ``gain``, ``roi_x0`` and ``roi_y0`` can be scanned and are
re-applied before the shot of a point that changes them; ``frames``,
``binning``, ``roi_width`` and ``roi_height`` change the shape of the recorded
``<camera>_images``, so scanning them is refused.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from pytweezer.analysis.cloud_fit import SPECIES_MASS_AMU, fit_frames
from pytweezer.experiment.arguments import Argument, Device, Integer, Number, collect
from pytweezer.experiment.motmaster import MotMasterExperiment
from pytweezer.experiment.scan import Point, Scan
from pytweezer.logging_utils import get_logger
from pytweezer.servers.device_client import get_device_config

logger = get_logger(__name__)

SETTINGS = ("frames", "exposure", "binning", "gain", "roi")

_ROI_FIELDS = ("roi_x0", "roi_width", "roi_y0", "roi_height")

#: Settings whose change alters the shape of a camera's frames.
_SHAPE_SETTINGS = ("frames", "binning", "roi_width", "roi_height")

_GAIN_TOOLTIP = (
    "In the camera's own unit: dB (ThorCams), contrast gain (ORCAs), "
    "sensitivity (HamCam, once EM gain is enabled)"
)


@dataclass(frozen=True)
class ShotSettings:
    """The settings a camera takes its frames with."""

    frames: int
    exposure: float | None
    binning: int | None
    gain: float | None
    roi: tuple[int, int, int, int] | None


class CameraShots(Device):
    """A camera and the frames it takes on each shot of a :class:`CameraExperiment`.

    Args:
        device_name: The camera's ``CONFIG["Devices"]`` name.
        frames: Frames per shot.
        exposure: Exposure time in seconds; ``None`` leaves it to the camera,
            as for a camera exposed by its trigger pulse.
        binning: Pixels per side binned together; ``None`` leaves it.
        gain: Gain in the camera's own unit; ``None`` leaves it.
        roi: ``(x0, width, y0, height)`` in unbinned sensor pixels; ``None``
            leaves it.
        trigger: Trigger source for ``set_trigger_source``; ``None`` keeps the
            one the camera was configured with.
        locked: Settings kept at their declared value and left out of the form.
        timeout: RPC timeout for the camera's client.

    Raises:
        ValueError: If a setting is out of range or ``locked`` names an
            unknown setting.
    """

    def __init__(
        self,
        device_name: str,
        *,
        frames: int = 1,
        exposure: float | None = None,
        binning: int | None = None,
        gain: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        trigger: str | None = "ext",
        locked: Iterable[str] = (),
        timeout: float | None = None,
    ) -> None:
        super().__init__(device_name, timeout=timeout)
        if int(frames) != frames or frames < 1:
            raise ValueError(
                f"{device_name!r}: frames must be at least 1, not {frames!r}"
            )
        if exposure is not None and exposure <= 0:
            raise ValueError(f"{device_name!r}: exposure must be positive")
        if binning is not None and (int(binning) != binning or binning < 1):
            raise ValueError(f"{device_name!r}: binning must be at least 1")
        if gain is not None and gain < 0:
            raise ValueError(f"{device_name!r}: gain must not be negative")
        if roi is not None and (
            len(roi) != 4 or any(int(value) != value for value in roi)
        ):
            raise ValueError(
                f"{device_name!r}: roi is (x0, width, y0, height) in whole pixels, "
                f"not {roi!r}"
            )
        locked = tuple(locked)
        if unknown := sorted(set(locked) - set(SETTINGS)):
            raise ValueError(
                f"{device_name!r}: cannot lock {unknown}; the settings are "
                f"{list(SETTINGS)}"
            )
        self.frames = int(frames)
        self.exposure = exposure
        self.binning = None if binning is None else int(binning)
        self.gain = gain
        self.roi = None if roi is None else tuple(int(value) for value in roi)
        self.trigger = trigger
        self.locked = locked

    def declared(self) -> ShotSettings:
        return ShotSettings(
            self.frames, self.exposure, self.binning, self.gain, self.roi
        )

    def arguments(self, name: str) -> dict[str, Argument]:
        """The form arguments of camera attribute ``name``, keyed by dotted name."""
        group = f"Camera: {name}"
        arguments: dict[str, Argument] = {
            "frames": Integer(self.frames, min=1, group=group),
        }
        if self.exposure is not None:
            arguments["exposure"] = Number(
                self.exposure, unit="ms", scale=1e-3, min=0, group=group
            )
        if self.binning is not None:
            arguments["binning"] = Integer(
                self.binning, min=1, group=group, tooltip="Pixels per side"
            )
        if self.gain is not None:
            arguments["gain"] = Number(
                self.gain, min=0, group=group, tooltip=_GAIN_TOOLTIP
            )
        if self.roi is not None:
            for field, value in zip(_ROI_FIELDS, self.roi, strict=True):
                arguments[field] = Integer(
                    value,
                    min=1 if field in ("roi_width", "roi_height") else 0,
                    unit="px",
                    group=group,
                    tooltip="Unbinned sensor pixels",
                )
            if "roi" in self.locked:
                arguments = {
                    key: value
                    for key, value in arguments.items()
                    if key not in _ROI_FIELDS
                }
        arguments = {
            f"{name}.{setting}": argument
            for setting, argument in arguments.items()
            if setting not in self.locked
        }
        for dotted, argument in arguments.items():
            argument.name = dotted
        return arguments

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "camera": {
                "frames": self.frames,
                "exposure": self.exposure,
                "binning": self.binning,
                "gain": self.gain,
                "roi": list(self.roi) if self.roi is not None else None,
                "trigger": self.trigger,
                "locked": list(self.locked),
            }
        }


def _all_setting_names(name: str) -> list[str]:
    return [f"{name}.{setting}" for setting in (*SETTINGS[:-1], *_ROI_FIELDS)]


class CameraExperiment(MotMasterExperiment):
    """A MOTMaster experiment that reads frames from its declared cameras on each shot.

    :meth:`prepare` applies each camera's settings, in this order: binning,
    ROI (``set_roi`` keeps the binning, while ``set_binning`` can reset the
    ROI), exposure, trigger source, :meth:`configure_camera`, gain, then
    ``setup_acquisition(frames, mode="snap")``. The origin each camera's ROI
    took is in :attr:`roi_origins`. The default
    :meth:`run_point` takes one shot and records each camera's frames as
    ``<camera>_images``, ``(frames, H, W)`` per point.

    :meth:`record_cloud_fit` fits the mean image of a camera's frames and
    records the cloud's parameters; ``cloud_species`` ("CaF" or "Rb") makes
    :meth:`prepare` record the mass as the constant ``cloud_mass_amu``.
    """

    cloud_species: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        inherited = collect(cls, Argument)
        for name, camera in vars(cls).copy().items():
            if not isinstance(camera, CameraShots):
                continue
            generated = camera.arguments(name)
            # setattr skips __set_name__; arguments() has set each name already.
            for dotted, argument in generated.items():
                setattr(cls, dotted, argument)
            for dotted in _all_setting_names(name):
                if dotted not in generated and dotted in inherited:
                    # collect() drops a name a subclass sets to anything else.
                    setattr(cls, dotted, None)
        super().__init_subclass__(**kwargs)

    @classmethod
    def camera_shots(cls) -> dict[str, CameraShots]:
        return collect(cls, CameraShots)

    @property
    def cameras(self) -> dict[str, Any]:
        """``{camera attribute: device client}``."""
        return {name: getattr(self, name) for name in self.camera_shots()}

    def shot_settings(self, name: str) -> ShotSettings:
        """Camera ``name``'s settings for the current point."""
        declared = self.camera_shots()[name]

        def value(setting: str, default: Any) -> Any:
            dotted = f"{name}.{setting}"
            return getattr(self, dotted) if dotted in self.arguments() else default

        roi = declared.roi
        if roi is not None:
            roi = tuple(
                value(field, default)
                for field, default in zip(_ROI_FIELDS, roi, strict=True)
            )
        return ShotSettings(
            frames=value("frames", declared.frames),
            exposure=value("exposure", declared.exposure),
            binning=value("binning", declared.binning),
            gain=value("gain", declared.gain),
            roi=roi,
        )

    def configure_camera(self, name: str, camera: Any) -> None:
        """Set driver-specific options of camera ``name``; called before its gain is set."""

    def apply_camera_settings(self, name: str) -> None:
        """Send stopped camera ``name`` its settings for the current point and set it up."""
        declared = self.camera_shots()[name]
        camera = self.cameras[name]
        settings = self.shot_settings(name)
        if settings.binning is not None:
            camera.set_binning(settings.binning)
        if settings.roi is not None:
            self.roi_origins[name] = tuple(camera.set_roi(*settings.roi))
        if settings.exposure is not None:
            camera.set_exposure_time(settings.exposure)
        if declared.trigger is not None:
            camera.set_trigger_source(declared.trigger)
        self.configure_camera(name, camera)
        if settings.gain is not None:
            camera.set_gain(settings.gain)
        camera.setup_acquisition(settings.frames, mode="snap")
        self._applied[name] = settings

    def planned_points(self, scan: Scan) -> list[Point]:
        for axis in scan.axes:
            camera, _, setting = axis.argument.partition(".")
            if camera in self.camera_shots() and setting in _SHAPE_SETTINGS:
                raise ValueError(
                    f"{axis.argument} changes the shape of {camera}_images; set it "
                    "for the whole task"
                )
        return super().planned_points(scan)

    def prepare(self) -> None:
        super().prepare()
        self.roi_origins: dict[str, tuple[int, int] | None] = dict.fromkeys(
            self.camera_shots()
        )
        self._applied: dict[str, ShotSettings] = {}
        self._was_live: list[str] = []
        for name, camera in self.cameras.items():
            if camera.is_live():
                self._was_live.append(name)
            camera.stop_live()
            self.apply_camera_settings(name)
        if self.cloud_species is not None:
            self.record(
                "cloud_mass_amu", SPECIES_MASS_AMU[self.cloud_species], unit="u"
            )

    def finish(self) -> None:
        try:
            self._restore_live()
        finally:
            super().finish()

    def _restore_live(self) -> None:
        errors = []
        # Unset if prepare() failed first.
        for name in getattr(self, "_was_live", []):
            try:
                self.cameras[name].start_live()
            except Exception as error:
                logger.exception("Could not put camera %s back in live mode", name)
                errors.append(error)
        if errors:
            raise errors[0]

    def take_shot(
        self, *, background: bool = False, **overrides: Any
    ) -> dict[str, np.ndarray]:
        """Run the sequences once and return every camera's frames, ``{name: (frames, H, W)}``.

        Args:
            background: Mark the broadcast frames as background, which the live
                shot-background analysis keeps for subtraction.
            **overrides: Script parameters for this run, as for
                :meth:`run_sequences`.
        """
        names = list(self.camera_shots())
        for name in names:
            if self.shot_settings(name) != self._applied[name]:
                self.apply_camera_settings(name)
        header: dict[str, Any] = {"rid": self.rid}
        if background:
            header["background"] = True
        frames: dict[str, np.ndarray] = {}
        failed = True
        try:
            for name in names:
                self.cameras[name].start_acquisition()
            self.run_sequences(**overrides)
            for name in names:
                frames[name] = np.asarray(
                    self.cameras[name].acquire_n_frames(
                        self._applied[name].frames,
                        broadcast=True,
                        header=header,
                    )
                )
            failed = False
        finally:
            self._stop_cameras(names, raise_errors=not failed)
        return frames

    def _stop_cameras(self, names: list[str], *, raise_errors: bool) -> None:
        errors = []
        for name in names:
            try:
                self.cameras[name].stop_acquisition()
            except Exception as error:
                logger.exception("Could not stop camera %s", name)
                errors.append(error)
        if errors and raise_errors:
            raise errors[0]

    def record_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Record each ``{camera: stack}`` entry as ``<camera>_images``."""
        for name, stack in frames.items():
            self.record(f"{name}_images", stack)

    def record_cloud_fit(
        self, name: str, frames: np.ndarray, background: np.ndarray | None = None
    ) -> None:
        """Fit the mean of camera ``name``'s ``frames`` and record the cloud.

        Records ``<name>_<field>`` for every field of
        :data:`~pytweezer.analysis.cloud_fit.FIELDS` and ``<name>_<field>_error``
        for each of :data:`~pytweezer.analysis.cloud_fit.ERROR_FIELDS`, in metres
        when the camera's CONFIG entry gives ``object_pixel_m`` and pixels
        otherwise; fields the fit could not find, or all of them for a frame
        with no cloud in it, are NaN. With a true ``save_mean_image`` argument
        it also records ``<name>_mean_image``.

        Args:
            name: The camera's attribute name.
            frames: ``(n, H, W)`` stack from :meth:`take_shot`.
            background: Frame subtracted from the mean, or ``None``.
        """
        resolution, origin = self._cloud_geometry(name)
        values, mean_image = fit_frames(frames, background, resolution, origin)
        length = "px" if resolution is None else "m"
        for field, value in values.items():
            quantity = field.removesuffix("_error")
            unit = {"angle": "rad", "amplitude": "counts", "offset": "counts"}.get(
                quantity, "counts" if quantity == "integral" else length
            )
            self.record(f"{name}_{field}", value, unit=unit)
        if getattr(self, "save_mean_image", False):
            self.record(f"{name}_mean_image", mean_image, unit="counts")

    def _cloud_geometry(
        self, name: str
    ) -> tuple[tuple[float, float] | None, tuple[float, float]]:
        """``(resolution, origin)`` of camera ``name``'s frames for :func:`fit_frames`.

        Binning and ROI origin are read from the camera, which may have been set
        outside this experiment.
        """
        conf = get_device_config(self.camera_shots()[name].device_name)
        binning, (x0, y0) = self.cameras[name].frame_geometry()
        pixel_m = conf.get("object_pixel_m")
        resolution = None if pixel_m is None else (pixel_m * binning,) * 2
        return resolution, (x0 / binning, y0 / binning)

    def run_point(self) -> None:
        self.record_frames(self.take_shot())
