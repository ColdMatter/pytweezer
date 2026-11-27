"""Generic camera abstraction shared by every real camera driver.

Rather than each camera module hand-rolling its own ROI/trigger/acquisition
logic and its own simulated twin, this module defines:

* :class:`Camera` -- an abstract base that owns everything *generic* to a
  camera server (broadcasting frames onto an image stream, TIFF autosave, the
  autosave/broadcast acquisition loop, connection lifecycle) and declares the
  small set of *hardware-specific* hooks a concrete driver must implement
  (connect/disconnect, ROI, trigger, exposure, acquisition start/stop, and the
  raw frame read).

* :class:`SimulatedCamera` -- a single, one-size-fits-all synthetic backend
  that implements those hooks with generated frames, usable in place of *any*
  real camera in simulation mode.

* :class:`Control` -- a declared camera setting (name, unit, range, choices),
  listed in a driver's :attr:`Camera.CONTROLS` so a GUI can build itself from
  the driver instead of hard-coding each camera.

* :func:`simulated_camera_for` -- adapts :class:`SimulatedCamera` to a specific
  real driver so that driver's *extra* methods (e.g. the ImagEM's EM-gain
  setters) are auto-stubbed via :func:`pytweezer.servers.simulated_device.simulate`,
  keeping the simulated surface interface-complete without drift.

A concrete driver ("shim") is therefore just the thin translation layer between
its vendor SDK (dcam, Spinnaker, ...) and the abstract hooks below.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np
import tifffile as tiff

from pytweezer.logging_utils import get_logger
from pytweezer.servers import ImageClient
from pytweezer.servers.simulated_device import simulate

LOGGER = get_logger("camera")


@dataclass(frozen=True)
class Control:
    """One camera setting a driver declares in :attr:`Camera.CONTROLS`.

    Drivers keep and return values in SI units; ``unit`` and ``scale`` say how to
    display them.

    Attributes:
        name: Key in :meth:`Camera.apply_controls` and in saved parameters.
        label: Text a GUI shows beside it.
        section: Group a GUI puts it under.
        kind: ``"number"``, ``"integer"``, ``"choice"`` or ``"bool"``.
        default: Value assumed until the camera has been read.
        unit: Display unit.
        scale: SI units per display unit (``1e-3`` for a value stored in seconds shown in ms).
        si_unit: Unit the driver stores the value in.
        minimum: Smallest value allowed, in stored units; ``None`` for no limit.
        maximum: Largest value allowed, in stored units; ``None`` for no limit.
        choices: The options of a ``"choice"`` control.
    """

    name: str
    label: str
    section: str
    kind: str
    default: Any
    unit: str = ""
    scale: float = 1.0
    si_unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    def describe(self) -> dict:
        """The control as a JSON-safe dict."""
        return {**asdict(self), "choices": list(self.choices)}


EXPOSURE = Control(
    "exposure",
    "Exposure",
    "Exposure & trigger",
    "number",
    10e-3,
    unit="ms",
    scale=1e-3,
    si_unit="s",
    minimum=1e-6,
)
TRIGGER_POLARITY = Control(
    "trigger_polarity",
    "Trigger polarity",
    "Exposure & trigger",
    "choice",
    "rise",
    choices=("rise", "fall"),
)
BINNING = Control("binning", "Binning", "Binning & ROI", "integer", 1, minimum=1)
#: Frames per second in internal trigger, so short exposures do not flood the
#: ring buffer and the stream; external triggers ignore it.
FRAME_RATE = Control(
    "frame_rate",
    "Frame rate (int.)",
    "Exposure & trigger",
    "number",
    10.0,
    unit="fps",
    si_unit="Hz",
    minimum=0.0,
)
ROI = (
    Control(
        "roi_x0",
        "ROI x0",
        "Binning & ROI",
        "integer",
        0,
        unit="px",
        si_unit="px",
        minimum=0,
    ),
    Control(
        "roi_width",
        "ROI width",
        "Binning & ROI",
        "integer",
        1024,
        unit="px",
        si_unit="px",
        minimum=1,
    ),
    Control(
        "roi_y0",
        "ROI y0",
        "Binning & ROI",
        "integer",
        0,
        unit="px",
        si_unit="px",
        minimum=0,
    ),
    Control(
        "roi_height",
        "ROI height",
        "Binning & ROI",
        "integer",
        1024,
        unit="px",
        si_unit="px",
        minimum=1,
    ),
)


def trigger_source(choices: tuple[str, ...]) -> Control:
    """The trigger-source control, offering ``choices`` (the first is its default)."""
    return Control(
        "trigger_source",
        "Trigger source",
        "Exposure & trigger",
        "choice",
        choices[0],
        choices=tuple(choices),
    )


def gain(
    label: str,
    kind: str,
    unit: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
    default: Any = 0,
) -> Control:
    """The gain control, labelled and ranged as the driver's gain is."""
    return Control(
        "gain",
        label,
        "Gain",
        kind,
        default,
        unit=unit,
        si_unit=unit,
        minimum=minimum,
        maximum=maximum,
    )


def roi_controls(roi: tuple) -> dict:
    """The ROI controls (and binning) for a pylablib ROI ``(hstart, hend, vstart, vend, hbin, vbin)``."""
    hstart, hend, vstart, vend, hbin, _ = roi
    return {
        "roi_x0": int(hstart),
        "roi_width": int(hend - hstart),
        "roi_y0": int(vstart),
        "roi_height": int(vend - vstart),
        "binning": int(hbin),
    }


_SETTER_NAMES = {
    "exposure": "set_exposure_time",
    "trigger_source": "set_trigger_source",
    "trigger_polarity": "set_trigger_polarity",
    "gain": "set_gain",
    "binning": "set_binning",
}
_ROI_NAMES = tuple(control.name for control in ROI)


class Camera(ABC):
    """Abstract base for a single-camera RPC target.

    Subclasses ("shims") implement the abstract hardware hooks; everything
    generic -- image broadcasting, TIFF autosave, the acquisition loop, and the
    connection lifecycle -- lives here so each driver stays a thin translation
    layer over its vendor SDK.

    The connected hardware handle is stored on :attr:`_backend`; ``None`` once
    :meth:`close` has run. Subclasses set it in :meth:`_connect`.

    A driver lists the settings it offers in :attr:`CONTROLS`, and :meth:`controls`
    and :meth:`apply_controls` work from that list. Each setter a driver
    implements takes :attr:`_lock` and records what the camera took with :meth:`_keep`.
    """

    #: The settings this camera offers, in the order a GUI shows them.
    CONTROLS: tuple[Control, ...] = ()
    #: Setter method for a control whose setter is not the shared ``set_<name>``
    #: (see :data:`_SETTER_NAMES`), by control name; a driver adds only its own.
    CONTROL_SETTERS: ClassVar[dict[str, str]] = {}
    #: The :class:`~pytweezer.drivers.temperature_monitor.TemperatureMonitor` a driver
    #: starts, if it watches the sensor temperature; :meth:`sensor_temperature` serves it.
    _temperature_monitor = None

    #: Filename prefix used by :meth:`save_tiff`; override per driver.
    image_prefix: str = "image"
    #: Sensor pixel ``(x0, y0)`` of the ROI's corner, as the last :meth:`set_roi` left it.
    roi_origin: tuple[int, int] = (0, 0)
    #: Pixels per side the frames are binned by, as the last :meth:`set_binning` or :meth:`set_roi` left it.
    binning: int = 1
    #: Metres per unbinned sensor pixel at the atoms (sensor pitch over magnification); ``None`` if unknown.
    object_pixel_m: float | None = None
    #: Whether the camera has an external trigger input. It starts on that
    #: trigger, or on software triggers without one.
    has_external_trigger: bool = True
    #: Sensor ``(width, height)`` in pixels, as :meth:`_sensor_shape` last read it; ``None`` if unknown.
    _sensor_size: tuple[int, int] | None = None
    #: The hardware's limits on the controls, as :meth:`_control_limits` last read them.
    _read_limits: dict[str, dict[str, Any] | None] | None = None
    #: Pixel value at which a live frame counts as saturated (see
    #: :meth:`saturated_frames_since`); ``None`` leaves saturation unwatched.
    saturation_level: int | None = None
    #: A simulated camera's frames never reach the lab's image streams
    #: (see :class:`~pytweezer.servers.clients.GenericClient`).
    simulated = False

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        object_pixel_m: float | None = None,
    ):
        """Set up broadcasting and autosave, then connect.

        The camera starts on its external trigger, or on software triggers if it
        has none, never free-running: a free-running camera floods the image server.

        Args:
            image_dir: Directory :meth:`save_tiff` writes to.
            timeout: Seconds to wait for frames.
            stream_name: Image stream to broadcast on; ``None`` disables broadcasting.
            object_pixel_m: Metres per unbinned sensor pixel at the atoms. When set,
                broadcast headers carry ``_imgresolution`` and ``_offset`` so
                consumers work in metres.
        """
        self.image_dir = image_dir
        self.object_pixel_m = object_pixel_m
        self.timeout = timeout
        self.stream_name = stream_name
        self.image_client = (
            ImageClient(stream_name, simulated=self.simulated) if stream_name else None
        )
        self._stream_clients: dict[str, ImageClient] = {}
        # The live broadcast thread and RPC calls share the backend.
        self._lock = threading.RLock()
        self._live_thread: threading.Thread | None = None
        self._pacer_thread: threading.Thread | None = None
        self._live_stop = threading.Event()
        self._live_args: tuple[int, float] = (1000, 0.05)
        # (frame number, peak value) of the saturated live frames of this acquisition.
        self._saturated: deque[tuple[int, int]] = deque(maxlen=1000)
        # (mark, header) of each shot whose frames live mode may not have sent yet,
        # in order; a None header ends tagging. See start_shot().
        self._shot_tags: list[tuple[int, dict | None]] = []
        self._backend = None
        self._kept: dict[str, Any] = {}
        self._connect()
        self._keep_hardware_limits()
        self.set_trigger_source("ext" if self.has_external_trigger else "software")

    # ------------------------------------------------------------------ #
    # Hardware-specific hooks -- implemented by each concrete shim.
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _connect(self) -> None:
        """Open the hardware and store the handle on ``self._backend``."""

    @abstractmethod
    def _disconnect(self) -> None:
        """Release the hardware handle (called by :meth:`close`)."""

    @abstractmethod
    def set_roi(self, x0: int, width: int, y0: int, height: int) -> tuple[int, int]:
        """Restrict readout to the given region of interest (pixels).

        Keeps the camera's binning. Returns the ``(x0, y0)`` the camera used, which it may have rounded,
        and keeps it as :attr:`roi_origin`. A driver keeps the four ROI values the camera took with :meth:`_keep`.
        """

    @abstractmethod
    def set_trigger_source(self, source: str) -> None:
        """Select the trigger mode/source (e.g. ``"int"``/``"ext"``).

        A driver keeps the source the camera took with :meth:`_keep`.
        """

    @abstractmethod
    def set_exposure_time(self, exposure: float) -> None:
        """Set the per-frame exposure time in seconds.

        A driver keeps the exposure the camera took, in seconds, with :meth:`_keep`.
        """

    def set_binning(self, binning: int) -> None:
        """Bin ``binning`` x ``binning`` pixels together.

        Changing it can reset the ROI, so call :meth:`set_roi` afterwards. A driver
        keeps the binning the camera took with :meth:`_keep`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no binning control")

    def set_gain(self, gain: float) -> None:
        """Set the camera's gain, in the unit its driver documents.

        A driver keeps the gain the camera took with :meth:`_keep`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no gain control")

    def set_trigger_polarity(self, polarity: str) -> None:
        """Trigger on the ``"rise"`` or ``"fall"`` edge of the external trigger.

        A driver keeps the polarity the camera took with :meth:`_keep`.
        """
        raise NotImplementedError(f"{type(self).__name__} has no trigger polarity")

    @abstractmethod
    def trigger(self) -> None:
        """Send one software trigger: with the trigger source on ``"software"``,
        the acquiring camera takes one frame. Live mode sends them at the
        ``frame_rate`` control's rate."""

    @abstractmethod
    def setup_acquisition(self, nframes: int, mode: str = "sequence") -> None:
        """Configure an acquisition of ``nframes``.

        ``mode="sequence"`` keeps acquiring into a ring buffer of ``nframes``;
        ``"snap"`` stops after ``nframes``. Cameras that only acquire into a
        ring buffer (Thorlabs) ignore ``mode``.
        """

    @abstractmethod
    def start_acquisition(self) -> None:
        """Begin acquiring frames."""

    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop acquiring frames."""

    @abstractmethod
    def _read_frames(self, nframes: int, start_frame: int) -> np.ndarray:
        """Grab ``nframes`` raw frames from the hardware and return them
        stacked as ``(nframes, H, W)``. May raise if not currently running."""

    def _acquisition_in_progress(self) -> bool:
        return bool(self._backend.acquisition_in_progress())

    def _count_frames(self) -> int:
        """Frames acquired since the acquisition started."""
        return int(self._backend.get_frames_status()[0])

    def _read_range(self, start: int, stop: int) -> np.ndarray:
        """Frames ``start`` to ``stop - 1`` without marking them read."""
        images = self._backend.read_multiple_images((start, stop), peek=True)
        return np.asarray(images if images is not None else [])

    def _read_range_timed(
        self, start: int, stop: int
    ) -> tuple[np.ndarray, list[float] | None]:
        """:meth:`_read_range` with each frame's camera-clock time in seconds.

        The times are ``None`` where the driver cannot give them. The clock's zero
        is arbitrary.
        """
        return self._read_range(start, stop), None

    # ------------------------------------------------------------------ #
    # Connection lifecycle (generic).
    # ------------------------------------------------------------------ #

    def is_connected(self) -> bool:
        return self._backend is not None

    def close(self) -> None:
        self._stop_live_thread()
        try:
            self._disconnect()
        except Exception:
            LOGGER.exception("Failed to close %s cleanly", type(self).__name__)
        finally:
            self._backend = None

    # ------------------------------------------------------------------ #
    # Acquisition orchestration (generic).
    # ------------------------------------------------------------------ #

    def acquire_n_frames(
        self,
        nframes: int,
        start_frame: int = 0,
        autosave: bool = False,
        broadcast: bool = False,
        header: dict | None = None,
        stream: str | None = None,
    ) -> np.ndarray:
        """Read ``nframes`` frames; ``header`` adds keys to each broadcast frame's header.

        ``stream`` broadcasts them on that image stream instead of the camera's own.
        """
        start_frame = int(start_frame)
        images = self._read_frames(int(nframes), start_frame)
        for i, image in enumerate(images):
            if autosave:
                self.save_tiff(image=image, image_dir=self.image_dir)
            if broadcast:
                info = {
                    **(header or {}),
                    "timestamp": time.time(),
                    "index": start_frame + i,
                }
                self.broadcast_image(image, info, stream)
        return np.asarray(images)

    def acquire_single_frame(
        self,
        autosave: bool = False,
        broadcast: bool = False,
        start_frame: int = 0,
        header: dict | None = None,
    ) -> np.ndarray:
        images = self.acquire_n_frames(
            nframes=1,
            start_frame=start_frame,
            autosave=autosave,
            broadcast=broadcast,
            header=header,
        )
        return images[0]

    def snap(self, broadcast: bool = True) -> np.ndarray:
        """One frame from a one-frame acquisition, which is stopped again afterwards.

        Raises:
            RuntimeError: if the camera is live; stop live first.
        """
        if self.is_live():
            raise RuntimeError(f"{type(self).__name__} is live; stop live to snap")
        self.setup_acquisition(1, mode="snap")
        try:
            self.start_acquisition()
            if self._software_triggered():
                self.trigger()
            return self.acquire_n_frames(1, broadcast=broadcast)[0]
        finally:
            self.stop_acquisition()

    def arm(self, buffer_frames: int = 1000) -> None:
        """Start a continuous ring-buffer acquisition unless one is already running.

        Frames are numbered from the start of that acquisition, which is what
        :meth:`frames_acquired` and :meth:`read_frames_from` count in.
        """
        with self._lock:
            if self._acquisition_in_progress():
                return
            self._shot_tags.clear()
            self.setup_acquisition(int(buffer_frames))
            self.start_acquisition()

    def start_shot(self, header: dict) -> int:
        """Start a shot: live frames from now on carry ``header`` and their index within the shot.

        Each live frame from the returned mark on is broadcast with ``header``
        added and ``index`` set to its frame number minus the mark (0, 1, ...),
        until the next :meth:`start_shot` or :meth:`end_shots`. Frames taken
        before the mark keep the tag they had.

        Returns:
            The mark: the frame number the shot's first frame will have, for
            :meth:`wait_frames`.

        Raises:
            RuntimeError: if the camera is not acquiring; start live first.
        """
        with self._lock:
            if not self._acquisition_in_progress():
                raise RuntimeError(
                    f"{type(self).__name__} is not acquiring; start live first"
                )
            mark = self._count_frames()
            self._shot_tags.append((mark, dict(header)))
        return mark

    def end_shots(self) -> None:
        """Live frames from now on are broadcast untagged, numbered from the acquisition's start."""
        with self._lock:
            if self._acquisition_in_progress():
                self._shot_tags.append((self._count_frames(), None))
            else:
                self._shot_tags.clear()

    def wait_frames(
        self, mark: int, n: int, timeout: float | None = None
    ) -> np.ndarray:
        """Frames ``mark`` to ``mark + n - 1`` of the current acquisition, once they exist.

        Args:
            mark: The first frame's number, as :meth:`start_shot` returned it.
            n: How many frames.
            timeout: Longest wait in seconds; the camera's :attr:`timeout` if None.

        Returns:
            ``(n, H, W)``.

        Raises:
            TimeoutError: if the frames have not all arrived in time.
            RuntimeError: if the acquisition restarted since the mark.
        """
        mark, n = int(mark), int(n)
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while (acquired := self.frames_acquired()) < mark + n:
            if acquired < mark:
                raise RuntimeError(
                    f"{type(self).__name__}'s acquisition restarted; frames from "
                    f"{mark} on are lost"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{type(self).__name__}: {acquired - mark} of {n} frames "
                    "arrived; was the camera triggered?"
                )
            time.sleep(0.005)
        return self.read_frames_from(mark)[:n]

    def frames_acquired(self) -> int:
        """Frames acquired since the current acquisition started."""
        with self._lock:
            return self._count_frames()

    def read_frames_from(self, start: int) -> np.ndarray:
        """Every frame from number ``start`` to the latest, as ``(n, H, W)``.

        Raises:
            RuntimeError: if some of those frames were already overwritten in the
                ring buffer.
        """
        return self._read_frames_timed_from(int(start))[0]

    def _read_frames_timed_from(
        self, start: int
    ) -> tuple[np.ndarray, list[float] | None]:
        """:meth:`read_frames_from` with each frame's camera-clock time (see :meth:`_read_range_timed`)."""
        with self._lock:
            stop = self._count_frames()
            if stop <= start:
                return np.empty((0, 0, 0), dtype=np.uint16), None
            images, times = self._read_range_timed(start, stop)
        if len(images) != stop - start:
            raise RuntimeError(
                f"{type(self).__name__}: only {len(images)} of frames {start}-"
                f"{stop - 1} were still in the buffer; arm() with more buffer_frames"
            )
        if times is not None and len(times) != len(images):
            raise RuntimeError(
                f"{type(self).__name__}: {len(times)} camera times for {len(images)} frames"
            )
        return images, times

    def settings(self) -> dict:
        """Camera settings worth keeping with saved frames (JSON-safe)."""
        return {}

    # ------------------------------------------------------------------ #
    # Declared controls (generic).
    # ------------------------------------------------------------------ #

    def controls(self) -> list[dict]:
        """Each of :attr:`CONTROLS` as :meth:`Control.describe` gives it, plus its ``"value"``.

        Values are those last kept or read back (the control's default before
        either); the hardware is not touched. Once the sensor size is known, the
        ROI controls' ``maximum`` is the sensor's; once the camera has been read,
        each control carries the limits :meth:`_control_limits` gave it (``binning``
        its allowed bin factors as integer ``choices``), and a control the camera
        lacks is left out.
        """
        limits = self._hardware_limits()
        return [
            {
                **control.describe(),
                **limits.get(control.name, {}),
                "value": self._kept.get(control.name, control.default),
            }
            for control in self.CONTROLS
            if control.name not in self._absent_controls()
        ]

    def _absent_controls(self) -> set[str]:
        """Names of the :attr:`CONTROLS` this camera's hardware turned out to lack."""
        return {
            name for name, limit in (self._read_limits or {}).items() if limit is None
        }

    def read_controls(self) -> list[dict]:
        """Read every control's value back from the camera, then return :meth:`controls`."""
        with self._lock:
            self._kept.update(self._read_control_values())
            self._keep_hardware_limits()
        return self.controls()

    def _sensor_shape(self) -> tuple[int, int] | None:
        """The sensor's ``(width, height)`` in pixels, read from the hardware; ``None`` if unknown.

        pylablib cameras give it as ``get_detector_size()``; a driver whose
        backend does not overrides this.
        """
        detector_size = getattr(self._backend, "get_detector_size", None)
        return None if detector_size is None else detector_size()

    def _control_limits(self) -> dict[str, dict[str, Any] | None]:
        """The hardware's limits on the controls, by control name, read from the camera.

        Each value overrides keys of the control's description: ``{"choices":
        [1, 2, 4]}`` for a ``binning`` that takes only those bin factors,
        ``{"minimum": ..., "maximum": ...}`` for a range. ``None`` means the camera
        lacks the control, which is then neither shown nor applied. Drivers
        override this.
        """
        return {}

    def _keep_hardware_limits(self) -> None:
        """Read the sensor size and allowed binnings, keeping the last good reading of each."""
        try:
            shape = self._sensor_shape()
        except Exception:
            LOGGER.warning(
                "%s could not read its sensor size", type(self).__name__, exc_info=True
            )
        else:
            if shape is not None:
                self._sensor_size = (int(shape[0]), int(shape[1]))
        try:
            self._read_limits = self._control_limits()
        except Exception:
            LOGGER.warning(
                "%s could not read its control limits",
                type(self).__name__,
                exc_info=True,
            )

    def _hardware_limits(self) -> dict[str, dict[str, Any]]:
        """The hardware's limits on the controls present, by control name: :meth:`_control_limits` and the ROI maxima."""
        limits = {
            name: dict(limit)
            for name, limit in (self._read_limits or {}).items()
            if limit is not None
        }
        if self._sensor_size is not None:
            width, height = self._sensor_size
            limits |= {
                "roi_x0": {"maximum": width - 1},
                "roi_width": {"maximum": width},
                "roi_y0": {"maximum": height - 1},
                "roi_height": {"maximum": height},
            }
        return limits

    def apply_controls(self, values: dict[str, Any]) -> list[dict]:
        """Set the controls named in ``values`` (SI units), then read them all back.

        A live camera is stopped while they are set and restarted after. The four
        ROI values are set together; any left out keep their current value.

        Returns:
            :meth:`controls`, as the camera now has them.

        Raises:
            KeyError: if a name is not one of this camera's controls.
        """
        by_name = {
            control.name: control
            for control in self.CONTROLS
            if control.name not in self._absent_controls()
        }
        unknown = sorted(set(values) - set(by_name))
        if unknown:
            raise KeyError(
                f"{type(self).__name__} has no control {unknown}; its controls are {sorted(by_name)}"
            )
        was_live = self.is_live()
        if was_live:
            self.stop_live()
        try:
            for name in self._application_order(by_name):
                if name == "roi":
                    if any(roi_name in values for roi_name in _ROI_NAMES):
                        self._apply_roi(values)
                elif name in values:
                    self._apply_control(by_name[name], values[name])
        finally:
            try:
                self.read_controls()
            finally:
                if was_live:
                    self.start_live(*self._live_args)
        return self.controls()

    @staticmethod
    def _application_order(by_name: dict[str, Control]) -> list[str]:
        """Control names in the order they are set: binning, ROI, exposure and trigger, the rest, gain last."""
        early = [
            "binning",
            "roi",
            "exposure",
            "exposure_mode",
            "trigger_source",
            "trigger_polarity",
        ]
        rest = [
            name
            for name in by_name
            if name not in early and name not in _ROI_NAMES and name != "gain"
        ]
        return early + rest + ["gain"]

    def _apply_roi(self, values: dict[str, Any]) -> None:
        missing = [name for name in _ROI_NAMES if name not in values]
        if any(name not in self._kept for name in missing):
            self.read_controls()
        x0, width, y0, height = (
            values[name] if name in values else self._kept[name] for name in _ROI_NAMES
        )
        self.set_roi(int(x0), int(width), int(y0), int(height))

    def _keep(self, **values: Any) -> None:
        """Record the values the camera took, in SI units, for :meth:`controls`."""
        self._kept.update(values)

    def _rebin_keeping_roi(self, binning: int) -> None:
        """Set the binning of a pylablib camera through its ``set_roi``, passing the current
        ROI back with it, and keep what the camera took."""
        with self._lock:
            hstart, hend, vstart, vend, *_ = self._backend.get_roi()
            roi = self._backend.set_roi(
                hstart, hend, vstart, vend, int(binning), int(binning)
            )
            self._keep_roi(roi)

    def _keep_roi(self, roi: tuple) -> tuple[int, int]:
        """Keep the pylablib ROI a setter got back, with the origin and binning frames are read with."""
        controls = roi_controls(roi)
        self.roi_origin = (controls["roi_x0"], controls["roi_y0"])
        self.binning = controls["binning"]
        self._keep(**controls)
        return self.roi_origin

    def _read_control_values(self) -> dict[str, Any]:
        """Every control's value as the camera has it now, keyed by control name.

        Drivers override this; the base knows nothing the setters did not :meth:`_keep`.
        """
        return {}

    def _apply_control(self, control: Control, value: Any) -> None:
        """Set one control through its setter, named in :attr:`CONTROL_SETTERS` or else the shared map."""
        name = self.CONTROL_SETTERS.get(control.name) or _SETTER_NAMES.get(
            control.name, f"set_{control.name}"
        )
        setter = getattr(self, name)
        setter(bool(value) if control.kind == "bool" else value)

    def sensor_temperature(self) -> dict | None:
        """The sensor temperature the camera's monitor last read, without touching the hardware.

        Returns:
            ``{"celsius": ..., "time": ..., "max_celsius": ...}``, ``time`` being when it
            was read (unix seconds) and ``max_celsius`` the monitor's limit (``None`` if
            it has none); ``None`` if the camera has no monitor or it has not read yet.
        """
        monitor = self._temperature_monitor
        reading = None if monitor is None else monitor.last_reading()
        if reading is None:
            return None
        celsius, read_at = reading
        return {"celsius": celsius, "time": read_at, "max_celsius": monitor.max_c}

    def frame_geometry(self) -> tuple[int, tuple[int, int]]:
        """``(binning, (x0, y0))`` the camera reads frames with now: :attr:`binning` and :attr:`roi_origin`."""
        return self.binning, self.roi_origin

    def start_live(self, buffer_frames: int = 1000, poll_s: float = 0.05) -> None:
        """Run free (see :meth:`arm`) and broadcast every new frame on the image stream.

        With the trigger source on ``"software"``, live mode triggers the camera at
        the ``frame_rate`` control's rate. A camera already live stays as it is.
        """
        self.arm(buffer_frames)
        if self.is_live():
            return
        self._live_args = (buffer_frames, poll_s)
        self._live_stop.clear()
        self._live_thread = threading.Thread(
            target=self._broadcast_live,
            args=(float(poll_s),),
            name=f"{type(self).__name__}-live",
            daemon=True,
        )
        self._live_thread.start()
        self._pacer_thread = threading.Thread(
            target=self._pace_software_triggers,
            name=f"{type(self).__name__}-pacer",
            daemon=True,
        )
        self._pacer_thread.start()

    def stop_live(self) -> None:
        """Stop broadcasting and stop the acquisition."""
        self._stop_live_thread()
        self.stop_acquisition()
        with self._lock:
            self._shot_tags.clear()

    def is_live(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def _stop_live_thread(self) -> None:
        self._live_stop.set()
        if self._live_thread is not None:
            self._live_thread.join(timeout=2.0)
            self._live_thread = None
        if self._pacer_thread is not None:
            self._pacer_thread.join(timeout=2.0)
            self._pacer_thread = None

    def _software_triggered(self) -> bool:
        return self._kept.get("trigger_source") == "software"

    def _pace_software_triggers(self) -> None:
        stop = self._live_stop
        while not stop.wait(1.0 / self._software_trigger_rate()):
            with self._lock:
                if not (self._software_triggered() and self._acquisition_in_progress()):
                    continue
                try:
                    self.trigger()
                except Exception:
                    LOGGER.warning(
                        "%s: software trigger failed",
                        type(self).__name__,
                        exc_info=True,
                    )

    def _software_trigger_rate(self) -> float:
        """Live mode's software triggers per second: the ``frame_rate`` control's value."""
        rate = self._kept.get("frame_rate")
        return float(rate) if rate else FRAME_RATE.default

    def saturated_frames_since(self, start: int) -> list[tuple[int, int]]:
        """``(frame number, peak value)`` of each live frame from number ``start``
        on that reached :attr:`saturation_level`, numbered as :meth:`frames_acquired`
        counts them. Frames are checked as live mode reads them, so a frame just
        acquired may not be checked yet.
        """
        with self._lock:
            return [entry for entry in self._saturated if entry[0] >= int(start)]

    def _broadcast_live(self, poll_s: float) -> None:
        sent, last_error = 0, None
        while not self._live_stop.wait(poll_s):
            sent, last_error = self._send_new_frames(sent, last_error)
        # Frames since the last poll would otherwise never be broadcast.
        self._send_new_frames(sent, last_error)

    def _send_new_frames(
        self, sent: int, last_error: str | None
    ) -> tuple[int, str | None]:
        """Broadcast every frame from number ``sent`` on; the next frame to send and the last error."""
        try:
            if self.frames_acquired() < sent:  # acquisition restarted
                sent = 0
                with self._lock:
                    self._saturated.clear()
            frames, times = self._read_frames_timed_from(sent)
        except Exception as error:
            if repr(error) != last_error:
                LOGGER.warning("Live broadcast failed: %r", error)
            return sent, repr(error)
        for i, frame in enumerate(frames):
            self._check_saturation(sent, frame)
            info = {"timestamp": time.time(), "index": sent}
            mark, header = self._shot_tag(sent)
            if header is not None:
                info.update(header, index=sent - mark)
            if times is not None:
                info["camera_timestamp"] = times[i]
            self.broadcast_image(frame, info)
            sent += 1
        self._forget_shot_tags_before(sent)
        return sent, None

    def _shot_tag(self, frame: int) -> tuple[int, dict | None]:
        """``(mark, header)`` of the shot that ``frame`` belongs to; ``(0, None)`` if none."""
        with self._lock:
            tag = (0, None)
            for entry in self._shot_tags:
                if entry[0] > frame:
                    break
                tag = entry
            return tag

    def _forget_shot_tags_before(self, frame: int) -> None:
        """Drop the tags of shots that ended before ``frame``, which live mode has sent."""
        with self._lock:
            while len(self._shot_tags) > 1 and self._shot_tags[1][0] <= frame:
                del self._shot_tags[0]

    def _check_saturation(self, index: int, frame: np.ndarray) -> None:
        if self.saturation_level is None or frame.size == 0:
            return
        peak = int(frame.max())
        if peak < self.saturation_level:
            return
        with self._lock:
            # Log the first of a run of saturated frames, not every one.
            if not self._saturated or self._saturated[-1][0] != index - 1:
                LOGGER.warning(
                    "Frame %d saturated: peak %d >= %d",
                    index,
                    peak,
                    self.saturation_level,
                )
            self._saturated.append((index, peak))

    def broadcast_image(
        self, image: np.ndarray, info, stream: str | None = None
    ) -> None:
        """Send ``image`` on the image hub, its header saying which ROI and binning it was read with.

        ``stream`` names another image stream to send it on, instead of the camera's own.

        With :attr:`object_pixel_m` set, the header also gives ``_imgresolution``
        (metres per binned pixel) and ``_offset`` (the ROI corner in binned pixels).
        """
        client = self.image_client if stream is None else self._client_for(stream)
        if client is None:
            LOGGER.error("tried to broadcast image but no client exists")
            return
        roi_x0, roi_y0 = self.roi_origin
        header = {"roi_x0": roi_x0, "roi_y0": roi_y0, "binning": self.binning}
        if self.object_pixel_m is not None:
            pixel_m = self.object_pixel_m * self.binning
            header["_imgresolution"] = [pixel_m, pixel_m]
            header["_offset"] = [roi_x0 / self.binning, roi_y0 / self.binning]
        client.send({**header, **info}, image)

    def _client_for(self, stream: str) -> ImageClient:
        if stream not in self._stream_clients:
            self._stream_clients[stream] = ImageClient(stream, simulated=self.simulated)
        return self._stream_clients[stream]

    @classmethod
    def save_tiff(
        cls, image: np.ndarray, image_dir: str | None = None, run_no: int = 0
    ) -> None:
        image_dir = image_dir or "."
        i = 1
        while os.path.exists(
            os.path.join(image_dir, f"{cls.image_prefix}{run_no:04d}_{i}.tif")
        ):
            i += 1
        filename = os.path.join(image_dir, f"{cls.image_prefix}{run_no:04d}_{i}.tif")
        tiff.imwrite(filename, image)


class SimulatedCamera(Camera):
    """One-size-fits-all synthetic camera backend.

    Implements every :class:`Camera` hardware hook with generated frames, so it
    can stand in for any real camera in simulation mode. Driver-specific *extra*
    methods (beyond the :class:`Camera` interface) are not defined here; use
    :func:`simulated_camera_for` to auto-stub those for a particular driver.

    While acquiring, frames "arrive" at :attr:`sim_frame_rate` per second, or one per
    :meth:`trigger` with the trigger source on ``"software"``.
    """

    simulated = True
    sim_frame_rate: float = 5.0
    #: Sensor ``(width, height)`` the simulated camera reports.
    sim_sensor_size: tuple[int, int] = (512, 512)

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        object_pixel_m: float | None = None,
    ):
        self._shape = (256, 256)
        self._sim_values: dict[str, Any] = {
            "roi_height": 256,
            "roi_width": 256,
            "frame_rate": self.sim_frame_rate,
        }
        self._nframes = 1
        self._running = False
        self._triggers = 0
        self._started = 0.0
        self._rng = np.random.default_rng()
        super().__init__(
            image_dir=image_dir,
            timeout=timeout,
            stream_name=stream_name,
            object_pixel_m=object_pixel_m,
        )

    def _connect(self) -> None:
        # No hardware; a non-None sentinel marks the camera as "connected".
        self._backend = object()

    def _disconnect(self) -> None:
        self._running = False

    def _sensor_shape(self) -> tuple[int, int]:
        return self.sim_sensor_size

    def _control_limits(self) -> dict[str, dict[str, Any] | None]:
        return {
            "binning": {"choices": [1, 2, 4]},
            "frame_rate": {"minimum": 0.1, "maximum": 100.0},
        }

    def set_frame_rate(self, rate: float) -> None:
        """Frames per second while acquiring, as an internal trigger would give them."""
        with self._lock:
            self.sim_frame_rate = float(rate)
            self._remember(frame_rate=self.sim_frame_rate)

    def _remember(self, **values: Any) -> None:
        """Keep ``values`` and hold them as what the simulated hardware reports."""
        self._keep(**values)
        self._sim_values.update(values)

    def _read_control_values(self) -> dict[str, Any]:
        return {
            control.name: self._sim_values.get(control.name, control.default)
            for control in self.CONTROLS
        }

    def _apply_control(self, control: Control, value: Any) -> None:
        with self._lock:
            super()._apply_control(control, value)
            self._sim_values[control.name] = value

    def set_roi(self, x0: int, width: int, y0: int, height: int) -> tuple[int, int]:
        with self._lock:
            self._shape = (int(height), int(width))
            self.roi_origin = (int(x0), int(y0))
            self._remember(
                roi_x0=int(x0),
                roi_width=int(width),
                roi_y0=int(y0),
                roi_height=int(height),
            )
            return self.roi_origin

    def set_trigger_source(self, source: str) -> None:
        with self._lock:
            self._trigger_source = source
            self._remember(trigger_source=source)

    def set_trigger_polarity(self, polarity: str) -> None:
        with self._lock:
            self._remember(trigger_polarity=polarity)

    def trigger(self) -> None:
        with self._lock:
            if self._running:
                self._triggers += 1

    def set_exposure_time(self, exposure: float) -> None:
        with self._lock:
            self._exposure = float(exposure)
            self._remember(exposure=self._exposure)

    def set_binning(self, binning: int) -> None:
        with self._lock:
            self.binning = int(binning)
            self._remember(binning=self.binning)

    def set_gain(self, gain: float) -> None:
        with self._lock:
            self._gain = float(gain)
            self._remember(gain=self._gain)

    def setup_acquisition(self, nframes: int, mode: str = "sequence") -> None:
        self._nframes = max(1, int(nframes))

    def start_acquisition(self) -> None:
        self._running = True
        self._started = time.monotonic()
        self._triggers = 0

    def stop_acquisition(self) -> None:
        self._running = False

    def _acquisition_in_progress(self) -> bool:
        return self._running

    def _count_frames(self) -> int:
        if not self._running:
            return 0
        if self._software_triggered():
            return self._triggers
        return int((time.monotonic() - self._started) * self.sim_frame_rate)

    def _read_range(self, start: int, stop: int) -> np.ndarray:
        return np.stack([self._generate_frame() for _ in range(stop - start)])

    def _read_range_timed(
        self, start: int, stop: int
    ) -> tuple[np.ndarray, list[float]]:
        times = [self._started + i / self.sim_frame_rate for i in range(start, stop)]
        return self._read_range(start, stop), times

    def settings(self) -> dict:
        return {"shape": list(self._shape)}

    def _generate_frame(self) -> np.ndarray:
        h, w = self._shape
        image = self._rng.normal(loc=120.0, scale=9.0, size=(h, w)).astype(np.float32)
        y, x = np.mgrid[0:h, 0:w]
        for _ in range(12):
            cy = self._rng.integers(20, max(21, h - 20))
            cx = self._rng.integers(20, max(21, w - 20))
            amp = self._rng.uniform(80.0, 280.0)
            sigma = self._rng.uniform(1.3, 2.5)
            image += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
        return np.clip(image, 0.0, 65535.0).astype(np.uint16)

    def _read_frames(self, nframes: int, start_frame: int) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Simulated camera is not running")
        count = max(1, int(nframes))
        time.sleep(0.1 * count)
        return np.stack([self._generate_frame() for _ in range(count)], axis=0)


def simulated_camera_for(
    real_cls: type, defaults: dict[str, Any] | None = None
) -> type:
    """Return a :class:`SimulatedCamera` subclass tailored to ``real_cls``.

    The returned class has the full generic simulated behaviour, plus a
    logging no-op stub for every public method ``real_cls`` adds beyond the
    :class:`Camera` interface (e.g. a driver's EM-gain setters), courtesy of
    :func:`~pytweezer.servers.simulated_device.simulate`. This keeps the
    simulated surface interface-complete with the real driver without any
    hand-written per-driver simulated class. ``defaults`` gives stubs a return
    value, as for :func:`~pytweezer.servers.simulated_device.simulate`.
    """

    @simulate(real_cls, defaults=defaults)
    class _Simulated(SimulatedCamera):
        pass

    _Simulated.__name__ = _Simulated.__qualname__ = f"Simulated{real_cls.__name__}"
    _Simulated.image_prefix = getattr(
        real_cls, "image_prefix", SimulatedCamera.image_prefix
    )
    _Simulated.CONTROLS = getattr(real_cls, "CONTROLS", SimulatedCamera.CONTROLS)
    _Simulated.CONTROL_SETTERS = getattr(
        real_cls, "CONTROL_SETTERS", SimulatedCamera.CONTROL_SETTERS
    )
    return _Simulated
