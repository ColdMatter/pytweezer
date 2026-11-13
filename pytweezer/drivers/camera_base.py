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
from dataclasses import asdict, dataclass
from typing import Any

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

    #: Filename prefix used by :meth:`save_tiff`; override per driver.
    image_prefix: str = "image"
    #: Sensor pixel ``(x0, y0)`` of the ROI's corner, as the last :meth:`set_roi` left it.
    roi_origin: tuple[int, int] = (0, 0)
    #: Pixels per side the frames are binned by, as the last :meth:`set_binning` or :meth:`set_roi` left it.
    binning: int = 1
    #: Metres per unbinned sensor pixel at the atoms (sensor pitch over magnification); ``None`` if unknown.
    object_pixel_m: float | None = None

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        object_pixel_m: float | None = None,
    ):
        """Set up broadcasting and autosave, then connect.

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
        self.image_client = ImageClient(stream_name) if stream_name else None
        self._stream_clients: dict[str, ImageClient] = {}
        # The live broadcast thread and RPC calls share the backend.
        self._lock = threading.RLock()
        self._live_thread: threading.Thread | None = None
        self._live_stop = threading.Event()
        self._backend = None
        self._kept: dict[str, Any] = {}
        self._connect()

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
            self.setup_acquisition(int(buffer_frames))
            self.start_acquisition()

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
        either); the hardware is not touched.
        """
        return [
            {
                **control.describe(),
                "value": self._kept.get(control.name, control.default),
            }
            for control in self.CONTROLS
        ]

    def read_controls(self) -> list[dict]:
        """Read every control's value back from the camera, then return :meth:`controls`."""
        with self._lock:
            self._kept.update(self._read_control_values())
        return self.controls()

    def apply_controls(self, values: dict[str, Any]) -> list[dict]:
        """Set the controls named in ``values`` (SI units), then read them all back.

        A live camera is stopped while they are set and restarted after. The four
        ROI values are set together; any left out keep their current value.

        Returns:
            :meth:`controls`, as the camera now has them.

        Raises:
            KeyError: if a name is not one of this camera's controls.
        """
        by_name = {control.name: control for control in self.CONTROLS}
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
                    self.start_live()
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

    def _read_control_values(self) -> dict[str, Any]:
        """Every control's value as the camera has it now, keyed by control name.

        Drivers override this; the base knows nothing the setters did not :meth:`_keep`.
        """
        return {}

    def _apply_control(self, control: Control, value: Any) -> None:
        """Set one control through its setter; drivers override where that does not fit."""
        setter = getattr(self, _SETTER_NAMES.get(control.name, f"set_{control.name}"))
        setter(bool(value) if control.kind == "bool" else value)

    def frame_geometry(self) -> tuple[int, tuple[int, int]]:
        """``(binning, (x0, y0))`` the camera reads frames with now: :attr:`binning` and :attr:`roi_origin`."""
        return self.binning, self.roi_origin

    def start_live(self, buffer_frames: int = 1000, poll_s: float = 0.05) -> None:
        """Run free (see :meth:`arm`) and broadcast every new frame on the image stream.

        A camera already live stays as it is.
        """
        self.arm(buffer_frames)
        if self.is_live():
            return
        self._live_stop.clear()
        self._live_thread = threading.Thread(
            target=self._broadcast_live,
            args=(float(poll_s),),
            name=f"{type(self).__name__}-live",
            daemon=True,
        )
        self._live_thread.start()

    def stop_live(self) -> None:
        """Stop broadcasting and stop the acquisition."""
        self._stop_live_thread()
        self.stop_acquisition()

    def is_live(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def _stop_live_thread(self) -> None:
        self._live_stop.set()
        if self._live_thread is not None:
            self._live_thread.join(timeout=2.0)
            self._live_thread = None

    def _broadcast_live(self, poll_s: float) -> None:
        sent = 0
        last_error = None
        while not self._live_stop.wait(poll_s):
            try:
                if self.frames_acquired() < sent:  # acquisition restarted
                    sent = 0
                frames, times = self._read_frames_timed_from(sent)
            except Exception as error:
                if repr(error) != last_error:
                    LOGGER.warning("Live broadcast failed: %r", error)
                    last_error = repr(error)
                continue
            last_error = None
            for i, frame in enumerate(frames):
                info = {"timestamp": time.time(), "index": sent}
                if times is not None:
                    info["camera_timestamp"] = times[i]
                self.broadcast_image(frame, info)
                sent += 1

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
            self._stream_clients[stream] = ImageClient(stream)
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

    While acquiring, frames "arrive" at :attr:`sim_frame_rate` per second.
    """

    sim_frame_rate: float = 5.0

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: str | None = None,
        object_pixel_m: float | None = None,
    ):
        self._shape = (256, 256)
        self._sim_values: dict[str, Any] = {"roi_height": 256, "roi_width": 256}
        self._nframes = 1
        self._running = False
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

    def stop_acquisition(self) -> None:
        self._running = False

    def _acquisition_in_progress(self) -> bool:
        return self._running

    def _count_frames(self) -> int:
        if not self._running:
            return 0
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
    return _Simulated
