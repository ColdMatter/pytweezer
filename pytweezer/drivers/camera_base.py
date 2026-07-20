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

* :func:`simulated_camera_for` -- adapts :class:`SimulatedCamera` to a specific
  real driver so that driver's *extra* methods (e.g. the ImagEM's EM-gain
  setters) are auto-stubbed via :func:`pytweezer.servers.simulated_device.simulate`,
  keeping the simulated surface interface-complete without drift.

A concrete driver ("shim") is therefore just the thin translation layer between
its vendor SDK (dcam, Spinnaker, ...) and the abstract hooks below.
"""

from __future__ import annotations

import os
import time
from functools import wraps
from typing import Optional

import numpy as np
import tifffile as tiff

from abc import ABC, abstractmethod

from pytweezer.servers import ImageClient
from pytweezer.servers.simulated_device import simulate

from pytweezer.logging_utils import get_logger

LOGGER = get_logger("camera")


def requires_camera(func):
    """Guard a method so it raises cleanly if the camera has been relinquished."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._require_camera()
        return func(self, *args, **kwargs)

    return wrapper


class Camera(ABC):
    """Abstract base for a single-camera RPC target.

    Subclasses ("shims") implement the abstract hardware hooks; everything
    generic -- image broadcasting, TIFF autosave, the acquisition loop, and the
    connection lifecycle -- lives here so each driver stays a thin translation
    layer over its vendor SDK.

    The connected hardware handle is stored on :attr:`_backend`; ``None`` means
    the camera has been relinquished. Subclasses set it in :meth:`_connect` and
    clear it (implicitly, via :meth:`close`) on teardown.
    """

    #: Filename prefix used by :meth:`save_tiff`; override per driver.
    image_prefix: str = "image"

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: Optional[str] = None,
    ):
        self.image_dir = image_dir
        self.timeout = timeout
        self.stream_name = stream_name
        self.image_client = ImageClient(stream_name) if stream_name else None
        self._backend = None
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
    def set_roi(self, x0: int, width: int, y0: int, height: int) -> None:
        """Restrict readout to the given region of interest (pixels)."""

    @abstractmethod
    def set_trigger_source(self, source: str) -> None:
        """Select the trigger mode/source (e.g. ``"int"``/``"ext"``)."""

    @abstractmethod
    def set_exposure_time(self, exposure: float) -> None:
        """Set the per-frame exposure time in seconds."""

    @abstractmethod
    def setup_acquisition(self, acq_mode: str, nframes: int) -> None:
        """Configure an acquisition of ``nframes`` in ``acq_mode``."""

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

    # ------------------------------------------------------------------ #
    # Connection lifecycle (generic).
    # ------------------------------------------------------------------ #

    def is_connected(self) -> bool:
        return self._backend is not None

    def _require_camera(self) -> None:
        if not self.is_connected():
            raise RuntimeError(
                f"{type(self).__name__} connection has been relinquished"
            )

    def close(self) -> None:
        try:
            self._disconnect()
        except Exception:
            LOGGER.exception("Failed to close %s cleanly", type(self).__name__)
        finally:
            self._backend = None

    def relinquish_camera(self) -> dict:
        self.close()
        return {"ok": True, "relinquished": True}

    def reacquire_camera(self) -> dict:
        if self._backend is None:
            self._connect()
        return {"ok": True, "relinquished": False}

    # ------------------------------------------------------------------ #
    # Acquisition orchestration (generic).
    # ------------------------------------------------------------------ #

    @requires_camera
    def acquire_n_frames(
        self,
        nframes: int,
        start_frame: int = 0,
        autosave: bool = False,
        broadcast: bool = False,
    ) -> np.ndarray:
        start_frame = int(start_frame)
        images = self._read_frames(int(nframes), start_frame)
        for i, image in enumerate(images):
            if autosave:
                self.save_tiff(image=image, image_dir=self.image_dir)
            if broadcast:
                info = {"timestamp": time.time(), "index": start_frame + i}
                self.broadcast_image(image, info)
        return np.asarray(images)

    @requires_camera
    def acquire_single_frame(
        self,
        autosave: bool = False,
        broadcast: bool = False,
        start_frame: int = 0,
    ) -> np.ndarray:
        images = self.acquire_n_frames(
            nframes=1,
            start_frame=start_frame,
            autosave=autosave,
            broadcast=broadcast,
        )
        return images[0]

    def broadcast_image(self, image: np.ndarray, info) -> None:
        if self.image_client is None:
            LOGGER.error("tried to broadcast image but no client exists")
            return
        self.image_client.send(info, image)

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
    """

    def __init__(
        self,
        image_dir: str | None = None,
        timeout: float = 5.0,
        stream_name: Optional[str] = None,
    ):
        self._shape = (256, 256)
        self._nframes = 1
        self._running = False
        self._rng = np.random.default_rng()
        super().__init__(image_dir=image_dir, timeout=timeout, stream_name=stream_name)

    def _connect(self) -> None:
        # No hardware; a non-None sentinel marks the camera as "connected".
        self._backend = object()

    def _disconnect(self) -> None:
        self._running = False

    def set_roi(self, x0: int, width: int, y0: int, height: int) -> None:
        self._shape = (int(height), int(width))

    def set_trigger_source(self, source: str) -> None:
        self._trigger_source = source

    def set_exposure_time(self, exposure: float) -> None:
        self._exposure = float(exposure)

    def setup_acquisition(self, acq_mode: str, nframes: int) -> None:
        self._nframes = max(1, int(nframes))

    def start_acquisition(self) -> None:
        self._running = True

    def stop_acquisition(self) -> None:
        self._running = False

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


def simulated_camera_for(real_cls: type) -> type:
    """Return a :class:`SimulatedCamera` subclass tailored to ``real_cls``.

    The returned class has the full generic simulated behaviour, plus a
    logging no-op stub for every public method ``real_cls`` adds beyond the
    :class:`Camera` interface (e.g. a driver's EM-gain setters), courtesy of
    :func:`~pytweezer.servers.simulated_device.simulate`. This keeps the
    simulated surface interface-complete with the real driver without any
    hand-written per-driver simulated class.
    """

    @simulate(real_cls)
    class _Simulated(SimulatedCamera):
        pass

    _Simulated.__name__ = _Simulated.__qualname__ = f"Simulated{real_cls.__name__}"
    _Simulated.image_prefix = getattr(real_cls, "image_prefix", SimulatedCamera.image_prefix)
    return _Simulated
