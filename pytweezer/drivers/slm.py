"""Meadowlark Blink Plus SLM driver.

Wraps the Blink C-wrapper SDK (a ctypes DLL) as a single-target device backend,
so a Spatial Light Modulator is a first-class device: ``get_device("Rb SLM")``.
This replaces the old standalone ZMQ ``SLMServer.py`` — the point of making it a
device is that a coordinator sharing its process (e.g. rearrangement) can call
``slm.update_mask(frame)`` directly, with no socket between the GPU-computed
phasemask and the hardware.

The Blink DLL and LUT paths are Windows-only and specific to the SLM PC; they are
loaded lazily in :meth:`_connect`, so importing this module never requires the SDK
to be present. Use :class:`SimulatedSLM` (``simulate: True``) anywhere else.

Masks are ``uint8`` arrays of shape ``(height, width)`` (or a flat
``height*width`` buffer) already mapped through the phase LUT by the caller; the
hardware applies its own voltage LUT on top.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from pytweezer.servers.simulated_device import simulate
from pytweezer.logging_utils import get_logger

LOGGER = get_logger("slm")

#: Defaults matching the lab's Blink Plus install; overridable via config.
DEFAULT_SDK_DLL = "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper"
DEFAULT_LUT_FILE = "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\slm40_at852.LUT"


class SLM:
    """Blink Plus SLM over its ctypes C-wrapper SDK.

    The live hardware handle (the loaded DLL) lives on :attr:`_lib`; ``None`` means
    disconnected. One board only (board number defaults to 1).
    """

    def __init__(
        self,
        sdk_dll: str = DEFAULT_SDK_DLL,
        lut_file: Optional[str] = DEFAULT_LUT_FILE,
        board_number: int = 1,
        timeout_ms: int = 5000,
        wait_for_trigger: bool = False,
        flip_immediate: bool = False,
        output_pulse: bool = False,
    ):
        self.sdk_dll = sdk_dll
        self.lut_file = lut_file
        self.board_number = int(board_number)
        self.timeout_ms = int(timeout_ms)
        self.wait_for_trigger = bool(wait_for_trigger)
        self.flip_immediate = bool(flip_immediate)
        self.output_pulse = bool(output_pulse)

        self._lib = None
        self.width = None
        self.height = None
        self.depth = None
        self._connect()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        # Imported lazily: ctypes is stdlib, but loading the Blink DLL is what
        # actually requires the SDK to be installed on this machine.
        from ctypes import CDLL, c_double, c_int, c_uint, byref, POINTER, c_ubyte, cdll

        cdll.LoadLibrary(self.sdk_dll)
        lib = CDLL("Blink_C_wrapper")
        lib.Read_SLM_temperature.argtypes = [c_int]
        lib.Read_SLM_temperature.restype = c_double

        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        lib.Create_SDK(byref(num_boards_found), byref(constructed_okay))
        if constructed_okay.value == 0:
            raise RuntimeError("Blink SDK did not construct successfully")
        if num_boards_found.value != 1:
            raise RuntimeError(
                f"Expected exactly one SLM board, found {num_boards_found.value}"
            )

        self.width = lib.Get_image_width(self.board_number)
        self.height = lib.Get_image_height(self.board_number)
        self.depth = lib.Get_image_depth(self.board_number)  # bits per pixel
        serial = lib.Read_Serial_Number(self.board_number)
        LOGGER.info(
            "SLM board %d: %dx%d @ %d-bit, serial %s, %.3f degC",
            self.board_number, self.width, self.height, self.depth, serial,
            lib.Read_SLM_temperature(self.board_number),
        )

        lib.SetWaitForTrigger(self.board_number, int(self.wait_for_trigger))
        lib.SetFlipImmediate(self.board_number, int(self.flip_immediate))
        lib.SetOutputPulse(self.board_number, int(self.output_pulse))

        if self.lut_file:
            status = lib.Load_LUT_file(self.board_number, self.lut_file.encode())
            if status != 1:
                raise RuntimeError(f"Failed to load LUT file {self.lut_file!r}")
            LOGGER.info("Loaded LUT %s", self.lut_file)

        self._lib = lib
        self._poi = POINTER(c_ubyte)  # cached for hot-path mask writes
        # Start from a blank (zero-phase) frame.
        self.update_mask(np.zeros((self.height, self.width), dtype=np.uint8))

    def is_connected(self) -> bool:
        return self._lib is not None

    def _require_slm(self) -> None:
        if self._lib is None:
            raise RuntimeError("SLM connection has been relinquished")

    def close(self) -> None:
        if self._lib is not None:
            try:
                self._lib.Delete_SDK()
            except Exception:
                LOGGER.exception("Failed to close Blink SDK cleanly")
        self._lib = None

    # ------------------------------------------------------------------ #
    # Mask / sequence operations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_c_uint8(mask_array: np.ndarray) -> np.ndarray:
        mask = np.ascontiguousarray(mask_array, dtype=np.uint8)
        return mask

    def update_mask(self, mask_array: np.ndarray) -> None:
        """Write one phase mask to the SLM (blocks until the write completes).

        Raises ``RuntimeError`` if the DMA or write-complete handshake fails.
        """
        self._require_slm()
        mask = self._as_c_uint8(mask_array)
        ret0 = self._lib.Write_image(
            self.board_number, mask.ctypes.data_as(self._poi), self.timeout_ms
        )
        if ret0 != 1:
            raise RuntimeError("SLM Write_image DMA failed")
        ret1 = self._lib.ImageWriteComplete(self.board_number, self.timeout_ms)
        if ret1 != 1:
            raise RuntimeError("SLM ImageWriteComplete failed")
        LOGGER.info("Updated mask to SLM board %d", mask.shape[1], mask.shape[0], self.board_number)

    def preload_sequence(self, mask_sequence: np.ndarray) -> None:
        """Upload a whole ``(n, H, W)`` sequence into the SLM's on-board memory.

        Does not display anything by itself (1024x1024 board only, up to 752
        frames); the upload time is logged. Raises on failure.
        """
        self._require_slm()
        seq = self._as_c_uint8(mask_sequence)
        list_length = int(seq.shape[0])
        t0 = time.perf_counter()
        ret = self._lib.PreLoad_sequence(
            self.board_number, seq.ctypes.data_as(self._poi), list_length, self.timeout_ms
        )
        if ret != 1:
            raise RuntimeError("SLM PreLoad_sequence failed")
        LOGGER.info(
            "Preloaded %d frames in %.3f ms.", list_length, (time.perf_counter() - t0) * 1000
        )

    def start_auto_increment(self, list_length: int) -> None:
        """Arm hardware auto-increment over a preloaded sequence.

        After :meth:`preload_sequence`, this makes the SLM listen for external
        triggers: frame 0 is live once armed, then each trigger advances to the
        next preloaded frame, looping back to frame 0 after the last one, until
        :meth:`stop_auto_increment`. So an ``n``-frame sequence needs ``n-1``
        triggers to end on the final frame; an ``n``-th trigger wraps to frame 0.

        1024x1024 board with firmware rev >= 2.4 only. Raises on failure.
        """
        self._require_slm()
        ret = self._lib.StartAutoIncrement(self.board_number, int(list_length))
        if ret != 1:
            raise RuntimeError("SLM StartAutoIncrement failed")
        LOGGER.info("Auto-increment armed over %d frames.", int(list_length))

    def stop_auto_increment(self) -> None:
        """Stop hardware auto-increment (see :meth:`start_auto_increment`).

        Leaves whichever frame was last triggered live on the SLM. Raises on failure.
        """
        self._require_slm()
        ret = self._lib.StopAutoIncrement(self.board_number)
        if ret != 1:
            raise RuntimeError("SLM StopAutoIncrement failed")
        LOGGER.info("Auto-increment stopped.")

    def run_sequence(self, mask_sequence: np.ndarray, fps: float = 1.0) -> None:
        """Display each frame of a ``(n, H, W)`` sequence at ``fps`` (software timed).

        Software-timed writes; for hardware-triggered playback use
        :meth:`preload_sequence`, :meth:`start_auto_increment`, and drive the
        SLM's external trigger.
        """
        self._require_slm()
        seq = self._as_c_uint8(mask_sequence)
        n = int(seq.shape[0])
        period = 1.0 / fps if fps else 0.0
        for i in range(n):
            self.update_mask(seq[i])
            if period:
                time.sleep(period)
            LOGGER.info("Updated sequence frame %d/%d", i + 1, n)

    def get_temperature(self) -> float:
        self._require_slm()
        return float(self._lib.Read_SLM_temperature(self.board_number))

    def get_dimensions(self) -> dict:
        return {"width": self.width, "height": self.height, "depth": self.depth}


class SimulatedSLM:
    """Synthetic SLM: remembers the last mask written and counts frames.

    Interface-complete with :class:`SLM` via :func:`simulate`, so anything the
    real driver adds is auto-stubbed. Fixed 1024x1024 8-bit geometry, matching the
    lab's board.
    """

    def __init__(self, width: int = 1024, height: int = 1024, depth: int = 8, **_ignored):
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        self.last_mask = None
        self.frames_written = 0
        self.preloaded_frames = 0
        self.auto_increment_length = 0
        self._temp = 35.0
        self._closed = False

    def is_connected(self) -> bool:
        return not self._closed

    def update_mask(self, mask_array: np.ndarray) -> None:
        self.last_mask = np.ascontiguousarray(mask_array, dtype=np.uint8)
        self.frames_written += 1

    def preload_sequence(self, mask_sequence: np.ndarray) -> None:
        self.preloaded_frames = int(np.asarray(mask_sequence).shape[0])

    def start_auto_increment(self, list_length: int) -> None:
        self.auto_increment_length = int(list_length)

    def stop_auto_increment(self) -> None:
        self.auto_increment_length = 0

    def run_sequence(self, mask_sequence: np.ndarray, fps: float = 1.0) -> None:
        for frame in np.asarray(mask_sequence, dtype=np.uint8):
            self.update_mask(frame)

    def get_temperature(self) -> float:
        return float(self._temp)

    def get_dimensions(self) -> dict:
        return {"width": self.width, "height": self.height, "depth": self.depth}

    def close(self) -> None:
        self._closed = True


SimulatedSLM = simulate(SLM)(SimulatedSLM)
