"""Locating a specific Hamamatsu camera among several on one DCAM host."""

import ctypes
import subprocess
import sys

import numpy as np
from pylablib.devices.DCAM import dcamapi4_lib
from pylablib.devices.DCAM.DCAM import DCAMCamera, lib, libctl

PROBE_TIMEOUT_S = 60

# Bare ctypes so the child skips pylablib's import, which pulls in pandas and
# scipy and would add ~6 s to every start. The struct is DCAMAPI_INIT.
_PROBE_SCRIPT = """
import ctypes
ctypes.windll.ole32.CoInitializeEx(None, 0)
class Init(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32), ("count", ctypes.c_int32),
                ("reserved", ctypes.c_int32), ("optionbytes", ctypes.c_int32),
                ("option", ctypes.c_void_p), ("guid", ctypes.c_void_p)]
lib = ctypes.CDLL("dcamapi.dll")
init = Init(size=ctypes.sizeof(Init))
if lib.dcamapi_init(ctypes.byref(init)) >= 0:
    lib.dcamapi_uninit()
"""


#: DCAM ``trigger_active`` for each external exposure mode.
_TRIGGER_ACTIVE = {"edge": 1, "level": 2}


def set_dcam_exposure_mode(camera: DCAMCamera, mode: str) -> None:
    """Put ``camera`` in exposure mode ``"internal"``, ``"edge"`` or ``"level"``.

    In ``"edge"`` each trigger starts an exposure of the set time; in
    ``"level"`` the trigger pulse's width sets it. Both take the trigger
    inverted (active low). A running acquisition is paused while the mode
    changes, since DCAM refuses trigger properties while capturing.

    Raises:
        ValueError: For any other ``mode``.
    """
    if mode != "internal" and mode not in _TRIGGER_ACTIVE:
        raise ValueError(
            f"exposure mode is 'internal', 'edge' or 'level', not {mode!r}"
        )
    with camera.pausing_acquisition():
        if mode == "internal":
            camera.set_trigger_mode("int")
            return
        camera.set_trigger_mode("ext")
        camera.set_attribute_value("trigger_active", _TRIGGER_ACTIVE[mode])
        camera.setup_ext_trigger(invert=True)


def get_dcam_exposure_mode(camera: DCAMCamera) -> str:
    """The mode :func:`set_dcam_exposure_mode` would have set: ``"internal"``
    for the internal trigger, otherwise ``"edge"`` or ``"level"`` by the
    camera's ``trigger_active`` (any other activation reads as ``"edge"``)."""
    if camera.get_trigger_mode() == "int":
        return "internal"
    active = int(camera.get_attribute_value("trigger_active"))
    return {value: mode for mode, value in _TRIGGER_ACTIVE.items()}.get(active, "edge")


def set_dcam_trigger_polarity(camera: DCAMCamera, polarity: str) -> None:
    """Trigger on the ``"rise"`` or ``"fall"`` edge of the external trigger.

    A running acquisition is paused while it changes.

    Raises:
        ValueError: For any other ``polarity``.
    """
    if polarity not in ("rise", "fall"):
        raise ValueError(f"trigger polarity is 'rise' or 'fall', not {polarity!r}")
    with camera.pausing_acquisition():
        camera.setup_ext_trigger(invert=polarity == "fall")


def dcam_trigger_controls(camera: DCAMCamera) -> dict:
    """The camera's ``exposure_mode``, ``trigger_source`` and ``trigger_polarity``, as the
    controls of that name hold them; they are coupled, so they are read together."""
    inverted, _delay = camera.get_ext_trigger_parameters()
    return {
        "exposure_mode": get_dcam_exposure_mode(camera),
        "trigger_source": camera.get_trigger_mode(),
        "trigger_polarity": "fall" if inverted else "rise",
    }


def dcam_roi_controls(roi: tuple) -> dict:
    """The ROI controls (and binning) for a pylablib ROI ``(hstart, hend, vstart, vend, hbin, vbin)``."""
    hstart, hend, vstart, vend, hbin, _ = roi
    return {
        "roi_x0": int(hstart),
        "roi_width": int(hend - hstart),
        "roi_y0": int(vstart),
        "roi_height": int(vend - vstart),
        "binning": int(hbin),
    }


def read_dcam_controls(camera: DCAMCamera) -> dict:
    """Every control the ImagEM and the ORCA share, read from ``camera``."""
    return {
        "exposure": float(camera.get_exposure()),
        **dcam_trigger_controls(camera),
        **dcam_roi_controls(camera.get_roi()),
    }


def read_dcam_range_timed(
    camera: DCAMCamera, start: int, stop: int
) -> tuple[np.ndarray, list[float]]:
    """Frames ``start`` to ``stop - 1`` without marking them read, and each one's
    camera timestamp in seconds. Both are empty when no acquisition is set up."""
    images, infos = camera.read_multiple_images(
        (start, stop), peek=True, return_info=True
    )
    if images is None:
        return np.asarray([]), []
    return np.asarray(images), [info.timestamp_us * 1e-6 for info in infos]


def _dcam_session():
    """A DCAM library session on the calling thread.

    With a USB3 camera (the ORCA-Fusion) attached, ``dcamapi_init`` fast-fails
    the process in ``combase.dll`` unless the thread has initialised COM. Any
    apartment will do, and initialising it again is harmless.
    """
    ctypes.windll.ole32.CoInitializeEx(None, 0)
    return libctl.temp_open()


def _normalise_serial(serial: str) -> str:
    return serial.removeprefix("S/N:").strip()


def list_dcam_cameras() -> list[dict[str, str]]:
    """Index, model and serial number of every connected DCAM camera.

    The cameras are queried without being opened, so this works while another
    process holds one of them.
    """
    idstr = dcamapi4_lib.DCAM_IDSTR
    cameras = []
    with _dcam_session():
        for index in range(libctl.cameras):
            model = lib.dcamdev_getstring(index, idstr.DCAM_IDSTR_MODEL)
            serial = lib.dcamdev_getstring(index, idstr.DCAM_IDSTR_CAMERAID)
            cameras.append(
                {
                    "index": index,
                    "model": model.decode().strip(),
                    "serial_number": _normalise_serial(serial.decode()),
                }
            )
    return cameras


def find_dcam_index(serial_number: str | None, model_prefix: str) -> int:
    """DCAM index of the camera with ``serial_number``, or, if that is None, of
    the only connected camera whose model starts with ``model_prefix``.

    DCAM indices depend on enumeration order, so they are never configured
    directly.
    """
    cameras = list_dcam_cameras()
    if serial_number is not None:
        wanted = _normalise_serial(str(serial_number))
        matches = [c for c in cameras if c["serial_number"] == wanted]
        description = f"serial number {wanted}"
    else:
        matches = [c for c in cameras if c["model"].startswith(model_prefix)]
        description = f"model {model_prefix}*"
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one DCAM camera with {description}, found {len(matches)}; "
            f"connected: {cameras}"
        )
    return matches[0]["index"]


def _probe_dcam_library() -> None:
    """Initialise the DCAM library in a throwaway process, raising if that dies.

    A broken DCAM install can kill the calling process outright (a fast-fail in
    ``combase.dll``, exit code ``0xC0000602``), which no ``except`` can catch. In
    a composite device that would take every other sub-device down with it, so
    the library is first exercised in a child that may die instead. A DCAM error
    code is not a failure here: the real open reports it by name.
    """
    probe = subprocess.run(
        [sys.executable, "-I", "-c", _PROBE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    if probe.returncode != 0:
        last_line = (probe.stderr.strip().splitlines() or ["no error output"])[-1]
        raise RuntimeError(
            f"DCAM library probe failed (exit code {probe.returncode & 0xFFFFFFFF:#x}): "
            f"{last_line}"
        )


def open_dcam_camera(serial_number: str | None, model_prefix: str) -> DCAMCamera:
    """Open the camera :func:`find_dcam_index` selects.

    A camera held by another process is not enumerated, so indices shift
    whenever the DCAM library is re-initialised; the lookup and the open
    therefore share one library session. The library is first probed in a
    child process (:func:`_probe_dcam_library`) so that a crash in it surfaces as
    a failed device instead of killing this one.
    """
    _probe_dcam_library()
    with _dcam_session():
        camera = DCAMCamera(find_dcam_index(serial_number, model_prefix))
    opened_serial = _normalise_serial(camera.get_device_info().serial_number)
    if serial_number is not None and opened_serial != _normalise_serial(
        str(serial_number)
    ):
        camera.close()
        raise RuntimeError(
            f"Opened DCAM camera {opened_serial}, expected {serial_number}"
        )
    return camera
