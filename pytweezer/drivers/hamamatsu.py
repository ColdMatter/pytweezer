"""Locating a specific Hamamatsu camera among several on one DCAM host."""

from pylablib.devices.DCAM import dcamapi4_lib
from pylablib.devices.DCAM.DCAM import DCAMCamera, lib, libctl


def _normalise_serial(serial: str) -> str:
    return serial.removeprefix("S/N:").strip()


def list_dcam_cameras() -> list[dict[str, str]]:
    """Index, model and serial number of every connected DCAM camera.

    The cameras are queried without being opened, so this works while another
    process holds one of them.
    """
    idstr = dcamapi4_lib.DCAM_IDSTR
    cameras = []
    with libctl.temp_open():
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


def open_dcam_camera(serial_number: str | None, model_prefix: str) -> DCAMCamera:
    """Open the camera :func:`find_dcam_index` selects.

    A camera held by another process is not enumerated, so indices shift
    whenever the DCAM library is re-initialised; the lookup and the open
    therefore share one library session.
    """
    with libctl.temp_open():
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
