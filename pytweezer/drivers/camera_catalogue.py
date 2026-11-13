"""Which CONFIG devices are cameras, and how to look one up by name."""

from __future__ import annotations

import re

from pytweezer.drivers.camera_base import Camera
from pytweezer.logging_utils import get_logger
from pytweezer.servers import device_server

logger = get_logger(__name__)


def camera_names() -> list[str]:
    """Every CONFIG device whose driver is a :class:`Camera`."""
    names = []
    for address in device_server.device_index().values():
        path = address.conf.get("class")
        if not path:
            continue
        try:
            driver = device_server._load(path)
        except Exception:
            logger.warning(
                "Cannot import the driver of %r", address.name, exc_info=True
            )
            continue
        if isinstance(driver, type) and issubclass(driver, Camera):
            names.append(address.name)
    return sorted(names)


def camera_conf(name: str) -> dict:
    """The CONFIG entry of camera ``name``, a composite's sub-device included."""
    return device_server.resolve_address(name).conf


def camera_driver(name: str) -> type[Camera]:
    """The real driver class of camera ``name``, even when it runs simulated."""
    return device_server._load(camera_conf(name)["class"])


def camera_key(name: str) -> str:
    """``name`` as a saved-parameter key prefix: ``"Rb HamCam"`` becomes ``"rb_hamcam"``."""
    return re.sub(r"\W+", "_", name).strip("_").lower()
