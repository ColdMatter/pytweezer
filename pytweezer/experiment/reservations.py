"""Which CONFIG devices an experiment reserves while it runs."""

from pytweezer.servers import device_server


class ReservationError(ValueError):
    """A declared device cannot be reserved: unknown, inactive or mis-configured."""


def _address(name: str) -> device_server.DeviceAddress:
    try:
        return device_server.resolve_address(name)
    except KeyError as error:
        raise ReservationError(error.args[0]) from None


def canonical_device(name: str) -> str:
    """Return the CONFIG spelling of ``name``."""
    return _address(name).name


def device_reservation(name: str) -> set[str]:
    """Return the canonical names reserved by using device ``name``.

    A composite reserves itself and every sub-device; a sub-device only itself.
    """
    address = _address(name)
    if address.owner_conf.get("active", True) is False:
        raise ReservationError(
            f"Device {address.name!r} is inactive in CONFIG['Devices']"
        )
    reserved = {address.name}
    if not address.is_sub_device:
        reserved.update(address.conf.get("devices") or {})
    return reserved
