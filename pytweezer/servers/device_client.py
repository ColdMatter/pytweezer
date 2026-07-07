"""Generic device RPC-client factory.

Every device in ``CONFIG["Devices"]`` runs a sipyco RPC server exposing a single
target (``motmaster``, ``camera``, ...). Rather than hand-writing a client class
per device, :func:`get_device` looks the device up by its config name, reads its
``host``/``port``, and returns a transparent sipyco proxy. Because each server
exposes exactly one target, :class:`~sipyco.pc_rpc.AutoTarget` selects it
automatically — the caller never needs to know the target name.

Example::

    cam = get_device("Rb HamCam")
    cam.acquire()                 # remote method call, transparently proxied
    cam.close_rpc()               # release the socket when done
"""

from sipyco.pc_rpc import AutoTarget
from sipyco.pc_rpc import Client as RPCClient

from pytweezer.servers.configreader import ConfigReader

from pytweezer.logging_utils import get_logger

logger = get_logger("device client")


def get_device_config(name):
    """Return the ``CONFIG["Devices"]`` entry for ``name``.

    Raises ``KeyError`` with the list of known devices if ``name`` is unknown.
    """
    devices = ConfigReader.getConfiguration().get("Devices", {})
    try:
        return devices[name]
    except KeyError:
        available = ", ".join(sorted(devices)) or "(none)"
        raise KeyError(
            f"Device {name!r} not found in config. Available devices: {available}"
        ) from None


def get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None):
    """Build a sipyco RPC client for the device named ``name`` in the config.

    Parameters
    ----------
    name:
        Key in ``CONFIG["Devices"]`` (e.g. ``"Rb MotMaster Server"``).
    host, port:
        Optional overrides; default to the device's configured ``host``/``port``.
    target_name:
        RPC target to bind. Defaults to :class:`~sipyco.pc_rpc.AutoTarget`, which
        auto-selects the server's sole target — override only for a multi-target
        server.
    timeout:
        Socket timeout in seconds (``None`` = block indefinitely).

    Returns
    -------
    sipyco.pc_rpc.Client
        A transparent proxy; remote methods are called as normal attributes.
        Call ``.close_rpc()`` when finished to release the socket.
    """
    device_conf = get_device_config(name)
    host = host or device_conf.get("host", "127.0.0.1")
    if port is None:
        port = device_conf.get("port")
    if port is None:
        raise ValueError(f"Device {name!r} has no 'port' configured and none was given")
    port = int(port)

    logger.debug("Connecting to device %r at %s:%s", name, host, port)
    return RPCClient(host, port, target_name=target_name, timeout=timeout)
