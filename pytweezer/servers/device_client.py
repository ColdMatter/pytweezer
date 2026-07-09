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

:func:`get_device_async` is the ``asyncio`` counterpart, for driving two or
more device servers concurrently (e.g. starting two MotMaster sequences in
parallel)::

    async def main():
        mm1 = await get_device_async("Rb MotMaster Server")
        mm2 = await get_device_async("CaF MotMaster Server")
        try:
            await asyncio.gather(
                mm1.start_motmaster_experiment(),
                mm2.start_motmaster_experiment(),
            )
        finally:
            await mm1.close_rpc()
            await mm2.close_rpc()

    asyncio.run(main())
"""

from sipyco.pc_rpc import AsyncioClient, AutoTarget
from sipyco.pc_rpc import Client as RPCClient

from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers.device_server import resolve_device

from pytweezer.logging_utils import get_logger

logger = get_logger("device client")


def get_device_config(name):
    """Return the ``CONFIG["Devices"]`` entry for ``name``.

    Matches the config key exactly first, then falls back to a
    whitespace-/case-insensitive match (same rule as ``device_server.resolve_device``)
    so callers can pass e.g. ``RbHamCam`` or ``rb hamcam`` for the config key
    ``"Rb HamCam"``. Raises ``KeyError`` with the list of known devices if
    ``name`` doesn't match anything.
    """
    _, conf = resolve_device(name)
    return conf


def _resolve_host_port(name, host, port):
    device_conf = get_device_config(name)
    host = host or device_conf.get("host", "127.0.0.1")
    if port is None:
        port = device_conf.get("port")
    if port is None:
        raise ValueError(f"Device {name!r} has no 'port' configured and none was given")
    return host, int(port)


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
    host, port = _resolve_host_port(name, host, port)
    logger.debug("Connecting to device %r at %s:%s", name, host, port)
    return RPCClient(host, port, target_name=target_name, timeout=timeout)


async def get_device_async(name, host=None, port=None, target_name=AutoTarget):
    """Build an :class:`~sipyco.pc_rpc.AsyncioClient` for the device named ``name``.

    Same config lookup as :func:`get_device`, but every RPC method on the
    returned client is a coroutine. Use this (with ``asyncio.gather``) to drive
    two independent device servers concurrently — e.g. starting two MotMaster
    sequences in parallel — since a blocking :func:`get_device` client waits for
    the full RPC round trip, including the remote call's execution time, before
    the next line runs.

    Call ``await client.close_rpc()`` when finished to release the socket.
    """
    host, port = _resolve_host_port(name, host, port)
    logger.debug("Connecting (async) to device %r at %s:%s", name, host, port)
    client = AsyncioClient()
    await client.connect_rpc(host, port, target_name=target_name)
    return client
