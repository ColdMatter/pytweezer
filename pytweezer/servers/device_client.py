"""Generic device RPC-client factory.

Every device in ``CONFIG["Devices"]`` is served by a sipyco RPC server. Rather than
hand-writing a client class per device, :func:`get_device` looks the device up by
its config name, resolves the endpoint that serves it, and returns a transparent
sipyco proxy::

    cam = get_device("Rb HamCam")
    cam.acquire()                 # remote method call, transparently proxied
    cam.close_rpc()               # release the socket when done

**Every device is addressed by its own name**, whether it has a server to itself or
shares one with other devices. A composite device (``"driver": "composite"``) runs
several devices in one process so they can drive each other without RPC, but its
sub-devices are named in config exactly like any other device and reached the same
way — the caller never needs to know which process a device lives in, or that
sipyco targets exist::

    cam = get_device("Rb Feedback Cam")     # sub-device of the "Rb Feedback Rig" composite
    dac = get_device("Rb Feedback DAC")     # ...same process, same port, different target
    rig = get_device("Rb Feedback Rig")     # the composite itself -> its coordinator

``target_name`` is resolved from the config and only needs passing to reach a target
the config doesn't name.

For driving two or more device servers concurrently (e.g. starting two MotMaster
sequences in parallel), prefer :func:`pytweezer.parallel.run_parallel` — it runs
blocking :func:`get_device` clients in threads with no ``async``/``await`` and
works from the GUI. :func:`get_device_async` is the lower-level ``asyncio``
counterpart if you want to drive the servers with coroutines directly::

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
from pytweezer.servers.device_server import resolve_address

from pytweezer.logging_utils import get_logger

logger = get_logger("device client")


def get_device_config(name):
    """Return the config entry for the device named ``name``.

    Matches leniently (whitespace-/case-insensitively), and finds composite
    sub-devices as well as top-level ones. Raises ``KeyError`` listing the known
    devices if nothing matches.

    A sub-device's entry carries only its own driver settings; ``host``/``port``
    belong to the composite that serves it. Use :func:`get_device` (or
    ``device_server.resolve_address``) rather than reading them off this dict.
    """
    return resolve_address(name).conf


def _resolve_endpoint(name, host, port, target_name):
    """Return ``(host, port, target_name)`` for connecting to ``name``.

    Explicit ``host``/``port``/``target_name`` arguments win; otherwise they come
    from the device's address. A sub-device's endpoint is its composite's
    ``host``/``port`` plus the target that selects it.
    """
    address = resolve_address(name)
    owner = address.owner_conf

    host = host or owner.get("host", "127.0.0.1")
    if port is None:
        port = owner.get("port")
    if port is None:
        raise ValueError(f"Device {name!r} has no 'port' configured and none was given")

    if target_name is AutoTarget:
        if address.target_name is not None:
            target_name = address.target_name
        elif owner.get("driver") == "composite":
            sub_devices = ", ".join(sorted(owner.get("devices") or {})) or "(none)"
            raise KeyError(
                f"Composite device {name!r} has no coordinator, so there is nothing to "
                f"address by that name. Name one of its sub-devices instead: {sub_devices}"
            )

    return host, int(port), target_name


#: Prefix of the ValueError sipyco's ``_validate_target_name`` raises when
#: ``AutoTarget`` meets a server with more than one target.
_MULTI_TARGET_PREFIX = "Server has multiple targets:"


def _reraise_multi_target(name, exc):
    """Turn sipyco's bare multi-target ValueError into an actionable one.

    Re-raises unchanged if ``exc`` is some other ValueError.
    """
    message = str(exc)
    if not message.startswith(_MULTI_TARGET_PREFIX):
        raise exc
    targets = sorted(message[len(_MULTI_TARGET_PREFIX) :].split())
    listed = ", ".join(repr(target) for target in targets)
    example = targets[0] if targets else "camera"
    raise ValueError(
        f"Device {name!r} serves multiple RPC targets ({listed}); AutoTarget cannot "
        f"choose between them. Name one explicitly, e.g. "
        f"get_device({name!r}, target_name={example!r})."
    ) from exc


def get_device(name, host=None, port=None, target_name=AutoTarget, timeout=None):
    """Build a sipyco RPC client for the device named ``name`` in the config.

    Parameters
    ----------
    name:
        Any device name: a top-level ``CONFIG["Devices"]`` key (``"Rb HamCam"``), a
        composite's sub-device (``"Rb Feedback Cam"``), or a composite itself
        (which resolves to its coordinator).
    host, port:
        Optional overrides; default to the serving process's ``host``/``port``.
    target_name:
        RPC target to bind. Defaults to :class:`~sipyco.pc_rpc.AutoTarget`, which
        selects the right target automatically — the server's sole target for a
        plain device, or the one belonging to ``name`` on a composite. Override
        only to reach a target the config doesn't name.
    timeout:
        Socket timeout in seconds (``None`` = block indefinitely).

    Returns
    -------
    sipyco.pc_rpc.Client
        A transparent proxy; remote methods are called as normal attributes.
        Call ``.close_rpc()`` when finished to release the socket.
    """
    host, port, target_name = _resolve_endpoint(name, host, port, target_name)
    logger.debug("Connecting to device %r at %s:%s (target %s)", name, host, port, target_name)
    try:
        return RPCClient(host, port, target_name=target_name, timeout=timeout)
    except ValueError as exc:
        _reraise_multi_target(name, exc)


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
    host, port, target_name = _resolve_endpoint(name, host, port, target_name)
    logger.debug("Connecting (async) to device %r at %s:%s", name, host, port)
    client = AsyncioClient()
    try:
        await client.connect_rpc(host, port, target_name=target_name)
    except ValueError as exc:
        _reraise_multi_target(name, exc)
    return client
