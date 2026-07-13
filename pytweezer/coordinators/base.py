"""Generic :class:`Coordinator` base class for composite devices.

A *coordinator* is an extra RPC target on a composite device server (see
``pytweezer/servers/device_server.py``'s ``_make_composite``) that holds direct
references to that server's device backends. Because the backends are ordinary
Python objects in the same process, a coordinator drives them with plain method
calls: a camera frame handed to a DAC never touches a socket and is never
serialized. That is the whole point of the composite — a step that would cost an
RPC round trip per device becomes an attribute lookup.

Subclasses reach their backends through ``self.targets``, keyed by **role** — each
sub-config's ``"role"`` value, defaulting to its device name::

    class MyLoop(Coordinator):
        def step(self):
            frame = self.targets["camera"].acquire_single_frame()
            self.targets["dac"].set_voltage("Dev1/ao0", frame.mean())

Roles exist so a coordinator asks for ``"camera"`` while the device it gets is
called ``"Rb Feedback Cam"`` in config and served under that name to RPC clients.
One coordinator class therefore works across rigs whose devices are named
differently.

Register the subclass in ``device_server.COORDINATOR_REGISTRY`` and name it from
the composite's ``"coordinator"`` config key.

Three constraints on subclasses:

* **Every public method is a synchronous RPC method and stalls the whole server
  for its duration.** The sipyco server is single-threaded asyncio and runs a
  plain ``def`` target method inline on the event loop, so while a coordinator
  method runs, no target on that server — camera, DAC, or coordinator — answers
  anything. Keep methods bounded; a batch method should take an iteration count
  rather than free-running.
* **Return values cross a PYON boundary.** Return plain types, dicts, lists and
  numpy arrays; never exception objects or backend handles. Returning a camera
  frame reintroduces the serialization cost the coordinator exists to avoid.
* **Do not define** ``__call__``. ``sipyco.pc_rpc.Server`` treats a callable
  target as a per-connection factory and invokes it instead of serving it.
"""

from pytweezer.logging_utils import get_logger

LOGGER = get_logger("coordinator")


class Coordinator:
    """Base for an in-process device coordinator.

    ``targets`` maps role to backend object; ``conf`` is the composite device's
    config entry.
    """

    def __init__(self, targets: dict, conf: dict):
        self.targets = targets
        self.conf = conf or {}

    def require_role(self, role):
        """Return the backend registered under ``role``, or raise a clear error.

        The error names the roles the composite actually provides and how to add
        the missing one, since a coordinator/config role mismatch is otherwise an
        opaque ``KeyError`` deep in construction.
        """
        try:
            return self.targets[role]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} needs a sub-device with role {role!r}; this "
                f"composite provides roles: {sorted(self.targets) or '(none)'}. Set "
                f'"role": {role!r} on the relevant entry in the composite\'s "devices" block.'
            ) from None

    def shutdown(self) -> None:
        """Release anything the coordinator holds on its backends.

        Called before the backends themselves are torn down. Default is a no-op;
        override to stop acquisitions, zero outputs, and so on.
        """
