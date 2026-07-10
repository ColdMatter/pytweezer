"""Camera-to-DAC feedback: the worked example of a composite coordinator.

Grabs a frame, reduces it to a scalar, and drives a DAC channel with a
proportional correction. The frame never leaves the process — only the scalar
summary crosses the RPC boundary — which is what makes iteration cheap enough to
sit in a control loop.

Wire it up with a composite device (``pytweezer/configuration/config.py``)::

    "Rb Feedback Rig": {
        "driver": "composite",
        "devices": {
            "Rb Feedback Cam": {"driver": "imagemx2", "role": "camera", ...},
            "Rb Feedback DAC": {"driver": "nidac", "role": "dac", "channels": ["Dev1/ao0"]},
        },
        "coordinator": "camera_dac_feedback",
    }

then drive it from a script::

    coord = get_device("Rb Feedback Rig")   # the composite resolves to its coordinator
    coord.run_n(50, setpoint=130.0, gain=0.01, channel="Dev1/ao0")

The ``"role"`` keys are what this class looks its backends up by; the device names
are what RPC clients address them by (``get_device("Rb Feedback Cam")``).

The control law is deliberately trivial (proportional only, on the frame mean).
Subclass or edit :meth:`measure` and :meth:`control` for a real experiment.
"""

import time

from pytweezer.coordinators.base import Coordinator
from pytweezer.logging_utils import get_logger

LOGGER = get_logger("coordinator")


class CameraDACFeedback(Coordinator):
    """Close a proportional loop from a camera's frame mean onto a DAC channel.

    ``run_n`` runs the loop entirely in-process: one RPC issues the whole batch,
    and each iteration costs a direct camera grab and a direct DAC write. The
    server answers nothing else until the batch finishes, so keep ``n`` bounded.
    """

    #: Output range clamped around every DAC write, in volts.
    default_limits = (-10.0, 10.0)

    #: Roles this coordinator expects the composite's sub-devices to declare.
    camera_role = "camera"
    dac_role = "dac"

    def __init__(self, targets, conf):
        super().__init__(targets, conf)
        self.camera = self._require_role(self.camera_role)
        self.dac = self._require_role(self.dac_role)
        self._armed = False

    def _require_role(self, role):
        try:
            return self.targets[role]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} needs a sub-device with role {role!r}; this "
                f"composite provides roles: {sorted(self.targets) or '(none)'}. Set "
                f'"role": {role!r} on the relevant entry in the composite\'s "devices" block.'
            ) from None

    # ---- camera arming -------------------------------------------------- #

    def arm(self, nframes: int = 1, acq_mode: str = "sequence") -> None:
        """Configure and start the camera. Idempotent."""
        if self._armed:
            return
        self.camera.setup_acquisition(acq_mode, int(nframes))
        self.camera.start_acquisition()
        self._armed = True

    def disarm(self) -> None:
        """Stop the camera if this coordinator armed it. Idempotent."""
        if not self._armed:
            return
        self.camera.stop_acquisition()
        self._armed = False

    def is_armed(self) -> bool:
        return self._armed

    # ---- control law ---------------------------------------------------- #

    def measure(self) -> float:
        """Reduce one frame to the scalar the loop controls on."""
        frame = self.camera.acquire_single_frame()
        return float(frame.mean())

    def control(self, measured: float, setpoint: float, gain: float, limits=None) -> float:
        """Proportional correction, clamped to the output range."""
        low, high = limits or self.default_limits
        return max(low, min(high, gain * (setpoint - measured)))

    # ---- RPC surface ----------------------------------------------------- #

    def image_to_dac(self, setpoint: float, gain: float, channel: str, limits=None) -> dict:
        """Run one measure/control/actuate step and return its summary.

        Returns ``{"mean", "voltage", "t"}``. The frame itself is intentionally not
        returned — serializing it back over RPC would cost more than the step.
        """
        self.arm()
        measured = self.measure()
        voltage = self.control(measured, float(setpoint), float(gain), limits)
        self.dac.set_voltage(channel, voltage)
        return {"mean": measured, "voltage": voltage, "t": time.time()}

    def run_n(
        self, n: int, setpoint: float, gain: float, channel: str, limits=None
    ) -> list:
        """Run :meth:`image_to_dac` ``n`` times in-process; return every summary.

        One RPC for the whole batch, so per-iteration cost is direct Python calls
        with no serialization. The batch cannot be aborted once started and blocks
        every other target on this server while it runs.
        """
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.arm()
        LOGGER.info("camera->dac feedback: %d iterations on %s", n, channel)
        return [
            self.image_to_dac(setpoint, gain, channel, limits=limits) for _ in range(n)
        ]

    def shutdown(self) -> None:
        self.disarm()
