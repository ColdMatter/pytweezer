"""Experiments that run a MOTMaster sequence once per point.

::

    class TofImaging(MotMasterExperiment):
        sequencer = Device("Rb MotMaster")
        motmaster_script = "RbTweezerBasic"

        tof = MotMasterInteger(100, parameter="tDelay1", unit="ms/100")
        coil_current = MotMasterNumber(1.5, unit="A")

The MOTMaster parameters are ordinary scannable arguments; each point sends
all of them to the script. The default :meth:`~MotMasterExperiment.run_point`
just runs the sequence; override it to do more around
:meth:`~MotMasterExperiment.run_sequence` (arm a camera, read it out, ...).
"""

from typing import Any, ClassVar

from pytweezer.experiment.arguments import Integer, Number
from pytweezer.experiment.experiment import Experiment, logger


class MotMasterParameter:
    """Marks an argument as a MOTMaster script parameter (default group "MOTMaster")."""

    def __init__(self, *args: Any, parameter: str | None = None, **kwargs: Any) -> None:
        self.parameter = parameter
        kwargs.setdefault("group", "MOTMaster")
        super().__init__(*args, **kwargs)

    @property
    def motmaster_name(self) -> str:
        return self.parameter or self.name

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {"motmaster_parameter": self.motmaster_name}


class MotMasterNumber(MotMasterParameter, Number):
    """A float MOTMaster parameter (sent to .NET as ``Double``)."""


class MotMasterInteger(MotMasterParameter, Integer):
    """An integer MOTMaster parameter (sent to .NET as ``Int32``).

    The script's parameter type must match: a float sent for an ``int``
    parameter is rejected by MOTMaster.
    """


class MotMasterExperiment(Experiment):
    """Base class for an experiment driven by one MOTMaster script.

    Subclasses must declare ``sequencer = Device("<MotMaster device>")`` and
    set ``motmaster_script``. :meth:`prepare` configures the sequencer
    completely on every task, since its settings persist between tasks.
    Repeated shots come from the scan's ``repetitions``, giving one stored
    point per shot; ``motmaster_iterations`` only exists for scripts that
    must run several times per point.
    """

    motmaster_script: ClassVar[str] = ""
    motmaster_iterations: ClassVar[int] = 1
    #: Whether MOTMaster also saves its own data files.
    motmaster_save: ClassVar[bool] = False
    #: Wait for an external trigger before each run; ``None`` leaves it as is.
    motmaster_triggered: ClassVar[bool | None] = None

    @classmethod
    def motmaster_arguments(cls) -> dict[str, MotMasterParameter]:
        return {
            name: argument
            for name, argument in cls.arguments().items()
            if isinstance(argument, MotMasterParameter)
        }

    def motmaster_parameters(self) -> dict[str, Any]:
        """``{script parameter: current value}`` for every declared MOTMaster parameter."""
        return {
            argument.motmaster_name: getattr(self, name)
            for name, argument in self.motmaster_arguments().items()
        }

    def prepare(self) -> None:
        if "sequencer" not in self.devices():
            raise TypeError(
                f"{type(self).__name__} must declare sequencer = Device(...)"
            )
        if not self.motmaster_script:
            raise TypeError(f"{type(self).__name__} must set motmaster_script")
        sequencer = self.sequencer
        sequencer.set_motmaster_experiment(self.motmaster_script)
        sequencer.set_run_until_stopped(False)
        sequencer.set_iterations(self.motmaster_iterations)
        sequencer.set_save_toggle(self.motmaster_save)
        if self.motmaster_triggered is not None:
            sequencer.set_trigger_mode(self.motmaster_triggered)
        self.record("motmaster_script", self.motmaster_script)
        try:
            defaults = sequencer.get_params()
        except Exception as error:
            # Provenance only: worth a warning, not worth refusing to run.
            logger.warning("Could not read MOTMaster script defaults: %s", error)
            defaults = {"error": f"could not read: {error}"}
        self.record("motmaster_script_defaults", defaults)

    def run_sequence(self, **overrides: Any) -> None:
        """Run the script once (blocking) with the current parameters plus ``overrides``."""
        self.sequencer.start_motmaster_experiment(
            self.motmaster_parameters() | overrides
        )

    def run_point(self) -> None:
        self.run_sequence()
