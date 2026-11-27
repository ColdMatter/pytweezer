"""Experiments that run a MOTMaster sequence once per point.

::

    class TofImaging(MotMasterExperiment):
        rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")

        tof = MotMasterInteger(100, parameter="tDelay1", unit="ms/100")

An experiment declares one :class:`MotMaster`: the device and the script it
runs. Parameters declared with :class:`MotMasterNumber`/:class:`MotMasterInteger`
are always shown; any other script parameter is set by passing its name as an
argument (``tPulse``), which the Experiments form offers through a search box.
A name the experiment declares as an argument is always that argument, never a
script parameter of the same name. Only MOTMaster knows its script's
parameters, so a misspelt one is accepted at submit and refused in
:meth:`MotMasterExperiment.prepare`, before any sequence runs. Parameters not
set are left at the script's own defaults.

``saved=`` binds script parameters to saved parameters, read once in
``prepare`` and sent with every run. A bare name binds the saved key of the same
name; a dict maps a script parameter to a differently named key::

    rb = MotMaster(
        "Rb MotMaster",
        script="RbTweezerBasic",
        saved=("tMOTLoadRepVCO", {"tImgCoolVCO1": "tImgCoolVCO"}),
    )

A value set in the form (or scanned) beats the saved one, and an override passed
to :meth:`MotMasterExperiment.run_motmaster_sequence` beats both. The saved values sent
are recorded as the constant ``saved_sequence``, keyed by script parameter.
With ``use_motmaster_parameter_overrides`` off, none are sent and the script's
own values run.

Every run stores MOTMaster's record of the sequence it built (every event,
the full parameter set and the script's text) in the measurement file; see
:mod:`pytweezer.experiment.storage` and ``Dataset.motmaster_sequence``. A
MOTMaster built without that record still runs; the file lists the run as
unrecorded.

Two machines are run together by MOTMaster's linked mode: the primary claims its
secondary itself and fires it by TTL, so the experiment declares only the
primary, and every parameter (both machines', plus ``SecondaryTriggerDelay``)
belongs to its script. The secondary is named by the primary's ``linked_partner``
CONFIG key and is reserved along with it::

    class LinkedLoad(MotMasterExperiment):
        mm = MotMaster("CaF MotMaster", script="LinkedTweezerLoad", linked=True)
        tof = MotMasterInteger(100, parameter="tDelay1")

``script_from=`` takes the script from a :class:`MotMasterScript` argument
instead, chosen when the task is submitted; it cannot be scanned, since
:meth:`MotMasterExperiment.prepare` loads the script once. The Experiments form
offers the scripts in the MOTMaster's own script folder, asked of the device;
a name that isn't there is refused in ``prepare``. ``linked_from=`` likewise
takes linked mode from a ``Bool`` argument, and reserves the secondary whether
or not the task links it::

    class RunScript(MotMasterExperiment):
        script = MotMasterScript("MERGE.CaFTweezer")
        linked = Bool(True, scannable=False)
        caf = MotMaster("CaF MotMaster", script_from="script", linked_from="linked")
"""

import difflib
import json
from collections.abc import Iterable
from typing import Any

from pytweezer.experiment.arguments import (
    Argument,
    Bool,
    Device,
    Integer,
    Number,
    Text,
    collect,
)
from pytweezer.experiment.client import (
    ManagerError,
    ManagerUnavailable,
    get_saved_parameters,
)
from pytweezer.experiment.experiment import Experiment, logger


class MotMaster(Device):
    """A MOTMaster sequencer device and the script it runs for this experiment."""

    def __init__(
        self,
        device_name: str,
        *,
        script: str | None = None,
        script_from: str | None = None,
        linked: bool = False,
        linked_from: str | None = None,
        iterations: int = 1,
        save: bool = False,
        saved: Iterable[str | dict[str, str]] = (),
        timeout: float | None = None,
    ) -> None:
        super().__init__(device_name, timeout=timeout)
        if (script is None) == (script_from is None):
            raise TypeError("MotMaster takes one of script= and script_from=")
        if linked and linked_from is not None:
            raise TypeError("MotMaster takes at most one of linked= and linked_from=")
        self.script = script
        #: Name of the :class:`MotMasterScript` argument the script is read from, or ``None``.
        self.script_from = script_from
        self.linked = linked
        #: Name of the ``Bool`` argument linked mode is read from, or ``None``.
        self.linked_from = linked_from
        self.iterations = iterations
        self.save = save
        #: ``{script parameter: saved key}``.
        self.saved = _saved_bindings(saved)

    def reserves(self) -> set[str]:
        from pytweezer.experiment.reservations import (
            ReservationError,
            device_reservation,
        )
        from pytweezer.servers.device_server import resolve_address

        reserved = super().reserves()
        if self.linked or self.linked_from:
            partner = resolve_address(self.device_name).conf.get("linked_partner")
            if not partner:
                raise ReservationError(
                    f"{self.device_name!r} is linked but its CONFIG entry has no "
                    "'linked_partner'"
                )
            reserved |= device_reservation(partner)
        return reserved

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "motmaster": {
                "script": self.script,
                "script_from": self.script_from,
                "linked": self.linked,
                "linked_from": self.linked_from,
                "iterations": self.iterations,
                "save": self.save,
                "saved": dict(self.saved),
            }
        }


def _saved_bindings(saved: Iterable[str | dict[str, str]]) -> dict[str, str]:
    if isinstance(saved, (str, dict)):
        saved = (saved,)
    bindings: dict[str, str] = {}
    for entry in saved:
        if isinstance(entry, str):
            pairs = {entry: entry}
        elif isinstance(entry, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in entry.items()
        ):
            pairs = entry
        else:
            raise TypeError(
                f"saved= takes script parameter names and dicts of "
                f"{{parameter: saved key}}, not {entry!r}"
            )
        for parameter, key in pairs.items():
            if bindings.get(parameter, key) != key:
                raise TypeError(
                    f"saved= binds {parameter!r} to both {bindings[parameter]!r} "
                    f"and {key!r}"
                )
            bindings[parameter] = key
    return bindings


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
    """An integer MOTMaster parameter (sent to .NET as ``Int32``)."""


class MotMasterScript(Text):
    """The name of the script a :class:`MotMaster` declared with ``script_from=`` runs.

    Never scannable. The Experiments form offers the scripts in the MOTMaster's
    script folder; any other name is refused when the task starts.
    """

    kind = "motmaster_script"

    def __init__(self, default: str, **kwargs: Any) -> None:
        kwargs.setdefault("tooltip", "Script in the MOTMaster's script folder")
        super().__init__(default, scannable=False, **kwargs)


class _MotMasterOverride(Number):
    """Placeholder for a script parameter chosen by name; the script gives its real type."""


class MotMasterExperiment(Experiment):
    """Base class for an experiment driven by one MOTMaster script.

    :meth:`prepare` configures the sequencer completely on each task, since its
    settings persist between tasks. Repeated shots come from the scan's
    ``repetitions``, giving one stored point per shot.

    An experiment whose MOTMaster binds saved parameters (``saved=``) is given
    the argument ``use_motmaster_parameter_overrides``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sequencers = collect(cls, MotMaster)
        if len(sequencers) > 1:
            raise TypeError(
                f"{cls.__name__} declares MotMasters {sorted(sequencers)}; an "
                "experiment drives one MOTMaster. Run two machines together in "
                "linked mode: MotMaster(..., linked=True) on the primary"
            )
        if any(sequencer.saved for sequencer in sequencers.values()) and (
            "use_motmaster_parameter_overrides" not in cls.arguments()
        ):
            overrides = Bool(
                True,
                scannable=False,
                tooltip="send the saved parameters bound to the sequence to "
                "MOTMaster as parameter overrides; off, the script's own values "
                "run (a value set in the form still wins)",
            )
            overrides.__set_name__(cls, "use_motmaster_parameter_overrides")
            cls.use_motmaster_parameter_overrides = overrides
        targets: dict[str, str] = {}
        for name, argument in cls.motmaster_arguments().items():
            parameter = argument.motmaster_name
            if parameter in targets:
                raise TypeError(
                    f"{cls.__name__}.{targets[parameter]} and {cls.__name__}.{name} "
                    f"both set {parameter}"
                )
            targets[parameter] = name
        if sequencers:
            cls._check_from_arguments(next(iter(sequencers.values())))

    @classmethod
    def _check_from_arguments(cls, motmaster: MotMaster) -> None:
        arguments = cls.arguments()
        script = arguments.get(motmaster.script_from)
        if motmaster.script_from is not None and not isinstance(
            script, MotMasterScript
        ):
            raise TypeError(
                f"{cls.__name__}: script_from={motmaster.script_from!r} must name a "
                "MotMasterScript argument"
            )
        linked = arguments.get(motmaster.linked_from)
        if motmaster.linked_from is not None and (
            not isinstance(linked, Bool) or linked.scannable
        ):
            raise TypeError(
                f"{cls.__name__}: linked_from={motmaster.linked_from!r} must name a "
                "Bool argument declared with scannable=False"
            )

    @classmethod
    def motmaster(cls) -> tuple[str, MotMaster]:
        """The attribute name and declaration of the experiment's MOTMaster."""
        sequencers = collect(cls, MotMaster)
        if not sequencers:
            raise TypeError(f"{cls.__name__} must declare a MotMaster(...)")
        return next(iter(sequencers.items()))

    @classmethod
    def motmaster_arguments(cls) -> dict[str, MotMasterParameter]:
        return {
            name: argument
            for name, argument in cls.arguments().items()
            if isinstance(argument, MotMasterParameter)
        }

    @classmethod
    def extra_argument(cls, name: str) -> Argument | None:
        """A script parameter for undeclared ``name``, unless it cannot be one.

        Dotted and other non-identifier names cannot, nor can a parameter a
        declared argument already sets.
        """
        if not collect(cls, MotMaster) or not name.isidentifier():
            return None
        if any(a.motmaster_name == name for a in cls.motmaster_arguments().values()):
            return None
        argument = _MotMasterOverride(0.0, group="MOTMaster")
        argument.name = name
        return argument

    def script_name(self) -> str:
        """The script this task runs."""
        motmaster = self.motmaster()[1]
        if motmaster.script_from is None:
            return motmaster.script
        return getattr(self, motmaster.script_from)

    def is_linked(self) -> bool:
        """Whether this task runs its MOTMaster in linked mode."""
        motmaster = self.motmaster()[1]
        if motmaster.linked_from is None:
            return motmaster.linked
        return getattr(self, motmaster.linked_from)

    def motmaster_parameters(self) -> dict[str, Any]:
        """``{script parameter: value}`` for everything set explicitly."""
        requested = {
            argument.motmaster_name: getattr(self, name)
            for name, argument in self.motmaster_arguments().items()
        }
        return requested | self.extra_values

    def prepare(self) -> None:
        attribute, motmaster = self.motmaster()
        self._linked = False
        client = getattr(self, attribute)
        linked = self.is_linked()
        # Linked mode persists in MOTMaster, so set it on every task.
        client.set_linked(linked)
        self._linked = linked
        client.set_motmaster_experiment(self.script_name())
        client.set_run_until_stopped(False)
        client.set_iterations(motmaster.iterations)
        client.set_save_toggle(motmaster.save)
        client.set_trigger_mode(False)
        self._script_parameters: dict[str, Any] = client.get_params()
        set_in_form = self.motmaster_parameters()
        self._check_parameters(set_in_form)
        self._bind_saved_parameters(set_in_form)

    def saved_bindings(self) -> dict[str, str]:
        """``{script parameter: saved key}`` this task sends: the ``saved=`` ones, or none with ``use_motmaster_parameter_overrides`` off."""
        saved = self.motmaster()[1].saved
        if saved and not self.use_motmaster_parameter_overrides:
            return {}
        return saved

    def _bind_saved_parameters(self, set_in_form: dict[str, Any]) -> None:
        bindings = self.saved_bindings()
        self.saved_parameters: dict[str, Any] = {}
        self.saved_sequence: dict[str, Any] = {}
        self._saved_values: dict[str, Any] = {}
        if not bindings:
            return
        try:
            self.saved_parameters = get_saved_parameters()
        except (ManagerUnavailable, ManagerError) as error:
            raise RuntimeError(
                f"the saved parameters could not be read: {error}"
            ) from error
        missing = sorted(set(bindings.values()) - set(self.saved_parameters))
        if missing:
            raise KeyError(f"no saved parameter {', '.join(map(repr, missing))}")
        self._check_parameters(bindings)
        self._saved_values = {
            parameter: self.saved_parameters[key] for parameter, key in bindings.items()
        }
        self.saved_sequence = {
            parameter: value
            for parameter, value in self._saved_values.items()
            if parameter not in set_in_form
        }
        self.record("saved_sequence", self.saved_sequence)

    def _check_parameters(self, parameters: Iterable[str]) -> None:
        known = self._script_parameters
        for parameter in parameters:
            if parameter in known:
                continue
            script = self.script_name()
            close = difflib.get_close_matches(
                parameter, [*known, *self.arguments()], n=3
            )
            hint = f"; did you mean {close}?" if close else ""
            raise ValueError(f"script {script!r} has no parameter {parameter!r}{hint}")

    def _typed(self, parameter: str, value: Any) -> Any:
        default = self._script_parameters[parameter]
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            return value
        if isinstance(default, float):
            return float(value)
        if not float(value).is_integer():
            script = self.script_name()
            raise ValueError(
                f"{parameter} is an Int32 in script {script!r}; "
                f"{value!r} is not a whole number"
            )
        return int(value)

    def run_motmaster_sequence(self, **overrides: Any) -> None:
        """Run the script once (blocking) with the current parameters plus ``overrides``.

        ``overrides`` are keyed by script parameter and beat the form's values,
        which beat the saved ones. MOTMaster's record of the sequence it ran
        goes into the measurement file's ``/motmaster`` group, so the file
        shows exactly what ran whatever later happens to the script.
        """
        if self._writer is None:
            raise RuntimeError(
                "run_motmaster_sequence() only works while the experiment is run "
                "by run_local() or the experiment manager"
            )
        requested = self._saved_values | self.motmaster_parameters() | overrides
        # A name that is only scanned reaches the experiment at its first point,
        # after prepare().
        self._check_parameters(requested)
        sent = {
            parameter: self._typed(parameter, value)
            for parameter, value in requested.items()
        }
        sequencer = getattr(self, self.motmaster()[0])
        record = sequencer.start_motmaster_experiment_recorded(sent)
        self._writer.record_motmaster_run(
            None if record is None else json.loads(record)
        )

    def run_point(self) -> None:
        self.run_motmaster_sequence()

    def close_devices(self) -> None:
        # Here rather than in finish(), which subclasses override: a link left on
        # would lock the secondary MOTMaster against every other task.
        if getattr(self, "_linked", False):
            try:
                getattr(self, self.motmaster()[0]).set_linked(False)
            except Exception:
                logger.warning("Could not release the MOTMaster link", exc_info=True)
        self._linked = False
        super().close_devices()
