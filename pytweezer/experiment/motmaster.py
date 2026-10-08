"""Experiments that run MOTMaster sequences once per point.

::

    class TofImaging(MotMasterExperiment):
        rb = MotMaster("Rb MotMaster", script="RbTweezerBasic", master=True)
        caf = MotMaster("CaF MotMaster", script="CaFTweezerLoad")

        tof = MotMasterInteger(100, device="rb", parameter="tDelay1", unit="ms/100")

Each :class:`MotMaster` names a device and the script it runs. Parameters
declared with :class:`MotMasterNumber`/:class:`MotMasterInteger` are always
shown; any other script parameter can be set by passing a dotted argument name
(``"rb.tPulse"``), which the Experiments form offers through a search box.
Parameters not set are left at the script's own defaults. One MOTMaster is the
master; the others are armed in trigger mode and started first.
"""

import difflib
import threading
import time
from typing import Any, ClassVar

from pytweezer.experiment.arguments import Argument, Device, Integer, Number, collect
from pytweezer.experiment.experiment import Experiment, logger


class MotMaster(Device):
    """A MOTMaster sequencer device and the script it runs for this experiment."""

    def __init__(
        self,
        device_name: str,
        *,
        script: str,
        master: bool = False,
        iterations: int = 1,
        save: bool = False,
        follower_timeout: float = 60.0,
        timeout: float | None = None,
    ) -> None:
        super().__init__(device_name, timeout=timeout)
        self.script = script
        self.master = master
        self.iterations = iterations
        self.save = save
        self.follower_timeout = follower_timeout

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "motmaster": {
                "script": self.script,
                "master": self.master,
                "iterations": self.iterations,
                "save": self.save,
                "follower_timeout": self.follower_timeout,
            }
        }


class MotMasterParameter:
    """Marks an argument as a MOTMaster script parameter (default group "MOTMaster").

    ``device`` is the name of a :class:`MotMaster` attribute; it may be omitted
    when the experiment has only one.
    """

    def __init__(
        self,
        *args: Any,
        device: str | None = None,
        parameter: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.parameter = parameter
        kwargs.setdefault("group", "MOTMaster")
        super().__init__(*args, **kwargs)

    @property
    def motmaster_name(self) -> str:
        return self.parameter or self.name

    def to_schema(self) -> dict[str, Any]:
        return super().to_schema() | {
            "motmaster_device": self.device,
            "motmaster_parameter": self.motmaster_name,
        }


class MotMasterNumber(MotMasterParameter, Number):
    """A float MOTMaster parameter (sent to .NET as ``Double``)."""


class MotMasterInteger(MotMasterParameter, Integer):
    """An integer MOTMaster parameter (sent to .NET as ``Int32``)."""


class _MotMasterOverride(Number):
    """Placeholder for a script parameter chosen by name; the script gives its real type."""


class MotMasterExperiment(Experiment):
    """Base class for an experiment driven by one or more MOTMaster scripts.

    :meth:`prepare` configures every sequencer completely on each task, since
    their settings persist between tasks. Repeated shots come from the scan's
    ``repetitions``, giving one stored point per shot.
    """

    #: Seconds to wait after arming the followers before the master starts.
    follower_arm_delay: ClassVar[float] = 0.5

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sequencers = cls.motmasters()
        if not sequencers:
            return
        masters = [name for name, sequencer in sequencers.items() if sequencer.master]
        if len(sequencers) > 1 and len(masters) != 1:
            raise TypeError(
                f"{cls.__name__} has {len(sequencers)} MOTMasters; exactly one "
                f"must have master=True (found {len(masters)})"
            )
        cls._declared_targets()

    @classmethod
    def motmasters(cls) -> dict[str, MotMaster]:
        return collect(cls, MotMaster)

    @classmethod
    def master_name(cls) -> str:
        sequencers = cls.motmasters()
        if not sequencers:
            raise TypeError(f"{cls.__name__} must declare at least one MotMaster(...)")
        flagged = [name for name, sequencer in sequencers.items() if sequencer.master]
        return flagged[0] if flagged else next(iter(sequencers))

    @classmethod
    def motmaster_arguments(cls) -> dict[str, MotMasterParameter]:
        return {
            name: argument
            for name, argument in cls.arguments().items()
            if isinstance(argument, MotMasterParameter)
        }

    @classmethod
    def _parameter_device(cls, name: str, argument: MotMasterParameter) -> str:
        sequencers = cls.motmasters()
        if argument.device is None:
            if len(sequencers) != 1:
                raise TypeError(
                    f"{cls.__name__}.{name} needs device= to say which of "
                    f"{sorted(sequencers)} it belongs to"
                )
            return next(iter(sequencers))
        if argument.device not in sequencers:
            raise TypeError(
                f"{cls.__name__}.{name}: device={argument.device!r} is not one of "
                f"the MOTMasters {sorted(sequencers)}"
            )
        return argument.device

    @classmethod
    def _declared_targets(cls) -> dict[tuple[str, str], str]:
        """``{(MotMaster attribute, script parameter): declared argument name}``."""
        targets: dict[tuple[str, str], str] = {}
        for name, argument in cls.motmaster_arguments().items():
            target = (cls._parameter_device(name, argument), argument.motmaster_name)
            if target in targets:
                raise TypeError(
                    f"{cls.__name__}.{targets[target]} and {cls.__name__}.{name} "
                    f"both set {target[0]}.{target[1]}"
                )
            targets[target] = name
        return targets

    @classmethod
    def extra_argument(cls, name: str) -> Argument | None:
        attribute, dot, parameter = name.partition(".")
        if not dot or not parameter or attribute not in cls.motmasters():
            return None
        if (attribute, parameter) in cls._declared_targets():
            return None
        argument = _MotMasterOverride(0.0, group=f"MOTMaster: {attribute}")
        argument.name = name
        return argument

    def motmaster_parameters(self) -> dict[str, dict[str, Any]]:
        """``{MotMaster attribute: {script parameter: value}}`` for everything set explicitly."""
        requested: dict[str, dict[str, Any]] = {name: {} for name in self.motmasters()}
        for name, argument in self.motmaster_arguments().items():
            device = self._parameter_device(name, argument)
            requested[device][argument.motmaster_name] = getattr(self, name)
        for name in self._extra_names:
            attribute, _, parameter = name.partition(".")
            requested[attribute][parameter] = getattr(self, name)
        return requested

    def prepare(self) -> None:
        master = self.master_name()
        self._script_parameters: dict[str, dict[str, Any]] = {}
        self._follower_clients: dict[str, Any] = {}
        for attribute, motmaster in self.motmasters().items():
            client = getattr(self, attribute)
            client.set_motmaster_experiment(motmaster.script)
            client.set_run_until_stopped(False)
            client.set_iterations(motmaster.iterations)
            client.set_save_toggle(motmaster.save)
            client.set_trigger_mode(attribute != master)
            self._script_parameters[attribute] = client.get_params()
        self._check_parameters(self.motmaster_parameters())

    def _check_parameters(self, requested: dict[str, dict[str, Any]]) -> None:
        for attribute, parameters in requested.items():
            for parameter in parameters:
                self._check_parameter(attribute, parameter)

    def _check_parameter(self, attribute: str, parameter: str) -> None:
        known = self._script_parameters[attribute]
        if parameter in known:
            return
        script = self.motmasters()[attribute].script
        close = difflib.get_close_matches(parameter, known, n=3)
        hint = f"; did you mean {close}?" if close else ""
        raise ValueError(
            f"{attribute}: script {script!r} has no parameter {parameter!r}{hint}"
        )

    def _typed(self, attribute: str, parameter: str, value: Any) -> Any:
        default = self._script_parameters[attribute][parameter]
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            return value
        if isinstance(default, float):
            return float(value)
        if not float(value).is_integer():
            script = self.motmasters()[attribute].script
            raise ValueError(
                f"{attribute}.{parameter} is an Int32 in script {script!r}; "
                f"{value!r} is not a whole number"
            )
        return int(value)

    def run_sequences(self, **overrides: Any) -> None:
        """Run every script once (blocking) with the current parameters plus ``overrides``.

        ``overrides`` are keyed ``"attribute.parameter"``. Followers are started
        in threads first, then the master; the first error is raised once all
        have finished or timed out.
        """
        requested = self.motmaster_parameters()
        for key, value in overrides.items():
            attribute, _, parameter = key.partition(".")
            if attribute not in requested:
                raise KeyError(f"no MotMaster attribute {attribute!r} for {key!r}")
            requested[attribute][parameter] = value
        # A name that is only scanned reaches the instance at its first point, after prepare().
        self._check_parameters(requested)
        sent = {
            attribute: {
                parameter: self._typed(attribute, parameter, value)
                for parameter, value in parameters.items()
            }
            for attribute, parameters in requested.items()
        }

        master = self.master_name()
        errors: list[tuple[str, BaseException]] = []

        def go(attribute: str, client: Any) -> None:
            try:
                client.start_motmaster_experiment(sent[attribute])
            except Exception as error:
                errors.append((attribute, error))

        follower_names = [name for name in self.motmasters() if name != master]
        for attribute in follower_names:
            if attribute not in self._follower_clients:
                motmaster = self.motmasters()[attribute]
                self._follower_clients[attribute] = self.device(
                    motmaster.device_name, fresh=True, timeout=motmaster.timeout
                )
        followers: dict[str, threading.Thread] = {}
        for attribute in follower_names:
            thread = threading.Thread(
                target=go,
                args=(attribute, self._follower_clients[attribute]),
                name=f"motmaster-{attribute}",
            )
            thread.daemon = True
            thread.start()
            followers[attribute] = thread
        if followers:
            time.sleep(self.follower_arm_delay)
        go(master, getattr(self, master))
        master_finished = time.monotonic()
        for attribute, thread in followers.items():
            timeout = self.motmasters()[attribute].follower_timeout
            thread.join(max(0.0, master_finished + timeout - time.monotonic()))
            if thread.is_alive():
                # Its client is still blocked in Go(); a later call needs another.
                del self._follower_clients[attribute]
                errors.append(
                    (
                        attribute,
                        TimeoutError(
                            f"{attribute} did not finish within {timeout} s; "
                            "it may still be waiting for its trigger"
                        ),
                    )
                )
        # A timed-out follower may still append to errors, so work on a copy.
        failures = sorted(list(errors), key=lambda entry: entry[0] != master)
        if failures:
            for attribute, error in failures[1:]:
                logger.error("MOTMaster %s also failed: %r", attribute, error)
            raise failures[0][1]

    def run_point(self) -> None:
        self.run_sequences()
