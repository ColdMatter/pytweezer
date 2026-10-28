"""First live check of linked MOTMaster driven from pytweezer, read on a scope."""

from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterInteger,
)


class LinkedScopeTest(MotMasterExperiment):
    """One 10 ms pulse on each of PH-BONESAW and RbTweezer from one linked script.

    The two pulses coincide on a scope once ``SecondaryTriggerDelay`` (10 us
    ticks) cancels the trigger and cable delay; scan it to find that value.
    """

    mm = MotMaster("CaF MotMaster", script="LinkedScopeTest", linked=True)

    secondary_trigger_delay = MotMasterInteger(
        0, parameter="SecondaryTriggerDelay", min=0, unit="10 us"
    )
    pulse_start = MotMasterInteger(1000, parameter="PulseStart", min=0)
