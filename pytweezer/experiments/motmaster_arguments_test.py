"""Exercise the MOTMaster argument interface without a camera."""

from pytweezer.experiment import Bool, Choice, Text
from pytweezer.experiment.motmaster import (
    MotMaster,
    MotMasterExperiment,
    MotMasterNumber,
)


class MotMasterArgumentsTest(MotMasterExperiment):
    """Run a script with parameters chosen from the form's search box.

    ``pulse_time`` is always shown; any other script parameter is added by name.
    ``dry_run``, ``mode`` and ``note`` are ordinary arguments and are not sent.
    """

    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")

    pulse_time = MotMasterNumber(
        20e-6, parameter="tPulse", unit="us", scale=1e-6, min=0
    )

    dry_run = Bool(False, tooltip="Record the parameters without running the script")
    mode = Choice(["fast", "slow"])
    note = Text("")

    def run_point(self):
        sent = self.motmaster_parameters()["rb"]
        for name, value in sent.items():
            self.record(f"sent_{name}", value)
        if not self.dry_run:
            self.run_sequences()
