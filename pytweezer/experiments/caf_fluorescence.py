"""CaF MOT fluorescence, background-subtracted, at chosen wavemeter-lock set points."""

from pytweezer.experiment import Number
from pytweezer.experiments.caf_mot import CaFMOT


class CaFFluorescence(CaFMOT):
    """:class:`~pytweezer.experiments.caf_mot.CaFMOT` less a background shot taken once per task.

    ``prepare`` takes the background shot with the MOT field at ``bg_field``;
    the mean of its frames is subtracted from every later frame before the
    ``counts`` and the cloud fit.
    """

    bg_field = Number(
        0.0,
        group="MOTMaster",
        scannable=False,
        tooltip="MOTCoilsCurrentValue for the background shot",
    )

    def prepare(self) -> None:
        super().prepare()
        background_frames = self.take_shot(MOTCoilsCurrentValue=self.bg_field)
        self.background = background_frames.astype(float).mean(axis=0)
