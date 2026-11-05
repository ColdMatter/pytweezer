"""CaF MOT fluorescence, background-subtracted, at chosen wavemeter-lock set points."""

import numpy as np

from pytweezer.experiment import Number
from pytweezer.experiments.caf_mot import CaFMOT, frame_counts


class CaFFluorescence(CaFMOT):
    """:class:`~pytweezer.experiments.caf_mot.CaFMOT` with one background shot per task.

    ``prepare`` takes a background shot with the MOT field at ``bg_field``,
    once for the whole task. Its frames go out on the image stream
    ``background_<camera stream>``, and every shot's frames minus their mean
    on ``subtracted_<camera stream>``, neither of which a viewer of the
    camera's own stream receives. ``counts`` and the cloud fit are of the
    background-subtracted frames.
    """

    bg_field = Number(
        0.0, group="MOTMaster", tooltip="MOTCoilsCurrentValue for the background shot"
    )

    def prepare(self) -> None:
        super().prepare()
        background_frames = self.take_shot(
            streams={"thorcam": self.stream_name("thorcam", "background")},
            **{"caf.MOTCoilsCurrentValue": self.bg_field},
        )["thorcam"]
        self.background = background_frames.astype(float).mean(axis=0)

    def analyse(self, frames: np.ndarray) -> dict[str, float]:
        subtracted = frames.astype(float) - self.background
        self.broadcast_frames(
            "thorcam", subtracted, self.stream_name("thorcam", "subtracted")
        )
        self.record_cloud_fit("thorcam", frames, self.background)
        return frame_counts(subtracted)
