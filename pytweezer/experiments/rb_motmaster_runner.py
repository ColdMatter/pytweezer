"""Run any Rb MOTMaster script and record every frame it triggers on the chosen cameras."""

from typing import ClassVar

from pytweezer.experiment import Bool
from pytweezer.experiment.motmaster import MotMaster, MotMasterScript
from pytweezer.experiment.motmaster_runner import MotMasterRunner


class RbMotMasterRunner(MotMasterRunner):
    """Run a script from the Rb MOTMaster once per point, recording each ticked camera's frames.

    Each ticked camera's frames from a run are recorded as ``<camera>_images``,
    ``(frames, H, W)``.
    """

    CAMERAS: ClassVar[dict[str, str]] = {"hamcam": "Rb HamCam", "orca": "ORCA"}

    script = MotMasterScript("RbTweezerBasic2026_2")
    rb = MotMaster("Rb MotMaster", script_from="script")

    use_hamcam = Bool(True, group="Cameras", scannable=False, tooltip="Rb HamCam")
    use_orca = Bool(False, group="Cameras", scannable=False, tooltip="ORCA")
