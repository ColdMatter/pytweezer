"""Run any CaF MOTMaster script, linked or not, and record every frame it triggers on the chosen cameras."""

from typing import ClassVar

from pytweezer.experiment import Bool
from pytweezer.experiment.motmaster import MotMaster, MotMasterScript
from pytweezer.experiment.motmaster_runner import MotMasterRunner


class CaFMotMasterRunner(MotMasterRunner):
    """Run a script from the CaF MOTMaster once per point, recording each ticked camera's frames.

    With ``linked`` the script also drives RbTweezer, so its parameters include
    the Rb machine's; the Rb MOTMaster is reserved either way. Each ticked
    camera's frames from a run are recorded as ``<camera>_images``,
    ``(frames, H, W)``.
    """

    CAMERAS: ClassVar[dict[str, str]] = {
        "cafmot_cam": "CaFMOT Cam",
        "hamcam": "Rb HamCam",
    }

    script = MotMasterScript("MERGE.CaFTweezer")
    linked = Bool(
        True,
        scannable=False,
        tooltip="Run in linked mode: the script drives RbTweezer too, fired by TTL",
    )
    caf = MotMaster("CaF MotMaster", script_from="script", linked_from="linked")

    use_cafmot_cam = Bool(
        True, group="Cameras", scannable=False, tooltip="CaFMOT Cam (ORCA-Fusion)"
    )
    use_hamcam = Bool(True, group="Cameras", scannable=False, tooltip="Rb HamCam")
