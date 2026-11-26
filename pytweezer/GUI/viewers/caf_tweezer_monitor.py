"""The HamCam image with a circle where each CaF tweezer of the SLM array should be.

The Tweezer Monitor, with predicted sites in place of FindTrapSites' accepted
ones: the traps of the array the rig (``rig``) has on the SLM, put on the
HamCam sensor by the saved CaF calibration (``caf_hamcam_centre_x``/``_y``,
``caf_hamcam_px_per_m``, ``caf_hamcam_angle``) as the "Light at the tweezer
sites" analysis puts them. Each circle is that analysis's summing disc,
``aperture`` pixels in radius. Subscribe it to the HamCam's image stream.

The sites are worked out when the viewer starts, when the saved parameters
change, and when an experiment starts or finishes, since that is when the
array on the SLM changes. A line on the image says why there are none: the rig
is not running, another program has written the SLM, or the calibration is
not saved.

Properties (besides the Tweezer Monitor's):
    *   rig: (str) device to ask for the array on the SLM.
    *   aperture: (float) radius of each circle, in pixels.
"""

import math
import threading

import numpy as np

from pytweezer.analyses.default.caf_tweezer_sites import SiteSettings
from pytweezer.analysis.trap_sites import geometry_to_pixels
from pytweezer.experiment.client import get_saved_parameters
from pytweezer.GUI.applet import run_applet
from pytweezer.GUI.viewers.tweezer_monitor import TweezerDisplay
from pytweezer.logging_utils import get_logger
from pytweezer.servers import PropertyAttribute
from pytweezer.servers.device_client import get_device

logger = get_logger(__name__)

#: Saved parameters that put the focal plane on the HamCam sensor.
CALIBRATION = (
    "caf_hamcam_centre_x",
    "caf_hamcam_centre_y",
    "caf_hamcam_px_per_m",
    "caf_hamcam_angle",
)


def caf_sites(
    array: dict | None, saved: dict, rig: str
) -> tuple[np.ndarray | None, str]:
    """The full-sensor ``(y, x)`` pixels of ``array``'s traps, from ``rig.get_array()``, and a message if there are none."""
    if array is None:
        return None, f"{rig} has no array on the SLM"
    missing = [key for key in CALIBRATION if key not in saved]
    if missing:
        return None, f"No CaF site calibration saved: {', '.join(missing)}"
    geometry = np.column_stack([array["x_um"], array["y_um"]])
    sites = geometry_to_pixels(
        geometry,
        float(saved["caf_hamcam_px_per_m"]) * 1e-6,
        math.degrees(float(saved["caf_hamcam_angle"])),
        float(saved["caf_hamcam_centre_x"]),
        float(saved["caf_hamcam_centre_y"]),
    )
    return sites, ""


class CaFTweezerDisplay(TweezerDisplay):
    """The Image Monitor with the predicted CaF tweezer sites drawn over each frame."""

    site_symbol = "o"

    _rig = PropertyAttribute("rig", "Rb Rearrangement Rig")
    _aperture = PropertyAttribute("aperture", SiteSettings.aperture)

    def init_gui(self):
        self._running = None
        self._requests = 0
        self._found = None
        super().init_gui()

    def _queue_changed(self, snapshot):
        super()._queue_changed(snapshot)
        running = {task.get("rid") for task in snapshot.get("running", [])}
        if running != self._running:
            self._running = running
            self._load_sites()

    def _load_sites(self):
        """Work the sites out off the GUI thread: the rig stalls while it computes a mask."""
        self._requests += 1
        threading.Thread(
            target=self._find_sites,
            args=(self._requests, self._rig),
            daemon=True,
            name="find CaF sites",
        ).start()

    def _find_sites(self, request: int, rig_name: str) -> None:
        sites, message = self._ask_for_sites(rig_name)
        if request == self._requests:
            self._found = (request, sites, message)

    @staticmethod
    def _ask_for_sites(rig_name: str) -> tuple[np.ndarray | None, str]:
        try:
            saved = get_saved_parameters()
        except Exception as error:
            logger.info("Could not read the saved parameters: %r", error)
            return None, "Could not read the saved parameters"
        try:
            rig = get_device(rig_name, timeout=10)
            try:
                array = rig.get_array()
            finally:
                rig.close_rpc()
        except Exception as error:
            logger.info("No array from %s: %r", rig_name, error)
            return None, f"Could not ask {rig_name} for its array"
        return caf_sites(array, saved, rig_name)

    def poll(self):
        found = self._found
        if found is not None and found[0] == self._requests:
            self._found = None
            _, sites, message = found
            self.overlay.set_sites(sites, 2 * float(self._aperture), message)
            self._placed = None
        super().poll()


def main(name):
    run_applet(CaFTweezerDisplay, default_name=name)


if __name__ == "__main__":
    run_applet(CaFTweezerDisplay, default_name="CaF Tweezer Monitor")
