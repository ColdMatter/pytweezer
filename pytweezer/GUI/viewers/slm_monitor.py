"""The SLM's phase mask, the light it makes, and how its optimisation converged.

Shows what ``experiments/slm_tweezer_array.py`` broadcasts for each new mask:

*   top left, the simulated intensity in the trap plane (``simulatedstreams``);
*   top right, the simulated intensity viewed top-down, summed over y with the
    propagation axis z up the panel (``topdownstreams``);
*   bottom left, the 8-bit phase mask on the SLM (``phasestreams``), drawn with
    the cyclic ``phase_cmap`` over a fixed range of one ``phase_period`` so
    that 0 and 2π share a colour and phase wraps are invisible;
*   bottom right, the trap uniformity and min/max ratio, and below them the
    mean-squared error, at each iteration of the weight optimisation
    (``convergencestreams``, a data stream: one row per quantity, named in the
    header's ``curves``).

The intensities are drawn with ``target_cmap``. Images are placed in metres
when their header carries ``_offset``/``_imgresolution``, in pixels
otherwise. Each panel pans and zooms on its own, and its context menu edits
its own subscriptions.

On opening it asks the rig (``rig``) for the array it last put on the SLM and
shows that until the streams send something, so it is not blank between
experiments. A rig that is not running, or whose SLM has since been written by
something else, leaves the panels empty.

Properties:
    *   simulatedstreams: ([str]) simulated trap-plane image streams.
    *   topdownstreams: ([str]) simulated top-down image streams.
    *   phasestreams: ([str]) phase-mask image streams.
    *   convergencestreams: ([str]) optimisation-convergence data streams.
    *   target_cmap: (str) matplotlib colormap for the intensities.
    *   phase_cmap: (str) matplotlib colormap for the phase; keep it cyclic.
    *   phase_period: (float) pixel value corresponding to 2π.
    *   rig: (str) device to ask for the array on the SLM when the window opens.
"""

import threading

import numpy as np
from PyQt6 import QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.theme import curve_color
from pytweezer.GUI.views.image import ImageView
from pytweezer.GUI.views.line_plot import LinePlot
from pytweezer.helper_scripts.array_hologram import CONVERGENCE, monitor_images
from pytweezer.logging_utils import get_logger
from pytweezer.servers import DataClient, ImageClient, PropertyAttribute
from pytweezer.servers.device_client import get_device

logger = get_logger(__name__)

#: The :func:`monitor_images` channel each image panel's default stream carries.
RIG_CHANNELS = {
    "simulatedstreams": "_simulated",
    "topdownstreams": "_top_down",
    "phasestreams": "_phasemask",
}
#: Convergence quantities drawn on the error plot (log scale) rather than the quality plot.
ERROR_CURVES = {"mse"}
CURVE_LABELS = {
    "uniformity": "uniformity",
    "min_max_ratio": "min/max ratio",
    "mse": "MSE",
}


class Convergence(QtWidgets.QWidget):
    """Quality (uniformity, min/max ratio) above error, against optimisation iteration."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.quality = LinePlot(title="Optimisation convergence", y_label="ratio")
        self.error = LinePlot(legend=False, x_label="iteration", y_label="MSE")
        self.error.setLogMode(y=True)
        self.error.getAxis("left").enableAutoSIPrefix(False)
        self.error.link_x(self.quality)
        self.quality.show_x_values(False)
        for plot in (self.quality, self.error):
            plot.set_y_axis_width(60)
        layout.addWidget(self.quality, 3)
        layout.addWidget(self.error, 2)

    def add_menu_action(self, text: str, slot) -> None:
        for plot in (self.quality, self.error):
            plot.add_menu_action(text, slot)

    def draw(self, names, rows: np.ndarray) -> None:
        iterations = np.arange(1, rows.shape[1] + 1)
        for index, (name, values) in enumerate(zip(names, rows)):
            plot = self.error if name in ERROR_CURVES else self.quality
            label = CURVE_LABELS.get(name, name)
            plot.curve(label, curve_color(index))
            plot.set_curve(label, iterations, values)


class SlmMonitor(Applet):
    """Simulated trap plane and top-down view on top, phase mask and convergence below."""

    stream_category = "Image"
    poll_interval = 50
    default_size = (1100, 900)

    _simulatedstreams = PropertyAttribute("simulatedstreams", ["slm_simulated"])
    _topdownstreams = PropertyAttribute("topdownstreams", ["slm_top_down"])
    _phasestreams = PropertyAttribute("phasestreams", ["slm_phasemask"])
    _convergencestreams = PropertyAttribute("convergencestreams", ["slm_convergence"])
    _target_cmap = PropertyAttribute("target_cmap", "magma")
    _phase_cmap = PropertyAttribute("phase_cmap", "twilight")
    _phase_period = PropertyAttribute("phase_period", 256.0)
    _rig = PropertyAttribute("rig", "Rb Rearrangement Rig")

    def init_gui(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.simulated = self._panel("Simulated trap plane")
        self.top_down = self._panel("Simulated top-down", vertical_axis="z")
        self.phase = self._panel("Phase mask")
        self.convergence = Convergence()
        grid.addWidget(self.simulated, 0, 0)
        grid.addWidget(self.top_down, 0, 1)
        grid.addWidget(self.phase, 1, 0)
        grid.addWidget(self.convergence, 1, 1)
        for index in (0, 1):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)

        self.panels = {
            "simulatedstreams": self.simulated,
            "topdownstreams": self.top_down,
            "phasestreams": self.phase,
        }
        self.streams = {key: ImageClient(f"{self.name}_{key}") for key in self.panels}
        self.convergence_stream = DataClient(f"{self.name}_convergence")
        for key, panel in self.panels.items():
            panel.add_menu_action(
                "subscriptions",
                lambda _=False, key=key: self.open_subscription_editor(key),
            )
            panel.add_menu_action("configure", self.open_config_editor)
        self.convergence.add_menu_action(
            "subscriptions",
            lambda _=False: self.open_subscription_editor(
                "convergencestreams", category="Data"
            ),
        )
        self.convergence.add_menu_action("configure", self.open_config_editor)

        self._streamed = set()
        self._from_rig = None
        threading.Thread(
            target=self._ask_rig, args=(self._rig,), daemon=True, name="ask rig"
        ).start()

    def _ask_rig(self, rig_name: str) -> None:
        """Fetch the rig's array off the GUI thread: the rig stalls while it computes a mask."""
        try:
            rig = get_device(rig_name, timeout=10)
            try:
                hologram = rig.get_array_hologram()
            finally:
                rig.close_rpc()
        except Exception as error:
            logger.info("No SLM array from %s: %r", rig_name, error)
            return
        if hologram is not None:
            images = monitor_images(
                hologram, hologram["blaze_um"], hologram["slm_pitch_um"]
            )
            self._from_rig = images, hologram.get("convergence")

    @staticmethod
    def _panel(title: str, vertical_axis: str = "y") -> ImageView:
        return ImageView(
            colour_bar="bar",
            units="px",
            title=title,
            row_zero_at_top=False,
            vertical_axis=vertical_axis,
        )

    def update_subscriptions(self):
        subscriptions = {
            "simulatedstreams": self._simulatedstreams,
            "topdownstreams": self._topdownstreams,
            "phasestreams": self._phasestreams,
        }
        for key, stream in self.streams.items():
            stream.unsubscribe()
            stream.subscribe(subscriptions[key])
        self.convergence_stream.unsubscribe()
        self.convergence_stream.subscribe(self._convergencestreams)
        for panel in (self.simulated, self.top_down):
            panel.set_colour_map(self._target_cmap)
        self.phase.set_colour_map(self._phase_cmap)
        self._relabel_phase_bar()

    def _relabel_phase_bar(self):
        period = float(self._phase_period)
        self.phase.set_bar_ticks([(0, "0"), (period / 2, "π"), (period, "2π")])

    @staticmethod
    def _placement(head: dict) -> tuple[str, dict]:
        """Axis units and header: metres only when the header has both resolution and offset."""
        if head.get("_imgresolution") is None or head.get("_offset") is None:
            return "px", {}
        return "m", head

    @staticmethod
    def _latest(stream):
        latest = None
        while stream.has_new_data(timeout_ms=0):
            message = stream.recv()
            if message is not None and len(message) >= 3 and message[2].ndim == 2:
                latest = message
        return latest

    def poll(self):
        if self._from_rig is not None:
            self._show_rig_array(*self._from_rig)
            self._from_rig = None
        for key, panel in self.panels.items():
            message = self._latest(self.streams[key])
            if message is not None:
                self._streamed.add(key)
                self._show(panel, message[1], message[2])
        message = self._latest(self.convergence_stream)
        if message is not None:
            self._streamed.add("convergencestreams")
            _, head, rows = message
            self.convergence.draw(head.get("curves", CONVERGENCE), rows)

    def _show_rig_array(self, images, convergence) -> None:
        """Draw what the rig says is on the SLM in every panel the streams have not drawn yet."""
        for key, panel in self.panels.items():
            if key not in self._streamed:
                self._show(panel, *images[RIG_CHANNELS[key]])
        if convergence is not None and "convergencestreams" not in self._streamed:
            self.convergence.draw(CONVERGENCE, np.asarray(convergence))

    def _show(self, panel: ImageView, head: dict, image) -> None:
        if panel is self.phase:
            levels = (0.0, float(self._phase_period))
        else:
            levels = (0.0, max(float(image.max()), 1e-12))
        units, head = self._placement(head)
        panel.set_units(units)
        panel.show_image(image, head, levels=levels)


def main(name):
    run_applet(SlmMonitor, default_name=name)


if __name__ == "__main__":
    run_applet(SlmMonitor, default_name="SLM Monitor")
