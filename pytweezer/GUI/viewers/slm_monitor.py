"""Target tweezer array, its simulated light and the SLM phase mask.

Subscribes to four image streams, one per panel, which
``experiments/slm_tweezer_array.py`` publishes:

*   the target trap intensities (``targetstreams``);
*   the simulated intensity in the trap plane (``simulatedstreams``);
*   the simulated intensity viewed top-down, summed over y with the
    propagation axis z up the panel (``topdownstreams``);
*   the 8-bit phase mask on the SLM (``phasestreams``), drawn with the cyclic
    ``phase_cmap`` over a fixed range of one ``phase_period`` so that 0 and 2π
    share a colour and phase wraps are invisible.

The three intensities are drawn with ``target_cmap``. Images are placed in
metres when their header carries ``_offset``/``_imgresolution``, in pixels
otherwise. Each panel pans and zooms on its own, and its context menu edits
its own subscriptions.

Properties:
    *   targetstreams: ([str]) target-array image streams.
    *   simulatedstreams: ([str]) simulated trap-plane image streams.
    *   topdownstreams: ([str]) simulated top-down image streams.
    *   phasestreams: ([str]) phase-mask image streams.
    *   target_cmap: (str) matplotlib colormap for the intensities.
    *   phase_cmap: (str) matplotlib colormap for the phase; keep it cyclic.
    *   phase_period: (float) pixel value corresponding to 2π.
"""

from PyQt6 import QtWidgets

from pytweezer.analysis.cloud_fit import frame_geometry
from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.GUI.views.image import ImageView
from pytweezer.servers import ImageClient, PropertyAttribute


class SlmMonitor(Applet):
    """Target and simulated trap plane on top, phase mask and top-down view below."""

    stream_category = "Image"
    poll_interval = 50
    default_size = (1100, 900)

    _targetstreams = PropertyAttribute("targetstreams", ["slm_target"])
    _simulatedstreams = PropertyAttribute("simulatedstreams", ["slm_simulated"])
    _topdownstreams = PropertyAttribute("topdownstreams", ["slm_top_down"])
    _phasestreams = PropertyAttribute("phasestreams", ["slm_phasemask"])
    _target_cmap = PropertyAttribute("target_cmap", "magma")
    _phase_cmap = PropertyAttribute("phase_cmap", "twilight")
    _phase_period = PropertyAttribute("phase_period", 256.0)

    def init_gui(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)

        self.target = self._panel("Target array")
        self.simulated = self._panel("Simulated trap plane")
        self.phase = self._panel("Phase mask")
        self.top_down = self._panel("Simulated top-down", vertical_axis="z")
        for row, panels in enumerate(
            ((self.target, self.simulated), (self.phase, self.top_down))
        ):
            for column, panel in enumerate(panels):
                grid.addWidget(panel, row, column)

        self.panels = {
            "targetstreams": self.target,
            "simulatedstreams": self.simulated,
            "topdownstreams": self.top_down,
            "phasestreams": self.phase,
        }
        self.streams = {key: ImageClient(f"{self.name}_{key}") for key in self.panels}
        for key, panel in self.panels.items():
            panel.add_menu_action(
                "subscriptions",
                lambda _=False, key=key: self.open_subscription_editor(key),
            )
            panel.add_menu_action("configure", self.open_config_editor)

        self.update_subscriptions()

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
            "targetstreams": self._targetstreams,
            "simulatedstreams": self._simulatedstreams,
            "topdownstreams": self._topdownstreams,
            "phasestreams": self._phasestreams,
        }
        for key, stream in self.streams.items():
            stream.unsubscribe()
            stream.subscribe(subscriptions[key])
        for panel in (self.target, self.simulated, self.top_down):
            panel.set_colour_map(self._target_cmap)
        self.phase.set_colour_map(self._phase_cmap)
        self._relabel_phase_bar()

    def _relabel_phase_bar(self):
        period = float(self._phase_period)
        self.phase.set_bar_ticks([(0, "0"), (period / 2, "π"), (period, "2π")])

    @staticmethod
    def _latest(stream):
        latest = None
        while stream.has_new_data(timeout_ms=0):
            message = stream.recv()
            if message is not None and len(message) >= 3 and message[2].ndim == 2:
                latest = message
        return latest

    def poll(self):
        for key, panel in self.panels.items():
            message = self._latest(self.streams[key])
            if message is None:
                continue
            _, head, image = message
            if panel is self.phase:
                levels = (0.0, float(self._phase_period))
            else:
                levels = (0.0, max(float(image.max()), 1e-12))
            panel.set_units("px" if frame_geometry(head)[0] is None else "m")
            panel.show_image(image, head, levels=levels)


def main(name):
    run_applet(SlmMonitor, default_name=name)


if __name__ == "__main__":
    run_applet(SlmMonitor, default_name="SLM Monitor")
