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

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.servers import ImageClient, PropertyAttribute


class ImagePanel:
    """One plot of a live image with its colour bar."""

    def __init__(self, layout: pg.GraphicsLayoutWidget, title: str, vertical="y"):
        self.plot: pg.PlotItem = layout.addPlot(title=title)
        self.plot.getViewBox().setAspectLocked(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.bar = pg.ColorBarItem(interactive=False, width=15, colorMapMenu=False)
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.vertical = vertical
        self.units = None
        self._set_units("px")

    def _set_units(self, units: str) -> None:
        if units == self.units:
            return
        self.units = units
        self.plot.setLabel("bottom", "x", units=units)
        self.plot.setLabel("left", self.vertical, units=units)

    def show(self, image, head: dict, levels: tuple[float, float]) -> None:
        self.image.setImage(image, autoLevels=False)
        self.bar.setLevels(levels)
        resolution = head.get("_imgresolution")
        offset = head.get("_offset")
        if resolution is None or offset is None:
            resolution, offset = (1, 1), (0, 0)
            self._set_units("px")
        else:
            self._set_units("m")
        ny, nx = image.shape
        self.image.setRect(
            QtCore.QRectF(
                offset[0] * resolution[0],
                offset[1] * resolution[1],
                nx * resolution[0],
                ny * resolution[1],
            )
        )


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
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(graphics)

        self.target = ImagePanel(graphics, "Target array")
        self.simulated = ImagePanel(graphics, "Simulated trap plane")
        graphics.nextRow()
        self.phase = ImagePanel(graphics, "Phase mask")
        self.top_down = ImagePanel(graphics, "Simulated top-down", vertical="z")

        self.panels = {
            "targetstreams": self.target,
            "simulatedstreams": self.simulated,
            "topdownstreams": self.top_down,
            "phasestreams": self.phase,
        }
        self.streams = {key: ImageClient(f"{self.name}_{key}") for key in self.panels}
        for key, panel in self.panels.items():
            menu = panel.plot.getViewBox().menu
            menu.addAction("subscriptions").triggered.connect(
                lambda _=False, key=key: self.open_subscription_editor(key)
            )
            menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

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
        intensity_cmap = pg.colormap.get(self._target_cmap, source="matplotlib")
        for panel in (self.target, self.simulated, self.top_down):
            panel.bar.setColorMap(intensity_cmap)
        self.phase.bar.setColorMap(
            pg.colormap.get(self._phase_cmap, source="matplotlib")
        )
        self._relabel_phase_bar()

    def _relabel_phase_bar(self):
        period = float(self._phase_period)
        self.phase.bar.axis.setTicks([[(0, "0"), (period / 2, "π"), (period, "2π")]])

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
            panel.show(image, head, levels)


def main(name):
    run_applet(SlmMonitor, default_name=name)


if __name__ == "__main__":
    run_applet(SlmMonitor, default_name="SLM Monitor")
