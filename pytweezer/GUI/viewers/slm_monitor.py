"""Target tweezer array and SLM phase mask, side by side.

Subscribes to two image streams: the target trap intensities (``targetstreams``,
drawn with ``target_cmap``) and the 8-bit phase mask on the SLM
(``phasestreams``), drawn with the cyclic ``phase_cmap`` over a fixed range of
one ``phase_period`` so that 0 and 2π share a colour and phase wraps are
invisible. ``experiments/slm_array.py`` publishes both. Images are placed in
metres when their header carries ``_offset``/``_imgresolution``, in pixels
otherwise.

Properties:
    *   targetstreams: ([str]) target-array image streams.
    *   phasestreams: ([str]) phase-mask image streams.
    *   target_cmap: (str) matplotlib colormap for the target.
    *   phase_cmap: (str) matplotlib colormap for the phase; keep it cyclic.
    *   phase_period: (float) pixel value corresponding to 2π.
"""

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from pytweezer.GUI.applet import Applet, run_applet
from pytweezer.servers import ImageClient, PropertyAttribute


class ImagePanel:
    """One plot of a live image with its colour bar."""

    def __init__(self, layout: pg.GraphicsLayoutWidget, title: str):
        self.plot: pg.PlotItem = layout.addPlot(title=title)
        self.plot.getViewBox().setAspectLocked(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.bar = pg.ColorBarItem(interactive=False, width=15, colorMapMenu=False)
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.units = None
        self._set_units("px")

    def _set_units(self, units: str) -> None:
        if units == self.units:
            return
        self.units = units
        self.plot.setLabel("bottom", "x", units=units)
        self.plot.setLabel("left", "y", units=units)

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
    """Target array on the left, phase mask on the SLM on the right."""

    stream_category = "Image"
    poll_interval = 50
    default_size = (1100, 520)

    _targetstreams = PropertyAttribute("targetstreams", ["slm_target"])
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
        self.phase = ImagePanel(graphics, "Phase mask")
        self.target_stream = ImageClient(self.name + "_target")
        self.phase_stream = ImageClient(self.name + "_phase")

        for panel in (self.target, self.phase):
            menu = panel.plot.getViewBox().menu
            menu.addAction("target subscriptions").triggered.connect(
                lambda: self.open_subscription_editor("targetstreams")
            )
            menu.addAction("phase mask subscriptions").triggered.connect(
                lambda: self.open_subscription_editor("phasestreams")
            )
            menu.addAction("configure").triggered.connect(self.open_config_editor)

        self.update_subscriptions()

    def update_subscriptions(self):
        self.target_stream.unsubscribe()
        self.target_stream.subscribe(self._targetstreams)
        self.phase_stream.unsubscribe()
        self.phase_stream.subscribe(self._phasestreams)
        self.target.bar.setColorMap(
            pg.colormap.get(self._target_cmap, source="matplotlib")
        )
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
        while stream.has_new_data():
            message = stream.recv()
            if message is not None and len(message) >= 3 and message[2].ndim == 2:
                latest = message
        return latest

    def poll(self):
        target = self._latest(self.target_stream)
        if target is not None:
            _, head, image = target
            self.target.show(image, head, (0.0, max(float(image.max()), 1e-12)))
        phase = self._latest(self.phase_stream)
        if phase is not None:
            _, head, image = phase
            self.phase.show(image, head, (0.0, float(self._phase_period)))


def main(name):
    run_applet(SlmMonitor, default_name=name)


if __name__ == "__main__":
    run_applet(SlmMonitor, default_name="SLM Monitor")
