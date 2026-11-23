"""A probability-density histogram plot styled like every pytweezer plot."""

import numpy as np
import pyqtgraph as pg

from pytweezer.GUI.theme import plot_title_style, style_plot


class HistogramPlot(pg.PlotItem):
    """A plot of step-drawn probability-density histograms.

    Args:
        title: Plot title.
        x_label: Bottom axis label.
        parent: Parent graphics item.
    """

    def __init__(self, title: str, x_label: str = "", parent=None):
        super().__init__(parent)
        self._histograms: list[pg.PlotDataItem] = []
        self.setTitle(title, **plot_title_style())
        style_plot(self)
        self.showGrid(x=True, y=True, alpha=0.15)
        if x_label:
            self.setLabel("bottom", x_label)
        self.setLabel("left", "probability density")
        self.getAxis("left").enableAutoSIPrefix(False)

    def add_histogram(
        self,
        colour,
        *,
        filled: bool = False,
        alpha: int = 255,
        width: float = 1,
        name: str | None = None,
    ) -> pg.PlotDataItem:
        """Add an empty step curve to fill with :meth:`draw`.

        ``alpha`` (0-255) makes the fill translucent when ``filled``, and
        the outline otherwise.
        """
        outline = pg.mkColor(colour)
        options = {"stepMode": "center", "name": name}
        if filled:
            fill = pg.mkColor(colour)
            fill.setAlpha(alpha)
            options |= {"fillLevel": 0, "brush": fill}
        else:
            outline.setAlpha(alpha)
        curve = self.plot(pen=pg.mkPen(outline, width=width), **options)
        self._histograms.append(curve)
        return curve

    @staticmethod
    def draw(
        curve: pg.PlotDataItem,
        values: np.ndarray,
        bins: int,
        value_range: tuple[float, float],
    ) -> None:
        """Histogram ``values`` into ``bins`` over ``value_range`` on ``curve``."""
        density, edges = np.histogram(
            values, bins=bins, range=value_range, density=True
        )
        curve.setData(edges, density)

    def clear_histograms(self) -> None:
        """Remove the curves :meth:`add_histogram` made, and only those."""
        for curve in self._histograms:
            self.removeItem(curve)
        self._histograms = []
