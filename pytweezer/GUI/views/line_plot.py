"""A plot of named curves, each a line or markers."""

import pyqtgraph as pg
from PyQt6.QtGui import QAction


class LinePlot(pg.PlotWidget):
    """A plot of curves, each drawn as a line or as markers and found by its key.

    Args:
        parent: Parent widget.
        legend: Show a legend naming each curve.
        x_label: Bottom axis label.
        y_label: Left axis label.
        title: Plot title.
    """

    def __init__(
        self,
        parent=None,
        *,
        legend: bool = True,
        x_label: str = "",
        y_label: str = "",
        title: str | None = None,
    ):
        super().__init__(parent)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self.showGrid(x=True, y=True, alpha=0.15)
        if legend:
            self.addLegend(offset=(-10, 10))
        if x_label:
            self.set_x_label(x_label)
        if y_label:
            self.set_y_label(y_label)
        if title is not None:
            self.set_title(title)

    def curve(
        self, key: str, colour, *, markers: bool = False, **style
    ) -> pg.PlotDataItem:
        """The curve under ``key``, made in ``colour`` if there is none yet.

        A new curve is a line, or ``"o"`` markers with no line if ``markers``;
        ``style`` goes to :meth:`pyqtgraph.PlotItem.plot`, and its legend name
        is ``key`` unless ``style`` gives a ``name``.
        """
        if (existing := self._curves.get(key)) is not None:
            return existing
        style.setdefault("name", key)
        if markers:
            style |= {"pen": None, "symbol": "o", "symbolSize": 6}
            style |= {"symbolPen": None, "symbolBrush": colour}
        else:
            style["pen"] = pg.mkPen(colour, width=1.5)
        curve = self._curves[key] = self.plot(**style)
        return curve

    def set_curve(self, key: str, x, y) -> None:
        """Draw ``y`` against ``x`` on the curve under ``key``."""
        self._curves[key].setData(x, y)

    def remove_curve(self, key: str) -> None:
        self.removeItem(self._curves.pop(key))

    def clear_curves(self) -> None:
        """Remove the curves :meth:`curve` made, and only those."""
        for key in list(self._curves):
            self.remove_curve(key)

    def set_x_label(self, text: str) -> None:
        self.setLabel("bottom", text)

    def set_y_label(self, text: str) -> None:
        self.setLabel("left", text)

    def set_title(self, text: str | None) -> None:
        """Title the plot, or hide the title if ``text`` is ``None``."""
        self.setTitle(text)

    def link_x(self, other: "LinePlot | None") -> None:
        """Pan and zoom the x axis with ``other``'s, or on its own if ``None``."""
        self.plotItem.setXLink(None if other is None else other.plotItem)

    def set_y_axis_width(self, px: int) -> None:
        """Fix the left axis's width, so stacked plots' left edges line up."""
        self.getAxis("left").setWidth(px)

    def show_x_values(self, show: bool) -> None:
        """Show or hide the tick values along the x axis."""
        self.getAxis("bottom").setStyle(showValues=show)

    def add_menu_action(self, text: str, slot) -> QAction:
        """Add ``text`` to the plot's right-click menu, calling ``slot``."""
        action = self.getViewBox().menu.addAction(text)
        action.triggered.connect(slot)
        return action
