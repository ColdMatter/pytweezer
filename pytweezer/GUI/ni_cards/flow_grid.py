"""A grid of equal cells that uses as many columns as the width allows."""

from PyQt6.QtWidgets import QGridLayout, QWidget

CELL_WIDTH = 170


class FlowGrid(QWidget):
    """Fills row by row with ``width // CELL_WIDTH`` columns, re-flowing on resize.

    The layout's minimum width is that of the current columns, so a grid inside a
    scroll area cannot narrow itself: its owner calls :meth:`set_columns` from the
    viewport's width instead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self._widgets: list[QWidget] = []
        self._columns = 1

    def add(self, widget: QWidget) -> None:
        self._widgets.append(widget)
        self._place()

    def clear(self) -> None:
        for widget in self._widgets:
            self._grid.removeWidget(widget)
            widget.deleteLater()
        self._widgets = []

    def column_count(self) -> int:
        return self._columns

    def set_columns(self, columns: int) -> None:
        columns = max(1, columns)
        if columns != self._columns:
            self._columns = columns
            self._place()

    def resizeEvent(self, event) -> None:
        self.set_columns(event.size().width() // CELL_WIDTH)
        super().resizeEvent(event)

    def _place(self) -> None:
        for widget in self._widgets:
            self._grid.removeWidget(widget)
        for index, widget in enumerate(self._widgets):
            self._grid.addWidget(widget, index // self._columns, index % self._columns)
        for column in range(max(self._grid.columnCount(), self._columns)):
            self._grid.setColumnStretch(column, 1 if column < self._columns else 0)
        self.updateGeometry()
