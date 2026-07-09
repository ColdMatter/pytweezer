"""Shared dark theme for the pytweezer GUIs.

Applied once, app-wide, via ``QApplication.setStyleSheet`` (see ``bin/gui.py``'s
``_run()``). Individual widgets opt into specific rules with ``setObjectName``
or the ``role``/``state`` dynamic properties referenced below, rather than each
carrying its own inline ``setStyleSheet`` call.

Status indicators (:data:`STATE_STYLE`) are deliberately colour *and* text: a
dot for an at-a-glance read plus an always-visible label, so state is never
conveyed by colour alone.
"""

# Traffic-light colour + label for every status a process/server/device can be
# in. Two mechanisms feed this table with different keys but the *same* user
# vocabulary, so a process reads identically no matter how its state was
# learned: a *reachability probe* yields ``up``/``down`` and a *managed
# subprocess* yields ``running``/``stopped``/``crashed`` — but ``up`` and
# ``running`` both display "Running", and ``down`` and ``stopped`` both display
# "Stopped". ``crashed`` (a subprocess that died unexpectedly, which only
# self-polling can distinguish) stays red as a genuine alert.
STATE_STYLE = {
    "running": ("#2ecc71", "Running"),
    "up": ("#2ecc71", "Running"),
    "stopped": ("#6b6c76", "Stopped"),
    "down": ("#6b6c76", "Stopped"),
    "crashed": ("#e74c3c", "Crashed"),
    "disabled": ("#6b6c76", "Disabled"),
    "unknown": ("#6b6c76", "Unknown"),
}


def resolve_state(status):
    """Normalize a bool/None/str status value to a :data:`STATE_STYLE` key."""
    if isinstance(status, str) and status in STATE_STYLE:
        return status
    if status is True:
        return "up"
    if status is False:
        return "down"
    return "unknown"


def state_style(status):
    """Return ``(color, label)`` for any status representation."""
    return STATE_STYLE[resolve_state(status)]


# Base UI font. Kept in sync with the ``QWidget`` rule in DARK_STYLESHEET;
# exposed as data so widgets that need accurate text measurements (e.g. row
# heights) can build a matching QFont — QSS font settings aren't reflected in
# a widget's QFontMetrics.
UI_FONT_FAMILY = "Segoe UI"
UI_FONT_POINT_SIZE = 10


DARK_STYLESHEET = """
QWidget {
    background-color: #1b1c22;
    color: #e6e6e6;
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}

QMainWindow, QScrollArea {
    background-color: #1b1c22;
    border: none;
}

QToolTip {
    background-color: #2f3038;
    color: #e6e6e6;
    border: 1px solid #3d3e47;
    padding: 4px;
}

QTabWidget::pane {
    background-color: #1b1c22;
    border-top: 1px solid #33343d;
}

QTabBar::tab {
    background-color: #24252c;
    color: #9a9aa5;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #2f3038;
    color: #e6e6e6;
    border-bottom: 2px solid #5b8def;
}

QTabBar::tab:hover {
    color: #e6e6e6;
}

/* Panels live in dock widgets so they can be dragged out into their own
   windows; style their title bars and the drag separators to match. Do NOT
   override titlebar-normal-icon here — that icon *is* the float/undock button,
   and hiding it removes the one-click way to pop a tab out into its own
   window. */
QDockWidget {
    color: #c8c8d0;
    font-weight: 600;
}

QDockWidget::title {
    background: #24252c;
    padding: 6px 26px 6px 10px;
    border-bottom: 1px solid #33343d;
}

QDockWidget::float-button {
    subcontrol-position: right center;
    subcontrol-origin: margin;
    right: 4px;
    icon-size: 14px;
}

QDockWidget::float-button:hover {
    background: #383944;
    border-radius: 3px;
}

QMainWindow::separator {
    background: #33343d;
    width: 4px;
    height: 4px;
}

QMainWindow::separator:hover {
    background: #5b8def;
}

QLabel {
    background: transparent;
}

QLabel[role="heading"] {
    color: #9a9aa5;
    font-weight: 600;
    font-size: 9pt;
}

QLabel#StatusLabel {
    font-weight: 600;
}

QFrame#ProcessTile {
    background-color: #24252c;
    border: 1px solid #33343d;
    border-radius: 8px;
}

/* Left-edge accent mirrors the status colour. up/running are the same state
   (see STATE_STYLE), so they share a rule; likewise down/stopped. */
QFrame#ProcessTile[state="running"],
QFrame#ProcessTile[state="up"] {
    border-left: 4px solid #2ecc71;
}

QFrame#ProcessTile[state="crashed"] {
    border-left: 4px solid #e74c3c;
}

QFrame#ProcessTile[state="stopped"],
QFrame#ProcessTile[state="down"],
QFrame#ProcessTile[state="disabled"],
QFrame#ProcessTile[state="unknown"] {
    border-left: 4px solid #6b6c76;
}

QPushButton {
    background-color: #2f3038;
    color: #e6e6e6;
    border: 1px solid #3d3e47;
    border-radius: 6px;
    padding: 6px 12px;
}

QPushButton:hover {
    background-color: #383944;
    border-color: #5b8def;
}

QPushButton:pressed {
    background-color: #23242b;
}

/* Single Start/Stop toggle: green when it will start a stopped process,
   red when it will stop a running one. */
QPushButton#ToggleButton[kind="start"] {
    color: #2ecc71;
    border-color: #2f5140;
}

QPushButton#ToggleButton[kind="start"]:hover {
    background-color: #2ecc71;
    color: #12351f;
    border-color: #2ecc71;
}

QPushButton#ToggleButton[kind="stop"] {
    color: #e74c3c;
    border-color: #5a2f2c;
}

QPushButton#ToggleButton[kind="stop"]:hover {
    background-color: #e74c3c;
    color: #3a1512;
    border-color: #e74c3c;
}

QTableWidget, QTreeWidget, QListWidget, QTableView, QTreeView, QListView {
    background-color: #24252c;
    alternate-background-color: #26272f;
    color: #e6e6e6;
    border: 1px solid #33343d;
    border-radius: 6px;
    gridline-color: #33343d;
}

QTreeView::item, QListView::item, QTableView::item {
    color: #e6e6e6;
}

QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {
    background-color: #5b8def;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #24252c;
    color: #9a9aa5;
    border: none;
    border-bottom: 1px solid #33343d;
    padding: 4px;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #24252c;
    border: 1px solid #3d3e47;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #5b8def;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #5b8def;
}

QScrollBar:vertical {
    background: #1b1c22;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3d3e47;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #5b8def;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #1b1c22;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #3d3e47;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #5b8def;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""


def apply_dot_style(label, status):
    """Style a status-dot ``QLabel`` (``●``) for the given status."""
    color, _ = state_style(status)
    label.setStyleSheet(f"color: {color}; font-size: 14px;")


def apply_label_style(label, status, text=None):
    """Style a status-text ``QLabel`` for the given status."""
    color, default_text = state_style(status)
    label.setText(text if text is not None else default_text)
    label.setStyleSheet(f"color: {color}; font-weight: 600;")
