"""The settings panel the Camera applet shows to the left of the live image.

A :class:`CameraSidebar` builds itself from a camera's control list (the dicts of
:meth:`pytweezer.drivers.camera_base.Camera.controls`) and makes no device calls:
it emits what the user asked for and shows what it is told. Values it receives
and emits are SI, as the drivers keep them; fields show the control's display
unit.
"""

from __future__ import annotations

import math
import time
from typing import Any

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.components import set_state
from pytweezer.GUI.theme import state_style

STATES = ("idle", "live", "held", "unknown_hold", "unavailable")

# The controls every camera driver may share. A section holding any other control
# is driver-specific and carries the camera's model.
SHARED_CONTROLS = frozenset(
    {
        "exposure",
        "exposure_mode",
        "trigger_source",
        "trigger_polarity",
        "gain",
        "binning",
        "roi_x0",
        "roi_width",
        "roi_y0",
        "roi_height",
    }
)
ROI_CONTROLS = ("roi_x0", "roi_width", "roi_y0", "roi_height")
ROI_ROWS = (
    ("x0, width", "roi_x0", "roi_width"),
    ("y0, height", "roi_y0", "roi_height"),
)

CHOICE_WORDS = {
    "int": "Internal",
    "internal": "Internal",
    "ext": "External",
    "rise": "Rising",
    "fall": "Falling",
}

SIDEBAR_WIDTH = 310
NUMBER_FIELD_WIDTH = 76
ROI_FIELD_WIDTH = 54
UNIT_WIDTH = 24
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


def choice_text(value: str) -> str:
    """How a choice control's stored value reads in the combo box."""
    return CHOICE_WORDS.get(value, str(value).capitalize())


def format_number(value: float) -> str:
    """A display-unit value without floating-point noise."""
    return f"{value:.6g}"


def age_text(seconds: float) -> str:
    """``seconds`` as "12 s ago", "3 min ago" or "2 h ago"."""
    seconds = max(0, int(seconds))
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds} s ago"
    if seconds < SECONDS_PER_HOUR:
        return f"{seconds // SECONDS_PER_MINUTE} min ago"
    return f"{seconds // SECONDS_PER_HOUR} h ago"


def celsius_text(celsius: float) -> str:
    return f"{celsius:.1f}".replace("-", "−") + " °C"


class _Field(QObject):
    """One control's editor, its error line, and the value last shown in it.

    Emits :attr:`edited` with the SI value when the user commits an edit (Enter,
    focus-out, selection or tick), never per keystroke.
    """

    edited = pyqtSignal(str, object)

    def __init__(self, control: dict, width: int | None = None):
        super().__init__()
        self.control = control
        self.name = control["name"]
        self.value = control.get("value", control.get("default"))
        self.scale = control.get("scale") or 1.0
        self.label = QLabel(control["label"])
        self.unit = QLabel(control.get("unit", ""))
        self.unit.setProperty("role", "regionHint")
        self.unit.setFixedWidth(UNIT_WIDTH)
        self.error = QLabel()
        self.error.setProperty("role", "regionHint")
        self.error.setProperty("state", "error")
        self.error.setWordWrap(True)
        self.error.hide()
        self._error_message = ""
        self._busy = False
        kind = control["kind"]
        if kind == "bool":
            self.editor: QWidget = QCheckBox()
            self.editor.clicked.connect(self._ticked)
        elif kind == "choice":
            self.editor = QComboBox()
            for choice in control["choices"]:
                self.editor.addItem(choice_text(choice), choice)
            self.editor.setMinimumWidth(NUMBER_FIELD_WIDTH + UNIT_WIDTH)
            self.editor.activated.connect(self._choice_activated)
        else:
            self.editor = QLineEdit()
            self.editor.setProperty("role", "cameraNumber")
            self.editor.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.editor.setFixedWidth(width or NUMBER_FIELD_WIDTH)
            self.editor.editingFinished.connect(self._commit_text)
        self.show_value(self.value)

    def is_being_edited(self) -> bool:
        """Whether the user has typed into the field, or has its drop-down open."""
        if isinstance(self.editor, QLineEdit):
            return self.editor.isModified()
        if isinstance(self.editor, QComboBox):
            return self.editor.view().isVisible()
        return False

    def show_value(self, value: Any) -> None:
        """Show ``value`` (SI) in the editor, without emitting."""
        self.value = value
        if isinstance(self.editor, QCheckBox):
            self.editor.setChecked(bool(value))
        elif isinstance(self.editor, QComboBox):
            self.editor.setCurrentIndex(self.editor.findData(value))
        else:
            self.editor.setText(self._display_text(value))
            self.editor.setModified(False)

    def update_value(self, value: Any) -> None:
        """Show a value read from the camera, unless the user is editing the field."""
        self.value = value
        if not self.is_being_edited():
            self.show_value(value)

    def show_error(self, message: str) -> None:
        """Say why a value was not accepted, and put the last accepted one back."""
        self.show_value(self.value)
        self._set_error(message)

    def _set_error(self, message: str) -> None:
        self._error_message = message
        self.error.setText(message)
        self.error.setVisible(bool(message))
        self._restyle()

    def set_enabled(self, enabled: bool, busy: bool = False) -> None:
        self._busy = busy
        for widget in (self.editor, self.label, self.unit):
            widget.setEnabled(enabled)
        self._restyle()

    def _restyle(self) -> None:
        state = "error" if self._error_message else "busy" if self._busy else ""
        set_state(self.editor, state)
        self.editor.setToolTip(self._error_message)

    def _display_text(self, value: Any) -> str:
        if self.control["kind"] == "integer":
            return str(round(value / self.scale))
        return format_number(value / self.scale)

    def _ticked(self, ticked: bool) -> None:
        self._set_error("")
        self.edited.emit(self.name, ticked)

    def _choice_activated(self, index: int) -> None:
        self._set_error("")
        self.edited.emit(self.name, self.editor.itemData(index))

    def _commit_text(self) -> None:
        edit: QLineEdit = self.editor
        if not edit.isModified():
            return
        text = edit.text().strip()
        edit.setModified(False)
        self._set_error("")
        integer = self.control["kind"] == "integer"
        try:
            display = float(text)
        except ValueError:
            display = None
        if display is None or not math.isfinite(display):
            self.show_error("Enter a whole number." if integer else "Enter a number.")
            return
        if integer and not display.is_integer():
            self.show_error("Enter a whole number." if integer else "Enter a number.")
            return
        value = float(f"{display * self.scale:.12g}")
        if integer:
            value = round(value)
        problem = self._range_problem(value)
        if problem:
            self.show_error(problem)
            return
        if value != self.value:
            self.edited.emit(self.name, value)

    def _range_problem(self, value: float) -> str:
        unit = f" {self.control['unit']}" if self.control.get("unit") else ""
        minimum, maximum = self.control.get("minimum"), self.control.get("maximum")
        if minimum is not None and value < minimum:
            return f"At least {format_number(minimum / self.scale)}{unit}."
        if maximum is not None and value > maximum:
            return f"At most {format_number(maximum / self.scale)}{unit}."
        return ""


class _Section(QFrame):
    """A titled group of settings that folds away, tagged with the camera model if driver-specific."""

    def __init__(self, title: str, model_tag: str, collapsed: bool):
        super().__init__()
        self.setObjectName("CameraSection")
        self.title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 6)
        outer.setSpacing(6)
        header = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setObjectName("SectionToggle")
        self.toggle.setText(title.replace("&", "&&"))
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.addWidget(self.toggle)
        header.addStretch(1)
        if model_tag:
            tag = QLabel(model_tag)
            tag.setProperty("role", "regionHint")
            tag.setToolTip(f"These settings are specific to the {model_tag}")
            header.addWidget(tag)
        outer.addLayout(header)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(5)
        self.grid.setColumnStretch(0, 1)
        outer.addWidget(self.body)
        self.toggle.toggled.connect(self._show_body)
        self.toggle.setChecked(not collapsed)
        self._show_body(not collapsed)

    def is_collapsed(self) -> bool:
        return not self.toggle.isChecked()

    def _show_body(self, open_: bool) -> None:
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(open_)

    def add_row(self, label: QLabel, fields: list[_Field], unit: QLabel | None) -> None:
        row = self.grid.rowCount()
        self.grid.addWidget(label, row, 0)
        holder = QHBoxLayout()
        holder.setSpacing(4)
        for field in fields:
            holder.addWidget(field.editor)
        self.grid.addLayout(holder, row, 1, Qt.AlignmentFlag.AlignRight)
        if unit is not None:
            self.grid.addWidget(unit, row, 2)
        for field in fields:
            self.grid.addWidget(field.error, self.grid.rowCount(), 0, 1, 3)


class CameraSidebar(QFrame):
    """Camera picker, live/snap buttons, status and the camera's settings.

    Snap is enabled when the camera is not live, held or unavailable and its
    ``trigger_source`` control reads ``"int"``; a camera with no such control
    keeps Snap enabled. Full sensor is enabled only when the ``roi_width`` and
    ``roi_height`` controls give a ``maximum``.

    Signals:
        camera_chosen(str): the user picked a camera.
        apply_requested(dict): the user committed an edit; ``{control name: SI value}``.
        live_toggled(bool): the user asked for live to start (True) or stop (False).
        snap_requested(): the user asked for one frame.
        roi_from_drawn_requested(): the user asked for the ROI drawn on the image.
        full_sensor_requested(): the user asked for the whole sensor.
        preset_chosen(str): the user picked a saved ROI; apply it.
        preset_save_requested(str): the user named the camera's current ROI to save.
        preset_delete_requested(str): the user confirmed deleting a saved ROI.
    """

    camera_chosen = pyqtSignal(str)
    apply_requested = pyqtSignal(dict)
    live_toggled = pyqtSignal(bool)
    snap_requested = pyqtSignal()
    roi_from_drawn_requested = pyqtSignal()
    full_sensor_requested = pyqtSignal()
    preset_chosen = pyqtSignal(str)
    preset_save_requested = pyqtSignal(str)
    preset_delete_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CameraSidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)
        self._state = "idle"
        self._fps: float | None = None
        self._detail = ""
        self._busy: set[str] = set()
        self._fields: dict[str, _Field] = {}
        self._collapsed: set[str] = set()
        self._presets: dict[str, dict[str, int]] = {}
        self._temperature: dict | None = None
        self._unavailable: dict[str, str] = {}
        self._roi_widgets: list[QWidget] = []
        self._full_sensor: QPushButton | None = None
        self._preset_picker: QComboBox | None = None
        self._preset_save: QPushButton | None = None
        self._preset_delete: QPushButton | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.picker = QComboBox()
        self.picker.activated.connect(self._camera_activated)
        layout.addWidget(self.picker)

        self.pill = QFrame()
        self.pill.setObjectName("CameraPill")
        pill_layout = QVBoxLayout(self.pill)
        pill_layout.setContentsMargins(10, 5, 10, 5)
        pill_layout.setSpacing(0)
        self.pill_title = QLabel()
        self.pill_title.setObjectName("CameraPillTitle")
        self.pill_title.setWordWrap(True)
        self.pill_detail = QLabel()
        self.pill_detail.setObjectName("CameraPillDetail")
        self.pill_detail.setWordWrap(True)
        pill_layout.addWidget(self.pill_title)
        pill_layout.addWidget(self.pill_detail)
        layout.addWidget(self.pill)

        self.temperature_label = QLabel()
        self.temperature_label.setObjectName("CameraTemperature")
        self.temperature_label.setWordWrap(True)
        self.temperature_label.hide()
        layout.addWidget(self.temperature_label)

        buttons = QHBoxLayout()
        self.live_button = QPushButton()
        self.live_button.setObjectName("ToggleButton")
        self.live_button.clicked.connect(
            lambda: self.live_toggled.emit(self._state != "live")
        )
        self.snap_button = QPushButton("Snap")
        self.snap_button.clicked.connect(self.snap_requested)
        buttons.addWidget(self.live_button, 1)
        buttons.addWidget(self.snap_button, 1)
        layout.addLayout(buttons)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("CameraSidebarScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setObjectName("CameraSidebarContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 4, 0)
        self.content_layout.setSpacing(0)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

        self._clock = QTimer(self, interval=1000)
        self._clock.timeout.connect(self._refresh_temperature)
        self._clock.start()

        self.set_cameras([], {})
        self.set_controls([])
        self.set_state("idle")

    # ---- camera picker -------------------------------------------------

    def set_cameras(self, names: list[str], unavailable: dict[str, str]) -> None:
        """List the cameras; those in ``unavailable`` (name to reason) are greyed.

        The chosen camera stays chosen if it is still listed. Nothing is emitted.
        """
        chosen = self.current_camera()
        self._unavailable = dict(unavailable)
        self.picker.clear()
        for name in names:
            self._add_camera(name, unavailable.get(name))
        if not names:
            self.picker.setPlaceholderText("No cameras")
        self.set_current_camera(chosen)

    def set_current_camera(self, name: str | None) -> None:
        """Show ``name`` as the chosen camera, without emitting.

        A camera that is not listed is added, marked unavailable, rather than
        replaced by another one whose settings are not on screen.
        """
        if name is None:
            if self.picker.count():
                self.picker.setCurrentIndex(0)
            return
        if self.picker.findData(name) < 0:
            self._add_camera(name, "It is not in the list of cameras")
        self.picker.setCurrentIndex(self.picker.findData(name))

    def _add_camera(self, name: str, unavailable_reason: str | None) -> None:
        text = name if unavailable_reason is None else f"{name} (unavailable)"
        self.picker.addItem(text, name)
        if unavailable_reason is not None:
            index = self.picker.count() - 1
            grey = QBrush(QColor(state_style("stopped")[0]))
            self.picker.setItemData(
                index, unavailable_reason, Qt.ItemDataRole.ToolTipRole
            )
            self.picker.setItemData(index, grey, Qt.ItemDataRole.ForegroundRole)

    def current_camera(self) -> str | None:
        """The name of the camera shown in the picker, or ``None`` if there is none."""
        return self.picker.currentData() if self.picker.count() else None

    def _camera_activated(self, index: int) -> None:
        self.camera_chosen.emit(self.picker.itemData(index))

    # ---- settings ------------------------------------------------------

    def set_controls(self, controls: list[dict], model: str = "") -> None:
        """Rebuild the settings from a camera's control list.

        Controls are grouped into one collapsible section per ``section``. A
        section holding a control that not every camera has is tagged with
        ``model``, the camera's model name.
        """
        for section in self._sections():
            self._collapsed.discard(section.title)
            if section.is_collapsed():
                self._collapsed.add(section.title)
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self._fields = {}
        self._roi_widgets = []
        self._full_sensor = None
        self._preset_picker = self._preset_save = self._preset_delete = None

        by_section: dict[str, list[dict]] = {}
        for control in controls:
            by_section.setdefault(control["section"], []).append(control)
        for title, section_controls in by_section.items():
            driver_specific = any(
                c["name"] not in SHARED_CONTROLS for c in section_controls
            )
            section = _Section(
                title, model if driver_specific else "", title in self._collapsed
            )
            self._fill_section(section, section_controls)
            self.content_layout.addWidget(section)
        if not controls:
            empty = QLabel("No settings to show.")
            empty.setProperty("role", "regionHint")
            self.content_layout.addWidget(empty)
        self.content_layout.addStretch(1)
        self.set_presets(self._presets)

    def show_values(self, controls: list[dict]) -> None:
        """Update the fields from a control list without rebuilding.

        A field the user is typing into, or whose drop-down is open, keeps what it shows.
        """
        for control in controls:
            field = self._fields.get(control["name"])
            if field is not None:
                field.update_value(control["value"])
        self._sync_preset_selection()

    def show_error(self, name: str, message: str) -> None:
        """Mark ``name`` as refused: its field returns to the last accepted value and says why."""
        field = self._fields.get(name)
        if field is not None:
            field.show_error(message)
        self._sync_preset_selection()

    def set_busy(self, names: set[str]) -> None:
        """Mark the controls with a call in flight; they cannot be edited meanwhile."""
        self._busy = set(names)
        self._refresh_enabled()

    def _sections(self) -> list[_Section]:
        return self.content.findChildren(_Section)

    def _fill_section(self, section: _Section, controls: list[dict]) -> None:
        names = {c["name"] for c in controls}
        paired = all(name in names for name in ROI_CONTROLS)
        by_name = {c["name"]: c for c in controls}
        roi_done = False
        for control in controls:
            name = control["name"]
            if paired and name in ROI_CONTROLS:
                if not roi_done:
                    roi_done = True
                    for label, first, second in ROI_ROWS:
                        fields = [
                            self._make_field(by_name[first], ROI_FIELD_WIDTH),
                            self._make_field(by_name[second], ROI_FIELD_WIDTH),
                        ]
                        fields[0].label.setText(label)
                        section.add_row(fields[0].label, fields, fields[0].unit)
                continue
            field = self._make_field(control)
            section.add_row(field.label, [field], field.unit)
        if paired:
            self._add_roi_tools(section)

    def _make_field(self, control: dict, width: int | None = None) -> _Field:
        field = _Field(control, width)
        field.edited.connect(self._field_edited)
        self._fields[field.name] = field
        return field

    def _field_edited(self, name: str, value: Any) -> None:
        self.apply_requested.emit({name: value})

    # ---- ROI tools and presets ----------------------------------------

    def _add_roi_tools(self, section: _Section) -> None:
        from_drawn = QPushButton("From drawn ROI")
        from_drawn.setToolTip("Use the ROI drawn on the image as the camera's ROI")
        from_drawn.clicked.connect(self.roi_from_drawn_requested)
        full = QPushButton("Full sensor")
        full.clicked.connect(self.full_sensor_requested)
        row = section.grid.rowCount()
        buttons = QHBoxLayout()
        buttons.addWidget(from_drawn, 1)
        buttons.addWidget(full, 1)
        section.grid.addLayout(buttons, row, 0, 1, 3)

        picker = QComboBox()
        picker.setMinimumContentsLength(6)
        picker.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        picker.activated.connect(self._preset_activated)
        save = QPushButton("Save as…")
        save.setToolTip("Save the camera's current ROI under a name")
        save.clicked.connect(self._save_preset)
        delete = QPushButton("Delete")
        delete.setObjectName("DangerButton")
        delete.clicked.connect(self._delete_preset)
        presets = QHBoxLayout()
        presets.addWidget(picker, 1)
        presets.addWidget(save)
        presets.addSpacing(6)
        presets.addWidget(delete)
        section.grid.addLayout(presets, row + 1, 0, 1, 3)
        self._roi_widgets = [from_drawn]
        self._full_sensor = full
        self._preset_picker, self._preset_save, self._preset_delete = (
            picker,
            save,
            delete,
        )

    def set_presets(self, presets: dict[str, dict[str, int]]) -> None:
        """List the saved ROIs, each as its ``roi_x0``, ``roi_width``, ``roi_y0``, ``roi_height``.

        The picker shows the saved ROI equal to the camera's current ROI, or its
        placeholder if there is none.
        """
        self._presets = {name: dict(values) for name, values in presets.items()}
        picker = self._preset_picker
        if picker is not None:
            picker.clear()
            picker.addItems(list(self._presets))
            picker.setPlaceholderText(
                "Saved ROIs" if self._presets else "No saved ROIs"
            )
        self._sync_preset_selection()

    def _matching_preset(self) -> str:
        """The saved ROI equal to the camera's ROI as last shown, or ``""``."""
        if not all(name in self._fields for name in ROI_CONTROLS):
            return ""
        current = {name: self._fields[name].value for name in ROI_CONTROLS}
        for name, values in self._presets.items():
            if all(values.get(key) == current[key] for key in ROI_CONTROLS):
                return name
        return ""

    def _sync_preset_selection(self) -> None:
        picker = self._preset_picker
        if picker is not None and not picker.view().isVisible():
            picker.setCurrentIndex(picker.findText(self._matching_preset()))
        self._refresh_enabled()

    def _preset_activated(self, index: int) -> None:
        self._refresh_enabled()
        self.preset_chosen.emit(self._preset_picker.itemText(index))

    def _save_preset(self) -> None:
        name, accepted = QInputDialog.getText(
            self,
            "Save ROI as",
            "Name for the camera's current ROI:",
            text=self._matching_preset(),
        )
        name = name.strip()
        if not accepted or not name:
            return
        if name in self._presets:
            replace = QMessageBox.question(
                self,
                "Replace saved ROI",
                f"A saved ROI called “{name}” already exists. Replace it?",
            )
            if replace != QMessageBox.StandardButton.Yes:
                return
        self.preset_save_requested.emit(name)

    def _delete_preset(self) -> None:
        name = self._preset_picker.currentText()
        if not name:
            return
        confirmed = QMessageBox.question(
            self,
            "Delete saved ROI",
            f"Delete the saved ROI “{name}”? It is shared with everyone "
            "using this camera and cannot be undone.",
        )
        if confirmed == QMessageBox.StandardButton.Yes:
            self.preset_delete_requested.emit(name)

    # ---- state ---------------------------------------------------------

    def set_state(self, state: str, detail: str = "") -> None:
        """Show who has the camera and what it is doing.

        Args:
            state: ``"idle"``, ``"live"``, ``"held"`` (a task has the camera: every
                control and Live/Snap are disabled), ``"unknown_hold"`` (cannot tell
                whether a task has it) or ``"unavailable"``.
            detail: A second line in the status pill; for ``"held"`` it is instead
                the experiment's name in the pill's title.
        """
        if state not in STATES:
            raise ValueError(f"unknown camera state {state!r}; use one of {STATES}")
        self._state = state
        self._detail = detail
        self._render_state()
        self._refresh_enabled()

    def set_fps(self, fps: float | None) -> None:
        """Show the live frame rate in the status pill (``None`` for none)."""
        self._fps = fps
        self._render_state()

    def _render_state(self) -> None:
        state, detail = self._state, self._detail
        second = detail
        if state == "idle":
            title = "Idle"
        elif state == "live":
            title = "Live" if self._fps is None else f"Live · {self._fps:.1f} fps"
        elif state == "held":
            title = f"Held by {detail}" if detail else "Held by an experiment"
            second = "Settings are read-only until it finishes"
        elif state == "unknown_hold":
            title = "Hold unknown"
            second = detail or "Cannot tell whether an experiment is using the camera"
        else:
            title = "Unavailable"
        self.pill_title.setText(f"● {title}")
        self.pill_detail.setText(second)
        self.pill_detail.setVisible(bool(second))
        set_state(self.pill, state)
        live = state == "live"
        self.live_button.setText("■ Stop live" if live else "▶ Live")
        self.live_button.setProperty("kind", "stop" if live else "start")
        set_state(self.live_button, state)

    def _refresh_enabled(self) -> None:
        locked = self._state in ("held", "unavailable")
        for name, field in self._fields.items():
            field.set_enabled(not locked and name not in self._busy, name in self._busy)
        for widget in self._roi_widgets:
            widget.setEnabled(not locked)
        if self._full_sensor is not None:
            sized = self._sensor_size_known()
            self._full_sensor.setEnabled(not locked and sized)
            self._full_sensor.setToolTip(
                "Read the whole sensor"
                if sized
                else "The camera does not say how large its sensor is"
            )
        picker, save, delete = (
            self._preset_picker,
            self._preset_save,
            self._preset_delete,
        )
        if picker is not None:
            picker.setEnabled(not locked and bool(self._presets))
            save.setEnabled(not locked)
            delete.setEnabled(not locked and picker.currentIndex() >= 0)
        self.live_button.setEnabled(not locked)
        reason = self._snap_blocked_reason()
        self.snap_button.setEnabled(not reason)
        self.snap_button.setToolTip(reason or "Take one frame")

    def _sensor_size_known(self) -> bool:
        return all(
            name in self._fields
            and self._fields[name].control.get("maximum") is not None
            for name in ("roi_width", "roi_height")
        )

    def _snap_blocked_reason(self) -> str:
        if self._state == "held":
            return "An experiment has the camera"
        if self._state == "unavailable":
            return "The camera is unavailable"
        if self._state == "live":
            return "Stop live first"
        trigger = self._fields.get("trigger_source")
        if trigger is not None and trigger.value != "int":
            return "Snap would wait for the trigger: set the trigger source to Internal"
        return ""

    # ---- sensor temperature -------------------------------------------

    def set_temperature(self, reading: dict | None) -> None:
        """Show the camera monitor's last sensor temperature, or hide it for ``None``.

        Args:
            reading: ``{"celsius", "time", "max_celsius"}`` as
                :meth:`~pytweezer.drivers.camera_base.Camera.sensor_temperature` gives it;
                the age shown is counted on from ``time`` by the sidebar itself.
        """
        self._temperature = reading
        self._refresh_temperature()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_temperature()

    def _refresh_temperature(self) -> None:
        reading = self._temperature
        if reading is None:
            self.temperature_label.hide()
            return
        if not self.isVisible():
            return
        celsius, limit = reading["celsius"], reading.get("max_celsius")
        age = age_text(time.time() - reading["time"])
        too_warm = limit is not None and celsius > limit
        text = f"Sensor {celsius_text(celsius)}"
        if too_warm:
            text += f", over {celsius_text(limit)}"
        self.temperature_label.setText(f"{text} · {age}")
        limit_text = (
            f"Limit {celsius_text(limit)}" if limit is not None else "No limit set"
        )
        self.temperature_label.setToolTip(
            f"Sensor temperature as last read by the camera's monitor, {age}. {limit_text}."
        )
        set_state(self.temperature_label, "alarm" if too_warm else "")
        self.temperature_label.show()
