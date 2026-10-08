"""Form for one experiment's arguments, scan and queue settings, built from its schema.

The schema is what :meth:`pytweezer.experiment.Experiment.schema` returns, as
served by the Experiment Manager's catalogue; the editor never imports
experiment code. Values are shown in each argument's display unit and stored
in SI.
"""

import math

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.components import CompactDoubleSpinBox, set_state
from pytweezer.GUI.experiments.motmaster_params import (
    DeviceParameterSource,
    MotMasterBox,
    ParameterFetcher,
)

_INT_LIMIT = 2**31 - 1
_FLOAT_LIMIT = 1e15


class ValueField(QWidget):
    """One argument's value in display units; :meth:`value` returns it in SI."""

    changed = QtCore.pyqtSignal()

    def __init__(self, schema, parent=None, *, show_unit=True, width=240):
        super().__init__(parent)
        self.schema = schema
        self.kind = schema["kind"]
        self.scale = schema.get("scale") or 1.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self.kind == "number":
            widget = CompactDoubleSpinBox()
            low, high = schema.get("min"), schema.get("max")
            widget.setRange(
                -_FLOAT_LIMIT if low is None else low / self.scale,
                _FLOAT_LIMIT if high is None else high / self.scale,
            )
            decimals = schema.get("ndecimals")
            widget.setDecimals(6 if decimals is None else decimals)
            if schema.get("step"):
                widget.setSingleStep(schema["step"] / self.scale)
            widget.valueChanged.connect(self.changed)
        elif self.kind == "integer":
            widget = QSpinBox()
            low, high = schema.get("min"), schema.get("max")
            widget.setRange(
                -_INT_LIMIT if low is None else low,
                _INT_LIMIT if high is None else high,
            )
            widget.valueChanged.connect(self.changed)
        elif self.kind == "bool":
            widget = QCheckBox()
            widget.toggled.connect(self.changed)
        elif self.kind == "choice":
            widget = QComboBox()
            widget.addItems(schema["options"])
            widget.currentTextChanged.connect(self.changed)
        else:
            widget = QLineEdit()
            widget.textChanged.connect(self.changed)
        if schema.get("tooltip"):
            widget.setToolTip(schema["tooltip"])
        if self.kind != "bool":
            widget.setFixedWidth(width)
        self.widget = widget
        layout.addWidget(widget)
        self.unit = QLabel(schema.get("unit") or "")
        self.unit.setProperty("role", "regionHint")
        self.unit.setVisible(show_unit)
        layout.addWidget(self.unit)
        layout.addStretch(1)
        self.set_value(schema["default"])

    def value(self):
        if self.kind == "number":
            return self.widget.value() * self.scale
        if self.kind == "integer":
            return self.widget.value()
        if self.kind == "bool":
            return self.widget.isChecked()
        if self.kind == "choice":
            return self.widget.currentText()
        return self.widget.text()

    def set_value(self, value):
        if self.kind == "number":
            self.widget.setValue(value / self.scale)
        elif self.kind == "integer":
            self.widget.setValue(int(value))
        elif self.kind == "bool":
            self.widget.setChecked(bool(value))
        elif self.kind == "choice":
            self.widget.setCurrentText(value)
        else:
            self.widget.setText(value)

    def is_default(self):
        default = self.schema["default"]
        if self.kind == "number":
            return math.isclose(self.value(), default, rel_tol=1e-9, abs_tol=1e-15)
        return self.value() == default


def parse_list(text, schema):
    """Parse comma-separated values typed in display units into SI values."""
    kind = schema["kind"]
    scale = schema.get("scale") or 1.0
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("the list of values is empty")
    values = []
    for item in items:
        if kind == "number":
            values.append(float(item) * scale)
        elif kind == "integer":
            values.append(int(item))
        elif kind == "bool":
            if item.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{item!r} is not true or false")
            values.append(item.lower() in ("true", "1"))
        elif kind == "choice" and item not in schema["options"]:
            raise ValueError(f"{item!r} is not one of {schema['options']}")
        else:
            values.append(item)
    if kind in ("number", "integer"):
        low, high = schema.get("min"), schema.get("max")
        for value in values:
            if (low is not None and value < low) or (high is not None and value > high):
                raise ValueError(f"{value / scale:g} is outside the allowed range")
    return values


def format_list(values, schema):
    scale = schema.get("scale") or 1.0
    if schema["kind"] == "number":
        return ", ".join(f"{value / scale:g}" for value in values)
    return ", ".join(str(value) for value in values)


class ScanField(QWidget):
    """Linear (start/stop/n) or explicit-list scan of one argument."""

    changed = QtCore.pyqtSignal()

    def __init__(self, name, schema, parent=None):
        super().__init__(parent)
        self.name = name
        self.schema = schema
        numeric = schema["kind"] in ("number", "integer")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.mode = QComboBox()
        self.mode.addItems(["linear", "list"] if numeric else ["list"])
        layout.addWidget(self.mode)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        linear = QWidget()
        linear_layout = QHBoxLayout(linear)
        linear_layout.setContentsMargins(0, 0, 0, 0)
        if numeric:
            self.start = ValueField(schema, show_unit=False, width=110)
            self.stop = ValueField(schema, show_unit=False, width=110)
            self.steps = QSpinBox()
            self.steps.setRange(1, 1_000_000)
            self.steps.setValue(11)
            self.steps.setPrefix("n = ")
            for label, widget in (("from", self.start), ("to", self.stop)):
                linear_layout.addWidget(QLabel(label))
                linear_layout.addWidget(widget)
            unit = QLabel(schema.get("unit") or "")
            unit.setProperty("role", "regionHint")
            linear_layout.addWidget(unit)
            linear_layout.addWidget(self.steps)
            linear_layout.addStretch(1)
            for signal in (
                self.start.changed,
                self.stop.changed,
                self.steps.valueChanged,
            ):
                signal.connect(self.changed)
        self.stack.addWidget(linear)

        self.values = QLineEdit()
        self.values.setPlaceholderText("comma-separated values")
        self.values.textChanged.connect(self.changed)
        self.stack.addWidget(self.values)

        self.mode.currentTextChanged.connect(self._mode_changed)
        self._mode_changed(self.mode.currentText())

    def _mode_changed(self, mode):
        self.stack.setCurrentIndex(0 if mode == "linear" else 1)
        self.changed.emit()

    def axis(self):
        if self.mode.currentText() == "linear":
            return LinearAxis(
                argument=self.name,
                start=self.start.value(),
                stop=self.stop.value(),
                n=self.steps.value(),
            )
        return ListAxis(
            argument=self.name, values=parse_list(self.values.text(), self.schema)
        )

    def n_values(self):
        if self.mode.currentText() == "linear":
            return self.steps.value()
        try:
            return len(parse_list(self.values.text(), self.schema))
        except ValueError:
            return 0

    def set_axis(self, axis):
        if isinstance(axis, LinearAxis) and self.mode.findText("linear") >= 0:
            self.mode.setCurrentText("linear")
            self.start.set_value(axis.start)
            self.stop.set_value(axis.stop)
            self.steps.setValue(axis.n)
        else:
            values = axis.values if isinstance(axis, ListAxis) else []
            self.mode.setCurrentText("list")
            self.values.setText(format_list(values, self.schema))


class ArgumentRow(QtCore.QObject):
    """Label, value-or-scan editor, unit and scan toggle for one argument."""

    changed = QtCore.pyqtSignal()

    def __init__(self, name, schema, parent=None):
        super().__init__(parent)
        self.name = name
        self.schema = schema
        self.label = QLabel(name)
        self.label.setProperty("role", "argument")
        self.label.setToolTip(schema.get("tooltip") or name)
        self.value = ValueField(schema)
        self.scan = ScanField(name, schema)
        self.stack = QStackedWidget()
        self.stack.setFixedWidth(560)
        self.stack.addWidget(self.value)
        self.stack.addWidget(self.scan)
        self.unit = self.value.unit
        self.scan_button = QToolButton()
        self.scan_button.setObjectName("ScanToggle")
        self.scan_button.setText("Scan")
        self.scan_button.setToolTip("Scan this argument instead of using one value")
        self.scan_button.setCheckable(True)
        self.scan_button.toggled.connect(self._scan_toggled)
        self.value.changed.connect(self._update)
        self.scan.changed.connect(self._update)
        self._update()

    def _scan_toggled(self, scanning):
        if (
            scanning
            and self.scan.mode.currentText() == "list"
            and not self.scan.values.text()
        ):
            self.scan.values.setText(format_list([self.value.value()], self.schema))
        if scanning and self.scan.mode.currentText() == "linear":
            self.scan.start.set_value(self.value.value())
            self.scan.stop.set_value(self.value.value())
        self.stack.setCurrentIndex(1 if scanning else 0)
        self._update()

    @property
    def scanning(self):
        return self.scan_button.isChecked()

    def _update(self):
        if self.scanning:
            state = "scanned"
        elif self.value.is_default():
            state = ""
        else:
            state = "modified"
        set_state(self.label, state)
        self.changed.emit()

    def add_to(self, grid, row):
        grid.addWidget(self.label, row, 0)
        grid.addWidget(self.stack, row, 1)
        grid.addWidget(self.scan_button, row, 2)


class ArgumentEditor(QWidget):
    """Edit and submit one experiment."""

    submit_requested = QtCore.pyqtSignal(object)
    last_requested = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.schema = None
        self.rows = {}
        self.fetcher = ParameterFetcher(DeviceParameterSource())
        self.fetcher.fetched.connect(self._parameters_fetched)
        self.fetcher.failed.connect(self._parameters_failed)
        self.motmaster_boxes = {}
        self._searched = {}
        self.setObjectName("ArgumentEditor")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        self.title = QLabel("Pick an experiment on the left to edit it")
        self.title.setProperty("role", "experimentTitle")
        self.module = QLabel()
        self.module.setProperty("role", "regionHint")
        title_row.addWidget(self.title)
        title_row.addStretch(1)
        title_row.addWidget(self.module)
        self.doc = QLabel()
        self.doc.setWordWrap(True)
        self.doc.setProperty("role", "regionHint")
        layout.addLayout(title_row)
        layout.addWidget(self.doc)

        self.arguments_box = QWidget()
        self.arguments_layout = QVBoxLayout(self.arguments_box)
        self.arguments_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.arguments_box)

        settings = QHBoxLayout()
        settings.setSpacing(10)
        scan_box = QGroupBox("Scan")
        scan_box.setObjectName("EditorGroup")
        scan_form = QFormLayout(scan_box)
        scan_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.order = QComboBox()
        self.order.addItems(["nested", "snake", "shuffle"])
        self.order.setToolTip(
            "nested: first scanned argument is the outer loop; snake: inner loops "
            "reverse on alternate passes; shuffle: random order (seed is stored)"
        )
        self.repeat = QComboBox()
        self.repeat.addItems(["point", "scan"])
        self.repeat.setToolTip(
            "point: repeat each point back to back; scan: repeat the whole sweep"
        )
        self.point_count = QLabel()
        self.point_count.setProperty("role", "regionHint")
        scan_form.addRow("Repetitions", self.repetitions)
        scan_form.addRow("Order", self.order)
        scan_form.addRow("Repeat by", self.repeat)
        self.repetitions.valueChanged.connect(self._update_count)
        for widget in (self.repetitions, self.order, self.repeat):
            widget.setMinimumWidth(140)
        settings.addWidget(scan_box, 1)

        queue_box = QGroupBox("Queue")
        queue_box.setObjectName("EditorGroup")
        queue_form = QFormLayout(queue_box)
        queue_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        self.priority = QSpinBox()
        self.priority.setRange(-1000, 1000)
        self.priority.setToolTip("higher runs first")
        self.label = QLineEdit()
        self.label.setPlaceholderText("note saved with the measurement")
        self.label.setMinimumWidth(300)
        self.start_at_enabled = QCheckBox("Start no earlier than")
        self.start_at = QDateTimeEdit(QtCore.QDateTime.currentDateTime())
        self.start_at.setCalendarPopup(True)
        self.start_at.setEnabled(False)
        self.start_at_enabled.toggled.connect(self.start_at.setEnabled)
        self.priority.setMinimumWidth(140)
        queue_form.addRow("Priority", self.priority)
        queue_form.addRow("Label", self.label)
        queue_form.addRow(self.start_at_enabled, self.start_at)
        settings.addWidget(queue_box, 1)
        layout.addLayout(settings)

        buttons = QHBoxLayout()
        self.defaults_button = QPushButton("Defaults")
        self.defaults_button.clicked.connect(self.reset_to_defaults)
        self.last_button = QPushButton("Last submitted")
        self.last_button.clicked.connect(
            lambda: (
                self.schema
                and self.last_requested.emit(
                    self.schema["module"], self.schema["class_name"]
                )
            )
        )
        self.submit_button = QPushButton("Submit")
        self.submit_button.setObjectName("PrimaryButton")
        self.submit_button.setDefault(True)
        self.submit_button.clicked.connect(self._submit)
        buttons.addWidget(self.defaults_button)
        buttons.addWidget(self.last_button)
        buttons.addStretch(1)
        buttons.addWidget(self.point_count)
        buttons.addWidget(self.submit_button)
        layout.addLayout(buttons)
        self.error = QLabel()
        self.error.setObjectName("StatusLabel")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        layout.addStretch(1)
        self._set_enabled(False)

    def _set_enabled(self, enabled):
        for widget in (self.defaults_button, self.last_button, self.submit_button):
            widget.setEnabled(enabled)

    # -- building --------------------------------------------------------

    def set_experiment(self, schema):
        self.schema = schema
        self.title.setText(schema["class_name"])
        self.module.setText(schema["module"])
        self.doc.setText(schema.get("doc", ""))
        self.doc.setVisible(bool(schema.get("doc")))
        while self.arguments_layout.count():
            item = self.arguments_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.rows = {}

        groups = {}
        for name, argument in schema["arguments"].items():
            groups.setdefault(argument.get("group") or "Arguments", []).append(
                (name, argument)
            )
        for group, arguments in groups.items():
            box = QGroupBox(group)
            box.setObjectName("EditorGroup")
            grid = QGridLayout(box)
            grid.setHorizontalSpacing(10)
            # Fields keep a readable width; the slack goes to a trailing column.
            grid.setColumnStretch(3, 1)
            for row, (name, argument) in enumerate(arguments):
                argument_row = ArgumentRow(name, argument, box)
                argument_row.add_to(grid, row)
                argument_row.changed.connect(self._update_count)
                self.rows[name] = argument_row
            self.arguments_layout.addWidget(box)
        self._add_motmaster_boxes(schema)
        if not self.rows and not self.motmaster_boxes:
            self.arguments_layout.addWidget(QLabel("No arguments."))
        self.error.clear()
        self._set_enabled(True)
        self._update_count()

    def reset_to_defaults(self):
        for name in list(self._searched):
            self.remove_motmaster_row(name)
        for row in self.rows.values():
            row.scan_button.setChecked(False)
            row.value.set_value(row.schema["default"])
        self.repetitions.setValue(1)
        self.order.setCurrentText("nested")
        self.repeat.setCurrentText("point")

    def load_request(self, request):
        """Fill the form from ``request``; arguments the experiment no longer has are ignored."""
        self.reset_to_defaults()
        for name in [*request.args, *(axis.argument for axis in request.scan.axes)]:
            attribute, _, parameter = name.partition(".")
            box = self.motmaster_boxes.get(attribute)
            if parameter and box and parameter not in box.declared:
                first_scanned = next(
                    (
                        axis.raw_values()[0]
                        for axis in request.scan.axes
                        if axis.argument == name
                    ),
                    0,
                )
                self.add_motmaster_row(
                    attribute, parameter, request.args.get(name, first_scanned)
                )
        for name, value in request.args.items():
            if name in self.rows:
                try:
                    self.rows[name].value.set_value(value)
                except (TypeError, ValueError):
                    pass
        for axis in request.scan.axes:
            if axis.argument in self.rows:
                row = self.rows[axis.argument]
                row.scan_button.setChecked(True)
                row.scan.set_axis(axis)
        self.repetitions.setValue(request.scan.repetitions)
        self.order.setCurrentText(request.scan.order)
        self.repeat.setCurrentText(request.scan.repeat)
        self.priority.setValue(request.priority)
        self.label.setText(request.label)

    # -- MOTMaster script parameters -------------------------------------

    def set_parameter_source(self, source):
        self.fetcher.source = source

    def _add_motmaster_boxes(self, schema):
        self.motmaster_boxes = {}
        self._searched = {}
        sequencers = {
            attribute: device
            for attribute, device in schema.get("devices", {}).items()
            if device.get("motmaster") is not None
        }
        declared = {attribute: set() for attribute in sequencers}
        for argument in schema["arguments"].values():
            if "motmaster_parameter" in argument and sequencers:
                attribute = argument["motmaster_device"] or next(iter(sequencers))
                declared.setdefault(attribute, set()).add(
                    argument["motmaster_parameter"]
                )
        for attribute, device in sequencers.items():
            box = MotMasterBox(
                attribute,
                device["device"],
                device["motmaster"]["script"],
                declared=declared[attribute],
            )
            box.parameter_chosen.connect(
                lambda name, attribute=attribute: self.add_motmaster_row(
                    attribute, name
                )
            )
            box.retry.clicked.connect(lambda _=False, box=box: self._retry(box))
            self.motmaster_boxes[attribute] = box
            self.arguments_layout.addWidget(box)
            self.fetcher.request(box.device, box.script)

    def _retry(self, box):
        box.set_loading()
        self.fetcher.request(box.device, box.script)

    def add_motmaster_row(self, attribute, parameter, value=None):
        """Add (or return) the row for script parameter ``parameter`` of ``attribute``.

        The row takes its type and default from the script; ``value`` stands in
        when the script's parameters are unknown or lack ``parameter``.
        """
        name = f"{attribute}.{parameter}"
        if name in self.rows:
            return self.rows[name]
        box = self.motmaster_boxes[attribute]
        default = box.defaults.get(parameter, value)
        if default is None:
            raise ValueError(
                f"{name}: script {box.script} has no known parameter "
                f"{parameter!r}; give a value"
            )
        kind = "integer" if isinstance(default, int) else "number"
        row = self._place_motmaster_row(box, name, kind, default, box.grid.rowCount())
        self._searched[name] = (attribute, parameter)
        if box.defaults and parameter not in box.defaults:
            self._flag_missing(row, box.script)
        self._update_count()
        return row

    def _place_motmaster_row(self, box, name, kind, default, grid_row):
        parameter = name.partition(".")[2]
        schema = {
            "kind": kind,
            "default": default,
            "tooltip": f"{parameter} in script {box.script}",
            "group": box.title(),
        }
        row = ArgumentRow(name, schema, box)
        row.add_to(box.grid, grid_row)
        remove = QToolButton()
        remove.setText("×")
        remove.setToolTip("Remove: the script's own value is used")
        remove.clicked.connect(lambda: self.remove_motmaster_row(name))
        box.grid.addWidget(remove, grid_row, 3)
        row.remove_button = remove
        row.changed.connect(self._update_count)
        self.rows[name] = row
        return row

    def remove_motmaster_row(self, name):
        row = self.rows.pop(name)
        attribute, _ = self._searched.pop(name)
        self._discard_row(row, self.motmaster_boxes[attribute].grid)
        self._update_count()

    @staticmethod
    def _discard_row(row, grid):
        for widget in (row.label, row.stack, row.scan_button, row.remove_button):
            grid.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        row.deleteLater()

    def _retype_motmaster_row(self, name, default):
        """Rebuild a row made before the script's parameters were known, keeping its edits."""
        old = self.rows[name]
        box = self.motmaster_boxes[self._searched[name][0]]
        kind = "integer" if isinstance(default, int) else "number"
        scan = old.scan
        kept = [old.value.value(), scan.start.value(), scan.stop.value()]
        try:
            kept += parse_list(scan.values.text(), {"kind": "number"})
        except ValueError:
            pass
        if kind == "integer" and not all(float(v).is_integer() for v in kept):
            # An integer field would round them; request() reports them instead.
            kind = "number"
        if (kind, default) == (old.schema["kind"], old.schema["default"]):
            return
        grid_row = box.grid.getItemPosition(box.grid.indexOf(old.label))[0]
        steps, mode, text = (
            scan.steps.value(),
            scan.mode.currentText(),
            scan.values.text(),
        )
        self._discard_row(old, box.grid)
        row = self._place_motmaster_row(box, name, kind, default, grid_row)
        row.value.set_value(kept[0])
        if old.scanning:
            row.scan_button.setChecked(True)
            row.scan.mode.setCurrentText(mode)
            row.scan.start.set_value(kept[1])
            row.scan.stop.set_value(kept[2])
            row.scan.steps.setValue(steps)
            row.scan.values.setText(text)

    def _flag_missing(self, row, script):
        row.label.setText(f"{row.name} (not in script)")
        row.label.setToolTip(
            f"{row.name}: not in script {script}; remove it or the task will fail"
        )

    def _parameters_fetched(self, device, script, parameters):
        for attribute, box in self.motmaster_boxes.items():
            if (box.device, box.script) != (device, script):
                continue
            box.set_parameters(parameters)
            for name, (owner, parameter) in list(self._searched.items()):
                if owner != attribute:
                    continue
                if parameter in box.defaults:
                    self._retype_motmaster_row(name, box.defaults[parameter])
                else:
                    self._flag_missing(self.rows[name], script)

    def _parameters_failed(self, device, script, error):
        for box in self.motmaster_boxes.values():
            if (box.device, box.script) == (device, script):
                box.set_error(error)

    # -- reading ---------------------------------------------------------

    def scan(self):
        axes = [row.scan.axis() for row in self.rows.values() if row.scanning]
        return Scan(
            axes=axes,
            repetitions=self.repetitions.value(),
            order=self.order.currentText(),
            repeat=self.repeat.currentText(),
        )

    def request(self, submitter=""):
        """The form as a :class:`TaskRequest`; raises ``ValueError`` if a field is invalid."""
        if self.schema is None:
            raise ValueError("no experiment selected")
        due = None
        if self.start_at_enabled.isChecked():
            due = self.start_at.dateTime().toPyDateTime().astimezone()
        self._check_integer_parameters()
        return TaskRequest(
            experiment=self.schema["module"],
            class_name=self.schema["class_name"],
            args={
                name: row.value.value()
                for name, row in self.rows.items()
                if not row.scanning
            },
            scan=self.scan(),
            priority=self.priority.value(),
            due_time=due,
            label=self.label.text(),
            submitter=submitter,
        )

    def _check_integer_parameters(self):
        """Raise ``ValueError`` if a searched Int32 parameter would be sent a non-whole value."""
        for name, (attribute, parameter) in self._searched.items():
            row = self.rows[name]
            script_default = self.motmaster_boxes[attribute].defaults.get(parameter)
            if row.schema["kind"] != "integer" and not isinstance(script_default, int):
                continue
            if not row.scanning:
                values = [row.value.value()]
            else:
                axis = row.scan.axis()
                values = axis.raw_values()
                if isinstance(axis, LinearAxis):
                    if all(float(v).is_integer() for v in values):
                        continue
                    raise ValueError(
                        f"{name} is an integer parameter: {axis.start:g} to "
                        f"{axis.stop:g} in {axis.n} steps gives non-whole values"
                    )
            if not all(float(v).is_integer() for v in values):
                raise ValueError(
                    f"{name} is an integer parameter: "
                    f"{format_list(values, row.schema)} is not whole"
                )

    def _update_count(self):
        count = self.repetitions.value()
        for row in self.rows.values():
            if row.scanning:
                count *= row.scan.n_values()
        self.point_count.setText(f"{count} point{'' if count == 1 else 's'}")

    def _submit(self):
        try:
            request = self.request()
        except ValueError as error:
            self.show_error(str(error))
            return
        self.error.clear()
        self.submit_requested.emit(request)

    def show_error(self, text):
        self.error.setText(text)
        set_state(self.error, "crashed" if text else "")
