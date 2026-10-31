"""The Saved parameters tab: the lab's calibrated values, edited by hand with a reason.

Values live in the Experiment Manager. The tab follows the manager's published
snapshot through an :class:`~pytweezer.GUI.experiments.feed.ExperimentFeed` and
refetches whenever ``saved_parameters_version`` changes while it is visible.
"""

import math
import re
from datetime import datetime

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    submitter_name,
)
from pytweezer.experiment.queue import now
from pytweezer.experiment.saved_parameters import SavedParameter, Value
from pytweezer.GUI.components import Region, set_state
from pytweezer.GUI.experiments.feed import ExperimentFeed

NOT_RESPONDING = "the Experiment Manager is not responding"
#: The newest changes read to show each key's last reason; older keys show none.
REASON_WINDOW = 1000
HISTORY_LIMIT = 200
_KEY = QtCore.Qt.ItemDataRole.UserRole
_NO_GROUP = "No group"
_INT_LITERAL = re.compile(r"[+-]?\d+")
_TYPE_WORDS = {
    bool: "true or false",
    int: "a whole number",
    float: "a number",
    str: "text",
}


def display_value(value: Value) -> str:
    """``value`` as it is shown and typed: bools as ``true``/``false``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_value(text: str, current: Value) -> Value:
    """``text`` read as the type of ``current``, or :class:`ValueError` saying why."""
    kind = type(current)
    stripped = text.strip()
    if kind is bool:
        if stripped.lower() in ("true", "false"):
            return stripped.lower() == "true"
    elif kind is int:
        if _INT_LITERAL.fullmatch(stripped):
            return int(stripped)
    elif kind is float:
        try:
            value = float(stripped)
        except ValueError:
            pass
        else:
            if math.isfinite(value):
                return value
    else:
        return text
    raise ValueError(f"'{text}' is not {_TYPE_WORDS[kind]}")


def when_text(time: datetime, current: datetime) -> str:
    """``time`` as a lab member says it: ``today 10:44``, ``3 days ago``, …"""
    time = time.astimezone(current.tzinfo)
    days = (current.date() - time.date()).days
    if days <= 0:
        return f"today {time:%H:%M}"
    if days == 1:
        return f"yesterday {time:%H:%M}"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{time:%d %b %Y}"


def _monospace():
    font = QtGui.QFont()
    font.setFamilies(["Cascadia Mono", "Consolas", "DejaVu Sans Mono"])
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font


def _tree(object_name, headers):
    tree = QTreeWidget()
    tree.setObjectName(object_name)
    tree.setHeaderLabels(headers)
    tree.setRootIsDecorated(False)
    tree.setUniformRowHeights(True)
    tree.setAlternatingRowColors(True)
    header = tree.header()
    header.setStretchLastSection(True)
    for column in range(len(headers) - 1):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    return tree


class SavedParametersPanel(QWidget):
    """View every saved parameter, edit one with a reason, and see its history.

    Args:
        client: Talks to the Experiment Manager; a short-timeout client by default.
        feed: The manager's published state, shared with other tabs; one of its
            own by default.
    """

    def __init__(self, client=None, feed=None, parent=None):
        super().__init__(parent)
        self.client = client or ExperimentManagerClient(timeout_ms=1000, retries=0)
        self._owns_feed = feed is None
        self.feed = feed or ExperimentFeed()
        self._version = None
        self._parameters: dict[str, SavedParameter] = {}
        self._groups: dict[str, QTreeWidgetItem] = {}
        self._items: dict[str, QTreeWidgetItem] = {}
        self._key = None
        self._shown_value = None

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter keys")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setFixedWidth(220)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.tree = _tree(
            "SavedParameterTree", ["Key", "Value", "Unit", "Last changed", "Why"]
        )
        self.tree.currentItemChanged.connect(self._current_changed)
        self.parameters_region = Region("well", "Saved parameters", "")
        self.parameters_region.header.addWidget(self.filter_edit)
        self.parameters_region.body.addWidget(self.tree, 1)

        self.sheet = Region("sheet", "Pick a saved parameter", "")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setHorizontalSpacing(12)
        self.value_edit = QLineEdit()
        self.value_edit.setFixedWidth(200)
        self.value_edit.returnPressed.connect(self.apply)
        self.unit_label = QLabel()
        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        value_row.addWidget(self.value_edit)
        value_row.addWidget(self.unit_label)
        form.addRow("Value", value_row)
        self.why_edit = QLineEdit()
        self.why_edit.setFixedWidth(360)
        self.why_edit.setPlaceholderText("saved with the change, e.g. lifetime scan")
        self.why_edit.returnPressed.connect(self.apply)
        form.addRow("Why", self.why_edit)
        self.sheet.body.addLayout(form)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("PrimaryButton")
        self.apply_button.setToolTip(
            "Save this value for the whole lab, with the reason"
        )
        self.apply_button.clicked.connect(self.apply)
        self.revert_button = QPushButton("Revert")
        self.revert_button.setToolTip("Put back the saved value and clear the reason")
        self.revert_button.clicked.connect(self._revert)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.revert_button)
        actions.addStretch(1)
        self.sheet.body.addLayout(actions)
        note = QLabel(
            "Read from now on; press Defaults in Experiments to use it in an open form"
        )
        note.setProperty("role", "regionHint")
        note.setWordWrap(True)
        self.sheet.body.addWidget(note)
        self.status = QLabel("Pick a saved parameter on the left to edit it")
        self.status.setObjectName("StatusLabel")
        self.status.setWordWrap(True)
        self.sheet.body.addWidget(self.status)

        self.history = _tree(
            "SavedParameterHistory", ["When", "Old", "New", "Who", "Why"]
        )
        self.history.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self.history_region = Region("well", "History", "")
        self.history_region.body.addWidget(self.history, 1)

        right = QSplitter(QtCore.Qt.Orientation.Vertical)
        right.setObjectName("PanelSplitter")
        right.addWidget(self.sheet)
        right.addWidget(self.history_region)
        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        right.setSizes([self.sheet.sizeHint().height(), 700])
        main = QSplitter(QtCore.Qt.Orientation.Horizontal)
        main.setObjectName("PanelSplitter")
        main.addWidget(self.parameters_region)
        main.addWidget(right)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        main.setSizes([900, 600])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(main, 1)

        self._set_editable(False)
        self.feed.queue_changed.connect(self._queue_changed)
        self.feed.connection_changed.connect(self._connection_changed)

    # -- following the manager -----------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _queue_changed(self, snapshot):
        version = snapshot.get("saved_parameters_version")
        if version is not None and version != self._version:
            self._version = version
            if self.isVisible():
                self.refresh()

    def _connection_changed(self, connected):
        if not connected:
            self._show_unreachable(NOT_RESPONDING)
        elif self.isVisible():
            self.refresh()

    def refresh(self):
        """Fetch every value and the selected key's history, updating in place."""
        try:
            parameters = self.client.saved_parameters()
            recent = self.client.saved_parameter_history(limit=REASON_WINDOW)
        except ManagerUnavailable:
            self._show_unreachable(NOT_RESPONDING)
            return
        except ManagerError as error:
            self._show_unreachable(str(error))
            return
        reasons = {}
        for change in recent:
            reasons.setdefault(change.key, change.reason)
        self._parameters = {parameter.key: parameter for parameter in parameters}
        self.tree.setEnabled(True)
        count = len(parameters)
        self.parameters_region.set_hint(
            f"{count} value{'' if count == 1 else 's'}; pick one to edit"
        )
        set_state(self.parameters_region.hint, "")
        self._show_parameters(parameters, reasons)
        if self._key is not None and self._key not in self._parameters:
            self.tree.setCurrentItem(None)
        elif self._key is not None:
            self._show_sheet(self._parameters[self._key])
            self._fetch_history()

    def _show_unreachable(self, why):
        self.parameters_region.set_hint(f"Saved parameters: {why}")
        set_state(self.parameters_region.hint, "error")
        self.tree.setEnabled(False)
        self.apply_button.setEnabled(False)

    def _show_parameters(self, parameters, reasons):
        current = now()
        by_group: dict[str, list[SavedParameter]] = {}
        for parameter in sorted(parameters, key=lambda p: (p.group, p.key)):
            by_group.setdefault(parameter.group or _NO_GROUP, []).append(parameter)
        for group in set(self._groups) - set(by_group):
            self.tree.takeTopLevelItem(
                self.tree.indexOfTopLevelItem(self._groups.pop(group))
            )
        for key in set(self._items) - set(self._parameters):
            item = self._items.pop(key)
            if item.parent() is not None:
                item.parent().removeChild(item)
        for group_index, (group, members) in enumerate(by_group.items()):
            group_item = self._group_item(group, group_index)
            for index, parameter in enumerate(members):
                item = self._items.get(parameter.key)
                if item is None:
                    item = self._items[parameter.key] = self._new_item(parameter.key)
                elif group_item.indexOfChild(item) != index:
                    item.parent().removeChild(item)
                if item.parent() is None:
                    group_item.insertChild(index, item)
                self._fill_item(
                    item, parameter, reasons.get(parameter.key, ""), current
                )
        self._apply_filter()

    def _group_item(self, group, index):
        item = self._groups.get(group)
        if item is None:
            item = self._groups[group] = QTreeWidgetItem([group])
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            self.tree.insertTopLevelItem(index, item)
            item.setFirstColumnSpanned(True)
            item.setExpanded(True)
        elif self.tree.indexOfTopLevelItem(item) != index:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
            self.tree.insertTopLevelItem(index, item)
            item.setFirstColumnSpanned(True)
            item.setExpanded(True)
        return item

    @staticmethod
    def _new_item(key):
        item = QTreeWidgetItem([key])
        item.setData(0, _KEY, key)
        item.setFont(0, _monospace())
        right = (
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        item.setTextAlignment(1, right)
        return item

    @staticmethod
    def _fill_item(item, parameter, reason, current):
        item.setText(1, display_value(parameter.value))
        item.setText(2, parameter.unit)
        item.setText(
            3, f"{when_text(parameter.updated_at, current)}, {parameter.updated_by}"
        )
        item.setToolTip(3, parameter.updated_at.isoformat(sep=" ", timespec="seconds"))
        item.setText(4, reason)
        item.setToolTip(4, reason)
        font = item.font(1)
        font.setBold(
            parameter.updated_at.astimezone(current.tzinfo).date() == current.date()
        )
        item.setFont(1, font)

    def _apply_filter(self):
        wanted = self.filter_edit.text().strip().lower()
        for group, group_item in self._groups.items():
            group_matches = wanted in group.lower()
            shown = 0
            for index in range(group_item.childCount()):
                child = group_item.child(index)
                hide = not (group_matches or wanted in child.data(0, _KEY).lower())
                child.setHidden(hide)
                shown += not hide
            group_item.setHidden(shown == 0)

    # -- the selected key ----------------------------------------------------

    def selected_key(self):
        """The key shown in the sheet, or ``None``."""
        return self._key

    def select(self, key: str) -> None:
        """Show ``key`` in the sheet and its history below it."""
        item = self._items.get(key)
        if item is not None:
            self.tree.setCurrentItem(item)

    def _current_changed(self, item, _previous):
        key = None if item is None else item.data(0, _KEY)
        if key == self._key:
            return
        self._key = key
        self._shown_value = None
        self.why_edit.clear()
        if key is None:
            self.sheet.title.setText("Pick a saved parameter")
            self.sheet.set_hint("")
            self.value_edit.clear()
            self.unit_label.clear()
            self.history.clear()
            self.history_region.set_hint("")
            self._set_editable(False)
            self._show_status("Pick a saved parameter on the left to edit it")
            return
        self._set_editable(True)
        self._show_status("")
        self._show_sheet(self._parameters[key])
        self._fetch_history()

    def _show_sheet(self, parameter):
        """Show ``parameter``, keeping a value the user has typed and not applied."""
        self.sheet.title.setText(parameter.key)
        self.sheet.set_hint(parameter.group)
        self.unit_label.setText(parameter.unit)
        self.value_edit.setToolTip(f"{_TYPE_WORDS[type(parameter.value)]}".capitalize())
        saved = display_value(parameter.value)
        if self._shown_value is None or self.value_edit.text() == self._shown_value:
            self.value_edit.setText(saved)
        self._shown_value = saved
        self.apply_button.setEnabled(True)

    def _fetch_history(self):
        key = self._key
        try:
            changes = self.client.saved_parameter_history(key, limit=HISTORY_LIMIT)
        except (ManagerUnavailable, ManagerError):
            self.history_region.set_hint(f"{key}: could not fetch the history")
            return
        current = now()
        self.history_region.set_hint(f"{key}, newest first")
        self.history.clear()
        right = (
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        for change in changes:
            old = "—" if change.old is None else display_value(change.old)
            item = QTreeWidgetItem(
                [
                    when_text(change.time, current),
                    old,
                    display_value(change.new),
                    change.who,
                    change.reason,
                ]
            )
            item.setToolTip(0, change.time.isoformat(sep=" ", timespec="seconds"))
            item.setToolTip(4, change.reason)
            item.setTextAlignment(1, right)
            item.setTextAlignment(2, right)
            self.history.addTopLevelItem(item)

    def _set_editable(self, editable):
        for widget in (
            self.value_edit,
            self.why_edit,
            self.apply_button,
            self.revert_button,
        ):
            widget.setEnabled(editable)

    def _show_status(self, text, state=""):
        self.status.setText(text)
        set_state(self.status, state)

    def _revert(self):
        if self._key is None:
            return
        self._shown_value = None
        self.why_edit.clear()
        self._show_sheet(self._parameters[self._key])
        self._show_status("")

    def apply(self) -> None:
        """Save the typed value for the selected key, with the typed reason.

        A value that does not parse as the key's type is refused here and
        nothing is sent.
        """
        if self._key is None:
            return
        key, parameter = self._key, self._parameters[self._key]
        try:
            value = parse_value(self.value_edit.text(), parameter.value)
        except ValueError as error:
            self._show_status(f"{key}: {error}", "error")
            return
        why = self.why_edit.text().strip()
        reason = f"set by hand: {why}" if why else "set by hand"
        try:
            changes = self.client.set_saved_parameters(
                {key: value}, reason, submitter_name()
            )
        except ManagerUnavailable:
            self._show_status(f"Applying: {NOT_RESPONDING}", "error")
            return
        except ManagerError as error:
            self._show_status(f"Applying: {error}", "error")
            return
        self._shown_value = None
        self.why_edit.clear()
        if changes:
            self._show_status(f"Saved {key} = {display_value(value)}")
        else:
            self._show_status(f"{key} already holds {display_value(value)}")
        self.refresh()

    def closeEvent(self, event):
        if self._owns_feed:
            self.feed.close()
        self.client.close()
        super().closeEvent(event)
