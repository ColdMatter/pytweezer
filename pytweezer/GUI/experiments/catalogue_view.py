"""Tree of the experiments the manager can run, one item each, in sections by folder."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.components import status_icon

_SCHEMA = QtCore.Qt.ItemDataRole.UserRole


def doc_summary(doc):
    """The part of a docstring before its first line break."""
    return (doc or "").strip().split("\n", 1)[0]


class CatalogueView(QWidget):
    experiment_selected = QtCore.pyqtSignal(dict)
    refresh_requested = QtCore.pyqtSignal()

    def __init__(self, package="pytweezer.experiments", parent=None):
        super().__init__(parent)
        self.package = package
        self.modules = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter experiments")
        self.filter.textChanged.connect(self._apply_filter)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_requested)
        top.addWidget(self.filter, 1)
        top.addWidget(refresh)
        layout.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._current_changed)
        layout.addWidget(self.tree, 1)

    def set_modules(self, modules):
        self.modules = modules
        self._rebuild()

    def _rebuild(self):
        selected = self.selected_key()
        self.tree.blockSignals(True)
        self.tree.clear()
        folders = {}
        entries = []
        for module in sorted(self.modules, key=lambda m: m["module"]):
            *folder, file = self._relative(module["module"]).split(".")
            # Optimisers are configured and run from the Optimisers tab.
            items = [
                (s["class_name"], _experiment_item(s, module))
                for s in _experiments(module)
            ]
            if module.get("error"):
                items.append((file, _broken_item(file, module)))
            # A module of base classes only, such as TweezerArrayMeasurement, offers nothing.
            if items:
                parent = self._folder(folders, tuple(folder))
                entries += [(parent, name, item) for name, item in items]
        reselect = None
        for parent, _name, item in sorted(entries, key=lambda e: e[1].lower()):
            parent.addChild(item)
            schema = item.data(0, _SCHEMA)
            if schema and _key(schema) == selected:
                reselect = item
        self.tree.expandAll()
        self._apply_filter(self.filter.text())
        if reselect is not None:
            self.tree.setCurrentItem(reselect)
        self.tree.blockSignals(False)

    def _relative(self, module):
        return module.removeprefix(self.package + ".")

    def _folder(self, folders, path):
        """The section item for folder ``path``, made (with its parents) on first use."""
        if not path:
            return self.tree.invisibleRootItem()
        if path not in folders:
            parent = self._folder(folders, path[:-1])
            item = QTreeWidgetItem([path[-1]])
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            # Sections come before the experiments beside them, which are added later.
            parent.addChild(item)
            folders[path] = item
        return folders[path]

    def schema_for(self, module, class_name):
        for entry in self.modules:
            if entry["module"] == module:
                for schema in _experiments(entry):
                    if schema["class_name"] == class_name:
                        return schema
        return None

    def select(self, module, class_name):
        iterator = QTreeWidgetItemIterator(self.tree)
        while item := iterator.value():
            schema = item.data(0, _SCHEMA)
            if schema and _key(schema) == (module, class_name):
                self.tree.setCurrentItem(item)
                return True
            iterator += 1
        return False

    def selected_key(self):
        item = self.tree.currentItem()
        schema = item.data(0, _SCHEMA) if item else None
        return _key(schema) if schema else None

    def _current_changed(self, item, _previous):
        if item is not None and (schema := item.data(0, _SCHEMA)):
            self.experiment_selected.emit(schema)

    def _apply_filter(self, text):
        self._filter_children(self.tree.invisibleRootItem(), text.lower(), False)

    def _filter_children(self, parent, text, shown_by_folder):
        any_visible = False
        for i in range(parent.childCount()):
            item = parent.child(i)
            match = shown_by_folder or text in item.text(0).lower()
            if item.data(0, _SCHEMA) is None and item.childCount():
                visible = self._filter_children(item, text, match) or match
            else:
                visible = match
            item.setHidden(not visible)
            any_visible |= visible
        return any_visible


def _experiments(module):
    return [s for s in module.get("classes", []) if not s.get("optimiser")]


def _key(schema):
    return (schema["module"], schema["class_name"])


def _experiment_item(schema, module):
    item = QTreeWidgetItem([schema["class_name"]])
    item.setData(0, _SCHEMA, schema)
    tooltip = "\n".join(
        filter(None, [doc_summary(schema.get("doc")), schema["module"]])
    )
    if module.get("warnings"):
        item.setIcon(0, status_icon("starting"))
        tooltip += "\n\n" + "\n".join(module["warnings"])
    item.setToolTip(0, tooltip)
    return item


def _broken_item(file, module):
    item = QTreeWidgetItem([f"{file}.py"])
    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
    item.setIcon(0, status_icon("crashed"))
    item.setToolTip(0, f"{module['module']} does not import:\n{module['error']}")
    return item
