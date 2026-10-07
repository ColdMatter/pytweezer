"""Tree of the experiments the manager can run, grouped by module."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.experiments.queue_view import status_icon

_SCHEMA = QtCore.Qt.ItemDataRole.UserRole


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
        selected = self.selected_key()
        self.modules = modules
        self.tree.blockSignals(True)
        self.tree.clear()
        reselect = None
        for module in modules:
            name = module["module"].removeprefix(self.package + ".")
            module_item = QTreeWidgetItem([name])
            module_item.setToolTip(0, module["module"])
            if module.get("error"):
                module_item.setIcon(0, status_icon("crashed"))
                module_item.setToolTip(0, module["error"])
            elif module.get("warnings"):
                module_item.setIcon(0, status_icon("starting"))
                module_item.setToolTip(0, "\n".join(module["warnings"]))
            for schema in module.get("classes", []):
                item = QTreeWidgetItem([schema["class_name"]])
                item.setData(0, _SCHEMA, schema)
                item.setToolTip(0, schema.get("doc") or schema["class_name"])
                module_item.addChild(item)
                if _key(schema) == selected:
                    reselect = item
            self.tree.addTopLevelItem(module_item)
        self.tree.expandAll()
        self._apply_filter(self.filter.text())
        if reselect is not None:
            self.tree.setCurrentItem(reselect)
        self.tree.blockSignals(False)

    def schema_for(self, module, class_name):
        for entry in self.modules:
            if entry["module"] == module:
                for schema in entry.get("classes", []):
                    if schema["class_name"] == class_name:
                        return schema
        return None

    def select(self, module, class_name):
        for i in range(self.tree.topLevelItemCount()):
            module_item = self.tree.topLevelItem(i)
            for j in range(module_item.childCount()):
                item = module_item.child(j)
                if _key(item.data(0, _SCHEMA)) == (module, class_name):
                    self.tree.setCurrentItem(item)
                    return True
        return False

    def selected_key(self):
        item = self.tree.currentItem()
        schema = item.data(0, _SCHEMA) if item else None
        return _key(schema) if schema else None

    def _current_changed(self, item, _previous):
        schema = item.data(0, _SCHEMA) if item else None
        if schema:
            self.experiment_selected.emit(schema)

    def _apply_filter(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            module_item = self.tree.topLevelItem(i)
            module_match = text in module_item.text(0).lower()
            any_child = False
            for j in range(module_item.childCount()):
                child = module_item.child(j)
                visible = module_match or text in child.text(0).lower()
                child.setHidden(not visible)
                any_child |= visible
            module_item.setHidden(not (module_match or any_child))


def _key(schema):
    return (schema["module"], schema["class_name"])
