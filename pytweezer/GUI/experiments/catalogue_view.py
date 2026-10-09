"""Tree of the experiments the manager can run, grouped by module, with their recipes."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.components import status_icon

_SCHEMA = QtCore.Qt.ItemDataRole.UserRole
_RECIPE = QtCore.Qt.ItemDataRole.UserRole + 1
_PRESS_EVENTS = (
    QtCore.QEvent.Type.MouseButtonPress,
    QtCore.QEvent.Type.MouseButtonDblClick,
)


class CatalogueView(QWidget):
    experiment_selected = QtCore.pyqtSignal(dict)
    recipe_selected = QtCore.pyqtSignal(dict)
    recipe_action_requested = QtCore.pyqtSignal(str, dict)
    refresh_requested = QtCore.pyqtSignal()

    def __init__(self, package="pytweezer.experiments", parent=None):
        super().__init__(parent)
        self.package = package
        self.modules = []
        self.recipes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter experiments and recipes")
        self.filter.textChanged.connect(self._apply_filter)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_requested)
        top.addWidget(self.filter, 1)
        top.addWidget(refresh)
        layout.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.viewport().installEventFilter(self)
        self._current_at_press = None
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.tree, 1)

    def set_modules(self, modules):
        self.modules = modules
        self._rebuild()

    def set_recipes(self, recipes):
        self.recipes = recipes
        self._rebuild()

    def _rebuild(self):
        selected = self._selection()
        by_class = {}
        for recipe in self.recipes:
            by_class.setdefault(
                (recipe["experiment"], recipe["class_name"]), []
            ).append(recipe)
        self.tree.blockSignals(True)
        self.tree.clear()
        reselect = None
        for module in self.modules:
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
                for recipe in by_class.get(_key(schema), []):
                    recipe_item = _recipe_item(recipe)
                    item.addChild(recipe_item)
                    if _recipe_key(recipe) == selected:
                        reselect = recipe_item
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

    def select_recipe(self, module, class_name, name):
        wanted = (module, class_name, name)
        iterator = QTreeWidgetItemIterator(self.tree)
        while item := iterator.value():
            recipe = item.data(0, _RECIPE)
            if recipe and _recipe_key(recipe) == wanted:
                self.tree.setCurrentItem(item)
                return True
            iterator += 1
        return False

    def selected_key(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        if recipe := item.data(0, _RECIPE):
            return (recipe["experiment"], recipe["class_name"])
        schema = item.data(0, _SCHEMA)
        return _key(schema) if schema else None

    def _selection(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        if recipe := item.data(0, _RECIPE):
            return _recipe_key(recipe)
        schema = item.data(0, _SCHEMA)
        return _key(schema) if schema else None

    def _current_changed(self, item, _previous):
        if item is None:
            return
        if recipe := item.data(0, _RECIPE):
            self.recipe_selected.emit(recipe)
        elif schema := item.data(0, _SCHEMA):
            self.experiment_selected.emit(schema)

    def eventFilter(self, watched, event):
        if watched is self.tree.viewport() and event.type() in _PRESS_EVENTS:
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                return True
            self._current_at_press = self.tree.currentItem()
        return super().eventFilter(watched, event)

    def _item_clicked(self, item):
        recipe = item.data(0, _RECIPE)
        if recipe and item is self._current_at_press:
            self.recipe_selected.emit(recipe)

    def recipe_menu(self, recipe):
        menu = QMenu(self)
        submit = menu.addAction("Submit now")
        submit.triggered.connect(
            lambda: self.recipe_action_requested.emit("submit", recipe)
        )
        menu.addSeparator()
        delete = menu.addAction("Delete…")
        delete.triggered.connect(
            lambda: self.recipe_action_requested.emit("delete", recipe)
        )
        return menu

    def _context_menu(self, position):
        item = self.tree.itemAt(position)
        recipe = item.data(0, _RECIPE) if item else None
        if recipe:
            self.recipe_menu(recipe).exec(self.tree.viewport().mapToGlobal(position))

    def _apply_filter(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            module_item = self.tree.topLevelItem(i)
            module_match = text in module_item.text(0).lower()
            any_class = False
            for j in range(module_item.childCount()):
                class_item = module_item.child(j)
                class_match = module_match or text in class_item.text(0).lower()
                any_recipe = False
                for k in range(class_item.childCount()):
                    recipe_item = class_item.child(k)
                    visible = class_match or text in recipe_item.text(0).lower()
                    recipe_item.setHidden(not visible)
                    any_recipe |= visible
                class_item.setHidden(not (class_match or any_recipe))
                any_class |= class_match or any_recipe
            module_item.setHidden(not (module_match or any_class))


def _key(schema):
    return (schema["module"], schema["class_name"])


def _recipe_key(recipe):
    return (recipe["experiment"], recipe["class_name"], recipe["name"])


def _recipe_item(recipe):
    item = QTreeWidgetItem([recipe["name"]])
    item.setData(0, _RECIPE, recipe)
    font = item.font(0)
    font.setItalic(True)
    item.setFont(0, font)
    saved = recipe.get("saved_at", "")[:16].replace("T", " ")
    tooltip = f"Recipe saved by {recipe.get('submitter') or 'unknown'} on {saved}"
    if recipe.get("label"):
        tooltip += f"\nLabel: {recipe['label']}"
    item.setToolTip(
        0,
        tooltip
        + "\nClick to load it into the form; right-click to submit or delete it",
    )
    return item
