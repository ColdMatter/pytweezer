"""The Experiments tab: catalogue, argument editor and queue in one panel."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pytweezer.experiment.client import (
    ExperimentManagerClient,
    ManagerError,
    ManagerUnavailable,
    submitter_name,
)
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.components import Region, set_state
from pytweezer.GUI.experiments.arg_editor import ArgumentEditor
from pytweezer.GUI.experiments.catalogue_view import CatalogueView
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.experiments.motmaster_params import DeviceParameterSource
from pytweezer.GUI.experiments.queue_view import QueueView, queue_summary
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments")


class ExperimentsPanel(QWidget):
    """Browse experiments, edit and submit them, and control the queue.

    All state shown comes from the manager's published state (an
    :class:`ExperimentFeed`, which may be shared with other tabs); requests go
    over a short-timeout REQ client so a down manager never freezes the GUI for
    long.
    """

    def __init__(self, client=None, feed=None, parent=None):
        super().__init__(parent)
        self.client = client or ExperimentManagerClient(timeout_ms=1000, retries=0)
        self._owns_feed = feed is None
        self.feed = feed or ExperimentFeed()
        self._catalogue_version = None
        self._recipes_version = None
        # Unsubmitted edits per experiment, restored when switching back.
        self._drafts = {}
        self._current_key = None
        self._simulated = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        status_bar = QHBoxLayout()
        self.status = QLabel("Waiting for the Experiment Manager…")
        self.status.setObjectName("StatusLabel")
        self.simulation_banner = QLabel(
            "Simulation: devices are simulated, data kept apart"
        )
        self.simulation_banner.setObjectName("SimulationBanner")
        self.simulation_banner.setVisible(False)
        status_bar.addWidget(self.status, 1)
        status_bar.addWidget(self.simulation_banner)
        layout.addLayout(status_bar)

        self.catalogue = CatalogueView()
        catalogue_region = Region("well", "Experiments", "pick one to edit")
        catalogue_region.body.addWidget(self.catalogue, 1)

        self.editor = ArgumentEditor()
        self.editor.set_parameter_source(DeviceParameterSource(lambda: self._simulated))
        editor_scroll = QScrollArea()
        editor_scroll.setObjectName("EditorScroll")
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        editor_scroll.setWidget(self.editor)
        editor_region = Region("sheet")
        editor_region.body.addWidget(editor_scroll, 1)

        self.queue_view = QueueView()
        self.queue_region = Region("well", "Queue", "")
        self.queue_region.body.addWidget(self.queue_view, 1)

        top = QSplitter(QtCore.Qt.Orientation.Horizontal)
        top.setObjectName("ExperimentsSplitter")
        top.addWidget(catalogue_region)
        top.addWidget(editor_region)
        top.setStretchFactor(1, 3)
        top.setSizes([260, 900])
        main = QSplitter(QtCore.Qt.Orientation.Vertical)
        main.setObjectName("ExperimentsSplitter")
        main.addWidget(top)
        main.addWidget(self.queue_region)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        layout.addWidget(main, 1)

        self.catalogue.experiment_selected.connect(self._experiment_selected)
        self.catalogue.refresh_requested.connect(self.refresh_catalogue)
        self.editor.submit_requested.connect(self._submit)
        self.editor.last_requested.connect(self._load_last)
        self.catalogue.recipe_selected.connect(self._recipe_selected)
        self.catalogue.recipe_action_requested.connect(self._recipe_action)
        self.editor.save_recipe_requested.connect(self._save_recipe)
        self.queue_view.action_requested.connect(self._action)
        self.queue_view.edit_requested.connect(
            lambda task: self.load_request(TaskRequest.model_validate(task))
        )
        self.feed.queue_changed.connect(self._queue_changed)
        self.feed.connection_changed.connect(self._connection_changed)
        QtCore.QTimer.singleShot(0, self.refresh_catalogue)

    # -- manager calls -------------------------------------------------------

    def _call(self, what, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ManagerUnavailable:
            self._show_status(
                f"{what}: the Experiment Manager is not responding", "crashed"
            )
        except ManagerError as error:
            self._show_status(f"{what}: {error}", "crashed")
        return None

    def _show_status(self, text, state=""):
        self.status.setText(text)
        set_state(self.status, state)

    def refresh_catalogue(self):
        reply = self._call("Listing experiments", self.client.call, "catalogue")
        if reply is None:
            return
        self._catalogue_version = reply.get("version")
        self.catalogue.set_modules(reply["modules"])
        self.refresh_recipes()

    def refresh_recipes(self):
        reply = self._call("Listing recipes", self.client.call, "recipes")
        if reply is None:
            return
        self._recipes_version = reply.get("version")
        self.catalogue.set_recipes(reply["recipes"])

    def _queue_changed(self, snapshot):
        self._simulated = bool(snapshot.get("simulated"))
        self.simulation_banner.setVisible(bool(snapshot.get("simulated")))
        self.queue_view.set_snapshot(snapshot)
        self.queue_region.set_hint(queue_summary(snapshot))
        version = snapshot.get("catalogue_version")
        if version is not None and version != self._catalogue_version:
            self.refresh_catalogue()
        recipes_version = snapshot.get("recipes_version")
        if recipes_version is not None and recipes_version != self._recipes_version:
            self.refresh_recipes()

    def _connection_changed(self, connected):
        if connected:
            self._show_status("Connected to the Experiment Manager")
        else:
            self._show_status("Experiment Manager not reachable", "crashed")

    # -- editing and submitting ----------------------------------------------

    def _experiment_selected(self, schema):
        key = (schema["module"], schema["class_name"])
        if key == self._current_key:
            return
        self._save_draft()
        self._current_key = key
        self.editor.set_experiment(schema)
        if key in self._drafts:
            self.editor.load_request(self._drafts[key])
        else:
            self._load_last(*key, quiet=True)

    def _save_draft(self):
        if self._current_key is None:
            return
        try:
            self._drafts[self._current_key] = self.editor.request(validate=False)
        except ValueError:
            pass

    def _load_last(self, module, class_name, quiet=False):
        try:
            request = self.client.last_request(module, class_name)
        except (ManagerUnavailable, ManagerError):
            if not quiet:
                self._show_status("Could not fetch the last submission", "crashed")
            return
        if request is not None:
            self.editor.load_request(request)

    def load_request(self, request):
        """Show ``request`` in the editor (e.g. to resubmit a past measurement)."""
        schema = self.catalogue.schema_for(request.experiment, request.class_name)
        if schema is None:
            self._show_status(
                f"{request.experiment}.{request.class_name} is not in the catalogue",
                "crashed",
            )
            return
        self.catalogue.select(request.experiment, request.class_name)
        if self._current_key != (request.experiment, request.class_name):
            self._experiment_selected(schema)
        self.editor.load_request(request)
        self._raise_dock()

    def _raise_dock(self):
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, QDockWidget):
            widget = widget.parentWidget()
        if widget is not None:
            widget.raise_()

    def _submit(self, request):
        request = request.model_copy(update={"submitter": submitter_name()})
        rid = self._call("Submitting", self.client.submit, request)
        if rid is not None:
            self._drafts.pop((request.experiment, request.class_name), None)
            self._show_status(f"Queued task {rid}")

    # -- recipes -------------------------------------------------------------

    def _recipe_selected(self, recipe):
        key = (recipe["experiment"], recipe["class_name"])
        schema = self.catalogue.schema_for(*key)
        if schema is None:
            return
        if key != self._current_key:
            self._save_draft()
            self._current_key = key
            self.editor.set_experiment(schema)
        self.editor.load_request(Recipe.model_validate(recipe))
        self.editor.set_recipe_name(recipe["name"])

    def _save_recipe(self, request):
        name = self.ask_recipe_name(self.editor.recipe_name).strip()
        if not name:
            return
        exists = any(
            (r["experiment"], r["class_name"], r["name"])
            == (request.experiment, request.class_name, name)
            for r in self.catalogue.recipes
        )
        if exists and not self.confirm(
            "Replace recipe", f"Replace the saved recipe '{name}' for every PC?"
        ):
            return
        recipe = Recipe(
            **request.model_dump(exclude={"submitter"}),
            name=name,
            submitter=submitter_name(),
        )
        reply = self._call(
            f"Saving recipe '{name}'",
            self.client.call,
            "save_recipe",
            recipe=recipe.model_dump(mode="json"),
            overwrite=exists,
        )
        if reply is None:
            return
        self.editor.set_recipe_name(name)
        self._show_status(f"Saved recipe '{name}'")
        self.refresh_recipes()

    def _recipe_action(self, action, recipe):
        name = recipe["name"]
        fields = {key: recipe[key] for key in ("experiment", "class_name", "name")}
        if action == "submit":
            reply = self._call(
                f"Submitting recipe '{name}'",
                self.client.call,
                "submit_recipe",
                submitter=submitter_name(),
                **fields,
            )
            if reply is not None:
                self._show_status(f"Queued task {reply['rid']} from recipe '{name}'")
        elif action == "delete":
            if not self.confirm(
                "Delete recipe",
                f"Delete the saved recipe '{name}' for {recipe['class_name']}? "
                "It is removed for every PC.",
            ):
                return
            reply = self._call(
                f"Deleting recipe '{name}'", self.client.call, "delete_recipe", **fields
            )
            if reply is not None:
                self._show_status(f"Deleted recipe '{name}'")
                self.refresh_recipes()

    def ask_recipe_name(self, default):
        name, ok = QInputDialog.getText(
            self, "Save as recipe", "Recipe name (shared with every PC):", text=default
        )
        return name if ok else ""

    def confirm(self, title, question):
        answer = QMessageBox.question(self, title, question)
        return answer == QMessageBox.StandardButton.Yes

    def _action(self, command, fields):
        self._call(
            f"{command} task {fields['rid']}", self.client.call, command, **fields
        )

    def closeEvent(self, event):
        if self._owns_feed:
            self.feed.close()
        self.client.close()
        super().closeEvent(event)
