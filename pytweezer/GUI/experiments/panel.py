"""The Experiments tab: catalogue, argument editor and queue in one panel."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QDockWidget,
    QLabel,
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
from pytweezer.experiment.task import TaskRequest
from pytweezer.GUI.experiments.arg_editor import ArgumentEditor, _set_state
from pytweezer.GUI.experiments.catalogue_view import CatalogueView
from pytweezer.GUI.experiments.feed import ExperimentFeed
from pytweezer.GUI.experiments.queue_view import QueueView
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.experiments")


class ExperimentsPanel(QWidget):
    """Browse experiments, edit and submit them, and control the queue.

    All state shown comes from the manager's PUB feed; requests go over a
    short-timeout REQ client so a down manager never freezes the GUI for long.
    """

    def __init__(self, client=None, feed=None, parent=None):
        super().__init__(parent)
        self.client = client or ExperimentManagerClient(timeout_ms=1000, retries=0)
        self.feed = feed or ExperimentFeed()
        self._catalogue_version = None
        # Unsubmitted edits per experiment, restored when switching back.
        self._drafts = {}
        self._current_key = None

        layout = QVBoxLayout(self)
        self.status = QLabel("Waiting for the Experiment Manager…")
        self.status.setObjectName("StatusLabel")
        layout.addWidget(self.status)
        self.simulation_banner = QLabel(
            "SIMULATION: experiments use simulated devices and their data is kept "
            "separately"
        )
        self.simulation_banner.setObjectName("SimulationBanner")
        self.simulation_banner.setVisible(False)
        layout.addWidget(self.simulation_banner)

        self.catalogue = CatalogueView()
        self.editor = ArgumentEditor()
        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(self.editor)
        self.queue_view = QueueView()

        top = QSplitter(QtCore.Qt.Orientation.Horizontal)
        top.addWidget(self.catalogue)
        top.addWidget(editor_scroll)
        top.setStretchFactor(1, 3)
        main = QSplitter(QtCore.Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.queue_view)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        layout.addWidget(main, 1)

        self.catalogue.experiment_selected.connect(self._experiment_selected)
        self.catalogue.refresh_requested.connect(self.refresh_catalogue)
        self.editor.submit_requested.connect(self._submit)
        self.editor.last_requested.connect(self._load_last)
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
        _set_state(self.status, state)

    def refresh_catalogue(self):
        reply = self._call("Listing experiments", self.client.call, "catalogue")
        if reply is None:
            return
        self._catalogue_version = reply.get("version")
        self.catalogue.set_modules(reply["modules"])

    def _queue_changed(self, snapshot):
        self.simulation_banner.setVisible(bool(snapshot.get("simulated")))
        self.queue_view.set_snapshot(snapshot)
        version = snapshot.get("catalogue_version")
        if version is not None and version != self._catalogue_version:
            self.refresh_catalogue()

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
            self._drafts[self._current_key] = self.editor.request()
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

    def _action(self, command, fields):
        self._call(
            f"{command} task {fields['rid']}", self.client.call, command, **fields
        )

    def closeEvent(self, event):
        self.feed.close()
        self.client.close()
        super().closeEvent(event)
