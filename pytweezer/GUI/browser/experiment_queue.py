# pylint: disable=C0302
from functools import partial
from enum import Enum, auto

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTableView, QPushButton, QAbstractItemView,
    QAction, QHBoxLayout, QVBoxLayout, QHeaderView, QGroupBox
)

from pytweezer.servers import icon_path, PropertyAttribute
from pytweezer.GUI.models import ScheduleModel
from pytweezer.servers.model_sync.client import ModelClient
from pytweezer.servers.command.client import CommandClient as QueueCommandClient


class QueueRole(Enum):
    """
    Determines how ExperimentQ interacts with the ScheduleModel.

    SERVER  — This process owns the authoritative model and the experiment
              manager. Mutations are applied directly and broadcast to clients.

    CLIENT  — This process holds a replica. All mutations are sent to the
              server via ModelClient, which applies them authoritatively and
              rebroadcasts. Direct model writes are forbidden.
    """
    SERVER = auto()
    CLIENT = auto()


class ExperimentQ(QGroupBox):
    """
    Experiment Queue widget — works in both SERVER and CLIENT roles.

    SERVER role (original machine):
        - Owns a ModelServer-registered ScheduleModel.
        - Mutations (delete, terminate, pause) are applied directly to the
          model, then broadcast to all remote clients by ModelServer.
        - expManager is accessible and called directly.

    CLIENT role (remote machine):
        - Holds a ModelClient whose.model is a replica ScheduleModel.
        - All mutations are sent to the server via ModelClient.set() /
          ModelClient.delete(), which ensures the server is the single
          source of truth.
                - expManager commands (pause, restart) are sent via CommandClient
                    to the server-side CommandServer.
        - The table is updated automatically when ModelClient.synced fires.

    Args:
        browser     : Parent browser object (provides props, expManager on server).
        model_client: A started ModelClient instance. Required for CLIENT role,
                      optional for SERVER (if provided, the server's own view
                      also goes through the client path for consistency).
        role        : QueueRole.SERVER or QueueRole.CLIENT.
        parent      : Qt parent widget.
        title       : GroupBox title string.
    """

    def __init__(self,
                 browser,
                 model_client: ModelClient = None,
                 command_client: QueueCommandClient = None,
                 role: QueueRole = QueueRole.SERVER,
                 parent=None,
                 title="ExperimentQ"):
        super().__init__(title, parent)

        self.browser = browser
        self.props = browser.props
        self._props = self.props
        self.role = role
        self._client = model_client   # None on server if not using loopback client
        self._command_client = command_client

        # ── Layouts ──────────────────────────────────────────────────────────
        self.qLayout = QHBoxLayout()
        self.buttonLayout = QVBoxLayout()
        self.create_buttons()
        self.qLayout.addLayout(self.buttonLayout)

        # ── Table ─────────────────────────────────────────────────────────────
        self.init_table()
        self.init_table_actions()
        self.qLayout.addWidget(self.table)
        self.setLayout(self.qLayout)

        # ── Properties ────────────────────────────────────────────────────────
        self.props.set('testval', 3)
        self.test = PropertyAttribute('testval', 3, parent=self)
        self._props.set('testlist', [])
        self.testlist = PropertyAttribute('testlist', [], parent=self)

        self.lock = False

        # ── Wire up sync signal if we have a client ───────────────────────────
        if self._client is not None:
            # synced fires on the Qt thread whenever the server broadcasts a
            # change. We use it to keep the status bar / logs up to date.
            # The model itself is already updated by ModelClient._qt_set/_qt_del
            # before this signal fires, so no further model work is needed here.
            self._client.synced.connect(self._on_remote_sync)
            self._client.connection_failed.connect(self._on_connection_failed)

    # ── Table initialisation ─────────────────────────────────────────────────

    def init_table(self):
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setContextMenuPolicy(Qt.ActionsContextMenu)

        if self._client is not None:
            # ── CLIENT or server-with-loopback-client path ────────────────────
            # The model is owned by ModelClient and already populated with the
            # server snapshot. We just point the table at it.
            self.tableModel = self._client.model
            self.expDict = self.tableModel.backing_store
        else:
            # ── Pure SERVER path (no client object) ───────────────────────────
            # Model is created locally; ModelServer holds a reference to it.
            self.expDict = {}
            self.tableModel = ScheduleModel(self.expDict)

        self.table.setModel(self.tableModel)

        cw = QtGui.QFontMetrics(self.font()).averageCharWidth()
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, h.count()):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    def set_model(self, model):
        """Replace the table model at runtime (e.g. after reconnect)."""
        self.tableModel = model
        self.expDict = model.backing_store
        self.table.setModel(self.tableModel)

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def _model_set(self, key, value):
        """
        Set a value in the model via the correct path for the current role.

        SERVER: write directly — ModelServer will broadcast the change.
        CLIENT: send to server via ModelClient — server applies and broadcasts.
                The local optimistic update in ModelClient.set() means the UI
                updates immediately without waiting for the round-trip.
        """
        if self._client is not None:
            self._client.set(key, value)
        else:
            self.tableModel[key] = value

    def _model_del(self, key):
        """
        Delete a row via the correct path for the current role.
        See _model_set for the SERVER/CLIENT distinction.
        """
        if self._client is not None:
            self._client.delete(key)
        else:
            del self.tableModel[key]

    def _model_setitem(self, key, field, value):
        """
        Mutate a single field within a row's value dict.

        We must read the full row value, patch it, then write it back —
        because the sync protocol only supports whole-row SET operations.
        This keeps the protocol simple and avoids partial-update races.

        Args:
            key   : Row key (task number).
            field : Field name within the row dict, e.g. 'terminated'.
            value : New value for that field.
        """
        # Read current value from backing_store (always up to date)
        row = dict(self.expDict[key])   # shallow copy — avoids mutating in place
        row[field] = value
        self._model_set(key, row)

    def add_task(self, task_nr, task_dict):
        """Public API for adding/replacing a task row."""
        self._model_set(task_nr, task_dict)

    def set_task_field(self, task_nr, field, value):
        """Public API for mutating one field in a task row."""
        if task_nr not in self.expDict:
            return
        self._model_setitem(task_nr, field, value)

    def delete_task(self, task_nr):
        """Public API for removing a task row."""
        if task_nr not in self.expDict:
            return
        self._model_del(task_nr)

    # ── Button creation ──────────────────────────────────────────────────────

    def create_buttons(self):
        self.pauseButton = QPushButton('')
        self.pauseButton.setMaximumWidth(30)
        self.pauseButton.setIcon(QtGui.QIcon(icon_path + 'pause.png'))
        self.pauseButton.clicked.connect(self.pause)
        self.buttonLayout.addWidget(self.pauseButton)

        terminateButton = QPushButton('')
        terminateButton.setMaximumWidth(30)
        terminateButton.setIcon(QtGui.QIcon(icon_path + 'terminate.png'))
        terminateButton.clicked.connect(self.terminate_clicked)
        self.buttonLayout.addWidget(terminateButton)

        terminateAllButton = QPushButton('')
        terminateAllButton.setMaximumWidth(30)
        terminateAllButton.setIcon(QtGui.QIcon(icon_path + 'nuke_icon.png'))
        terminateAllButton.clicked.connect(self.terminate_all)
        self.buttonLayout.addWidget(terminateAllButton)

        self.restartButton = QPushButton('Restart Q')
        self.restartButton.setMaximumWidth(30)
        self.restartButton.clicked.connect(self.restart)
        self.buttonLayout.addWidget(self.restartButton)

        self.buttonLayout.setAlignment(Qt.AlignLeft)

        # ── Disable controls that require server-side resources ───────────────
        # pause and restart require expManager, which only exists on the server.
        # We grey them out on clients so the intent is visually clear.
        if self.role == QueueRole.CLIENT:
            self.pauseButton.setToolTip('Pause (remote)')
            self.restartButton.setToolTip('Restart queue (remote)')
            # Buttons remain enabled — commands are sent via Properties.
            # Disable only if you want a strict read-only remote view:
            # self.pauseButton.setEnabled(False)
            # self.restartButton.setEnabled(False)

    # ── Table actions ────────────────────────────────────────────────────────

    def init_table_actions(self):
        delete_action = QAction("Delete", self.table)
        delete_action.triggered.connect(partial(self.delete_clicked))
        delete_action.setShortcut("SHIFT+DELETE")
        delete_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(delete_action)

        terminate_action = QAction("Terminate", self.table)
        terminate_action.triggered.connect(partial(self.terminate_clicked))
        terminate_action.setShortcut("DELETE")
        terminate_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(terminate_action)

        terminate_all_action = QAction("Terminate all", self.table)
        terminate_all_action.triggered.connect(partial(self.terminate_all))
        self.table.addAction(terminate_all_action)

        sleep_action = QAction("Sleep", self.table)
        sleep_action.triggered.connect(partial(self.set_sleeping))
        sleep_action.setShortcut("SHIFT+SPACE")
        sleep_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(sleep_action)

        pause_action = QAction("Pause", self.table)
        pause_action.triggered.connect(partial(self.pause))
        pause_action.setShortcut("SPACE")
        pause_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(pause_action)

    # ── Queue actions ────────────────────────────────────────────────────────

    def delete_clicked(self):
        """
        Delete a task or flag it for termination if already running.
        Routes through _model_del / _model_setitem so both roles work correctly.
        """
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            status = self.expDict[taskNr]['status']
            if status in ('Running', 'Scanning'):
                # Can't hard-delete a running task — flag it instead
                self._model_setitem(taskNr, 'terminated', True)
            else:
                self._model_del(taskNr)
            self.table.setCurrentIndex(idx[0])

    def terminate_clicked(self):
        """
        Gracefully terminate the selected task (or the first task if none selected).
        Sets 'terminated' and 'status' fields via _model_setitem so the change
        is routed correctly in both SERVER and CLIENT roles.
        """
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            self._model_setitem(taskNr, 'terminated', True)
            self._model_setitem(taskNr, 'status', 'Termination Pending')
        else:
            if self.tableModel.row_to_key:
                taskNr = self.tableModel.row_to_key[0]
                self._model_setitem(taskNr, 'terminated', True)
                self._model_setitem(taskNr, 'status', 'Termination Pending')

    def terminate_all(self):
        """
        Flag every task for termination.
        Iterates over a snapshot of row_to_key to avoid mutation-during-iteration.
        """
        for taskNr in list(self.tableModel.row_to_key):
            self._model_setitem(taskNr, 'terminated', True)
            self._model_setitem(taskNr, 'status', 'Termination Pending')

    def set_sleeping(self):
        """Toggle the selected task between Sleeping and Queued."""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            status = self.expDict[taskNr]['status']
            new_status = 'Queued' if status == 'Sleeping' else 'Sleeping'
            self._model_setitem(taskNr, 'status', new_status)

    def pause(self):
        """
        SERVER: call expManager.pause() directly.
        CLIENT: send pause command via CommandClient.
        """
        if self.role == QueueRole.SERVER:
            self.browser.expManager.pause()
            paused = self.browser.expManager.paused
        else:
            if self._command_client is not None:
                self._command_client.send('pause')
            current = self.props.get('/ExperimentQ/paused', False)
            paused = not current

        icon = 'run.png' if paused else 'pause.png'
        self.pauseButton.setIcon(QtGui.QIcon(icon_path + icon))

    def restart(self):
        """
        SERVER: start the queue directly if not already running.
        CLIENT: send a restart command via CommandClient.
        """
        if self.role == QueueRole.SERVER:
            if not self.browser.expManager.queueRunning:
                self.browser.expManager.start_queue()
            else:
                print('queue already running')
        else:
            if self._command_client is not None:
                self._command_client.send('restart')

    # ── Slots ────────────────────────────────────────────────────────────────

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def update_item(self, k, v):
        """
        Legacy slot — kept for compatibility with any signals still connected.
        In the new scheme, ModelClient handles model updates directly.
        Direct callers should prefer _model_set().
        """
        self._model_set(k, v)
        self.lock = False

    @QtCore.pyqtSlot(int)
    def delete_item(self, k):
        """Legacy slot — see update_item."""
        self._model_del(k)
        self.lock = False

    @QtCore.pyqtSlot(str, str, object)
    def _on_remote_sync(self, model_name: str, op: str, key):
        """
        Called on the Qt thread whenever ModelClient receives a broadcast.
        The model is already updated at this point — use this for side-effects
        like logging, status bar updates, or triggering downstream logic.
        """
        pass  # extend as needed

    @QtCore.pyqtSlot(str)
    def _on_connection_failed(self, endpoint: str):
        """Called if ModelClient could not reach the server during start()."""
        self.setTitle(f'ExperimentQ  ⚠ disconnected ({endpoint})')
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(False)