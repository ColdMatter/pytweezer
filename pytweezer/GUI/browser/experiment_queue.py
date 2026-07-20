#pylint: disable=<C0302>

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableView, QPushButton, QAbstractItemView, QAction, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QHeaderView, QGroupBox

from pytweezer.servers import icon_path, PropertyAttribute
from pytweezer.servers.model_sync import SyncedScheduleModel


class ExperimentQ(QGroupBox):
    """
    The Experiment Queue has a dictionary of tasks expDict
    Each task is a dictionary containing the information needed to run the experiment.
    expDict is displayed in a table sorted by 1) priority and 2) task number.
    The table uses the model ScheduleModel defined in GUI/models.
    Tasks can be paused, terminated, or deleted prematurely.
    """
    def __init__(self, browser, parent=None, title="ExperimentQ"):
        super().__init__(title, parent)
        self.browser = browser
        self.props = browser.props
        self._props = self.props
        self.qLayout = QHBoxLayout()
        self.buttonLayout = QVBoxLayout()
        self.create_buttons()
        self.qLayout.addLayout(self.buttonLayout)

        self.init_table()
        self.init_table_actions()
        self.qLayout.addWidget(self.table)
        self.setLayout(self.qLayout)
        self.props.set('testval',3)
        self.test = PropertyAttribute('testval', 3, parent=self)
        self._props.set('testlist', [])
        self.testlist = PropertyAttribute('testlist', [], parent=self)

        self.lock = False
        self._selected_task_key = None
        self._selected_column = 0

    def create_buttons(self):
        self.pauseButton = QPushButton('')
        self.pauseButton.setMaximumWidth(30)
        self.pauseButton.setIcon(QtGui.QIcon(icon_path+'pause.png'))
        self.pauseButton.clicked.connect(self.pause)
        self.buttonLayout.addWidget(self.pauseButton)

        terminateButton = QPushButton('')
        terminateButton.setMaximumWidth(30)
        terminateButton.setIcon(QtGui.QIcon(icon_path+'terminate.png'))
        terminateButton.clicked.connect(self.terminate_clicked)
        self.buttonLayout.addWidget(terminateButton)

        terminateAllButton = QPushButton('')
        terminateAllButton.setMaximumWidth(30)
        terminateAllButton.setIcon(QtGui.QIcon(icon_path+'nuke_icon.png'))
        terminateAllButton.clicked.connect(self.terminate_all)
        self.buttonLayout.addWidget(terminateAllButton)


        self.buttonLayout.setAlignment(Qt.AlignLeft)

    def init_table_actions(self):

        delete_action = QAction("Delete", self.table)
        delete_action.triggered.connect(self.delete_clicked)
        delete_action.setShortcut("SHIFT+DELETE")
        delete_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(delete_action)

        terminate_action = QAction("Terminate", self.table)
        terminate_action.triggered.connect(self.terminate_clicked)
        terminate_action.setShortcut("DELETE")
        terminate_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(terminate_action)

        terminate_all_action = QAction("Terminate all", self.table)
        terminate_all_action.triggered.connect(self.terminate_all)
        self.table.addAction(terminate_all_action)

        sleep_action = QAction("Sleep", self.table)
        sleep_action.triggered.connect(self.set_sleeping)
        sleep_action.setShortcut("SHIFT+SPACE")
        sleep_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(sleep_action)

        pause_action = QAction("Pause", self.table)
        pause_action.triggered.connect(self.pause)
        pause_action.setShortcut("SPACE")
        pause_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(pause_action)

    def init_table(self):
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.tableModel = SyncedScheduleModel()
        self.table.setModel(self.tableModel)
        self._connect_model_selection_preservation(self.tableModel)

        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1,h.count()):
            h.setSectionResizeMode(1, QHeaderView.ResizeToContents)

    def set_model(self, model):
        """Sets a new model for the table"""
        self.tableModel = model
        self.table.setModel(self.tableModel)
        self._connect_model_selection_preservation(self.tableModel)

    @property
    def expDict(self):
        """Return live task dictionary from the synced schedule model."""
        return self.tableModel.backing_store

    def _connect_model_selection_preservation(self, model):
        model.modelAboutToBeReset.connect(self._remember_selection)
        model.modelReset.connect(self._restore_selection)

    def _remember_selection(self):
        idx = self.table.selectedIndexes()
        if not idx:
            self._selected_task_key = None
            self._selected_column = 0
            return

        current = idx[0]
        row = current.row()
        if row < 0 or row >= len(self.tableModel.row_to_key):
            self._selected_task_key = None
            self._selected_column = 0
            return

        self._selected_task_key = self.tableModel.row_to_key[row]
        self._selected_column = current.column()

    def _restore_selection(self):
        if self._selected_task_key is None:
            return

        try:
            row = self.tableModel.row_to_key.index(self._selected_task_key)
        except ValueError:
            self._selected_task_key = None
            return

        column = min(self._selected_column, max(0, self.tableModel.columnCount() - 1))
        new_index = self.tableModel.index(row, column)
        self.table.setCurrentIndex(new_index)
        self.table.selectRow(row)

    def delete_clicked(self):
        """Deletes a task from the queue, or terminates gracefully if already running"""
        current = self._selected_index()
        if current is None:
            return

        taskNr = self.tableModel.row_to_key[current.row()]
        status = self.expDict[taskNr].get('status')
        if status in ('Running', 'Scanning'):
            self.tableModel[taskNr]['terminated'] = True
        else:
            del self.tableModel[taskNr]
        self.table.setCurrentIndex(current)

    def terminate_clicked(self):
        """Selected task will be gracefully terminated (allowed to finish run)"""
        taskNr = self._get_selected_task_key(default_first=True)
        if taskNr is None:
            return

        self.tableModel[taskNr]['terminated'] = True
        self.tableModel[taskNr]['status'] = 'Termination Pending'

    def terminate_all(self):
        for taskNr in self.tableModel.row_to_key:
            self.tableModel[taskNr]['terminated'] = True
            self.tableModel[taskNr]['status'] = 'Termination Pending'

    def set_sleeping(self):
        """Selected task will be ignored by the manager"""
        taskNr = self._get_selected_task_key(default_first=False)
        if taskNr is None:
            return

        status = self.expDict[taskNr].get('status')
        self.tableModel[taskNr]['status'] = 'Queued' if status == 'Sleeping' else 'Sleeping'

    def pause(self):
        """Toggle pause state by editing the active task status in the synced model."""
        taskNr = self._get_active_task_key()
        if taskNr is None:
            # No active task to pause/resume.
            self.pauseButton.setIcon(QtGui.QIcon(icon_path + 'pause.png'))
            return

        status = self.expDict[taskNr].get('status', '')
        if status == 'Paused':
            # Resume into queued state; manager will set Running/Scanning on next run.
            self.tableModel[taskNr]['status'] = 'Queued'
            self.pauseButton.setIcon(QtGui.QIcon(icon_path + 'pause.png'))
        elif status in ('Running', 'Scanning'):
            self.tableModel[taskNr]['status'] = 'Paused'
            self.pauseButton.setIcon(QtGui.QIcon(icon_path + 'run.png'))

    def _get_active_task_key(self):
        """Return the currently active task key (running/scanning/paused), if any."""
        taskNr = self._get_selected_task_key(default_first=False)
        if taskNr is not None:
            status = self.expDict.get(taskNr, {}).get('status', '')
            if status in ('Running', 'Scanning', 'Paused'):
                return taskNr

        for taskNr in self.tableModel.row_to_key:
            status = self.expDict.get(taskNr, {}).get('status', '')
            if status in ('Running', 'Scanning', 'Paused'):
                return taskNr
        return None

    def _selected_index(self):
        """Return current selected model index, if any."""
        selected = self.table.selectedIndexes()
        if not selected:
            return None
        return selected[0]

    def _get_selected_task_key(self, default_first=False):
        """Return selected task key, optionally falling back to first task."""
        current = self._selected_index()
        if current is not None:
            row = current.row()
            if 0 <= row < len(self.tableModel.row_to_key):
                return self.tableModel.row_to_key[row]

        if default_first and self.tableModel.row_to_key:
            return self.tableModel.row_to_key[0]
        return None

    # @QtCore.pyqtSlot(int, QtCore.QVariant)
    # def update_item(self, k, v):
    #     self.tableModel[k] = v
    #     self.lock = False

    # @QtCore.pyqtSlot(int)
    # def delete_item(self, k):
    #     del self.tableModel[k]
    #     self.lock = False