#pylint: disable=<C0302>
from functools import partial

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QTableView, QPushButton, QAbstractItemView, QAction, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QHeaderView, QGroupBox, QDialog

from pytweezer.servers import Properties, tweezerpath, icon_path, PropertyAttribute, DataClient
from pytweezer.GUI.models import ScheduleModel, PrepModel


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

        self.restartButton = QPushButton('Restart Q')
        self.restartButton.setMaximumWidth(30)
        self.restartButton.clicked.connect(self.restart)
        self.buttonLayout.addWidget(self.restartButton)

        self.buttonLayout.setAlignment(Qt.AlignLeft)

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

    def init_table(self):
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.expDict = {}
        self.tableModel = ScheduleModel(self.expDict)
        self.table.setModel(self.tableModel)

        cw = QtGui.QFontMetrics(self.font()).averageCharWidth()
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1,h.count()):
            h.setSectionResizeMode(1, QHeaderView.ResizeToContents)

    def set_model(self, model):
        """Sets a new model for the table"""
        self.tableModel = model
        self.table.setModel(self.tableModel)

    def delete_clicked(self):
        """Deletes a task from the queue, or terminates gracefully if already running"""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            status = self.expDict[taskNr]['status']
            if status == 'Running' or status == 'Scanning':
                self.tableModel[taskNr]['terminated'] = True
            else:
                del self.tableModel[taskNr]
            self.table.setCurrentIndex(idx[0])

    def terminate_clicked(self):
        """Selected task will be gracefully terminated (allowed to finish run)"""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            self.tableModel[taskNr]['terminated'] = True
            self.tableModel[taskNr]['status'] = 'Termination Pending'
        else:
            if len(self.tableModel.row_to_key) > 0:
                taskNr = self.tableModel.row_to_key[0]
                self.tableModel[taskNr]['terminated'] = True
                self.tableModel[taskNr]['status'] = 'Termination Pending'
            else:
                pass

    def terminate_all(self):
        for taskNr in self.tableModel.row_to_key:
            self.tableModel[taskNr]['terminated'] = True
            self.tableModel[taskNr]['status'] = 'Termination Pending'

    def set_sleeping(self):
        """Selected task will be ignored by the manager"""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            taskNr = self.tableModel.row_to_key[row]
            status = self.expDict[taskNr]['status']
            if status == 'Sleeping':
                self.tableModel[taskNr]['status'] = 'Queued'
            else:
                self.tableModel[taskNr]['status'] = 'Sleeping'

    def pause(self):
        """Pause the currently running experiment"""
        self.browser.expManager.pause()
        if self.browser.expManager.paused:
            self.pauseButton.setIcon(QtGui.QIcon(icon_path+'run.png'))
        else:
            self.pauseButton.setIcon(QtGui.QIcon(icon_path+'pause.png'))

    @QtCore.pyqtSlot(int, QtCore.QVariant)
    def update_item(self, k, v):
        self.tableModel[k] = v
        self.lock = False

    @QtCore.pyqtSlot(int)
    def delete_item(self, k):
        del self.tableModel[k]
        self.lock = False

    def restart(self):
        if not self.browser.expManager.queueRunning:
            self.browser.expManager.start_queue()
        else:
            print('queue already running')

    def props_test_func(self):
        """
        I've been using this for testing things because the button was there and unused
        """
        #self.test.value = 5
        #test = self._props.get('testval')
        #print('Set value to 5. get property. result: {}'.format(test))
        #self.test.value += 1
        #test2 = self._props.get('testval')
        #print('Increment value by 1. get property. result: {}'.format(test2))
        #self._props.set('testval', 1)
        #test3 = self.test.value
        #print('Set property to 1. get value. result: {}'.format(test3))
        #test4 = self.test.value*3
        #print('multiply value by 3. result: {}'.format(test4))
        self.testlist.value = [1, 2, 3]
        test = self._props.get('testlist')
        print('Set value to [1,2,3]. get property. result: {}'.format(test))
        self.testlist.value.append(4)
        test2 = self._props.get('testlist')
        print('append 4. get property. result: {}'.format(test2))
        self._props.set('testlist', ['a', 'b', 'c'])
        test3 = self.testlist.value
        print('Set property to [a, b, c]. get value. result: {}'.format(test3))
        test4 = self.testlist.value[2]
        print('get index 2 result: {}'.format(test4))
        self.testlist.value[2] = 'd'
        test5 = self._props.get('testlist')
        print('set index 2 to d. get property. result: {}'.format(test5))


        # self.test = 5
        # test = self._props.get('testval')
        # print('Set value to 5. get property. result: {}'.format(test))
        # self._props.set('testval', 1)
        # test3 = self.test
        # print('Set property to 1. get value. result: {}'.format(test3))
