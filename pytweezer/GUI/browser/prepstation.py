#pylint: disable=<C0302>
import os
from functools import partial
import json
from datetime import datetime

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QTableView, QPushButton, QHBoxLayout, QVBoxLayout, QDialog
from PyQt5.QtWidgets import QAction, QLabel, QCheckBox, QGridLayout, QHeaderView, QGroupBox
from PyQt5.QtCore import QDateTime

import pytweezer
from pytweezer.analysis.floating_point_arithmetics import round_floating_prec
from pytweezer.analysis.print_messages import print_error
from pytweezer.servers import tweezerpath, icon_path
from pytweezer.GUI.models import PrepModel
from pytweezer.servers.model_sync import SyncedPrepModel
from pytweezer.GUI.arg_boxes import FloatBox, BoolBox, ComboBox


class PrepStation(QGroupBox):
    """
    The Prep Station is a staging area for the experiment queue. It has a prepList and table
    similar to the ExperimentQ. It's a way to 'store' tasks before they're ready to be used.
    The task can be copied before sending to the queue so the same parameter set can be
    used multiple times.
    Tasks of the same experiment with different parameters can be distinguished by their name.

    TO DO:
    At some point ExperimentQ and PrepStation could be subclassed, as they share many methods and properies.
    The prep station should be able to build experiment loops, which run experiments conditionally on
    the outcomes of measurements.
    """
    def __init__(self, browser=None, title="PrepStation"):
        super().__init__(title, parent=browser)
        self.browser = browser
        self.main = True
        self.queue = self.browser.queue
        self.expDict = self.queue.expDict
        self.qWidget = QWidget()
        self.prep_layout = QHBoxLayout()
        self.buttonLayout = QVBoxLayout()
        self.props = browser.props
        self._task = browser._task
        self.paramDir = browser.paramDir
        self.table = QTableView()

        self.prepList = []
        self.init_ui()

        self.table.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.init_table_actions()

        self.setLayout(self.prep_layout)

    def init_ui(self):
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()

        pushButton = QPushButton('')
        pushButton.setMaximumWidth(30)
        pushButton.setIcon(QtGui.QIcon(icon_path + 'run.svg'))
        pushButton.clicked.connect(self.push)
        self.buttonLayout.addWidget(pushButton)

        # loopButton = QPushButton('')
        # loopButton.setMaximumWidth(30)
        # loopButton.setIcon(QtGui.QIcon(icon_path + 'right_arrow.svg'))
        # loopButton.clicked.connect(self.push_to_looper)
        # self.buttonLayout.addWidget(loopButton)

        # pushAllButton = QPushButton('Push All')
        # pushAllButton.clicked.connect(self.push_all_clicked)
        # self.buttonLayout.addWidget(pushAllButton)

        # pushKeepButton = QPushButton('Push and Keep')
        # pushKeepButton.clicked.connect(self.push_keep)
        # self.buttonLayout.addWidget(pushKeepButton)

        moveUpButton = QPushButton('')
        moveUpButton.setMaximumWidth(30)
        moveUpButton.setIcon(QtGui.QIcon(icon_path + 'up_arrow.svg'))
        moveUpButton.clicked.connect(self.move_up)
        moveUpButton.setShortcut("PgUp")
        self.buttonLayout.addWidget(moveUpButton)

        moveDownButton = QPushButton('')
        moveDownButton.setMaximumWidth(30)
        moveDownButton.setIcon(QtGui.QIcon(icon_path + 'down_arrow.svg'))
        moveDownButton.clicked.connect(self.move_down)
        moveDownButton.setShortcut("PgDown")
        self.buttonLayout.addWidget(moveDownButton)

        self.buttonLayout.setAlignment(Qt.AlignLeft)
        self.prep_layout.addLayout(self.buttonLayout)

    def init_table_actions(self):
        delete_action = QAction("Delete", self.table)
        delete_action.triggered.connect(partial(self.delete_clicked))
        delete_action.setShortcuts(["SHIFT+DELETE", "Ctrl+DELETE"])
        delete_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(delete_action)

        sleep_action = QAction("Sleep", self.table)
        sleep_action.triggered.connect(partial(self.set_sleeping))
        sleep_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(sleep_action)

        push_action = QAction("Push to Q", self.table)
        push_action.triggered.connect(partial(self.push))
        push_action.setShortcut("SPACE")
        push_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(push_action)

        # loop_action = QAction("Loop", self.table)
        # loop_action.triggered.connect(partial(self.push_to_looper))
        # loop_action.setShortcuts(["SHIFT+SPACE", "Ctrl+SPACE"])
        # loop_action.setShortcutContext(Qt.WidgetShortcut)
        # self.table.addAction(loop_action)

        edit_action = QAction("Edit", self.table)
        edit_action.triggered.connect(self.open_editor)
        edit_action.setShortcut("e")
        edit_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(edit_action)

        view_action = QAction("View", self.table)
        view_action.triggered.connect(self.open_viewer)
        view_action.setShortcut("v")
        view_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(view_action)

        save_action = QAction("Save", self.table)
        save_action.triggered.connect(self.save_params)
        save_action.setShortcut("s")
        save_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(save_action)

    def set_model(self):
        self.tableModel = SyncedPrepModel(self.prepList)
        self.table.setModel(self.tableModel)
        self.prep_layout.addWidget(self.table)

        h = self.table.horizontalHeader()
        for i in range(h.count()):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        size = self.size()
        size.setHeight(self.table.height())
        self.resize(size)
        self.adjustSize()

    def load_previous(self):
        if os.path.exists(tweezerpath+'/configuration/tweezer_browser/prepfile.json'):
            with open(tweezerpath+'/configuration/tweezer_browser/prepfile.json') as f:
                prepList = json.load(f)
                for d in prepList:
                    args_round = {}
                    for arg in d['args']:
                        args_round[arg] = round_floating_prec(d['args'][arg])
                    d['args'] = args_round
                    if 'dueDateTime' not in d.keys():
                        d['dueDateTime'] = QDateTime.currentDateTime().toString()
                    opened = False
                    for window in self.browser.openWindows:
                        if d['filepath'] == window.filepath:
                            opened = True
                    if not opened:
                        self.browser.open_experiment(d['filepath'])
                self.prepList = prepList
        else:
            return

    def push(self):
        """Push the task and keep a copy of it in the prepstation"""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            self.push_row(row)
        else:
            self.push_first()

    def push_row(self, row):
        self._task.value += 1
        newTaskNr = self._task.value
        self.tableModel[row]['task'] = newTaskNr
        if self.tableModel[row]['status'] != 'Sleeping':
            if QDateTime.fromString(self.tableModel[row]['dueDateTime']) < QDateTime.currentDateTime():
                self.tableModel[row]['status'] = 'Queued'
            else:
                self.tableModel[row]['status'] = 'Waiting'
        task_dict = self.tableModel[row].copy()
        # task_dict['experiment'] = get_experiment(task_dict['filepath'], self.browser)
        print_error('\nprepstation.py - push_row(): Submitted task dict:\n{0}\n'.format(task_dict), 'weak')
        self.queue.tableModel[newTaskNr] = task_dict
        self.update_prep_file()

    def push_first(self):
        """Push the first task to the queue"""
        self.push_row(0)

    def delete_clicked(self):
        idx = self.table.selectedIndexes()
        if idx:
            for i in idx[::-1]:
                row = i.row()
                if row > self.tableModel.rowCount()-1:
                    pass
                else:
                    del self.tableModel[row]
                    if row == self.tableModel.rowCount():
                        index = self.tableModel.index(self.tableModel.rowCount()-1, idx[0].column())
                        self.table.setCurrentIndex(index)
                    else:
                        self.table.setCurrentIndex(i)
        self.update_prep_file()

        # self.browser.looper.startup = True
        # for group in self.browser.looper.groupDict.values():
        #     for item in group.groupItemList:
        #         item.update_nr_box()
        # self.browser.looper.startup = False
        # self.browser.looper.update_loop_file()

    def set_sleeping(self):
        """Set the task as sleeping before sending"""
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            status = self.tableModel[row]['status']
            if status == 'Sleeping':
                self.tableModel[row]['status'] = 'Queued'
            else:
                self.tableModel[row]['status'] = 'Sleeping'

    # def push_to_looper(self):
    #     idx = self.table.selectedIndexes()
    #     if idx:
    #         row = idx[0].row()
    #         self.browser.looper.tabWidget.currentWidget().baustelle.add_task_item(row)

    def move_up(self):
        idx = self.table.selectedIndexes()
        if idx:
            idx = idx[0]
            row = idx.row()
            column = idx.column()
            if row != 0:
                # Swap in the model's backing store
                self.tableModel.backing_store[row], self.tableModel.backing_store[row-1] = \
                    self.tableModel.backing_store[row-1], self.tableModel.backing_store[row]
                self.tableModel.layoutChanged.emit()
                newIndex = self.tableModel.index(row-1, column)
                self.table.setCurrentIndex(newIndex)
            else:
                pass

    def move_down(self):
        idx = self.table.selectedIndexes()
        if idx:
            idx = idx[0]
            row = idx.row()
            column = idx.column()
            if row != self.tableModel.rowCount()-1:
                # Swap in the model's backing store
                self.tableModel.backing_store[row], self.tableModel.backing_store[row+1] = \
                    self.tableModel.backing_store[row+1], self.tableModel.backing_store[row]
                self.tableModel.layoutChanged.emit()
                newIndex = self.tableModel.index(row+1, column)
                self.table.setCurrentIndex(newIndex)
            else:
                pass

    def update_prep_file(self):
        with open(tweezerpath+'/configuration/tweezer_browser/prepfile.json', 'w') as f:
            # Save from the synced model's backing store, not the stale prepList
            json.dump(self.tableModel.backing_store, f, indent=4)
            

    def open_editor(self):
        taskList = []
        idx = self.table.selectedIndexes()
        if idx:
            for index in idx:
                row = index.row()
                taskList.append(self.tableModel[row])
        else:
            taskList.append(self.tableModel[0])
        name = taskList[0]['expName']
        if not all(task['expName'] == name for task in taskList):
            print('Multiple experiments selected. Cannot open editor')
            return
        expEditor = ExpEditWindow(taskList, parent=self)
        expEditor.exec_()

    def open_viewer(self):
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            task = self.tableModel[row]
        else:
            task = self.tableModel[0]
        self.viewer = ParViewWindow(task, parent=self)
        self.viewer.exec_()

    def save_params(self):
        """
        saves experiment params to a json as in the balibrowser
        """
        idx = self.table.selectedIndexes()
        if idx:
            row = idx[0].row()
            task = self.tableModel[row]
        else:
            task = self.tableModel[0]
        saved_params = task['args']

        if task['label'] != '':
            label = task['label']
        else:
            label = 'unlabelled'
        saved_params['label'] = label

        dateDir = datetime.today().strftime('%Y_%m_%d')
        paramTime = datetime.today().strftime('%Hh%M')
        fullDir = self.paramDir + '/' + task['expName'] + '/' + dateDir
        if not os.path.exists(fullDir):
            os.mkdir(fullDir)
        with open(fullDir + '/' + paramTime + '_' + label + '.json', 'w') as outfile:
            json.dump(saved_params, outfile, indent=4)


class ExpEditWindow(QDialog):
    def __init__(self, taskList, parent=None):
        super().__init__(parent)
        self.browser = parent.browser
        self.props = parent.browser.props
        self.taskList = taskList
        self.experiment = get_experiment(taskList[0]['filepath'], self.browser)
        self.setWindowTitle("Experiment Editor")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.submitButton = QPushButton("Submit Changes")
        self.submitButton.clicked.connect(self.submit)
        layout.addWidget(self.submitButton)
        gridlayout = QGridLayout()
        ncol = self.experiment._gui_columns
        self.argument_boxes = []

        # building the argument view
        for i, argument in enumerate(self.experiment._arguments):
            if type(argument) == pytweezer.experiment.experiment.NumberValue:
                box = FloatBox(props=None, **argument.__dict__.copy())
            elif type(argument) == pytweezer.experiment.experiment.BoolValue:
                box = BoolBox(props=None, value=True, parName=argument.name)
            elif type(argument) == pytweezer.experiment.experiment.StringCombo:
                box = ComboBox(props=None, parName=argument.name, **argument.__dict__.copy())
            else:
                print('tweezer_browser error argument type not known')
                box = None
            box = CheckLayout(box)
            self.argument_boxes.append(box)
            gridlayout.addWidget(box, int(i/ncol), i%ncol)
        layout.addLayout(gridlayout)
        self.setLayout(layout)

    def submit(self):
        for box in self.argument_boxes:
            if box.checkBox.isChecked():
                for task in self.taskList:
                    task['args'][box.widget.name] = box.widget.value
        self.parent().update_prep_file()
        self.accept()


class ParViewWindow(QDialog):
    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.browser = parent.browser
        self.props = parent.browser.props
        self.task = task
        self.experiment = get_experiment(task['filepath'], self.browser)
        self.setWindowTitle("Parameter Viewer")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        gridlayout = QGridLayout()
        ncol = self.experiment._gui_columns
        self.argumentBoxes = []

        # building the argument view
        for i, argument in enumerate(self.experiment._arguments):
            parName = argument.name
            if parName in self.task['args']:
                parVal = self.task['args'][parName]
                if hasattr(argument, 'display_multiplier'):
                    parVal = round_floating_prec(parVal / argument.display_multiplier)
            else:
                parVal = '--'
            argDict = argument.__dict__.copy()
            if 'unit' in argDict:
                parUnit = ' ' + argDict['unit']
            else:
                parUnit = ''
            parVal = str(parVal)
            box = DisplayBox(parName, parVal, parUnit)
            self.argumentBoxes.append(box)
            gridlayout.addWidget(box, int(i/ncol), i % ncol)
        layout.addLayout(gridlayout)
        self.setLayout(layout)


class CheckLayout(QWidget):
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.checkBox = QCheckBox()
        self.widget = widget
        layout.addWidget(self.checkBox)
        layout.addWidget(self.widget)
        self.setLayout(layout)


class DisplayBox(QWidget):
    def __init__(self, parName, parVal, parUnit, parent=None):
        super().__init__(parent)
        self.parName = parName
        self.parVal = parVal
        self.parUnit = parUnit
        self.init_ui()

    def init_ui(self):
        layout=QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(QLabel(self.parName))
        layout.addWidget(QLabel(self.parVal + self.parUnit))
        self.setLayout(layout)


def get_experiment(filepath, browser):
    opened = False
    for window in browser.openWindows:
        if filepath == window.filepath:
            experiment = window.experiment
            opened = True
    if not opened:
        exwin = browser.open_experiment(filepath)
        experiment = exwin.experiment
    return experiment
