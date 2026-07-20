from PyQt5 import QtGui
import time
import os
import json
from datetime import datetime
import copy
import numpy as np
import operator as op
import re

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QDialog, QScrollArea, QSpinBox, QCheckBox
from PyQt5.QtWidgets import QPushButton, QLabel, QAction, QComboBox, QLineEdit, QFrame, QFileDialog, QGroupBox, QTextEdit
from PyQt5.QtWidgets import QGridLayout, QApplication, QTabBar, QTabWidget, QStyle
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer, QEvent, QTime, QDate, QDateTime
from pytweezer.servers import Properties, PropertyAttribute
from pytweezer.servers.clients import DataClient
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.GUI.browser.prepstation import PrepStation
from pytweezer.servers import tweezerpath, icon_path
from pytweezer.GUI.browser.browser_workers import Worker
from pytweezer.GUI.models import PrepModel
from pytweezer.servers.model_sync import SyncedPrepModel
from pytweezer.GUI.qled import LedIndicator
from pytweezer.GUI.pytweezerQt import SearchComboBox
from pytweezer.analysis.print_messages import print_error

from functools import partial

propName = 'BaliBrowser/Looper'
_props = Properties(propName)
operators = {'>': op.gt, '>=': op.ge, '==': op.eq, '!=': op.ne, '=<': op.le, '<': op.lt}


class Looper(QGroupBox):
    """
    Looper allows groups of tasks to be run dependent on the outcome of measurements.
    """
    def __init__(self, parent=None, title="Looper"):
        super().__init__(title, parent)
        self.props = _props
        self._props = _props
        # self.parent = parent
        self.browser = parent
        self._task = self.browser._task
        self.loopDir = tweezerpath + '/configuration/loops'
        # Use the synced model's backing store for consistency with prep station changes
        self.tableModel = self.browser.prepStation.tableModel
        self.prepList = self.tableModel.backing_store
        self.keyList = PropertyAttribute('keyList', [], parent=self)
        self.dataManager = DataManager(self.keyList, self.props)
        self.dataDict = self.dataManager.dataDict  # dictionary for storing streamed data
        self.groupDict = {}
        self.loopManager = LoopManager(self.browser, self.props, parent=self)

        self.qlayout = QVBoxLayout()
        self.qlayout.setAlignment(Qt.AlignTop)

        self.buttonLayout = QHBoxLayout()
        self.create_buttons()
        self.qlayout.addLayout(self.buttonLayout)

        self.tabWidget = LoopTabWidget(self)
        self.qlayout.addWidget(self.tabWidget)

        self.setLayout(self.qlayout)

        self.autoTerminating = False
        self.temrinationTimer = QtCore.QTimer()
        self.temrinationTimer.timeout.connect(self.terminate_by_time)
        self.temrinationTimer.start(20)

    def add_tab(self, groupName=None):
        if not groupName:
            n = len(self.groupDict)
            groupName = 'New Tab #{}'.format(n)
            while groupName in self.groupDict:  # in case e.g. NewTab #1 is the only existing tab
                groupName = 'New Tab #{}'.format(n := n+1)
        group = LoopGroup(name=groupName, parent=self)
        self.tabWidget.addTab(group, groupName)
        for group in self.groupDict.values():
            group.baustelle.groupItem.update_boxes()
        self.tabWidget.setCurrentWidget(group)

    def delete_tab(self, index):
        if len(self.groupDict) > 1:  # things break if you delete the only tab
            group = self.tabWidget.widget(index)
            group.delete_group()

    def create_buttons(self):
        runButton = QPushButton('')
        runButton.setIcon(QtGui.QIcon(icon_path + 'run.svg'))
        runButton.clicked.connect(self.loopManager.run_loop)
        self.buttonLayout.addWidget(runButton)

        terminateButton = QPushButton('')
        terminateButton.setIcon(QtGui.QIcon(icon_path + 'terminate.svg'))
        terminateButton.clicked.connect(self.loopManager.terminate)
        self.buttonLayout.addWidget(terminateButton)

        subButton = QPushButton('Subs')
        subButton.clicked.connect(self.dataManager.edit_subscriptions)
        self.buttonLayout.addWidget(subButton)

        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.save_loop)
        self.buttonLayout.addWidget(saveButton)

        loadButton = QPushButton('Load')
        loadButton.clicked.connect(self.load_loop)
        self.buttonLayout.addWidget(loadButton)

        self.terminationButton = QPushButton('Enable AutoTermination:')
        self.terminationButton.setMinimumWidth(200)
        self.terminationButton.clicked.connect(self.toggle_auto_termination)
        self.buttonLayout.addWidget(self.terminationButton)
        self.terminationTimeBox = QLineEdit()
        self.buttonLayout.addWidget(self.terminationTimeBox)
        self.terminationLabel = QLabel('')
        self.buttonLayout.addWidget(self.terminationLabel)

    def update_loop_file(self):
        """convenience method for autosaving"""
        self.save_loop(auto=True)

    def load_previous(self):
        """convenience method for autoloading"""
        self.load_loop(auto=True)

    def save_loop(self, auto=False):
        """
        creates a json file of the current loop. if auto, this goes to the looplist.
        if not, a file is created based on the current date and time.
        """
        if auto:
            fullDir = tweezerpath + '/configuration/'
            path = fullDir + 'loopfile.json'
        else:
            dateDir = datetime.today().strftime('%Y_%m_%d')
            timeDir = datetime.today().strftime('%Hh%M')
            fullDir = self.loopDir + '/' + dateDir + '/'
            path = fullDir + timeDir + '.json'
        if not os.path.exists(fullDir):
            os.makedirs(fullDir)

        groupStore = {}
        for groupName in self.groupDict:
            itemStore = []
            group = self.groupDict[groupName]
            for item in group.groupItemList:
                itemDict = item.setup_item_dict()
                itemStore.append(itemDict)
            groupStore[groupName] = itemStore

        with open(path, 'w') as outfile:
            json.dump(groupStore, outfile, indent=4)

    def load_loop(self, auto=False):
        """
        loads a loop from a json file. erases the current loop.
        called with auto=True on browser startup to get looplist.
        """
        if auto:
            filepath = tweezerpath + '/configuration/loopfile.json'
            if not os.path.exists(filepath):
                print_error("Looper: loopfile not found", 'warning')
                self.add_tab()
                return
        else:
            filepath = self.get_files()
            if not filepath:
                return
        print(f'{filepath=}')
        with open(filepath) as json_file:
            try:
                loadedGroups = json.load(json_file)
            except Exception as e:
                print_error('Error loading loopfile', 'error')
                print(e)
                return

        if not loadedGroups:
            self.add_tab()
            return

        for group in list(self.groupDict.values()):
            group.delete_group()

        for groupName in loadedGroups:
            self.add_tab(groupName)
            items = loadedGroups[groupName]
            self.groupDict[groupName].baustelle.load_from_list(items)

    def get_files(self):
        return QFileDialog.getOpenFileName(self, "Select Loop File", self.loopDir, filter="Loop Files (*.json)")[0]

    def toggle_auto_termination(self):
        if not self.autoTerminating:
            self.terminationButton.setText('Disable AutoTermination:')
        else:
            self.terminationButton.setText('Enable AutoTermination:')
        self.autoTerminating = not self.autoTerminating

    def terminate_by_time(self):
        valTxt = self.terminationTimeBox.text()
        valTxt = parse_text(valTxt, startDateTime=self.loopManager.startDateTime)  # returns either '+x' seconds or an epoch time
        self.terminationLabel.setText(valTxt)
        if self.loopManager.terminated or not self.autoTerminating:
            return

        if len(valTxt) >= 2:

            now = QDateTime.currentDateTime().toSecsSinceEpoch()

            # calculate the Unix timestamp of the termination time:
            if valTxt[0] == '+':
                try:
                    val = self.loopManager.startDateTime.toSecsSinceEpoch() + int(valTxt[1:])
                except:
                    return
            else:
                try:
                    val = int(valTxt)
                except:
                    return

            if now > val:
                print_error('terminate_by_time: Terminating ...', 'warning')
                self.loopManager.terminate()
                print('terminate_by_time: valTxt={0},\tval={1}'.format(valTxt, val))
        return


class DataManager:
    def __init__(self, keyList, props):
        self.props = props
        self.streamNames = self.props.get('datastreams', [''])
        self.dataClient = DataClient(propName)
        self.dataClient.subscribe(self.streamNames)
        self.keyList = keyList
        self.dataDict = {}

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_new_data)
        self.timer.start(20)

    def set_new_data(self):
        if self.dataClient.has_new_data():
            recvmsg = self.dataClient.recv()
            if len(recvmsg) == 2:
                msg, di = recvmsg
            elif len(recvmsg) == 3:
                msg, di, A = recvmsg
            else:
                di = {}
            keys = list(di.keys())
            for key in keys:
                self.dataDict[key] = di[key]
            self.keyList.value = list(self.dataDict.keys())
            return di

    def edit_subscriptions(self):
        sub = QDialog()
        layout = QVBoxLayout()
        sub.setWindowTitle("Looper Subscriptions")
        editor = SubscriptionEditor(self.props, 'Data')
        layout.addWidget(editor)
        sub.setLayout(layout)
        sub.exec_()


class LoopManager:
    """
    The LoopManager handles the running of loops via the creation of workers on the browser's threadpool.
    It has a currentItemIndex and currentGroupName, which tells the loop worker which loop item to run from which
    loop group. The first item (index 0) of the first (leftmost) group will be run first. From there, the loop items set
    currentItemIndex when their run function is called.
    """

    def __init__(self, browser, props, parent=None):
        self.browser = browser
        self.props = props
        self.looper = parent
        self.threadPool = browser.threadPool
        self.expManager = browser.expManager
        self.loopWorker = None
        self.terminated = False
        self.currentGroupName = '0'
        self.currentIndex: 0  # the index of the current loop item
        self.currentItem = None  # the current loop item object
        self.lastGroupName = ''
        self.lastIndex = 0  # used for the max runs check
        self.lastTaskIndex = 0  # index of the last TaskItem or ListItem. used for the con runs check
        self.startDateTime = None

    def loop_fn(self, progress_callback):
        """The function run by the LoopWorker"""
        while self.terminated is False:  # loop stops when last item runs
            group = self.looper.groupDict[self.currentGroupName]
            if self.currentGroupName != self.lastGroupName:
                group.conRuns = 0
            itemList = group.groupItemList
            while self.expManager.running: # make sure last experiment is finished before starting
                time.sleep(0.1)
            self.currentItem = itemList[self.currentIndex]
            self.loopWorker.signals.taskStart.emit()
            self.currentItem.run(self.loopWorker.signals)  # different item types have their own run functions
            time.sleep(0.1)
            while self.expManager.running:  # make sure current experiment is finished before moving on
                time.sleep(0.1)
            self.currentItem.end_run()  # cleanup function
            self.loopWorker.signals.taskDone.emit()
            self.loopWorker.signals.update_ui.emit()
            self.lastIndex = self.currentItem
            time.sleep(0.1)  # some of these pauses might not be necessary

    def run_loop(self):
        """Sets all values to default then starts the loop worker in the browser threadpool."""
        self.terminated = False
        self.currentIndex = 0  # start a fresh loop
        self.currentGroupName = self.looper.tabWidget.widget(0).groupName
        self.startDateTime = QDateTime.currentDateTime()
        for group in self.looper.groupDict.values():
            group.conRuns = 0
            group.groupItemList[-1].update_con_check()
            group.groupItemList[-1].update_con_max()
            group.groupItemList[-1].update_con_box()
            idx = self.looper.tabWidget.indexOf(group)
            self.looper.tabWidget.setIconOff(idx)
            for item in group.groupItemList:
                item.totalRuns = 0
                item.update_runs_box()
                item.conRuns = 0
                item.update_con_box()
        self.loopWorker = Worker(self.loop_fn)
        self.loopWorker.signals.error.connect(self.loop_error)
        self.loopWorker.signals.finished.connect(self.loop_done)
        self.loopWorker.signals.taskStart.connect(self.task_start)
        self.loopWorker.signals.taskDone.connect(self.task_end)
        self.loopWorker.signals.addItem.connect(self.browser.queue.update_item)
        self.threadPool.start(self.loopWorker)

    def terminate(self):
        self.terminated = True

    # @QtCore.pyqtSlot()
    def task_start(self):
        self.currentItem.task_start()

    # @QtCore.pyqtSlot()
    def task_end(self):
        self.currentItem.task_end()

    # @QtCore.pyqtSlot()
    def loop_done(self):
        print('\n\033[1mloop done\n\033[0m')

    # @QtCore.pyqtSlot(tuple)
    def loop_error(self, args):
        """not sure if this does anything"""
        print_error('loop_error', 'error')
        print(args)


class LoopGroup(QFrame):
    """
    Container for a group of loop items. Exists as  a tab in the looper's tabWidget.
    Has a name, a list of items, and a baustelle for building its loop.
    Individual loop groups can be saved and loaded, as well as the entire loop.
    """
    def __init__(self, name='', parent=None):
        super().__init__(parent)
        self.groupName = name
        self.groupItemList = []
        self.looper = parent
        self.looper.groupDict[name] = self
        self.browser = parent.browser

        self._props = parent._props
        self.props = self._props
        self.baustelle = Baustelle(self.browser, self.props, parent=self)  # the loop builder
        self.qlayout = QVBoxLayout()

        self.buttonLayout = QHBoxLayout()
        self.create_buttons()

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.baustelle)

        self.qlayout.addLayout(self.buttonLayout)
        self.qlayout.addWidget(self.scrollArea)
        self.setLayout(self.qlayout)

        self.conRuns = 0
        self.conCheck = False
        self.maxCon = 0

    def create_buttons(self):
        itemButton = QPushButton('New Task')
        itemButton.clicked.connect(self.baustelle.add_task_item)
        self.buttonLayout.addWidget(itemButton)

        listButton = QPushButton('New List')
        listButton.clicked.connect(self.baustelle.add_list_item)
        self.buttonLayout.addWidget(listButton)

        condButton = QPushButton('Conditional')
        condButton.clicked.connect(self.baustelle.add_conditional)
        self.buttonLayout.addWidget(condButton)

        saveButton = QPushButton('Save Group')
        saveButton.clicked.connect(self.baustelle.save_loop_group)
        self.buttonLayout.addWidget(saveButton)

        loadButton = QPushButton('Load Group')
        loadButton.clicked.connect(self.baustelle.load_loop_group)
        self.buttonLayout.addWidget(loadButton)

    def delete_group(self):
        for item in self.looper.groupDict[self.groupName].groupItemList:
            item.delete_item()
        del self.looper.groupDict[self.groupName]
        idx = self.looper.tabWidget.indexOf(self)
        self.looper.tabWidget.removeTab(idx)
        self.deleteLater()

    def update_loop_group_file(self):
        self.baustelle.save_loop_group(auto=True)


class Baustelle(QWidget):
    """
    The Baustelle is the place for building loops and displaying loops. It contains the methods for creating loop
    items and adding them to its layout.
    Loops are composed of a list of loop items. There are three types of loop item: tasks, lists, and conditionals.
    All loops also have a GroupItem, which tells looper which group to go to next.
    Loop items all have an index, which is used both to tell the LooperManager when to run them and also the order
    in which they're displayed in the GUI
    Loops can be stored and loaded. A backup file of the current loop is updated when something is changed in the loop.
    The backup is automatically loaded when the browser is started.
    """

    def __init__(self, browser, props, parent=None):
        super().__init__(parent)
        self.group = parent
        self.groupName = parent.groupName
        self.groupItemList = parent.groupItemList
        self.looper = parent.looper
        self._props = parent._props
        self.browser = browser
        self._task = self.browser._task
        self.props = props
        self.loopDir = tweezerpath + '/configuration/loops'
        self.qlayout = QVBoxLayout()
        self.qlayout.setAlignment(Qt.AlignTop)
        self.add_group_item()
        self.setLayout(self.qlayout)

    def update_keylist(self):
        """list of labels of streamed data"""
        self.keyList = self.props.get('keyList', [])

    def get_files(self):
        return QFileDialog.getOpenFileName(self, "Select Loop File", self.loopDir, filter="Loop Files (*.json)")[0]

    def add_task_item(self, taskNr=0):
        item = TaskItem(taskNr, parent=self)
        self.add_item(item)
        return item

    def add_list_item(self):
        item = ListItem(parent=self)
        self.add_item(item)
        return item

    def add_conditional(self):
        item = ConditionalItem(parent=self)
        self.add_item(item)
        return item

    def add_group_item(self):
        self.groupItem = GroupItem(parent=self)
        self.qlayout.addWidget(self.groupItem)
        self.groupItemList.append(self.groupItem)

    def add_item(self, item):
        # self.qlayout.addWidget(item)
        self.qlayout.insertWidget(len(self.groupItemList)-1, item)
        self.groupItemList.insert(-1, item)
        self.update_idx()
        self.looper.update_loop_file()

    def update_idx(self):
        for i, item in enumerate(self.groupItemList):
            item.idx = i
            item.update_idx_label()
            item.update_boxes()

    def save_loop_group(self, auto=False):
        """
        creates a json file of the current loop. if auto, this goes to the looplist. if not, a browser window opens.
        """
        configDir = tweezerpath + '/configuration/'
        if auto:
            fullDir = configDir
            path = fullDir + 'looplist_' + self.groupName + '.json'
        else:
            dateDir = datetime.today().strftime('%Y_%m_%d')
            timeHM = datetime.today().strftime('%Hh%M')
            fullDir = self.loopDir + '/' + dateDir + '/'
            path = fullDir + self.groupName + '_' + timeHM + '.json'
        if not os.path.exists(fullDir):
            os.makedirs(fullDir)

        commandStore = []
        for item in self.groupItemList:
            itemDict = item.setup_item_dict()
            commandStore.append(itemDict)

        with open(path, 'w') as outfile:
            json.dump(commandStore, outfile, indent=4)

    def load_loop_group(self, auto=False):
        """
        loads a loop from a json file. erases the current loop.
        called with auto=True on browser startup to get looplist.
        """
        if auto:
            filepath = tweezerpath + '/configuration/looplist.json'
            if not os.path.exists(filepath):
                return
        else:
            filepath = self.get_files()
            if not filepath:
                return

        with open(filepath) as json_file:
            loadedItems = json.load(json_file)

        self.load_from_list(loadedItems)

    def load_from_list(self, loadedItems):
        for item in list(self.groupItemList):  # using list() creates a copy of the list to iterate while modifying
            if not isinstance(item, GroupItem):
                item.delete_item()

        for i, itemDict in enumerate(loadedItems):
            itemType = itemDict['type']
            if itemType in ('Task', 'Item'):  # need both for legacy reasons
                item = self.add_task_item()
            elif itemType == 'List':
                item = self.add_list_item()
            elif itemType == 'Conditional':
                item = self.add_conditional()
            elif itemType == 'Group':
                item = self.groupItem
            elif itemType == 'Name':
                continue
            else:
                print_error('Looper tried to load unrecognised item type: {}. '
                            'Please check the loop file'.format(itemType), 'error')
                continue
            item.load_item_dict(itemDict, loadedItems)


class LoopItem(QFrame):
    """
    This is the base class for the other loop items. It isn't ever used for loops.
    create_widgets creates the widgets shared by all loop items. The different loop items also create their own
    widgets.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemType = None
        self.setFrameShape(QFrame.Panel | QFrame.Sunken)
        self.setLineWidth(3)
        self.browser = parent.browser
        self.looper = parent.looper
        self.group = parent.group
        self.loopManager = self.looper.loopManager
        self._task = self.browser._task
        # Use the synced model's backing store
        self.tableModel = self.browser.prepStation.tableModel
        self.prepList = self.tableModel.backing_store
        self.props = parent.props
        self.groupItemList = parent.groupItemList
        self._props = parent._props
        self.idx = len(self.groupItemList)-1
        self.totalRuns = 0
        self.maxRunsCheck = False
        self.maxRuns = 0
        self.conRuns = 0
        self.maxConCheck = False
        self.maxCon = 0
        self.qlayout = QGridLayout()
        self.qlayout.setAlignment(Qt.AlignLeft)
        self.setLayout(self.qlayout)
        self.hidx = 0  # widgets added to the item layout from left to right by incrementing the hidx.

    def create_widgets(self):
        self.led = LedIndicator(self)
        self.led.setDisabled(True)

        self.idxLabel = QLabel('')
        self.idxLabel.setAlignment(Qt.AlignCenter)
        self.idxLabel.setText(str(self.idx))

        self.delButton = QPushButton("Del")
        self.delButton.clicked.connect(self.delete_item)

        self.upButton = QPushButton()
        self.upButton.setIcon(QtGui.QIcon(icon_path + 'up_arrow.svg'))
        self.upButton.clicked.connect(self.move_up)

        self.downButton = QPushButton()
        self.downButton.setIcon(QtGui.QIcon(icon_path + 'down_arrow.svg'))
        self.downButton.clicked.connect(self.move_down)

    def add_widget(self, widget, label):
        """convenience function for adding widgets to QGridLayout with labels"""
        label = QLabel(label)
        label.setStyleSheet("font-weight: bold")
        self.qlayout.addWidget(label, 0, self.hidx)
        self.qlayout.addWidget(widget, 1, self.hidx)
        self.hidx += 1

    def move_up(self):
        if self.idx == 0:
            return
        else:
            oldIdx = self.idx
            newIdx = self.idx - 1
            self.move_item(oldIdx, newIdx)

    def move_down(self):
        if self.idx == len(self.groupItemList)-2:
            return
        else:
            oldIdx = self.idx
            newIdx = self.idx + 1
            self.move_item(oldIdx, newIdx)

    def move_item(self, oldIdx, newIdx):
        """widgets are moved by taking them from the layout and reinserting them.
        they are then removed from and reinserted into the command list
        TODO: click and drag?"""
        # TODO: This doesn't move items of lists within the Baustelle
        widgetItem = self.parent().layout().takeAt(oldIdx)
        self.parent().layout().insertWidget(newIdx, widgetItem.widget())
        self.idx = newIdx
        self.groupItemList.insert(newIdx, self.groupItemList.pop(oldIdx))
        self.parent().update_idx() # when a task moves, the labels of the other tasks need to be updated
        self.update_loop_file()

    def update_idx_label(self):
        """updates the displayed index label."""
        self.idxLabel.setText(str(self.idx))

    def delete_item(self):
        """
        deleting widgets in pyqt is difficult and prone to errors. this function likely has some overkill.
        """
        self.groupItemList.remove(self)  # remove the item from the command list
        self.parent().layout().removeWidget(self)  # remove the item widget from the layout
        for i, item in enumerate(self.groupItemList):  # update labels
            item.idx = i
            item.update_idx_label()
        self.parent().children().remove(self)  # remove the item widget from the layout again, but different?
        self.deleteLater()  # remove the widget from memory, later ??
        self.update_loop_file()

    def update_loop_file(self):
        self.looper.update_loop_file()
        self.group.update_loop_group_file()

    def switch_led(self):
        self.led.setChecked(not self.led.isChecked())

    def led_on(self):
        self.led.setChecked(True)

    def led_off(self):
        self.led.setChecked(False)


    # sometimes for convenience we call a method from all loop items, even if not implemented
    # this saves having to check the type of every loop item. therefore we define the methods here
    # some of these may not be necessary or might not be used anymore. should be cleaned up.

    def run(self, signals):
        pass

    def end_run(self):
        pass

    def update_ui(self):
        pass

    def update_vals(self):
        pass

    def task_start(self):
        pass

    def task_end(self):
        pass

    def update_runs_box(self):
        pass

    def update_con_box(self):
        pass

    def update_boxes(self):
        pass

    def update_nr_box(self):
        pass

    def setup_item_dict(self):
        pass

    def load_item_dict(self, itemDict, loadedItems):
        pass


class TaskItem(LoopItem):
    """
    A TaskItem points to a single task in the preplist, which it runs once before incrementing the currentItem pointer.
    TaskItems track how many times they've been run this loop, as well as how many times consecutively (not counting
    conditionals). There is an option to set a max value for each of these. When reached, the loop terminates.
    """
    def __init__(self, taskNr=0, parent=None):
        super().__init__(parent)
        self.itemType = 'Task'
        self.taskNr = taskNr
        self.task = self.prepList[taskNr] if self.prepList else {}
        self.init_ui()
        self.setLayout(self.qlayout)

    def init_ui(self):

        self.create_widgets()
        self.create_task_widgets()

        self.qlayout.addWidget(self.idxLabel, 0, self.hidx)
        self.qlayout.addWidget(self.led, 1, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.nrBox,'Task:')

        self.add_widget(self.taskLabel, 'Label:')

        self.expLabel.setText((self.task.get('expName')))
        self.add_widget(self.expLabel, 'Experiment:')

        self.qlayout.setColumnStretch(self.hidx, 1)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.totalRunsBox, 'Run:')
        self.totalRunsBox.valueChanged.connect(self.update_runs)
        self.add_widget(self.maxRunsCheck, 'Check?')
        self.add_widget(self.maxRunsBox, 'Max:')

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.conRunsBox, 'Consecutive:')
        self.conRunsBox.valueChanged.connect(self.update_con)
        self.add_widget(self.maxConCheck, 'Check?')
        self.add_widget(self.maxConBox, 'Max consecutive:')

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.delButton, '')

        self.qlayout.addWidget(self.upButton, 0, self.hidx)
        self.qlayout.addWidget(self.downButton, 1, self.hidx)

    def create_task_widgets(self):
        self.nrBox = QComboBox()  # select a task from its preplist index
        for i, _ in enumerate(self.prepList):
            self.nrBox.addItem(str(i))
        self.nrBox.currentIndexChanged.connect(self.update_task)

        self.taskLabelLabel = QLabel('Label:')
        self.taskLabel = QLabel(self.task.get('label'))
        self.expLabel = QLabel(self.task.get('expName'))
        self.totalRunsBox = QSpinBox()
        self.totalRunsBox.setMaximum(99999)
        self.maxRunsCheck = QCheckBox()
        self.maxRunsBox = QSpinBox()
        self.maxRunsBox.setMaximum(99999)
        self.conRunsBox = QSpinBox()
        self.conRunsBox.setMaximum(99999)
        self.maxConCheck = QCheckBox()
        self.maxConBox = QSpinBox()
        self.maxConBox.setMaximum(99999)

    def run(self, signals):
        """
        creates a copy of the task from the preplist and adds it to the ExperimentQ. Adding to the Q is done via a
        pyqt signal, since this function is called from within a worker thread.
        """
        self._task.value += 1
        newTaskNr = self._task.value
        task_dict = self.task.copy()
        # task_dict['experiment'] = get_experiment(task_dict['filepath'], self.browser)
        signals.addItem.emit(newTaskNr, task_dict)

    def end_run(self):
        """increments run number and consecutive runs if applicable"""
        self.totalRuns += 1
        if (self.loopManager.lastGroupName == self.group.groupName) and (self.loopManager.lastTaskIndex == self.idx):
            self.conRuns += 1
        else:
            self.conRuns = 0

        if self.loopManager.lastGroupName == self.group.groupName:
            self.group.conRuns += 1

        self.loopManager.lastGroupName = self.group.groupName
        self.loopManager.lastIndex = self.loopManager.currentIndex
        self.loopManager.lastTaskIndex = self.loopManager.currentIndex
        self.loopManager.currentIndex += 1
        self.max_check()

    def task_start(self):
        """to be accessed from the worker signal taskStart"""
        self.led_on()

    def task_end(self):
        """to be accessed from the worker signal taskDone"""
        self.led_off()
        self.update_runs_box()
        self.update_con_box()
        self.update_browser_task()
        self.group.groupItemList[-1].update_con_box()

    def update_task(self):
        """updates the item's current task info"""
        nr = self.nrBox.currentText()
        if nr != '':
            taskNr = int(nr)
            self.taskNr = taskNr
            newTask = self.prepList[taskNr]
            self.task = newTask
            self.update_task_labels()

    def update_task_labels(self):
        """updates the item's task and display labels based on the selected task"""
        taskNr = self.taskNr
        newTask = self.prepList[taskNr]
        self.task = newTask
        idx = self.nrBox.findText(str(taskNr))
        self.nrBox.setCurrentIndex(idx)
        label = newTask['label']
        self.taskLabel.setText(label)
        name = newTask['expName']
        self.expLabel.setText(name)
        self.update_loop_file()

    def update_nr_box(self):
        """update the list of indices to choose from the preplist"""
        try:
            idx = int(self.nrBox.currentText())
        except:
            idx = 0
        self.nrBox.clear()
        for i, _ in enumerate(self.prepList):
            self.nrBox.addItem(str(i))
        self.nrBox.setCurrentIndex(idx)

    def max_check(self):
        """check if either max value has been reached"""
        if self.maxRunsCheck.isChecked() and self.totalRuns >= self.maxRunsBox.value():
            print_error('Looper: item {} reached max runs'.format(self.idx), 'warning')
            self.loopManager.terminate()
            self.loopManager.currentIndex = 0
        if self.maxConCheck.isChecked() and self.conRuns >= self.maxConBox.value():
            print_error('Looper: item {} reached max consecutive runs'.format(self.idx), 'warning')
            self.loopManager.terminate()
            self.loopManager.currentIndex = 0
        if self.group.conRuns >= self.group.maxCon and self.group.conCheck:
            print_error('Looper: item {} reached max consecutive group runs'.format(self.idx), 'warning')
            self.loopManager.terminate()
            self.loopManager.currentIndex = 0

    def update_runs(self):
        """called when the runs box is updated"""
        self.totalRuns = self.totalRunsBox.value()
        self.update_loop_file()

    def update_con(self):
        """called when the con box is updated"""
        self.conRuns = self.conRunsBox.value()
        self.update_loop_file()

    def update_runs_box(self):
        self.totalRunsBox.setValue(self.totalRuns)

    def update_con_box(self):
        self.conRunsBox.setValue(self.conRuns)

    def setup_item_dict(self):
        itemDict = {'type': 'Task',
                    'nr': self.nrBox.currentText(),
                    'maxRuns': self.maxRunsBox.value(),
                    'maxRunsCheck': self.maxRunsCheck.isChecked(),
                    'maxCon': self.maxConBox.value(),
                    'maxConCheck': self.maxConCheck.isChecked()}
        return itemDict

    def load_item_dict(self, itemDict, loadedItems):
        find_or_add(self.nrBox, itemDict['nr'])
        self.maxRunsBox.setValue(itemDict['maxRuns'])
        self.maxRunsCheck.setChecked(itemDict['maxRunsCheck'])
        self.maxConBox.setValue(itemDict['maxCon'])
        self.maxConCheck.setChecked(itemDict['maxConCheck'])


class ListItem(TaskItem):
    """
    A ListItem contains a list of tasks from the preplist. When the looper runs a ListItem, it runs a single task
    from the taskList and then increments the currentTaskIndex.
    These tasks can be individual scan runs "unpacked" from a prepared task.
    """
    def __init__(self, parent=None):
        super().__init__(taskNr=0, parent=parent)
        self.itemType = 'List'

    def init_ui(self):
        self.taskList = []
        self.currentTaskIndex = 0
        self.create_widgets()
        self.create_task_widgets()
        self.create_list_widgets()
        self.update_nr_box()
        self.qlayout.addWidget(self.led, 1, self.hidx)

        self.qlayout.addWidget(self.idxLabel, 0, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 3, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(self.listButton, 2, self.hidx)
        self.add_widget(self.nrBox, 'Current:')
        self.qlayout.addWidget(self.expLabel, 2, self.hidx)
        self.add_widget(self.taskLabel, 'Label:')

        self.qlayout.setColumnStretch(self.hidx, 1)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 3, self.hidx)
        self.hidx += 1

        self.add_widget(self.totalRunsBox, 'Run:')
        self.add_widget(self.maxRunsCheck, 'Check?')
        self.add_widget(self.maxRunsBox, 'Max:')

        self.qlayout.addWidget(VLine(), 0, self.hidx, 3, self.hidx)
        self.hidx += 1

        self.add_widget(self.conRunsBox, 'Consecutive:')
        self.add_widget(self.maxConCheck, 'Check?')
        self.add_widget(self.maxConBox, 'Max consecutive:')

        self.qlayout.addWidget(VLine(), 0, self.hidx, 3, self.hidx)
        self.hidx += 1

        self.add_widget(self.delButton, '')

        self.qlayout.addWidget(self.upButton, 0, self.hidx)
        self.qlayout.addWidget(self.downButton, 1, self.hidx)

    def create_list_widgets(self):
        self.listButton = QPushButton("Edit List")
        self.listButton.clicked.connect(self.edit_list)

    def edit_list(self):
        listEdit = ListEdit(browser=self.browser, parent=self)
        listEdit.show()

    def run(self, signals):
        """like TaskItem, but needs to get the current task from its tasklist and increment the currentTaskIndex."""
        if not self.nrBox.currentText():
            print_error('List in loop item {} in group {} is empty'
                        .format(self.looper.loopManager.currentIndex, self.looper.loopManager.currentGroupName), 'error')
            return
        self.currentTaskIndex = int(self.nrBox.currentText())
        task = self.taskList[self.currentTaskIndex]
        self._task.value += 1
        newTaskNr = self._task.value
        task_dict = task.copy()
        experiment = get_experiment(task_dict['filepath'], self.browser)
        # task_dict['experiment'] = experiment
        if not experiment:
            print_error('Looper: experiment missing!', 'error')
            self.terminate()
        signals.addItem.emit(newTaskNr, task_dict)

    def end_run(self):
        if (self.loopManager.lastGroupName == self.group.groupName) and (self.loopManager.lastTaskIndex == self.idx):
            self.conRuns += 1
        else:
            self.conRuns = 0
        if self.currentTaskIndex < len(self.taskList)-1:
            self.currentTaskIndex += 1
        else:
            self.totalRuns += 1
            self.currentTaskIndex = 0

        self.loopManager.lastGroupName = self.group.groupName
        self.loopManager.lastIndex = self.loopManager.currentIndex
        self.loopManager.lastTaskIndex = self.loopManager.currentIndex
        self.loopManager.currentIndex += 1
        self.max_check()

    def task_end(self):
        self.led_off()
        self.update_nr_box()
        self.update_runs_box()
        self.update_con_box()
        self.update_browser_task()

    def update_nr_box(self):
        self.nrBox.clear()
        for i, _ in enumerate(self.taskList):
            self.nrBox.addItem(str(i))
        if self.currentTaskIndex == '':
            return
        else:
            idx = self.nrBox.findText(str(self.currentTaskIndex))
            self.nrBox.setCurrentIndex(idx)
        self.update_task_labels()

    def update_task_labels(self):
        taskLabel = ''
        expLabel = ''
        if self.currentTaskIndex != '':
            nr = int(self.currentTaskIndex)
            if nr < len(self.taskList):
                taskLabel = self.taskList[nr]['label']
                expLabel = self.taskList[nr]['expName']
        self.taskLabel.setText(taskLabel)
        self.expLabel.setText(expLabel)

    def update_task(self):
        nr = self.nrBox.currentText()
        if nr != '' and self.taskList:
            currentTask = int(nr)
            self.currentTask = currentTask
            newTask = self.taskList[currentTask]
            self.task = newTask
            self.update_task_labels()
            self.update_loop_file()

    def setup_item_dict(self):
        itemDict = {'type': 'List',
                    'nr': self.taskNr,
                    'taskList': self.taskList,
                    'maxRuns': self.maxRunsBox.value(),
                    'maxRunsCheck': self.maxRunsCheck.isChecked(),
                    'maxCon': self.maxConBox.value(),
                    'maxConCheck': self.maxConCheck.isChecked()}
        return itemDict

    def load_item_dict(self, itemDict, loadedItems):
        self.taskList = itemDict['taskList']
        self.update_nr_box()
        self.taskNr = itemDict['nr']
        self.maxRunsBox.setValue(itemDict['maxRuns'])
        self.maxRunsCheck.setChecked(itemDict['maxRunsCheck'])
        self.maxConBox.setValue(itemDict['maxCon'])
        self.maxConCheck.setChecked(itemDict['maxConCheck'])
        for d in self.taskList:
            if d['filepath'] not in self.browser.openWindowNames.value:
                self.browser.open_experiment(d['filepath'])


class ConditionalItem(LoopItem):
    """
    Conditional items take some streamed data, a comparitor, and a value, form a conditional statement,
    and set the LoopManager's currentItemIndex based on the outcome.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemType = 'Conditional'
        self.keyList = self.looper.keyList
        self.dataDict = self.looper.dataDict
        self.init_ui()

    def init_ui(self):
        self.create_widgets()
        self.create_cond_widgets()

        self.qlayout.addWidget(self.idxLabel, 0, self.hidx)
        self.qlayout.addWidget(self.led, 1, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(QLabel("If"), 0, 2)
        self.qlayout.addWidget(self.dataBox, 0, 3)
        self.qlayout.addWidget(self.compBox, 0, 4)
        self.qlayout.addWidget(self.valBox, 0, 5)
        self.qlayout.addWidget(self.parsedBox, 0, 6)

        self.qlayout.addWidget(QLabel("go to"), 1, 2)
        self.qlayout.addWidget(self.ifBox, 1, 3)
        self.qlayout.addWidget(QLabel("else"), 1, 4)
        self.qlayout.addWidget(self.elseBox, 1, 5)

        self.hidx = 7

        self.qlayout.setColumnStretch(self.hidx, 1)

        self.hidx += 1
        self.descriptionTextEdit = QTextEdit('')
        self.descriptionTextEdit.setFixedWidth(500)
        self.qlayout.addWidget(self.descriptionTextEdit, 0, self.hidx, 2, 1)

        self.hidx += 1
        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.delButton, '')

        self.qlayout.addWidget(self.upButton, 0, self.hidx)
        self.qlayout.addWidget(self.downButton, 1, self.hidx)

    def run(self, signals):
        self.loopManager.lastIndex = self.loopManager.currentItem

        datatxt = self.dataBox.currentText()
        if datatxt in self.dataDict.keys():
            data = self.dataDict[datatxt]
        else:
            data = 0
            print_error('looper: data {} not found'.format(datatxt), 'error')

        comp = self.compBox.currentText()
        operator = operators[comp]  # get comparison function from operators module

        valTxt = self.valBox.text()

        if valTxt == '':
            valTxt = '0'
        valTxt = parse_text(valTxt, startDateTime=self.loopManager.startDateTime)
        if valTxt[0] == '+':
            try:
                valTxt = str(self.loopManager.startDateTime.toSecsSinceEpoch() + int(valTxt[1:]))
            except:
                valTxt = 0
        if isinstance(data, int):
            val = int(valTxt)
        elif isinstance(data, float):
            val = float(valTxt)
        elif isinstance(data, str):
            val = valTxt
        else:
            val = valTxt

        idxIf = self.ifBox.currentText()
        idxElse = self.elseBox.currentText()

        if operator(data, val):
            nextIndex = idxIf
            self.result = True  # we use this to set the led color, which has to be done outside the worker thread
        else:
            nextIndex = idxElse
            self.result = False

        if nextIndex in ('end loop', 'end'):
            self.loopManager.terminate()
        elif nextIndex == 'end box':
            groupIndex = len(self.group.groupItemList) - 1
            self.loopManager.currentIndex = groupIndex
        else:
            self.loopManager.currentIndex = int(nextIndex)

    def task_end(self):
        if self.result:
            self.led.set_green()
        else:
            self.led.set_red()
        self.led_on()
        QApplication.processEvents()  # update GUI to turn led on NOW
        QTimer.singleShot(50, self.led_off)  # turn led off after 50 ms

    def end_run(self):
        pass

    def update_boxes(self):
        for box in [self.ifBox, self.elseBox]:
            item = box.currentText()
            if not item: item = '0'  # protect from empty strings when creating the item
            box.clear()
            for i, _ in enumerate(self.groupItemList):
                box.addItem(str(i))
            box.addItem('end box')
            box.addItem('end loop')
            find_or_add(box, item)

    def update_runs_box(self):
        pass

    def update_data_box(self):
        dat = self.dataBox.currentText()
        self.dataBox.clear()
        self.keyList = self.parent.keyList
        keyListSorted = sorted(self.keyList.value)
        for i in keyListSorted:
            self.dataBox.addItem(i)
        idx = self.dataBox.findText(dat)
        self.dataBox.setCurrentIndex(idx)

    def create_cond_widgets(self):
        self.dataBox = SearchComboBox()
        for key in sorted(self.keyList.value):
            self.dataBox.addItem(key)
        self.dataBox.currentIndexChanged.connect(self.update_loop_file)
        self.compBox = QComboBox()
        for i in operators.keys():
            self.compBox.addItem(i)
        self.compBox.currentIndexChanged.connect(self.update_loop_file)
        self.valBox = QLineEdit()
        self.valBox.textChanged.connect(self.update_parsed_box)
        self.valBox.setMaximumWidth(200)

        self.parsedBox = QLabel('')

        self.ifBox = QComboBox()
        self.ifBox.currentIndexChanged.connect(self.update_loop_file)
        self.elseBox = QComboBox()
        self.elseBox.currentIndexChanged.connect(self.update_loop_file)
        self.update_boxes()

    def setup_item_dict(self):
        itemDict = {'type': 'Conditional',
                    'data': self.dataBox.currentText(),
                    'comp': self.compBox.currentText(),
                    'val': self.valBox.text(),
                    'if': self.ifBox.currentText(),
                    'else': self.elseBox.currentText(),
                    'description': self.descriptionTextEdit.toPlainText()}
        return itemDict

    def load_item_dict(self, itemDict, loadedItems=None):
        find_or_add(self.dataBox, itemDict['data'])
        find_or_add(self.compBox, itemDict['comp'])
        self.valBox.setText(itemDict['val'])

        for j, _ in enumerate(loadedItems):  # need fully populate combo boxes before the loop is finished loading
            j = str(j)
            if self.ifBox.findText(j) < 0:
                self.ifBox.addItem(j)
                self.elseBox.addItem(j)
        idx = self.ifBox.findText(itemDict['if'])
        self.ifBox.setCurrentIndex(idx)
        idx = self.elseBox.findText(itemDict['else'])
        self.elseBox.setCurrentIndex(idx)
        self.update_boxes()
        self.descriptionTextEdit.setText(itemDict.get('description'))

    def update_parsed_box(self, text):
        parsedText = parse_text(text)
        self.parsedBox.setText(parsedText)


class GroupItem(LoopItem):
    """Tells the loop which group to go to next. All groups have exactly one group item at the end of their lists"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemType = 'Group'
        self.groupName = parent.groupName
        self.idx = 0
        self.init_ui()

    def create_group_widgets(self):
        self.conRunsBox = QSpinBox()
        self.conRunsBox.setMaximum(99999)
        self.maxConCheck = QCheckBox()
        self.maxConBox = QSpinBox()
        self.maxConBox.setMaximum(99999)
        self.conRunsBox.valueChanged.connect(self.update_con_box)
        self.maxConCheck.stateChanged.connect(self.update_con_check)
        self.maxConBox.valueChanged.connect(self.update_con_max)

    def init_ui(self):
        self.create_widgets()
        self.create_group_widgets()
        self.qlayout.addWidget(self.idxLabel, 0, self.hidx)
        self.qlayout.addWidget(self.led, 1, self.hidx)
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1

        self.add_widget(self.conRunsBox, 'Consecutive:')
        self.add_widget(self.maxConCheck, 'Check?')
        self.add_widget(self.maxConBox, 'Max consecutive:')
        self.hidx += 1

        self.qlayout.addWidget(VLine(), 0, self.hidx, 2, self.hidx)
        self.hidx += 1
        self.qlayout.addWidget(QLabel('Go to box:'), 0, self.hidx, 2, self.hidx)
        self.hidx += 15

        self.groupBox = QComboBox()
        self.update_boxes()
        self.groupBox.currentIndexChanged.connect(self.update_loop_file)
        self.qlayout.addWidget(self.groupBox, 0, self.hidx, 2, self.hidx)

    def update_boxes(self):
        text = self.groupBox.currentText()
        if not text: text = 'end loop'  # protect from empty strings when creating the item
        self.groupBox.clear()
        for groupName in self.looper.groupDict.keys():
            self.groupBox.addItem(groupName)
        self.groupBox.addItem('end loop')
        find_or_add(self.groupBox, text)

    def update_con_box(self):
        self.conRunsBox.setValue(self.group.conRuns)

    def update_con_check(self):
        self.group.conCheck = self.maxConCheck.isChecked()

    def update_con_max(self):
        self.group.maxCon = self.maxConBox.value()

    def run(self, signals):
        print(f'{self.groupName = }')
        if self.groupBox.currentText() == 'end loop':
            self.loopManager.terminate()
        else:
            self.loopManager.currentGroupName = self.groupBox.currentText()

            self.loopManager.currentIndex = 0

    def task_end(self):
        text = self.groupBox.currentText()
        if text == 'end loop':
            self.led.set_red()
        else:
            self.led.set_green()
            group = self.looper.groupDict[text]
            idx = self.looper.tabWidget.indexOf(group)
            self.looper.tabWidget.setIconOn(idx)
        self.led_on()
        idx = self.looper.tabWidget.indexOf(self.group)
        self.looper.tabWidget.setIconOff(idx)
        QApplication.processEvents()  # update GUI to turn led on NOW
        QTimer.singleShot(50, self.led_off)  # turn led off after 50 ms

    def setup_item_dict(self):
        itemDict = {'type': 'Group',
                    'nextBox': self.groupBox.currentText(),
                    'name': self.group.groupName,
                    'maxCon': self.group.maxCon,
                    'maxConCheck': self.group.conCheck}
        return itemDict

    def load_item_dict(self, itemDict, loadedItems):
        find_or_add(self.groupBox, itemDict['nextBox'])
        if 'maxConCheck' in itemDict:
            self.maxConCheck.setChecked(itemDict['maxConCheck'])
        if 'maxCon' in itemDict:
            self.maxConBox.setValue(itemDict['maxCon'])
        self.conRunsBox.setValue(0)
        self.update_con_check()
        self.update_con_max()

        name = itemDict.get('name')
        if name == self.group.groupName:
            return
        if name in self.looper.groupDict:
            name += '#1'
        idx = self.looper.tabWidget.indexOf(self.group)
        self.looper.tabWidget.setTabText(idx, name, self.group.groupName)
        self.looper.tabWidget.setTabText(idx, name, self.group.groupName)


class LoopSubMgr(QWidget):
    """GUI for editing datastream subscriptions"""
    def __init__(self, parent = None):
        super().__init__(parent)
        self.qlayout = QVBoxLayout()
        self._props = _props
        self.parent = parent
        self.dataClient = parent.dataClient
        self.editor = SubscriptionEditor(self._props, 'Data', propprefix='')
        self.editor.subscriptionsChanged.connect(self.update_subscriptions)
        self.qlayout.addWidget(self.editor)
        self.setLayout(self.qlayout)

    def update_subscriptions(self):
        self.streamNames = self._props.get('datastreams', [''])
        # we first unsubscribe from the old datastream, then subscribe to the new one
        self.dataClient.unsubscribe()
        self.dataClient.subscribe(self.streamNames)


class ListEdit(QDialog):
    """GUI for editing the taskLsit of a ListItem"""
    def __init__(self, browser=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('ListItem List Editor')
        self.browser = browser
        self.parent = parent
        layout = QHBoxLayout()
        self.listBox = ListBox(browser=self.browser, parent=self)
        self.prepBox = PrepBox(browser=self.browser, parent=self)
        layout.addWidget(self.listBox)
        layout.addWidget(self.prepBox)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.parent.taskList = self.listBox.taskList.copy()
        self.parent.update_nr_box()


class PrepBox(PrepStation):
    """Displays the prepList in a ListEdit window"""
    def __init__(self, browser=None, parent=None):
        super().__init__(browser=browser, parent=parent, title='PrepBox')
        self.main = False
        self.prepList = browser.prepStation.prepList
        self.set_model()
        self.qlayout.setDirection(2)

    def init_ui(self):
        pushButton = QPushButton('Push Selected')
        pushButton.clicked.connect(self.push_selection)
        self.buttonLayout.addWidget(pushButton)

        unpackButton = QPushButton('Unpack Selected')
        unpackButton.clicked.connect(self.unpack_selection)
        self.buttonLayout.addWidget(unpackButton)

        self.qlayout.addLayout(self.buttonLayout)

    def init_table_actions(self):
        push_action = QAction("Add to List", self.table)
        push_action.triggered.connect(partial(self.push_selection))
        push_action.setShortcut("SPACE")
        push_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(push_action)

        view_action = QAction("View", self.table)
        view_action.triggered.connect(self.open_viewer)
        view_action.setShortcut("v")
        view_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(view_action)

    def push_selection(self):
        idx = self.table.selectedIndexes()
        if idx:
            for i in idx:
                row = i.row()
                task = self.prepList[row]
                self.push(task)
        else:
            pass

    def unpack_selection(self):
        idx = self.table.selectedIndexes()
        if idx:
            for i in idx:
                row = i.row()
                task = self.prepList[row]
                if task['nRuns'] > 1:
                    copyTask = copy.deepcopy(task)
                    self.unpack(copyTask)
                else:
                    self.push(task)
        else:
            pass

    def push(self, task):
        if isinstance(task, list):
            self.parent().listBox.taskList.extend(task)  # layoutChanged only emitted once
        else:
            self.parent().listBox.taskList.append(task)
        self.parent().listBox.tableModel.layoutChanged.emit()

    def unpack(self, task):
        nRuns = task['nRuns']
        scanPars = task['scanpars']
        scanVals = np.array(task['scanvals'])
        scanSequence = np.array(task['scansequence'])
        taskList = list(range(nRuns))
        for run in range(nRuns):
            newTask = copy.deepcopy(task)
            for i, parname in enumerate(scanPars):  # modify value(s) for scan
                if parname != '--NONE--':
                    self.set_dict(newTask, parname, scanVals[scanSequence[run], i])
            newTask['nRuns'] = 1
            newTask['scanpars'] = []
            newTask['scanvals'] = []
            newTask['scansequence'] = []
            label = newTask['label']
            newTask['label'] = 'run: ' + str(run) + ' ' + label
            taskList[run] = newTask
        self.push(taskList)

    @staticmethod
    def set_dict(task, parname, value):
        """Sets experiment arguments."""
        task['args'][parname] = value

    def update_prep_file(self):
        """we don't want PrepBox to overwrite the prepfile"""
        pass


class ListBox(PrepStation):
    """Displays the taskList of a ListItem in a ListEdit window"""
    def __init__(self, browser=None, parent=None):
        super().__init__(browser=browser, parent=parent, title='ListBox')
        self.main = False
        self.taskList = parent.parent.taskList.copy()
        self.prepList = self.taskList
        self.set_model()
        self.parent = parent
        self.qlayout.setDirection(2)

    def init_ui(self):
        moveUpButton = QPushButton('Move Up')
        moveUpButton.clicked.connect(self.move_up)
        moveUpButton.setShortcut("PgUp")
        self.buttonLayout.addWidget(moveUpButton)

        moveDownButton = QPushButton('Move Down')
        moveDownButton.clicked.connect(self.move_down)
        moveDownButton.setShortcut("PgDown")
        self.buttonLayout.addWidget(moveDownButton)

        self.qlayout.addLayout(self.buttonLayout)

    def init_table_actions(self):
        view_action = QAction("View", self.table)
        view_action.triggered.connect(self.open_viewer)
        view_action.setShortcut("v")
        view_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(view_action)

        delete_action = QAction("Delete", self.table)
        delete_action.triggered.connect(partial(self.delete_clicked))
        delete_action.setShortcut("SHIFT+DELETE")
        delete_action.setShortcutContext(Qt.WidgetShortcut)
        self.table.addAction(delete_action)

    def set_model(self):
        self.tableModel = SyncedPrepModel(self.taskList)
        self.table.setModel(self.tableModel)
        self.qlayout.addWidget(self.table)

        cw = QtGui.QFontMetrics(self.font()).averageCharWidth()
        h = self.table.horizontalHeader()
        h.resizeSection(0, 7*cw)
        h.resizeSection(1, 40*cw)
        h.resizeSection(2, 20*cw)
        h.resizeSection(3, 12*cw)
        h.resizeSection(4, 7*cw)
        h.resizeSection(5, 7*cw)
        h.resizeSection(6, 8*cw)

    def update_prep_file(self):
        """we don't want ListEdit to overwrite the prepfile"""
        pass


class VLine(QFrame):
    """convenience class for creating separators in the looper UI"""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Raised)


class LoopTabWidget(QTabWidget):
    """Custom TabWidget class"""
    def __init__(self, parent):
        super().__init__()
        self.looper = parent
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        delete_action = QAction("Delete", self)
        delete_action.setShortcutContext(Qt.WidgetShortcut)
        delete_action.triggered.connect(parent.delete_tab)
        self.tabCloseRequested.connect(parent.delete_tab)
        self.addAction(delete_action)
        self.setTabBar(EditableTabBar(self, looper=parent))
        self.addTabButton = QPushButton('+')
        self.addTabButton.clicked.connect(parent.add_tab)
        self.setCornerWidget(self.addTabButton)
        pixmapiOn = QStyle.SP_DialogYesButton
        self.onIcon = self.style().standardIcon(pixmapiOn)
        pixmapiOff = QStyle.SP_DialogNoButton
        self.offIcon = self.style().standardIcon(pixmapiOff)

    def addTab(self, widget, name):
        super().addTab(widget, self.offIcon, name)

    def setIconOn(self, idx):
        self.setTabIcon(idx, self.onIcon)

    def setIconOff(self, idx):
        self.setTabIcon(idx, self.offIcon)

    def setTabText(self, index, newText, oldText=''):
        """sets when tabText is set, also update the groupName and groupDict"""
        super().setTabText(index, newText)
        self.widget(index).groupName = newText
        edit_key_in_place(self.looper.groupDict, oldText, newText)
        for group in self.looper.groupDict.values():
            group.baustelle.groupItem.update_boxes()


class EditableTabBar(QTabBar):
    """Custom TabBar which allows the tab name to be edited. The new tab name is also applied to the
    corresponding loop group"""
    def __init__(self, parent, looper):
        QTabBar.__init__(self, parent)
        self.looper = looper
        self._editor = QLineEdit(self)
        self._editor.setWindowFlags(Qt.Popup)
        self._editor.setFocusProxy(self)
        self._editor.editingFinished.connect(self.handleEditingFinished)
        self._editor.installEventFilter(self)
        self.setTabsClosable(True)
        self.setMovable(True)

    def eventFilter(self, widget, event):
        if ((event.type() == QEvent.MouseButtonPress and not self._editor.geometry().contains(event.globalPos()))
                or (event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape)):
            self._editor.hide()
            return True
        return QTabBar.eventFilter(self, widget, event)

    def mouseDoubleClickEvent(self, event):
        index = self.tabAt(event.pos())
        if index >= 0:
            self.editTab(index)

    def editTab(self, index):
        rect = self.tabRect(index)
        self._editor.setFixedSize(rect.size())
        self._editor.move(self.parent().mapToGlobal(rect.topLeft()))
        self._editor.setText(self.tabText(index))
        if not self._editor.isVisible():
            self._editor.show()

    def handleEditingFinished(self):
        index = self.currentIndex()
        oldText = self.tabText(index)#
        newText = self._editor.text()
        if index >= 0:
            self._editor.hide()
            self.looper.tabWidget.setTabText(index, newText, oldText=oldText)


def get_experiment(filepath, browser):
    """
    gets a bali experiment object from a browser window
    :param filepath: full path to bali experiment
    :param browser: bali browser object
    :return: the experiment object if a window for the experiment is already open, else None
    """
    opened = False
    for window in browser.openWindows:
        if filepath == window.filepath:
            return window.experiment
    if not opened:
        print_error('Looper tried to run an unopened experiment', 'error')
        return None


def edit_key_in_place(d, oldKey, newKey):
    """
    Changes the key of a dictionary in place (does not create a new dictionary) while preserving order

    :param d: dict
    :param oldKey: key to be replaced
    :param newKey: new key
    :return:
    """
    replacement = {oldKey: newKey}
    for k, v in list(d.items()):
        d[replacement.get(k, k)] = d.pop(k)


def find_or_add(box, text):
    """if it's in the box, set it, if not, add and set it"""
    idx = box.findText(text)
    if idx < 0:
        box.addItem(text)
        idx = box.findText(text)
    box.setCurrentIndex(idx)


def parse_text(text, startDateTime=None):
    """
    :param text: format of text is checked to see if it corresponds to a (date)time, and converts appropriately
    accepted formats:
        - +Xh: converts X in hours to seconds, returns '+seconds'
        - +Xm: converts X in minutes to seconds, returns '+seconds'
        - hh:mm: converts to epoch time. if time is before startTime, converts to tomorrow's time
        - yyyy.MM.dd_hh:mm: converts to epoch time
    :param startDateTime: the looper passes the start datetime of the loop to the function
    :return: returns converted text
    """
    if not startDateTime:
        startDateTime = QDateTime.currentDateTime()
    startTime = startDateTime.time()
    futureHPattern = re.compile('^\+[0-9]+h$')
    if futureHPattern.match(text):
        return '+' + str(int(text[1:-1]) * 60 * 60)
    futureMPattern = re.compile('^\+[0-9]+m$')
    if futureMPattern.match(text):
        return '+' + str(int(text[1:-1]) * 60)
    timePattern = re.compile('^[0-9][0-9][:h][0-9][0-9]$')
    if timePattern.match(text):
        checkTime = QTime.fromString(text, "hh:mm")
        checkDateTime = QDateTime()
        checkDateTime.setTime(checkTime)
        if checkTime < startTime:
            checkDateTime.setDate(QDate.currentDate().addDays(1))
        else:
            checkDateTime.setDate(QDate.currentDate())
        return str(checkDateTime.toSecsSinceEpoch())
    datetimePattern = re.compile('^[0-9][0-9][0-9][0-9][.][0-9][0-9][.][0-9][0-9][_][0-9][0-9][:h][0-9][0-9]$')
    if datetimePattern.match(text):
        checkDateTime = QDateTime.fromString(text, "yyyy.MM.dd_hh:mm")
        return str(checkDateTime.toSecsSinceEpoch())
    else:
        return text
