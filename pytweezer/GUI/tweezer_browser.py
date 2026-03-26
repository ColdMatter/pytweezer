import os
import importlib.util
import inspect
import traceback

import numpy as np
import json
from datetime import datetime
import time
import signal

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QGroupBox, QWidget, QPushButton
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QMainWindow, QDialog, QDockWidget
from PyQt5.QtWidgets import QMdiArea, QTreeView, QDirModel, QSpinBox
from PyQt5.QtWidgets import QMdiSubWindow, QLabel, QCompleter
from PyQt5.QtWidgets import QSizePolicy, QComboBox, QGridLayout, QDialogButtonBox
from PyQt5.QtWidgets import QMenu, QCheckBox, QLineEdit, QDoubleSpinBox
from PyQt5.QtWidgets import QFileDialog, QSplitter, QDateTimeEdit
from PyQt5.QtCore import QThreadPool, Qt, QDateTime, QEvent

import pytweezer
from pytweezer.analysis.floating_point_arithmetics import round_floating_prec
from pytweezer.experiment.experiment import Experiment
from pytweezer.servers import (
    Properties,
    tweezerpath,
    icon_path,
    PropertyAttribute,
    DataClient,
)
from pytweezer.servers import send_info, send_error
from pytweezer.GUI.browser.editor_sequencer import CodeEditorParser, CodeEditor
from pytweezer.GUI.arg_boxes import FloatBox, BoolBox, ComboBox
from pytweezer.servers.experiment_manager import ExperimentManager
from pytweezer.GUI.browser.prepstation import PrepStation
from pytweezer.GUI.looper import Looper
from pytweezer.GUI.browser.experiment_queue import ExperimentQ
from pytweezer.GUI.pytweezerQt import SearchComboBox
from pytweezer.analysis.print_messages import print_error
from bin.processmanager import backup_file

from pytweezer.experiment.motmaster_client import MotMasterClient


class BaliBrowser(QMainWindow):
    """
    The Bali Browser is the interface for running all experiments other than the Helpers.
    The File Selector browses the experiments in scripts/pytweezer/experiments.
    The Experiment Manager handles the running of experiments from the Experiment Queue.
    The Experiment Queue has a dictionary of tasks, displayed in a table.
    The Prep Station is for storing and editing tasks before sending them to the queue.
    Looper allows groups of tasks to be run dependent on the outcome of measurments.
    """

    experimentOpened = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.props = Properties("BaliBrowser")
        self._name = "BaliBrowser"
        self._props = self.props
        self._task = PropertyAttribute("/Experiments/_task", 0, parent=self)
        self.paramDir = (
            tweezerpath + "/configuration/tweezer_browser/experiment_params"
        )  # path for storing parameters
        self.motmaster_interface = MotMasterClient()

        self.threadPool = QThreadPool.globalInstance()
        self.threadPool.setMaxThreadCount(3)
        self.fileSelector = BaliFileSelector(self)
        self.queue = ExperimentQ(self)
        self.prepStation = PrepStation(browser=self)
        # self.looper = Looper(parent=self)

        self.init_ui()
        g = self.geometry()
        geo = self.props.get("Geometry", g.getRect())
        self.setGeometry(*geo)
        self.openWindows = []
        self.openWindowNames = PropertyAttribute("OpenWindows", [], parent=self)
        for window in self.openWindowNames.value:
            self.open_experiment(window, startup=True)

        self.prepStation.load_previous()
        self.prepStation.set_model()
        # self.looper.load_previous()

    def closeEvent(self, event):
        # Backup the following two files, since they are often corrupt after browser restart:
        # /home/bali/scripts/pytweezer/configuration/tweezer_browser/loopfile/loopfile.json
        # /home/bali/scripts/pytweezer/configuration/tweezer_browser/prepfile/prepfile.json

        path = tweezerpath + "/configuration/browser/"
        for fname in [
            "loopfile_backups/loopfile.json",
            "prepfile_backups/prepfile.json",
        ]:
            backup_file(path, fname)
        # TODO: Auto-load last backup when crashed while restart
        # TODO: call close event at restart

        super().closeEvent(event)

    def init_ui(self):
        self.statusBar().showMessage("Ready")
        self.setGeometry(100, 100, 1050, 550)
        self.setWindowTitle("pytweezer Browser")
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.create_dock_widgets()
        self.show()

    def create_dock_widgets(self):
        fileselectorDock = QDockWidget("File Selector", self)
        fileselectorDock.setWidget(self.fileSelector)
        fileselectorDock.show()
        self.addDockWidget(Qt.LeftDockWidgetArea, fileselectorDock)
        self.qDock = QDock(parent=self)
        self.queue.setMinimumSize(QtCore.QSize(800, 200))
        self.prepStation.setMinimumSize(QtCore.QSize(800, 200))
        self.addDockWidget(Qt.TopDockWidgetArea, self.qDock)

    def open_experiment(self, filepath, startup=False):
        """
        Opens a new experiment window.
        Does nothing if a window already exists for that experiment.
        """
        path, ext, name, filename = filepath_split(filepath)
        if ext != ".py":
            # print_error('Browser attempted to open a non-.py file', 'error')
            return
        for window in self.openWindows:
            if filepath == window.filepath:
                print_error(
                    "Browser attempted to open an already-open experiment", "warning"
                )
                window.subWindow.close()
        sub = ExperimentSubWindow(name, self.props, parent=self)
        exwin = ExperimentWindow(filepath, self.props, parent=sub, browser=self)
        sub.setWidget(exwin)
        self.mdi.addSubWindow(sub)
        p = sub.geometry()
        geo = self.props.get(exwin.name + "/Geo", p.getRect())
        if geo is not None:
            sub.setGeometry(*geo)
        sub.show()

        self.experimentOpened.emit()
        self.openWindows.append(exwin)
        if not startup:
            openWindowNames = self.openWindowNames.value
            openWindowNames.append(filepath)
            self.openWindowNames.value = openWindowNames
        return exwin


class ExperimentSubWindow(QMdiSubWindow):
    """
    This is a graphical container for an experiment window widget.
    Its only job is setting the title and storing the position of the window in the browser.
    """

    def __init__(self, name, props, parent=None):
        super().__init__(parent)
        self.name = name
        self.setWindowTitle(name)
        self.props = props
        self.browser = parent
        rect = self.props.get(self.name + "/Geo")
        if rect:
            self.geometry().setRect(*rect)
        self.geo = self.geometry()
        if self.props.get(self.name + "/minimized", False):
            self.showMinimized()

    def store_geometry(self):
        self.geo = self.geometry()
        self.props.set(self.name + "/Geo", self.geo.getRect())
        self.props.set(self.name + "/minimized", self.isMinimized())
        mainGeo = self.browser.geometry()
        self.props.set("Geometry", mainGeo.getRect())

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self.store_geometry()
        super().changeEvent(event)


class ExperimentWindow(QWidget):
    """Control a single experiment including sequences.
        Parameter sets may be saved to and loaded from json files. A new folder
        is made for each experiment each day (when saving for the first time).
        The file names are the current time and the "label" from the experiment
        window

    Args:
        filepath: full file path of the experiment
        props:(Properties)
        parent: the subwindow
        browser: bali browser
        hand over property class in order not to create too many instances
    """

    def __init__(self, filepath, props, parent=None, browser=None):
        super().__init__(parent)
        self.filepath = filepath
        self.path, ext, self.name, self.filename = filepath_split(filepath)
        self.subWindow = parent
        self.browser = browser
        self.props = props
        self._props = self.props
        self._task = self.browser._task
        self.queue = self.browser.queue
        self.prepStation = self.browser.prepStation
        self.paramDir = self.browser.paramDir
        self.style_sheets = None
        from pytweezer.experiment.experiment import get_experiment

        try:
            experiment_cls = get_experiment(filepath, self.name)
            experiment = experiment_cls(self.props, self.browser.motmaster_interface)
            experiment.build()
            self.experiment: Experiment = experiment
            self.init_ui()
        except Exception as e:
            print(e)
            print_error("error opening the experiment", "error")
            layout = QVBoxLayout()
            print(e.__dict__)
            layout.addWidget(QLabel("Syntax Error:  "))
            layout.addWidget(QLabel(str(e)))
            layout.addWidget(QLabel(str("".join(traceback.format_tb(e.__traceback__)))))
            layout.addWidget(QLabel("Don't worry everyone makes mistakes ;-) "))
            self.setLayout(layout)

    def init_ui(self):
        layout = QVBoxLayout()

        # first row: submission settings
        submitButton = QPushButton("")
        submitButton.setMaximumWidth(30)
        submitButton.setIcon(QtGui.QIcon(icon_path + "run.svg"))
        submitButton.clicked.connect(self.submit_to_queue)
        self.submitButton = submitButton

        runNextButton = QPushButton("Run Next")
        runNextButton.clicked.connect(self.submit_next)
        self.runNextButton = runNextButton

        prepButton = QPushButton("Prep")
        prepButton.clicked.connect(self.submit_to_prepper)
        self.prepButton = prepButton

        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.save_params)
        self.saveButton = saveButton

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.load_params)
        self.loadButton = loadButton

        defaultButton = QPushButton("Define default")
        defaultButton.clicked.connect(self.make_default)
        self.defaultButton = defaultButton

        resetButton = QPushButton("Reset")
        resetButton.clicked.connect(self.reset)
        self.resetButton = resetButton

        self.prioQSB = QSpinBox()
        self.prioQSB.setToolTip("Priority")
        self.prioQSB.setRange(0, 999)

        self.scheduleBox = QCheckBox(self)
        self.dateTimeEdit = QDateTimeEdit(self)
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())

        hlayout = QHBoxLayout()
        hlayout.addWidget(submitButton)
        hlayout.addWidget(runNextButton)
        hlayout.addWidget(prepButton)
        hlayout.addWidget(saveButton)
        hlayout.addWidget(loadButton)
        hlayout.addWidget(defaultButton)
        hlayout.addWidget(resetButton)
        hlayout.addStretch()
        hlayout.addWidget(QLabel("task:"))
        hlayout.addStretch()
        hlayout.addWidget(QLabel("prio:"))
        hlayout.addWidget(self.prioQSB)
        hlayout.addStretch()
        hlayout.addWidget(self.scheduleBox)
        hlayout.addWidget(self.dateTimeEdit)
        layout.addLayout(hlayout)

        # second row: scan and repetition info
        hlayout = QHBoxLayout()

        hlayout.addWidget(QLabel("label:"))
        self.taskLabel = QLineEdit()
        hlayout.addWidget(self.taskLabel)

        hlayout.addWidget(QLabel("repeat:"))
        self.nRepsQSB = QSpinBox()
        self.nRepsQSB.setRange(0, 100000)
        hlayout.addWidget(self.nRepsQSB)
        self.nRepsQSB.valueChanged.connect(self.check_defaults)

        label = QLabel("Scan:")
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        hlayout.addWidget(label)
        self.scanCombo = SearchComboBox(self)
        self.scanCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.scanCombo.addItem("--NONE--")
        self.scanCombo.currentTextChanged.connect(self.check_defaults)

        scanNames = sorted(
            list(self.experiment._props.get("scans", {}).keys()), key=str.casefold
        )
        for scan in scanNames:
            self.scanCombo.addItem(scan)

        hlayout.addWidget(self.scanCombo)
        editButton = QPushButton("Edit")
        editButton.clicked.connect(self.edit_sequence)
        hlayout.addWidget(editButton)
        layout.addLayout(hlayout)

        # other rows: items defined in Experiment
        arg_layout = QGridLayout()
        ncol = self.gui_columns()
        self.argument_boxes = []

        # building the argument view
        for i, argument in enumerate(self.experiment._arguments):
            if type(argument) == pytweezer.experiment.experiment.NumberValue:
                box = FloatBox(self.experiment._props, **argument.__dict__.copy())
                box.spin.valueChanged.connect(self.check_defaults)
            elif type(argument) == pytweezer.experiment.experiment.BoolValue:
                box = BoolBox(self.experiment._props, value=True, parName=argument.name)
                box.stateChanged.connect(self.check_defaults)
            elif type(argument) == pytweezer.experiment.experiment.StringCombo:
                box = ComboBox(
                    self.experiment._props,
                    parName=argument.name,
                    **argument.__dict__.copy()
                )
                box.spin.activated.connect(self.check_defaults)
            else:
                print_error("tweezer_browser error argument type not known", "error")
                box = None
            if box:
                self.argument_boxes.append(box)
            arg_layout.addWidget(box, int(i / ncol), i % ncol)
        layout.addLayout(arg_layout)

        mm_param_group = QGroupBox("MM Params")
        mm_param_layout = QGridLayout()
        mm_param_group.setLayout(mm_param_layout)
        self.mm_boxes = []
        for i, argument in enumerate(self.experiment.mm_params):
            if type(argument) == pytweezer.experiment.experiment.NumberValue:
                box = FloatBox(self.experiment._props, **argument.__dict__.copy())
                box.spin.valueChanged.connect(self.check_defaults)
            elif type(argument) == pytweezer.experiment.experiment.BoolValue:
                box = BoolBox(self.experiment._props, value=True, parName=argument.name)
                box.stateChanged.connect(self.check_defaults)
            elif type(argument) == pytweezer.experiment.experiment.StringCombo:
                box = ComboBox(
                    self.experiment._props,
                    parName=argument.name,
                    **argument.__dict__.copy()
                )
                box.spin.activated.connect(self.check_defaults)
            else:
                print_error("tweezer_browser error argument type not known", "error")
                box = None
            if box:
                self.mm_boxes.append(box)
            mm_param_layout.addWidget(box, int(i / ncol), i % ncol)
        layout.addWidget(mm_param_group)

        self.setLayout(layout)
        self.check_defaults(None)

    def rounded_box_val(self, box, use_ndecimals=True):
        if type(box) != FloatBox:
            return box.value
        val = round_floating_prec(box.value)
        if (
            use_ndecimals
            and "ndecimals" in box.__dict__
            and "display_multiplier" in box.__dict__
        ):
            # print('rounded_box_val', val, box.value, box.__dict__['ndecimals'], int(np.log10(box.__dict__['display_multiplier'])), np.round(val, box.__dict__['ndecimals'] - int(np.log10(box.__dict__['display_multiplier']))))
            val = np.round(
                val,
                box.__dict__["ndecimals"]
                - int(np.log10(box.__dict__["display_multiplier"])),
            )
            # print('rounded_box_val', val)
        return val

    def make_default(self):
        self.save_params(defaults=True)
        self.check_defaults(None)

    def reset(self):
        filepath = self.paramDir + "/" + self.name + "/defaults.json"
        if not os.path.isfile(filepath):
            print_error(
                "tweezer_browser.py - reset(): Save defaults before restoring defaults!",
                "warning",
            )
            return
        self.load_params(filepath=filepath)
        self.check_defaults(None)
        self.scanCombo.setCurrentText("--NONE--")
        self.nRepsQSB.setValue(0)

    def check_defaults(self, value, parambox=None):
        if parambox:
            parambox.updateValue(
                value
                / (
                    parambox.__dict__["display_multiplier"]
                    if "display_multiplier" in parambox.__dict__
                    else 1
                )
            )
        if not self.style_sheets:
            self.style_sheets = {}
            for box in self.argument_boxes:
                self.style_sheets[box.name] = box.styleSheet()
            self.style_sheets["nRepsQSB"] = self.nRepsQSB.styleSheet()
            self.style_sheets["scanCombo"] = self.scanCombo.styleSheet()

        filepath = self.paramDir + "/" + self.name + "/defaults.json"
        if not os.path.isfile(filepath):
            return
        with open(filepath, "r") as json_file:
            loaded_params = json.load(json_file)

        for box in self.argument_boxes:
            if box.parName in loaded_params and loaded_params[
                box.parName
            ] != self.rounded_box_val(box):
                if type(box) == FloatBox:
                    style_sheet = self.style_sheets[box.name]
                    for entry in style_sheet.split(";"):
                        if "background-color" in entry:
                            style_sheet = style_sheet.replace(
                                entry, entry.split(":")[0] + ": rgb(240,230,140)"
                            )
                            break
                    box.setStyleSheet(style_sheet)
                elif type(box) == BoolBox:
                    box.setStyleSheet("BoolBox {background-color: rgb(240,230,140);}")
            else:
                box.setStyleSheet(self.style_sheets[box.name])

        if self.nRepsQSB.value() == 0:
            self.nRepsQSB.setStyleSheet(self.style_sheets["nRepsQSB"])
        else:
            self.nRepsQSB.setStyleSheet(
                "QSpinBox {background-color: rgb(240,230,140);}"
            )

        if self.scanCombo.currentText() == "--NONE--":
            self.scanCombo.setStyleSheet(self.style_sheets["scanCombo"])
        else:
            self.scanCombo.setStyleSheet(
                "SearchComboBox {background-color: rgb(240,230,140);}"
            )

    def save_params(self, backup=False, defaults=False):
        """
        creates a dictionary of experiments arguments and values and dumps to a json file.
        File name has format HHhMM_label. A new directory is made for each new date and within that for each experiment.
        The function is called with backup=True whenever params are loaded.
        """
        # setting up directory, getting current time

        dateDir = datetime.today().strftime("%Y_%m_%d")
        paramTime = datetime.today().strftime("%Hh%M")
        fullDir = self.paramDir + "/" + self.name + "/" + dateDir
        os.makedirs(fullDir, exist_ok=True)
        # write the parameters to a dictionary
        saved_params = {}
        for box in self.argument_boxes:
            saved_params[box.parName] = self.rounded_box_val(box)

        # write file
        if not backup:  # we make a backup every time we load
            if self.taskLabel.text():
                label = self.taskLabel.text()
            else:
                label = "unlabelled"
            saved_params["label"] = label
            if not defaults:
                with open(
                    fullDir + "/" + paramTime + "_" + label + ".json", "w"
                ) as outfile:
                    json.dump(saved_params, outfile, indent=4)
            else:
                with open(
                    self.paramDir + "/" + self.name + "/defaults.json", "w"
                ) as outfile:
                    json.dump(saved_params, outfile, indent=4)
        else:
            saved_params["label"] = "backup"
            with open(fullDir + "/backup.json", "w") as outfile:
                json.dump(saved_params, outfile, indent=4)

    def load_params(self, filepath=None):
        """loads experiments parameter from the file created by save_params"""
        if not filepath:
            filepath = self.get_files()  # open file window
        if not filepath:  # protects from closing without selecting file
            return
        with open(filepath) as json_file:
            loaded_params = json.load(json_file)
        self.taskLabel.setText(loaded_params["label"])
        self.save_params(backup=True)  # create backup
        for box in self.argument_boxes:
            if box.parName in loaded_params:
                val = loaded_params[box.parName]
            else:
                val = box.value
                print_error(
                    "tweezer_browser.py - load_params(): Param {0} not in save file, setting to default value {1}.".format(
                        box.parName, box.value
                    ),
                    "warning",
                )
            if isinstance(val, bool):
                box.updateValue(val)
            else:
                box.updateValue(
                    val
                    / (
                        box.__dict__["display_multiplier"]
                        if "display_multiplier" in box.__dict__
                        else 1
                    )
                )
            # print('load_params val, rounded 1', val, box.value, self.rounded_box_val(box))
            box.updateValue(
                self.rounded_box_val(box)
                / (
                    box.__dict__["display_multiplier"]
                    if "display_multiplier" in box.__dict__
                    else 1
                )
            )
            # print('load_params val, rounded 2', val, self.rounded_box_val(box))
        self.check_defaults(None)

    def get_files(self):
        filepath = QFileDialog.getOpenFileName(
            self,
            "Select Parameter File",
            self.paramDir + "/" + self.name,
            filter="Param Files (*.json)",
        )[0]
        return filepath

    def submit_to_queue(self):
        """submits the experiment to the experiment queue"""
        self.store_geometry()
        self.setup_scan()
        self._task.value += 1  # #increase global task number
        task, task_dict = self.setup_task_dict()
        # task_dict['experiment'] = self.experiment
        print_error(
            "\ntweezer_browser.py - submit_to_queue(): Submitted task dict:\n{0}\n".format(
                task_dict
            ),
            "weak",
        )
        self.queue.tableModel[task] = task_dict
        self.check_defaults(None)

    def submit_next(self):
        """submits the experiment to the experiment queue with highest priority"""
        self.submit_to_queue()
        self.queue.tableModel[self._task.value][
            "priority"
        ] = 1000  # 1000 is the biggest number I could think of
        self.check_defaults(None)

    def submit_to_prepper(self):
        """submits the experiment to the prep station"""
        # self.browser.looper.startup = True
        self.store_geometry()
        self.setup_scan()
        task, task_dict = self.setup_task_dict()
        print_error(
            "\ntweezer_browser.py - submit_to_prepper(): Submitted task dict:\n{0}\n".format(
                task_dict
            ),
            "weak",
        )
        # Use tableModel.append() to sync through RPC instead of prepList.append()
        self.prepStation.tableModel.append(task_dict)
        self.prepStation.update_prep_file()
        # for group in self.browser.looper.groupDict.values():
        #     for item in group.groupItemList:
        #         item.update_nr_box()
        # self.browser.looper.startup = False
        # self.browser.looper.update_loop_file()
        self.check_defaults(None)

    def setup_scan(self):
        """build scan sequence"""
        self.scan = self.scanCombo.currentText()
        if self.scan != "--NONE--":
            self.scan_settings = self.experiment._props.get("scans/" + self.scan, {})

            # generate numpy array containing scan values for example
            # np.array([[0,0],[1,10],[2,20]]) for two parameters scanned
            if "listgenerators" in self.scan_settings:
                generators = self.scan_settings.get("listgenerators", [""])
                scan_values = []
                self.scanPars = self.scan_settings.get("parameters", [])
                for i, gen in enumerate(generators):
                    # print(self.scanPars[i])
                    if self.scanPars[i] != "--NONE--":
                        scan_values.append(eval(gen))
                    else:
                        scan_values.append(np.zeros_like(eval(generators[0])))
                scan_values = np.vstack(scan_values).T
            else:
                scan_values = eval(self.scan_settings.get("listgenerator", ""))
                if scan_values.ndim == 1:
                    scan_values = scan_values[:, np.newaxis]
            send_info(
                "bali Experiment scanning: {} {}".format(self.scanPars, scan_values)
            )

            self.n_runs = scan_values.shape[0]
            self.scan_values = scan_values.tolist()
            self.scan_values = round_floating_prec(
                self.scan_values
            )  # TODO: use n_digits
            self.scan_sequence = np.arange(self.n_runs)
            self.scan_sequence = self.scan_sequence.tolist()
            if self.scan_settings.get("Randomize", True):
                np.random.shuffle(self.scan_sequence)
        else:
            self.n_runs = 1
            self.scanPars = []
            self.scan_values = []
            self.scan_sequence = []
        print_error(
            "tweezer_browser.py - setup_scan: Scan values are {0}".format(
                self.scan_values
            ),
            "weak",
        )

    def setup_task_dict(self):
        """creates task dictionary to be stored in the experimentQ or prepstation"""
        task = self._task.value
        task_dict = {}
        task_dict["task"] = task
        lastSet = self.experiment._props.get("last_set", "")
        task_dict["label"] = lastSet
        if self.taskLabel.text():
            task_dict["label"] = self.taskLabel.text() + " " + task_dict["label"]
        if self.scanCombo.currentText() != "--NONE--":
            task_dict["label"] = "scan: {} {}".format(
                self.scanCombo.currentText(), task_dict["label"]
            )
        task_dict["expName"] = self.name
        task_dict["run"] = 0
        task_dict["priority"] = self.prioQSB.value()
        task_dict["args"] = self.build_argument_dict()
        task_dict["nRuns"] = self.n_runs
        task_dict["scanpars"] = self.scanPars
        task_dict["scanvals"] = self.scan_values
        task_dict["nReps"] = self.nRepsQSB.value()
        task_dict["scansequence"] = self.scan_sequence
        task_dict["repetition"] = 0
        task_dict["terminated"] = False
        task_dict["filepath"] = self.filepath
        if self.scheduleBox.isChecked():
            task_dict["status"] = "Waiting"
            task_dict["dueDateTime"] = self.dateTimeEdit.dateTime().toString()
        else:
            task_dict["status"] = "Queued"
            task_dict["dueDateTime"] = QDateTime.currentDateTime().toString()
        return task, task_dict

    def build_argument_dict(self):
        """creates a dictionary of experiment parameters from an experiment window"""
        argDict = {}
        for arg in self.argument_boxes:
            argDict[arg.name] = self.rounded_box_val(arg)
            if argDict[arg.name] == 0 and arg.value != 0:
                print_error(
                    "tweezer_browser.py - build_argument_dict(): Error while rounding!",
                    "error",
                )
                print(arg.name, arg.value, arg.__dict__["ndecimals"])
        return argDict

    def edit_sequence(self):
        """opens a popup window for editing experiment sequences, then applies the sequence on closing"""
        d = SequenceEditor(self)
        d.exec_()
        currentsequence = self.scanCombo.currentText()
        self.scanCombo.clear()
        self.scanCombo.addItem("--NONE--")
        scanNames = sorted(
            list(self.experiment._props.get("scans", {}).keys()), key=str.casefold
        )
        for scan in scanNames:
            self.scanCombo.addItem(scan)
        self.scanCombo.setCurrentText(currentsequence)

    def closeEvent(self, event):
        """when an exp window is closed, removes the window from the open window list"""
        openWindowNames = self.browser.openWindowNames.value
        openWindowNames.remove(self.filepath)
        self.browser.openWindowNames.value = openWindowNames
        self.browser.openWindows.remove(self)
        self.subWindow.close()

    def gui_columns(self):
        return self.experiment._gui_columns

    def arguments(self):
        return self.experiment._arguments

    def argument_names(self):
        return [arg.name for arg in self.experiment._arguments]

    def argument_dicts(self):
        """returns a list containing copies of the argument dictionaries"""
        arg_dict = [arg.__dict__.copy() for arg in self.experiment._arguments]
        mm_dict = [arg.__dict__.copy() for arg in self.experiment.mm_params]
        return arg_dict + mm_dict

    def store_geometry(self):
        self.subWindow.store_geometry()


class SequenceEditor(QDialog):
    """
    docstring
    """

    def __init__(self, experiment_window: ExperimentWindow):
        super().__init__(experiment_window)
        self.exp_window = experiment_window
        self.setWindowTitle("Sequence editor")
        self.qlayout = QVBoxLayout()
        self.qlayout.addWidget(QLabel("Sequence Name:"))
        self.nameEdit = QLineEdit(self.exp_window.scanCombo.currentText())
        self.qlayout.addWidget(self.nameEdit)

        self.number_scans = 3
        snake_scan = False
        if self.nameEdit.text() != "--NONE--":
            defaults = self.exp_window.experiment._props.get(
                "scans/" + self.nameEdit.text(), {}
            )
            self.number_scans = int(defaults.get("number_scans", 3))
            snake_scan = defaults.get("snake_scan", False)

        self.startSpins = []
        self.stopSpins = []
        self.stepsSpins = []
        self.parameterCombos = []
        self.listGeneratorEdits = []

        btn_layout = QHBoxLayout()
        self.randomize = QCheckBox("Randomize")
        btn_layout.addWidget(self.randomize)

        self.tickbox_snake_scan = QCheckBox("Snake-like scan")
        self.tickbox_snake_scan.setChecked(snake_scan)
        btn_layout.addWidget(self.tickbox_snake_scan)
        self.tickbox_snake_scan.stateChanged.connect(self.update_list_generator)

        self.n_dim = QCheckBox("{0}D scan".format(self.number_scans))
        btn_layout.addWidget(self.n_dim)
        btn_layout.addWidget(QLabel("Dimensions:"))
        self.n_dim_box = QDoubleSpinBox()
        self.n_dim_box.setDecimals(0)
        self.n_dim_box.setRange(1, 10)
        self.n_dim_box.setSingleStep(1)

        btn_layout.addWidget(self.n_dim_box)

        self.qlayout.addLayout(btn_layout)

        self.glayout = QGridLayout()
        self.glayout.setSpacing(0)
        self.glayout.addWidget(QLabel("Parameter"), 2, 1)
        self.glayout.addWidget(QLabel("Start"), 2, 2)
        self.glayout.addWidget(QLabel("End"), 2, 3)
        self.glayout.addWidget(QLabel("Steps"), 2, 4)

        if self.nameEdit.text() != "--NONE--":
            self.randomize.setChecked(defaults.get("Randomize", True))
            self.n_dim.setChecked(defaults.get("nDscan", False))
        self.n_dim.toggled.connect(self.update_list_generator)

        self.init_scans()
        self.n_dim_box.setValue(self.number_scans)
        self.n_dim_box.valueChanged.connect(self.update_ndim)

        dbb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dbb.accepted.connect(self.update_values)
        dbb.rejected.connect(self.close)
        self.qlayout.addWidget(dbb)
        self.setLayout(self.qlayout)

    def init_scans(self):
        defaults = {}
        params = []
        starts = []
        ends = []
        stepsl = []
        listgenerators = []
        if self.nameEdit.text() != "--NONE--":
            defaults = self.exp_window.experiment._props.get(
                "scans/" + self.nameEdit.text(), {}
            )
            params = defaults.get("parameters", ["--NONE--"])
            starts = defaults.get("starts", [0])
            ends = defaults.get("ends", [0])
            stepsl = defaults.get("stepsl", [0])
            listgenerators = defaults.get("listgenerators", [""])

        for i in range(len(self.parameterCombos), self.number_scans):
            self.parameterCombos.append(QComboBox())
            self.parameterCombos[-1].addItem("--NONE--")
            paramNames = [
                box.parName
                for box in self.exp_window.argument_boxes + self.exp_window.mm_boxes
            ]
            for name in sorted(paramNames, key=str.casefold):
                self.parameterCombos[-1].addItem(name)
            self.startSpins.append(QDoubleSpinBox())
            self.startSpins[-1].multiplier = 1
            self.stopSpins.append(QDoubleSpinBox())
            self.stopSpins[-1].multiplier = 1
            stepsSpinBox = QSpinBox()
            stepsSpinBox.setMaximum(int(10e3))
            self.stepsSpins.append(stepsSpinBox)
            self.listGeneratorEdits.append(QLineEdit(""))

            self.glayout.addWidget(self.parameterCombos[i], 3 + i * 2, 1)
            self.glayout.addWidget(self.startSpins[i], 3 + i * 2, 2)
            self.glayout.addWidget(self.stopSpins[i], 3 + 2 * i, 3)
            self.glayout.addWidget(self.stepsSpins[i], 3 + 2 * i, 4)
            self.glayout.addWidget(self.listGeneratorEdits[i], 4 + 2 * i, 0, 1, 5)

            self.parameterCombos[i].currentTextChanged.connect(
                lambda text: self.update_parameter(text, i)
            )
            self.parameterCombos[i].setCurrentText(
                params[i] if i < len(params) else "--NONE--"
            )
            self.startSpins[i].setValue(
                (starts[i] if i < len(starts) else 0) / self.startSpins[i].multiplier
            )
            self.stopSpins[i].setValue(
                (ends[i] if i < len(ends) else 0) / self.stopSpins[i].multiplier
            )
            self.stepsSpins[i].setValue(stepsl[i] if i < len(stepsl) else 0)
            self.listGeneratorEdits[i].setText(
                listgenerators[i] if i < len(listgenerators) else ""
            )
            self.startSpins[i].valueChanged.connect(self.update_list_generator)
            self.stopSpins[i].valueChanged.connect(self.update_list_generator)
            self.stepsSpins[i].valueChanged.connect(self.update_list_generator)

        self.qlayout.addLayout(
            self.glayout
        )  # TODO: this is needed but somehow a reduced version of it
        self.setLayout(self.qlayout)

    def update_ndim(self):
        n_old = self.number_scans
        self.number_scans = int(self.n_dim_box.value())
        if n_old > self.number_scans:
            for enum in range(self.number_scans, n_old):
                self.glayout.removeWidget(self.parameterCombos[enum])
                self.glayout.removeWidget(self.startSpins[enum])
                self.glayout.removeWidget(self.stopSpins[enum])
                self.glayout.removeWidget(self.stepsSpins[enum])
                self.glayout.removeWidget(self.listGeneratorEdits[enum])
            self.parameterCombos = self.parameterCombos[: self.number_scans]
            self.startSpins = self.startSpins[: self.number_scans]
            self.stopSpins = self.stopSpins[: self.number_scans]
            self.stepsSpins = self.stepsSpins[: self.number_scans]
            self.listGeneratorEdits = self.listGeneratorEdits[: self.number_scans]
        self.n_dim.setText("{0}D scan".format(self.number_scans))
        self.init_scans()
        self.update_list_generator()

    def update_parameter(self, parname, n):
        """sets range and steps of the scanned parameter according to the parameters settings"""
        print("_updateParameter {}: {}".format(n, parname))
        i = n
        if parname == "--NONE--":
            return
        for arg in self.exp_window.argument_dicts():
            if arg["name"] == parname:
                argument = arg
                break
        else:
            send_error("tweezer_browser coding error")
            return
        if i >= len(self.parameterCombos):
            print_error(
                "SequenceEditor - update_parameter(): Trying to update box {0} in box array of length {1}.".format(
                    i, len(self.parameterCombos)
                ),
                "error",
            )
            return
        parname = self.parameterCombos[i].currentText()
        is_bool = type(argument["value"]) == bool
        for box in [self.startSpins[n], self.stopSpins[i]]:
            print_error(
                "tweezer_browser.py - update_parameter(): Argument dict for {0}: {1}".format(
                    parname, argument
                ),
                "weak",
            )

            if "step" in argument:
                box.setSingleStep(argument["step"])
            elif is_bool:
                box.setSingleStep(1)
            else:
                box.setSingleStep(1)
                print_error(
                    "tweezer_browser.py - update_parameter(): Set step=1 for box {0}.".format(
                        parname
                    ),
                    "weak",
                )

            if "ndecimals" in argument:
                box.setDecimals(argument["ndecimals"])
            elif is_bool:
                box.setDecimals(0)
            else:
                box.setDecimals(0)
                print_error(
                    "tweezer_browser.py - update_parameter(): Set decimals=0 for box {0}.".format(
                        parname
                    ),
                    "weak",
                )

            if "unit" in argument:
                box.setSuffix(argument["unit"])
            elif is_bool:
                box.setSuffix(" (bool)")
            else:
                box.setSuffix("")

            if "minval" in argument and "maxval" in argument:
                box.setRange(argument["minval"], argument["maxval"])
            elif is_bool:
                box.setRange(0, 1)
            else:
                box.setRange(0, 1)
                print_error(
                    "tweezer_browser.py - update_parameter(): Set range=(0, 1) for box {0}.".format(
                        parname
                    ),
                    "weak",
                )

            if "display_multiplier" in argument:
                box.multiplier = argument["display_multiplier"]
            elif is_bool:
                box.multiplier = 1
            else:
                box.multiplier = 1
                print_error(
                    "tweezer_browser.py - update_parameter(): Set multiplier=1 for box {0}.".format(
                        parname
                    ),
                    "weak",
                )

            val = argument["value"]
            box.setValue(val / box.multiplier)

    def round_n_dec(self, value, n_dec):
        value = round_floating_prec(value)
        value = np.round(value, n_dec)
        return value

    def update_list_generator(self):
        starts = [
            self.round_n_dec(
                self.startSpins[i].value() * self.startSpins[i].multiplier,
                self.startSpins[i].decimals()
                - int(np.log10(self.startSpins[i].multiplier)),
            )
            for i in range(self.number_scans)
        ]
        stops = [
            self.round_n_dec(
                self.stopSpins[i].value() * self.stopSpins[i].multiplier,
                self.stopSpins[i].decimals()
                - int(np.log10(self.stopSpins[i].multiplier)),
            )
            for i in range(self.number_scans)
        ]
        steps = [self.stepsSpins[i].value() for i in range(self.number_scans)]

        for k in range(self.number_scans):
            axis = k
            if not self.n_dim.isChecked():
                start = starts[k]
                stop = stops[k]
                step = steps[0]
                # generator = 'np.linspace('+repr(start)+','+repr(stop)+','+repr(steps)+')'
                generator = "np.linspace({},{},{})".format(start, stop, step)
            else:
                lens = int(np.prod(steps[k + 1 :]))
                reps = int(np.prod(steps[:k]))
                # [x1, x1, x2, x2, x1, x1, x2, x2, x1, x1, x2, x2] has lens=2 and reps=3
                # Create sequence like [x1, x1, x2, x2]:
                arr = np.repeat(
                    np.linspace(starts[k], stops[k], steps[k]).reshape(steps[k], 1),
                    lens,
                    axis=1,
                ).flatten()
                # Create sequence like [x1, x1, x2, x2, x1, x1, x2, x2, x1, x1, x2, x2]:
                # arr = np.repeat(arr.reshape(1, len(arr)), reps, axis=0).flatten()

                if self.tickbox_snake_scan.isChecked():
                    generator = (
                        "np.array([np.flip(rep) if enum % 2 == 0 else rep for enum, rep in enumerate(np.repeat("
                        "np.repeat(np.linspace({0}, {1}, {2}).reshape({2}, 1), {3},"
                        " axis=1).flatten().reshape(1, {5}), {4}, axis=0))]).flatten()"
                    ).format(starts[k], stops[k], steps[k], lens, reps, len(arr))
                else:
                    generator = (
                        "np.repeat(np.repeat(np.linspace({0}, {1}, {2}).reshape({2}, 1), {3}, axis=1).flatten()"
                        ".reshape(1, {5}), {4}, axis=0).flatten()"
                    ).format(starts[k], stops[k], steps[k], lens, reps, len(arr))
            self.listGeneratorEdits[k].setText(generator)

    def update_values(self):
        parnames = [
            self.parameterCombos[i].currentText() for i in range(self.number_scans)
        ]
        if self.nameEdit.text() != "--NONE--":
            scannedvalues = {
                "parameters": parnames,
                "starts": [
                    self.startSpins[i].value() * self.startSpins[i].multiplier
                    for i in range(self.number_scans)
                ],
                "ends": [
                    self.stopSpins[i].value() * self.stopSpins[i].multiplier
                    for i in range(self.number_scans)
                ],
                "stepsl": [
                    self.stepsSpins[i].value() for i in range(self.number_scans)
                ],
                "listgenerators": [
                    self.listGeneratorEdits[i].text() for i in range(self.number_scans)
                ],
                "Randomize": self.randomize.isChecked(),
                "number_scans": self.number_scans,
                "nDscan": self.n_dim.isChecked(),
                "snake_scan": self.tickbox_snake_scan.isChecked(),
            }
            self.exp_window.experiment._props.set(
                "scans/" + self.nameEdit.text(), scannedvalues
            )
        self.done(0)


class QDock(QDockWidget):
    """
    The dock widget containing the ExperimentQ, PrepStation, and Looper.
    These are contained in QSplitters, allowing the sizes of the elements to be adjusted.
    """

    def __init__(self, parent=None):
        super().__init__("", parent)

        qSplitter = QSplitter(Qt.Horizontal)
        qSplitter.addWidget(parent.prepStation)
        qSplitter.addWidget(parent.queue)

        # loopSplitter = QSplitter(Qt.Horizontal)
        # loopSplitter.addWidget(qSplitter)
        # loopSplitter.addWidget(parent.looper)

        self.setWidget(qSplitter)


class BaliFileSelector(QWidget):
    """
    Experiment file selector
    """

    def __init__(self, browser, parent=None):
        super().__init__(parent)
        self.browser = browser
        self.props = browser.props
        layout = QVBoxLayout()
        tree = QTreeView()
        self.tree = tree
        model = QDirModel()
        tree.setModel(model)
        tree.setColumnWidth(0, 250)
        tree.setRootIndex(
            model.index(self.props.get("experiments_dir", tweezerpath + "/experiments"))
        )
        tree.setColumnHidden(1, True)
        tree.doubleClicked.connect(self.open_file)
        tree.header().hideSection(2)
        tree.setContextMenuPolicy(Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self.create_context_menu)

        layout.addWidget(tree)
        self.setLayout(layout)

    def create_context_menu(self, position):
        index = self.tree.indexAt(position)
        filename = self.tree.model().filePath(index)
        menu = QMenu()
        editAction = menu.addAction("Edit")
        CombiAction = menu.addAction("Combi")
        action = menu.exec_(self.tree.mapToGlobal(position))
        if action == editAction:
            self.open_editor(filename)
        if action == CombiAction:
            d = QDialog()
            d.setWindowTitle("Dialog")
            layout = QVBoxLayout()
            editor = CodeEditorParser(filename=filename)
            layout.addWidget(editor)
            d.setLayout(layout)
            d.exec_()

    @staticmethod
    def open_editor(filename):
        d = QDialog()
        d.setWindowTitle("Dialog")
        layout = QVBoxLayout()
        editor = CodeEditor(filename=filename)
        layout.addWidget(editor)
        d.setLayout(layout)
        d.exec_()

    def open_file(self):
        index = self.tree.currentIndex()
        filename = self.tree.model().filePath(index)
        self.browser.open_experiment(filename)

    def terminate(self):
        pass


def filepath_split(filepath):
    """
    takes /dir/name.ext
    returns path, ext, name, filename
    = /dir/name, ext, name, name.ext
    """
    path, ext = os.path.splitext(filepath)  # /dir/name, ext
    name = os.path.basename(path)  # name
    filename = os.path.basename(filepath)  # name.ext
    return path, ext, name, filename


def main():
    qApp = QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile(icon_path + "pytweezer_experiment_browser_icon.svg")
    qApp.setWindowIcon(icon)
    Win = BaliBrowser()
    Win.show()
    sys._excepthook = sys.excepthook

    sys.excepthook = exception_hook

    # def on_exit(_signo, _stack_frame):
    #     Win.close()
    #     sys.exit(0)  # TODO: Confirm termination
    # signal.signal(signal.SIGTERM, on_exit)
    qApp.exec_()


def exception_hook(exctype, value, traceback):
    print_error("tweezer_browser.py: error {0} {1}".format(exctype, value), "error")
    print(traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(Qt, "PYQT_VERSION"):
        main()
