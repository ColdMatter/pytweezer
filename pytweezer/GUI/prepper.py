import pytweezer
#from pytweezer.servers.configreader import ConfigReader
#import  subprocess
import os
import importlib.util
import inspect
import numpy as np
import time
import threading
import traceback

from copy import copy

from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import PyQt5.QtCore as Qt
from PyQt5 import QtCore
from pytweezer import *
from pytweezer.servers import Properties,tweezerpath,PropertyAttribute,DataClient
from pytweezer.servers import send_info,send_error
from pytweezer.GUI.editor_sequencer import CodeEditorParser,CodeEditor
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.GUI.simple_defaults import BasicManager,FloatManager,BoolManager,ComboManager
from pytweezer.GUI.helper_workers import Worker
from pytweezer.experiment.experiment import NumberValue

from functools import partial

from PyQt5 import QtCore
from pytweezer.GUI.helper_workers import Worker
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QThread, QThreadPool, QRunnable, QObject

import cgitb
cgitb.enable(format = 'text')

'''
Contains the functions for scheduling and running experiments
'''

class Prepper:
    def __init__(self, browser=None):
        pass
        # self.browser = browser
        # self.queue = browser.queue
        # self.expDict = browser.queue.expDict
        # self.threadpool = QThreadPool()
        # self.threadpool.setMaxThreadCount(1)
    #     self.start_looper()
    #
    # def start_looper(self):
    #     self.loopWorker = Worker(self.queue_fn)
    #     self.loopWorker.signals.tableUpdate.connect(self.update_table)
    #     self.threadpool.start(self.loopWorker)
    #
    # def looper_fn(self, progress_callback=None):
    #     timer = QtCore.QTimer(self)
    #     timer.timeout.connect(self.updateData)
    #     timer.start(50)
    #     self.timer = timer
