#! /usr/bin/python3
import shutil
import time
from datetime import datetime

from PyQt5 import QtGui, QtCore,QtWidgets
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QGridLayout, QLabel
import numpy as np
from os.path import isfile
from os import listdir, remove
from pytweezer.GUI.pytweezerQt import BWidget
from pytweezer.servers.configreader import ConfigReader
from pytweezer.servers import tweezerpath, icon_path
import subprocess
from pytweezer.analysis.print_messages import print_error
import signal


class SingleProcess(QFrame):
    def __init__(self,script='',name='',active=False,category='',parent=None, tooltip=None):
        self.process=None
        self.script=script
        self.processname=name
        self.category=category
        super().__init__(parent)

        self.setStyleSheet("SingleProcess {background-color: rgb(210,230,240);color:blue; margin:7px; border:7px solid rgb(220, 240, 255); } QPushButton {background-color: rgb(210,230,240);color:#000000; margin:1px; border:0px solid rgb(20, 240, 255);} ")
        self.setStyleSheet("SingleProcess {background-color: rgb(210,230,240);color:blue; margin:1px; border:2px solid rgb(220, 240, 255); } ")
        layout=QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)
        startButton=QPushButton(name)
        if tooltip is not None:
            startButton.setToolTip(tooltip)
        startButton.setStyleSheet("QPushButton {"
                            "color: blue;"
                            "background-color: rgb(0, 255, 127);"
                            "}")
        self.startButton=startButton
        startButton.clicked.connect(self.startProcess)
        layout.addWidget(startButton)
        killButton=QPushButton('')
        killButton.setMaximumWidth(20)
        killButton.setIcon(QtGui.QIcon(icon_path+'terminate.png'))
        killButton.clicked.connect(self.terminateProcess)
        layout.addWidget(killButton)
        self.setLayout(layout)
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        if active:
            self.startProcess()
        ## Process monitoring
        self.timer  = QtCore.QTimer(self)
        self.timer.setInterval(1000)          # Throw event timeout with an interval of 1000 milliseconds
        self.timer.timeout.connect(self.updateStatus) # each time timer counts a second, call self.blink
        self.timer.start()

    def __del__(self):
        print('__del__',self.processname)
        self.killProcess()

    def startProcess(self):
        self.terminateProcess()
        print_error('processmanager.py - startProcess(): Starting {0}.'.format(self.processname), 'bold')
        self.process = subprocess.Popen(['python3', self.script, self.category+self.processname])

    def terminateProcess(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except Exception as e:
                print_error('process termination failed killing process {0}'.format(self.processname), 'error')
                print(e)
                self.process.kill()
            if self.process.poll()==-15:
                self.startButton.setStyleSheet("color: gray")
                self.process=None
    def killProcess(self):
        if self.process is not None:
            print('killing',self.processname)
            self.process.kill()
            if self.process.poll()==-15:
                self.startButton.setStyleSheet("color: gray")
                self.process=None

    def updateStatus(self):
        if self.process is not None:
            if self.process.poll()==None:
                self.startButton.setStyleSheet("color: blue")
            else:
                self.startButton.setStyleSheet("color: red")
        else:
                self.startButton.setStyleSheet("color: gray")

class ProcessManager(BWidget):

    def __init__(self, parent=None):
        super().__init__('ProcessManager', parent, create_props=False)
        self.setStyleSheet("ProcessManager {background-color: rgb(195,205,230);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} ")
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        conf = ConfigReader.getConfiguration()
        cmdlist = []
        self.processlist = []   # ensure child classes are properly destroyed
        
        # Only show GUI and Viewer categories (not Servers)
        for cnum, category in enumerate(['GUI', 'Viewer']):
            if category in conf:
                line = 1
                layout.addWidget(QLabel(category), line, cnum)
                for name, param in sorted(conf[category].items())[::-1]:
                    tooltip = None
                    if 'tooltip' in param:
                        tooltip = param['tooltip']
                    process = SingleProcess(tweezerpath + '/bin/' + param['script'], name,
                                            param['active'], category + '/', tooltip=tooltip)
                    line = line + 1
                    layout.addWidget(process, line, cnum)
                    self.processlist.append(process)
        self.setLayout(layout)

    def move(self,x,y):
        super().move(x,y)
        print(x)

    def closeEvent(self, event):
        '''on shutdown terminate all GUI/Viewer processes'''
        for p in self.processlist:
            p.terminateProcess()
        print_error('processmanager.py: Terminated all GUI/Viewer processes.', 'info')

        # Backup the following two files, since they are often corrupt after restart:
        # /home/bali/scripts/pytweezer/configuration/properties/properties.json

        path = '../pytweezer/configuration/properties/'
        # for fname in ['properties.json']:
        #     backup_file(path, fname)
        # TODO: Auto-load last backup when crashed while restart

        event.accept()

    def __del__(self):
        # now the class is deleted
        print('deleting')


def backup_file(path, fname, keep_old=10):
    """
        This method creates a backup copy of path+fname including the current datetime.
        Olf backup files will be deleted except the last files (n=keep_old).
        If keep_old == -1, nothing will be deletede.
    """

    date_format = '__%Y_%m_%d__%H_%M_%S'

    date = datetime.now().strftime(date_format)
    src = path + fname.split('.')[0] + '.' + fname.split('.')[1]
    dst = path + fname.split('.')[0] + date + '.' + fname.split('.')[1]
    if isfile(src):
        shutil.copyfile(src, dst)

    if keep_old >= 0:
        directory = path + fname
        if '/' not in directory:
            print_error('processmanager.py - backup_file(): Couldn\'t remove files in {0} + {1},'
                        ' since no \'/\' is present.'.format(path, fname), 'error')
            return

        file = directory[directory.rfind('/') + 1:]
        directory = directory[:directory.rfind('/') + 1]

        if '.' not in file:
            print_error('processmanager.py - backup_file(): Couldn\'t remove files in {0} + {1},'
                        ' since no \'.\' is present.'.format(path, fname), 'error')
            return
        f0 = file.split('.')[0]
        f1 = file.split('.')[-1]
        files = [f for f in listdir(directory) if f.startswith(f0) and f.endswith(f1)]
        dates_str = [s[len(f0):-len(f1) - 1] for s in files]
        dates = []
        for date in dates_str:
            try:
                dates = np.append(dates, datetime.datetime.strptime(date, date_format))
            except:
                continue
        dates = sorted(dates)

        for date in dates[: max(0, len(dates) - keep_old)]:
            file_to_delete = directory + f0 + date.strftime(date_format) + '.' + f1
            if isfile(file_to_delete):
                print_error('processmanager.py - backup_file(): Deleting file {0} ...'.format(file_to_delete),
                            'warning')
                remove(file_to_delete)
                # TODO: debug why deletion doesn't work


def main():
    app = QtWidgets.QApplication(sys.argv)
    Win = ProcessManager()
    Win.show()

    def on_exit(_signo, _stack_frame):
        print('closing')
        Win.close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_exit)

    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    print(icon_path)
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
    else:
        print('Running in interactive mode. ProcessManager won\'t be started.')
