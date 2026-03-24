from PyQt5 import QtGui, QtCore,QtWidgets
from PyQt5.QtWidgets import *
import PyQt5.QtCore as Qt
from pytweezer import *
from pytweezer.servers import Properties,PropertyAttribute
from pytweezer.servers.configreader import ConfigReader
from pytweezer.GUI.viewers.image_monitor import ImageDisplay
from pytweezer.GUI.viewers.updating_plot import LivePlot
from pytweezer.GUI.pytweezerQt import BMainWindow
from pytweezer.GUI.viewers.parameterbox import ParameterBox, ImageDataBox, CamPropsBox
import  subprocess
import os
import importlib.util
import inspect
import argparse

class ImageWindow(BMainWindow):
    ''' Containing an image viewer plus two plots on the sides for line plots
    '''


    def __init__(self,name,parent=None):
        super().__init__(name,parent)
        type(self)._displayname  = PropertyAttribute('ImageDisplay',self._name+'/Viewer')
        type(self)._topPlotterName = PropertyAttribute('Plottertop',self._name+'/Plotline')
        type(self)._rightPlotterName= PropertyAttribute('Plotterright',self._name+'/Plotcol')
        type(self)._parameterBoxName= PropertyAttribute('Parameterbox',self._name+'/ParBox')
        type(self)._imDataBoxName= PropertyAttribute('ImageDataBox',self._name+'/ImDatBox')
        type(self)._propBoxName= PropertyAttribute('PropDataBox',self._name+'/PropBox')
        self.initUI()



    def initUI(self):
        self.statusBar().showMessage('Ready')
        self.createDockWidgets()
        self.image = ImageDisplay(self._displayname, parent=self)
        self.setCentralWidget(self.image)
        #self.image.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
        #self.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
        #self.linedock.setMaximumWidth(100)
        #self.linedock.setMaximumHeight(100)
        #Qt.QTimer.singleShot(1000,self.res)



    def createDockWidgets(self):
        linedock=QDockWidget(self._topPlotterName,self)
        self.lineplot=LivePlot(self._topPlotterName)
        linedock.setWidget(self.lineplot)
        self.lineplot.setSizeHint(700,200)
        self.linedock=linedock
        self.addDockWidget(Qt.Qt.TopDockWidgetArea,linedock)
        self.linedock.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)

        fileselectorDock=QDockWidget(self._rightPlotterName,self)
        self.colplot=LivePlot(self._rightPlotterName)
        fileselectorDock.setWidget(self.colplot)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea,fileselectorDock)

        dock=QDockWidget("Image Info", self)
        dockWidget = QFrame()
        dockLayout = QVBoxLayout()

        self.parbox=ParameterBox(self._parameterBoxName)
        dockLayout.addWidget(self.parbox)

        #dock=QDockWidget(self._imDataBoxName,self)
        self.imDataBox=ImageDataBox(self._imDataBoxName)
        dockLayout.addWidget(self.imDataBox)

        self.propBox=CamPropsBox(self._propBoxName)
        dockLayout.addWidget(self.propBox)


        dockWidget.setLayout(dockLayout)
        dock.setWidget(dockWidget)
        self.addDockWidget(Qt.Qt.TopDockWidgetArea,dock)
        dock.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)


def main(name):
    app = QtWidgets.QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile('../pytweezer/icons/pytweezer_viewer_icon.svg')
    app.setWindowIcon(icon)
    Win = ImageWindow(name)
    Win.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        parser = argparse.ArgumentParser()
        parser.add_argument('name', nargs=1, help='name of this program instance')
        args = parser.parse_args()
        name=args.name[0]
        main(name)
