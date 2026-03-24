from PyQt5 import QtWidgets
import numpy as np
from pytweezer.servers import DataClient
from pytweezer.servers.properties import Properties
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.GUI.property_editor import PropEdit
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
class UpdatingPlot(pg.PlotItem):
    ''' plot subscribes to channels and updates '''
    def __init__(self,name,**kargs):
        super().__init__(**kargs) 
        self.datastream=DataClient(name)
        self.props=Properties(name)
        self.name=name
        vb=self.getViewBox()
        subscribe_menu=vb.menu.addAction('subscriptions')
        subscribe_menu.triggered.connect(self.subscribe_window)
        subscribe_menu=vb.menu.addAction('configure')
        subscribe_menu.triggered.connect(self.configureWindow)

        self.updateConfiguration()

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.setnewData)
        timer.start(10)

    def setnewData(self):
        #print('called')
        if self.datastream.has_new_data():
            msg,di,A=self.datastream.recv()    
            #print(A.shape)
            # generate x axis in case array is onedimensional
            if A.ndim==1:
                x=range(len(A))
                A=np.vstack(x,A)

            if msg in self.curvedict:
                if self.props.get('rotated',False):
                    self.curvedict[msg].setData(A[1],A[0])
                else:
                    self.curvedict[msg].setData(A[0],A[1])
    def subscribe_window(self):
        d = QtGui.QDialog()
        layout=QVBoxLayout()
        b1 = QtGui.QPushButton("ok",d)
        b1.move(50,50)
        d.setWindowTitle("Dialog")
        editor=SubscriptionEditor(self.props,'Data')
        layout.addWidget(editor)
        d.setLayout(layout)
        #d.setWindowModality(QtGui.ApplicationModal)
        d.exec_()
        self.updateConfiguration()

    def updateConfiguration(self):
        self.datastream.unsubscribe()
        self.curvedict={}
        datastreams=self.props.get('datastreams',['Axial_slice'])
        self.datastream.subscribe(datastreams)
        print('updating_plot.py subscribed to:  ',datastreams)
    
        #self.setRotation(-90i)
        self.clear()
        for stream in datastreams:
            col=QColor(self.props.get(stream+'/color',int(255*256**3+255)))
            pen=pg.mkPen(col, width=1)
            self.curvedict[stream] = self.plot(pen=pen)


    def configureWindow(self):
            d = QtGui.QDialog()
            d.setWindowTitle("Dialog")
            layout=QVBoxLayout()
            editor=PropEdit('/'+self.name+'/')            
            layout.addWidget(editor)
            d.setLayout(layout)
            d.exec_()
            self.updateConfiguration()



class LivePlot(QWidget):
    def __init__(self,name,parent=None,imagestreams=[]):
        super().__init__(parent)
        self.name=name
        self.widthHint=200
        self.heightHint=200

        layout=QtWidgets.QVBoxLayout()

        win = pg.GraphicsLayoutWidget(title="Basic plotting examples")
        win.setWindowTitle('pyqtgraph example: Plotting')
        layout.addWidget(win)
        self.setLayout(layout)
        self.show()
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        p6=UpdatingPlot(name)
        win.addItem(p6)
        #p6 = win.addPlot(title="Updating plot",viewBox=vb)
        #updatePlot()
    def sizeHint(self):
        return QtCore.QSize(self.widthHint,self.heightHint)
    def setSizeHint(self,width,height):
        self.widthHint=width
        self.heightHint=height

'''
win.nextRow()

p7 = win.addPlot(title="Filled plot, axis disabled")
y = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(size=1000, scale=0.1)
p7.plot(y, fillLevel=-0.3, brush=(50,50,200,100))
p7.showAxis('bottom', False)


x2 = np.linspace(-100, 100, 1000)
data2 = np.sin(x2) / x2
p8 = win.addPlot(title="Region Selection")
p8.plot(data2, pen=(255,255,255,200))
lr = pg.LinearRegionItem([400,700])
lr.setZValue(-10)
p8.addItem(lr)

p9 = win.addPlot(title="Zoom on selected region")
p9.plot(data2)
def updatePlot():
    p9.setXRange(*lr.getRegion(), padding=0)
def updateRegion():
    lr.setRegion(p9.getViewBox().viewRange()[0])
lr.sigRegionChanged.connect(updatePlot)
p9.sigXRangeChanged.connect(updateRegion)
updatePlot()
'''

def main(name):
    #QtGui.QApplication.setGraphicsSystem('raster')
    app = QtWidgets.QApplication([])
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)
    #init
    camWin = LivePlot(name)
    camWin.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()
    main('dummyname')




