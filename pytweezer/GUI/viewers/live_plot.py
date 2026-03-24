from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import pyqtgraph as pg
from pytweezer.servers import DataClient, Properties
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.pytweezerQt import BMainWindow
import numpy as np
import time


class PlotDataEditor(QWidget):
    """ Manages the programs subscription to data or image streams """
    def __init__(self,props,data={},category='Image',parent=None, preselect=None):
        """
        ARGS:
            category: 'Image' od 'Data' list image or data streams
            props:    link to property class
            parent:   see QWidget for information
        """
        super().__init__()
        self.parent = parent
        self.data=data
        self.props=props
        layout=QVBoxLayout()
        self.setLayout(layout)
        hlayout=QHBoxLayout()
        hlayout.addWidget(QLabel('y1: '))

        # make the current default the preselected value
        self.preselect = preselect
        self.streamlist = QComboBox()
        self.streamlist.currentTextChanged.connect(self.updateDatachannel)
        hlayout.addWidget(self.streamlist)
        layout.addLayout(hlayout)
        self.update_streamlist()
        self.show()

    def update_streamlist(self):
        """ update the ComboBox with available Streams"""
        self.streamlist.clear()
        data=self.data
        data['None']=0
        for i, (name,value) in enumerate(sorted(data.items())):
            self.streamlist.addItem(name+': '+repr(value))
            if name == self.preselect:
                self.streamlist.setCurrentIndex(i)


    def updateDatachannel(self,s):
        dataname = s.split(':')[0]
        self.props.set('ydata', [dataname])
        self.parent.dataname = dataname
        if 'lplot' in self.parent.__dict__.keys():
            self.parent.lplot.name = dataname

class ScrollPlot(pg.PlotItem):
    ''' plot subscribes to channels and updates '''
    def __init__(self, name, parent, **kargs):
        super().__init__(**kargs)

        self.parent = parent
        self.propname = parent.name + '/' + name
        self.props = Properties(self.propname)

        # set names
        self.name = name
        self.datastream = DataClient(self.propname)
        self.subscription_name = self.props.get('datastreams', [''])[0]
        self.datalist = self.props.get('ydata', [''])
        self.dataname = self.datalist[0]

        self.datadict={}
        self.feed_count = 0

        self.initDataBuffer()
        self.setTitle(name + ': ' + self.subscription_name)
        self.legend = self.addLegend()
        lc = self.props.get('linecolors',[[55,126,184],[166,206,227], [255,127,0]])
        self.qualitative_colors = [pg.mkColor(c[0],c[1],c[2]) for c in lc]

        # pens for mean and data points
        self.pen0 = pg.mkPen(self.qualitative_colors[1], width=1)
        self.pen1 = pg.mkPen(self.qualitative_colors[0], width=3)

        # create scatter plot for data points
        self.setScatterplot()

        # running average plot
        self.running_average_window_size = self.props.get('running_average_window_size', 5)
        self.running_average = self.props.get('running_average', True)

        if self.running_average:
            self.avplot = self.plot(pen=self.pen1, name='Mean')

        # general plot setting
        self.setDownsampling(mode='peak')
        self.setClipToView(True)
        self.setLimits(xMax=max(self.x))

        self.datastream.subscribe([self.subscription_name])

        # modify right click menu for plots
        vb=self.getViewBox()
        subscribe_menu=vb.menu.addAction('Subscriptions')
        subscribe_menu.triggered.connect(self.subscribe_window)
        subscribe_menu=vb.menu.addAction('Select Data')
        subscribe_menu.triggered.connect(self.dataSelectDialog)
        subscribe_menu=vb.menu.addAction('Configure')
        subscribe_menu.triggered.connect(self.configureWindow)
        subscribe_menu=vb.menu.addAction('Delete')
        subscribe_menu.triggered.connect(self.deletePlot)
        subscribe_menu=vb.menu.addAction('Clear')
        subscribe_menu.triggered.connect(self.clearPlot)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.setnewData)
        timer.start(10)
        self.timer=timer

    def initDataBuffer(self):
        self.x = np.arange(self.props.get('rolling_buffer_length',30))
        self.data_array = np.zeros((self.props.get('N_curves',1), len(self.x)))

    def feedData(self, data_chunk, data_slice=[0,1], count=True):
        """Feeds a new chunk of data to the data array.
        The chunk is pushed into data_array.
        :data_chunk: np.ndarray of length n_pars
        :data_slice: length 2, contains indices of the plots to which the data is appended

        """
        if self.feed_count < len(self.x) and count:
            self.feed_count += 1
        data_array = self.data_array
        data_array[0:1] = np.roll(data_array[0:1], shift=-1, axis=1)
        data_array[0:1,-1] = [data_chunk]

        # compute running average
        if self.running_average:
            self.running_average_array = self.running_mean(data_array[0], self.running_average_window_size)

        # update data array
        self.data_array = data_array

    def clearPlot(self):
        self.updateDataLists()
        self.updatePlotContent()

    def updatePlot(self):
        indx_feed_count = len(self.x) - self.feed_count
        self.lplot.setData(x=self.x[:self.feed_count], y=self.data_array[0, indx_feed_count:])
        if self.running_average:
            if self.feed_count > self.running_average_window_size:
                x = self.x[self.running_average_window_size-1:self.feed_count]
                y = self.running_average_array[indx_feed_count:]
                self.avplot.setData(x, y)

    def updatePlotContent(self):
        '''
        This method deletes the current content from the plot to make space for fresh content.
        Can be used e.g. when selecting a new datastream.
        '''
#<<<<<<< Updated upstream
        for sample, label in self.legend.items.copy():
            self.legend.removeItem(label.text)
#||||||| merged common ancestors
        #self.legend.scene().removeItem(self.legend)
        self.legend = self.addLegend()
#=======
        #self.legend.scene().removeItem(self.legend)
        #for i in self.legend.items:
        #    i.removeItem()
        #self.legend = self.addLegend()
#>>>>>>> Stashed changes
        self.lplot.clear()
        self.setScatterplot()
        self.feed_count = 0
        self.initDataBuffer()
        try:
            self.avplot.clear()
        except:
            pass
        if not self.running_average:
            try:
                self.avplot.clear()
            except:
                pass
            self.avplot = self.plot(pen=None)
        else:
            self.avplot = self.plot(pen=self.pen1, name='Mean')

    def updateTitle(self):
        title = self.name + ': ' + self.props.get('datastreams', [''])[0]
        self.setTitle(title)

    def updateSubscription(self):
        self.subscription_name = self.props.get('datastreams', [''])[0]
        # we first unsubsribe from the old datastream, then subsribe to the new one
        self.datastream.unsubscribe()
        self.datastream.subscribe([self.subscription_name])

    def updateDataLists(self):
        self.datalist = self.props.get('ydata', [''])
        self.dataname = self.datalist[0]
        self.running_average = self.props.get('running_average', True)
        self.running_average_window_size = self.props.get('running_average_window_size')

    def setnewData(self):
        if self.datastream.has_new_data():
            recvmsg=self.datastream.recv()
            A=None
            if len(recvmsg)==2:
                msg,di=recvmsg
            elif len(recvmsg)==3:
                msg,di,A=recvmsg
            self.datadict = di
            try:
                self.feedData(di[self.dataname])
                self.updatePlot()
            except Exception as e:
                if self.dataname == '':
                    print('no dataname given')
                else:
                    print(e)
                    print('scrollwindow.py selected data not found in dataset: ',
                            self.dataname,di)

    def setScatterplot(self):
        self.lplot = self.plot(pen=None, name=self.dataname, symbol='o',)
        self.lplot.setSymbolPen(self.pen0)
        self.lplot.setSymbolBrush(self.qualitative_colors[1])

    def deletePlot(self):
        # delete the chosen plot from the properties of parent scroll window
        oldprops = self.props.get('/'+self.parent.name)
        oldprops.pop(self.name)
        self.props.set('/'+self.parent.name, oldprops)

        # mark the plot widget for deletion
        self.deleteLater()

    def dataSelectDialog(self):
        d = QDialog()
        layout=QVBoxLayout()
        d.setWindowTitle("Select Data")
        editor = PlotDataEditor(self.props, self.datadict, parent=self, preselect=self.dataname)
        layout.addWidget(editor)
        d.setLayout(layout)
        d.exec_()

        # update configured parameters and plot
        self.updateDataLists()
        self.updatePlotContent()

    def subscribe_window(self):
        d = QDialog()
        layout=QVBoxLayout()
        d.setWindowTitle("Subscriptions")
        editor = SubscriptionEditor(self.props,'Data')
        layout.addWidget(editor)
        d.setLayout(layout)
        #d.setWindowModality(QtGui.ApplicationModal)
        d.exec_()

        # after the window is closed, update all plot properties
        self.updateTitle()
        self.updateSubscription()

    def configureWindow(self):
        self.timer.stop()
        d = QDialog()
        d.setWindowTitle("Configure")
        layout=QVBoxLayout()
        editor = PropEdit('/' + self.propname + '/')
        layout.addWidget(editor)
        d.setLayout(layout)
        d.exec_()
        self.initDataBuffer()
        self.timer.start(10)

        # update plot content so the new buffer length can be adapted
        self.updateDataLists()
        self.updatePlotContent()

    @staticmethod
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)


class ScrollWindow(BMainWindow):
    def __init__(self,name, parent=None, n_plots=0):
        """TODO: to be defined1. """
        super().__init__(name,parent)
        self.name = name
        self.props = Properties(self.name)

        # initialize the window
        win = pg.GraphicsLayoutWidget()
        win.resize(1000,600)
        self.setCentralWidget(win)
        self.win = win
        self.plotcount = n_plots
        self.plots = []
        self.restorePlots()
        # describe how the plotted parameters behave, i.e. how many they are
        # and initialize structure to store them
        self.init_toolbar()
        self.show()

    def init_toolbar(self):
        self.addstreamaction = QAction('Add Stream', self)
        self.addstreamaction.triggered.connect(self.addPlot)

        self.streambar = self.addToolBar('Stream')
        self.streambar.addAction(self.addstreamaction)

    def updateAllSubscriptions(self):
        '''
        Deprecated?
        '''
        for p in self.plots:
            p.updateSubscription()

    def restorePlots(self):
        storedplots = self.props.get('/' + self.name)
        newprops = {}
        # we need to remap the numbering to stay consistent
        for i, pn in enumerate(storedplots):
            newname = 'Plot {}'.format(i)
            newprops[newname] = storedplots[pn]
        self.props.set('/'+self.name, newprops)

        for pn in storedplots:
            self.addPlot()

    def addPlot(self):
        p = ScrollPlot('Plot {}'.format(self.plotcount), parent=self)
        self.win.addItem(p)
        if (self.plotcount+1)%2 == 0:
            self.win.nextRow()
        self.plotcount +=1
        self.plots.append(p)


def main(name):
    #QtGui.QApplication.setGraphicsSystem('raster')
    app = QApplication([])
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)
    #init
    Win = ScrollWindow(name)
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()
    main('Live Plot')
