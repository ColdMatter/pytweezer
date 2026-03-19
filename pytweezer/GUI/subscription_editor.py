from PyQt5 import QtGui, QtCore,QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pytweezer.servers import Properties
import copy
#app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from math import *
import numpy as np
from pytweezer.GUI.table_parameter import *
import time

## test subclassing parameters


class SubscriptionEditor(QWidget):
    subscriptionsChanged=QtCore.pyqtSignal()
    propertiesChanged = QtCore.pyqtSignal()
    """ Manages the programs subscription to data or image streams """
    def __init__(self,props,category='Image',parent=None,propprefix='',streamkey=None, fullList=[]):
        """
        ARGS:
            category: 'Image' od 'Data' list image or data streams
            props:    link to property class
            parent:   see QWidget for information
            propprefix: if not '' the propertyentry will be [propprefix]+category+'streams'
            streamkey : if set it will be used as property name instead of the category
        """
        super().__init__(parent)
        self.props=props
        self._props=props
        self.category=category
        self.fullList = fullList
        if streamkey is not None:
            self.streamkey=propprefix+streamkey
        else:
            self.streamkey=propprefix+self.category.lower()+'streams'
        layout=QVBoxLayout()
        self.setLayout(layout)
        qlist=QListWidget()
        self.qlist=qlist
        self.loadSubscriptions()
        layout.addWidget(qlist)
        self.streamlist=QComboBox()
        layout.addWidget(self.streamlist)
        self.update_streamlist()
        addButton=QPushButton('add')
        addButton.clicked.connect(self.add)
        layout.addWidget(addButton)
        delButton=QPushButton('del')
        delButton.clicked.connect(self.deletemarked)
        layout.addWidget(delButton)

        self.show()

    def loadSubscriptions(self):
        """ load subscriptions from properties into QListWidget """
        for stream in self.props.get(self.streamkey,[]):
            QListWidgetItem(stream,self.qlist)

    def handle_property_changes(self,changes):
        ''' Check the properties for changes and update subscriptions if properties have changed'''
        index='/'+self._props.name+'/'+self.streamkey
        if index in changes:
            self.qlist.clear()
            self.loadSubscriptions()
        self.propertiesChanged.emit()


    def update_streamlist(self):
        """ update the ComboBox with available Streams"""
        for i in range(self.streamlist.count()): #remove all items first
            self.streamlist.removeItem(0)
        if self.category == 'List':
            for name in self.fullList:
                self.streamlist.addItem(name)
        else:
            di = self.props.get('/Servers/'+self.category+'Stream/active', {})
            for name,value in sorted(di.items()):
                timedelta=time.time()-value['timestamp']
                self.streamlist.addItem(name+'[%i s]'%timedelta)

    def add(self):
        """Add the Stream in the Combobox to the list of Streams"""
        streaml=self.streamlist.currentText()
        stream=streaml.split('[')[0]
        streamlist= self.props.get(self.streamkey,[])
        if not stream in streamlist:
            QListWidgetItem(stream,self.qlist)
            self.props.set(self.streamkey,streamlist+[stream])
        self.subscriptionsChanged.emit()

    def deletemarked(self):
        """ delete the marked Item from the list of streams """
        row=self.qlist.currentRow()
        if row >=0:
            self.qlist.takeItem(row)
            streams=self.props.get(self.streamkey,[])
            del streams[row]
            streams=self.props.set(self.streamkey,streams)
        self.subscriptionsChanged.emit()

if __name__ == '__main__':
    import sys
    props=Properties('Tests/SubscriptionEditor')
    pp=SubscriptionEditor(props,'Data')
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
