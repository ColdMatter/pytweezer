
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pytweezer.GUI.pytweezerQt import BWidget, BFrame
from PyQt5.QtCore import *
import PyQt5.QtCore as QtCore
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.GUI.viewers.live_plot import PlotDataEditor
from pytweezer.GUI.property_editor import PropEdit, PropSelector
from pytweezer.servers import DataClient, Properties

class ParameterBox(BFrame):
    ''' Show individual parameters '''
    def __init__(self,name,parent=None):
        super().__init__(name,parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(1)
        self.datadict={}
        self.stream_data = {}
        self.propname = name
        self.props = Properties(self.propname)
        self.datalist = self.props.get('ydata', [''])
        self.dataname = self.datalist[0]


        # set names
        self.name = name
        self.datastream = DataClient(self.propname)
        self.subscription_names = self.props.get('datastreams', [''])
        self.dataBoxes = []
        self.row_config = []

        layout = QVBoxLayout()
        for _ in range(10):
            box = DataBox()
            box.hide()
            self.dataBoxes.append(box)
            layout.addWidget(box)
        self.setLayout(layout)

        #Context Menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.openMenu)
        self.updateSubscription()
        self.updateDataLists()

        timer = QTimer(self)
        timer.timeout.connect(self.setNewData)
        timer.start(500)
        self.timer=timer

    def setNewData(self):
        got_new = False
        while self.datastream.has_new_data():
            recvmsg=self.datastream.recv()
            if recvmsg is None:
                continue
            A=None
            if len(recvmsg)==2:
                msg,di=recvmsg
            elif len(recvmsg)==3:
                msg,di,A=recvmsg
            else:
                continue
            if isinstance(di, dict):
                self.stream_data[msg] = di
                self.datadict = di
                got_new = True

        if got_new:
            self._update_rows()

    def openMenu(self, position):
        menu=QMenu()
        subscribe_menu=menu.addAction('subscriptions')
        subscribe_menu.triggered.connect(self.subscribe_window)
        subscribe_menu=menu.addAction('selectData')
        subscribe_menu.triggered.connect(self.dataSelectDialog)
        menu.exec_(self.mapToGlobal(position))

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
        #self.updatePlotContent()

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
        self.updateSubscription()

    def updateSubscription(self):
        self.subscription_names = self.props.get('datastreams', [''])
        stream_names = [s for s in self.subscription_names if isinstance(s, str) and s]
        # we first unsubsribe from the old datastream, then subsribe to the new one
        self.datastream.unsubscribe()
        if len(stream_names) > 0:
            self.datastream.subscribe(stream_names)
        self.stream_data = {}

    def updateDataLists(self):
        self.datalist = self.props.get('ydata', [''])
        if len(self.datalist) == 0:
            self.datalist = ['']
        self.dataname = self.datalist[0]

        streams = [s for s in self.props.get('datastreams', ['']) if isinstance(s, str) and s]
        if len(streams) == 0:
            streams = ['']

        n_rows = max(len(streams), len(self.datalist))
        if len(self.datalist) == 1 and n_rows > 1:
            keys = self.datalist * n_rows
        else:
            keys = list(self.datalist)
            while len(keys) < n_rows:
                keys.append(keys[-1] if len(keys) > 0 else '')

        stream_list = list(streams)
        while len(stream_list) < n_rows:
            stream_list.append(stream_list[-1] if len(stream_list) > 0 else '')

        self.row_config = list(zip(stream_list, keys))
        self._update_rows()

    def _update_rows(self):
        for box in self.dataBoxes:
            box.hide()

        for i, (stream_name, data_key) in enumerate(self.row_config[:len(self.dataBoxes)]):
            box = self.dataBoxes[i]
            label = stream_name if stream_name else 'stream'
            if data_key:
                label = f'{label}: {data_key}'
            box.label.setText(label)

            di = self.stream_data.get(stream_name, None)
            val = '---'
            if isinstance(di, dict) and data_key in di:
                raw = di[data_key]
                if isinstance(raw, (float, int)):
                    val = str(round(raw, 5))
                else:
                    val = str(raw)
            box.value.setText(val)
            box.show()

class ImageDataBox(BFrame):
    def __init__(self,name,parent=None):
        super().__init__(name,parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.propname = name
        self.props = Properties(self.propname)
        self.datalist = self.props.get('ydata', [''])
        self.dataname = self.datalist[0]
        self.imDataDict = {}
        self.dataBoxes = []
        self.qlayout = QHBoxLayout()
        for i in range(10):
            box = DataBox()
            self.dataBoxes.append(box)
            self.qlayout.addWidget(box)
            box.hide()
            self.setLayout(self.qlayout)

        # set names

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.openMenu)
        self.updateSubscription()


    def openMenu(self, position):
        menu=QMenu()
        subscribe_menu=menu.addAction('subscriptions')
        subscribe_menu.triggered.connect(self.subscribe_window)
        # subscribe_menu=menu.addAction('selectData')
        # subscribe_menu.triggered.connect(self.dataSelectDialog)
        # subscribe_menu=menu.addAction('configure')
        # subscribe_menu.triggered.connect(self.configureWindow)
        menu.exec_(self.mapToGlobal(position))

    def subscribe_window(self):
        d = QDialog()
        layout=QVBoxLayout()
        d.setWindowTitle("Subscriptions")
        editor = SubscriptionEditor(self.props, category='List', fullList=self.imDataDict.keys())
        layout.addWidget(editor)
        d.setLayout(layout)
        #d.setWindowModality(QtGui.ApplicationModal)
        d.exec_()

        # after the window is closed, update all plot properties
        self.updateSubscription()

    def updateSubscription(self):
        # self.subscription_name = self.props.get('datastreams', [''])[0]
        # # we first unsubsribe from the old datastream, then subsribe to the new one
        # self.datastream.unsubscribe()
        # self.datastream.subscribe([self.subscription_name])
        pass

    def dataSelectDialog(self):
        d = QDialog()
        layout=QHBoxLayout()
        d.setWindowTitle("Select Data")
        layout.addWidget(QLabel('y1: '))
        dataCombo = QComboBox()
        for dataname in self.imDataDict.keys():
            dataCombo.addItem(dataname)
        dataCombo.currentTextChanged.connect(self.setNewDataName)
        layout.addWidget(dataCombo)
        d.setLayout(layout)
        d.exec_()

        # update configured parameters and plot
        self.updateDataLists()
        #self.updatePlotContent()

    def setNewData(self,imDataDict):
        for box in self.dataBoxes:
            box.hide()
        self.imDataDict = imDataDict
        #print('datadict:',imDataDict)
        streams = self.props.get('liststreams',[])
        for i,key in enumerate(streams):
            box = self.dataBoxes[i]
            box.label.setText(key)
            val = imDataDict[key]
            box.value.setText(str(round(val,5)))
            box.show()
        #val = self.datadict[self.dataname]
        #self.dataValue.setText(str(val))

    def setNewDataName(self, name):
        self.dataname = name

class CamPropsBox(ImageDataBox):
    def __init__(self,name,parent=None):
        super().__init__(name,parent)
        self.keyList = self.props.get('keyList',[])
        self.camName = self.props.get('camName','Axial')
        self.subtree = self.props.get('subtree','/Cameras/Axial/DefaultConfig/')

        timer = QTimer(self)
        timer.timeout.connect(self.setNewData)
        timer.start(1000)
        self.timer=timer

    def openMenu(self, position):
        menu=QMenu()
        subscribe_menu=menu.addAction('props')
        subscribe_menu.triggered.connect(self.props_window)
        subscribe_menu=menu.addAction('select camera')
        subscribe_menu.triggered.connect(self.cams_window)
        # subscribe_menu=menu.addAction('configure')
        # subscribe_menu.triggered.connect(self.configureWindow)
        menu.exec_(self.mapToGlobal(position))

    def props_window(self):
        d = QDialog()
        layout=QVBoxLayout()
        d.setWindowTitle("Subscriptions")
        self.setNewSubtree()
        editor = PropSelector(self.propname, self.props, subtree=self.subtree, parent=self)
        layout.addWidget(editor)
        d.setLayout(layout)
        #d.setWindowModality(QtGui.ApplicationModal)
        d.exec_()

    def cams_window(self):
        d = QDialog()
        layout=QHBoxLayout()
        d.setWindowTitle("Select Camera")
        layout.addWidget(QLabel('Camera Name:'))
        camCombo = QComboBox()
        camlist = self.props.get('/Cameras',{})
        #print(camlist.keys())
        for camName in camlist.keys():
            camCombo.addItem(camName)
        idx = camCombo.findText(self.camName)
        if idx != -1:
            camCombo.setCurrentIndex(idx)
        camCombo.currentTextChanged.connect(self.setNewCamName)
        layout.addWidget(camCombo)
        d.setLayout(layout)
        d.exec_()


    def setNewData(self):
        self.setNewSubtree()
        if self.subtree != '':
            for box in self.dataBoxes:
                box.hide()
            self.keyList = self.props.get('keyList', [])
            for i,key in enumerate(self.keyList):
                box = self.dataBoxes[i]
                if isinstance(key, str):
                    box.label.setText(key)
                    val = self.props.get(self.subtree+key,'param not set')
                    if isinstance(val, float) or isinstance(val, int):
                        box.value.setText(str(round(val, 1)))
                    elif isinstance(val, str):
                        box.value.setText(val)
                    else:
                        box.value.setText('nan')
                    box.show()
        else:
            print('no subtree set')

    def setNewCamName(self, name):
        self.props.set('camName', name)
        self.camName = name
        self.setNewSubtree()

    def setNewSubtree(self):
        self.currentConfig = self.props.get('/Cameras/'+self.camName+'/Configuration_name', 'DefaultConfig')
        self.subtree='/Cameras/'+self.camName+'/'+self.currentConfig+'/'
        self.props.set('subtree', self.subtree)

class DataBox(QWidget):
    def __init__(self,name='',value='',parent=None):
        super().__init__(parent)
        self.qlayout = QHBoxLayout()
        self.label = QLabel(name)
        self.qlayout.addWidget(self.label)
        self.value = QLabel(value)
        self.qlayout.addWidget(self.value)
        self.setLayout(self.qlayout)
        # set names

    def sizeHint(self):
        return QtCore.QSize(200,200)
