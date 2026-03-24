from PyQt5 import QtGui, QtCore,QtWidgets
from PyQt5 import QtWidgets  as QtW
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from pytweezer import *
from pytweezer.servers import Properties,tweezerpath, icon_path, PropertyAttribute,send_info,send_warning
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.pytweezerQt import BWidget
#from pytweezer.servers.configreader import ConfigReader
import  subprocess
import os
import time
import signal


class checkableModel(QtGui.QStandardItemModel):
    def test(self):
        pass
#    def isBooleanColumn(self,index):
#        if not index.isValid():
#            return False
#        else:
#            return True
#    def flags(self,index):
#        if not index.isValid():
#            return 0
#        return Qt.ItemIsUserCheckable
#    def data(self,index,role):
        #return Qt.Checked
        #item=self.getItem(index)
#        return 'dsdasd'


class TreeViewWidget(QWidget):
    FROM, SUBJECT, DATE, STREAM  = range(4)
    def __init__(self,props,parent=None):
        super().__init__(parent)
        self.props=props
        self._props=props
        self.dataView = QTreeView()
        self.dataView.setRootIsDecorated(False)
        self.dataView.setAlternatingRowColors(True)
        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self.dataView)
        self.setLayout(dataLayout)
        model = self.createMailModel(self)
        self.dataView.setModel(model)
        self.model=model
        self.itemdict={}
        self.filters={}
        self.processes={} # objects for running processes


        self.timer  = QtCore.QTimer(self)
        self.timer.setInterval(300)          # Throw event timeout with an interval of 1000 milliseconds
        self.timer.timeout.connect(self.updateStatus) # each time timer counts a second, call self.blink
        self.timer.start()


    def createMailModel(self,parent):
        model = checkableModel(0, 3, parent)
        model.setHeaderData(self.FROM, Qt.Horizontal, "Name")
        model.setHeaderData(self.SUBJECT, Qt.Horizontal, "Script")
        model.setHeaderData(self.DATE, Qt.Horizontal, "Category")
        model.setHeaderData(self.STREAM, Qt.Horizontal, "Inputstream")
        model.itemChanged.connect(self.itchange)
        return model

    def itchange(self,item):
        name=item.text()
        index=item.index()
        category=self.model.data(self.model.index(index.row(),2))
        if name not in self.itemdict:
            self.itemdict[name]=item
        if item.checkState()==Qt.Checked:
            print('starting process', name)
            send_info('starting process'+name)
            self.startProcess(name)

            self._props.set(category+'/'+name+'/active',True)
        else:
            send_info('stopping '+name)
            self.endProcess(name)
            self._props.set(category+'/'+name+'/active',False)
            #help(item)

    def killVagabonding(self,pname,pargs):
        """ if there is a prgramm running with python3 pname pargs it will be killed
        Sometimes the subprocesses can be still running if the Analysismanager has been killed and wasnt  given the time
        to shut down its subprocesses

        """
        pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]

        for pid in pids:
            try:
                s=open(os.path.join('/proc', pid, 'cmdline'), 'rb').read().decode("utf-8",'replace').replace('\x00',' ')
                s2='python3'+' '+pname+' '+pargs+' '
                if s.encode('UTF-8')==s2.encode('UTF-8'):
                    send_warning('Analysismanager: found Vagabonding Process: PID ='+repr(pid))
                    print('Analysismanager: found Vagabonding Process: PID ='+repr(pid))
                    print('Analysismanager: Vagabonding processname',s.encode('UTF-8'))
                    print('Analysismanager: killing:',pid)
                    os.kill(int(pid), signal.SIGKILL)
            except IOError: # proc has already terminated
                continue


        #print(sh.grep(sh.ps("x"), 'python3 '+pname+' '))


    def startProcess(self,name):

        self.endProcess(name)
        directory,script,category=self.filters[name]
        send_info(directory+' script: '+script+'name: '+name)
        arg='Analysis/'+category+'/'+name
        self.killVagabonding(directory+script,arg)
        self.processes[name]=subprocess.Popen(['python3',directory+script,'Analysis/'+category+'/'+name])



    def endProcess(self,name):

        if name in self.processes:
            send_info('terminating '+name)
            process=self.processes[name]
            process.kill()
            try:
                process.wait(timeout=0.5)
            except:
                send_warning("process termination failed killing process")
                process.kill()


    def updateStatus(self):
        ''' check wether processes are still running '''
        for name,process in self.processes.items():
            if process is not None:
                if process.poll()==None:
                    pass
                    #self.startButton.setStyleSheet("color: blue")
                else:
                    item=self.itemdict[name]
                    if item.checkState()==Qt.Checked:
                        item.setCheckState(False)
                    #self.startButton.setStyleSheet("color: red")
            else:
                send_warning('other '+name)
                self.startButton.setStyleSheet("color: gray")





    def addItem(self,direc,script, Filtername, active,category,streams):
        self.filters[Filtername]=[direc,script,category]
        model=self.model
        #model.insertRow(0)
        parent_item = model.invisibleRootItem()
        check_item = QtGui.QStandardItem(Filtername)
        check_item.setCheckable(True)
        check_item.setCheckState(Qt.Unchecked)
        parent_item.appendRow([check_item,
            QtGui.QStandardItem(script),
            QtGui.QStandardItem(category),
            QtGui.QStandardItem(','.join(streams))])
        if active:
            self.startProcess(Filtername)
            check_item.setCheckState(Qt.Checked)
        #imodel.setData(model.index(0, self.FROM), Filtername)
        #model.setData(model.index(0, self.SUBJECT), script)

    def del_current(self):
        ''' delete currently selected entry

        '''
        selmodel=self.dataView.selectionModel()
        indexlist=selmodel.selectedRows()
        for index in indexlist:
            delname=self.model.data(index)
            #check if the name is in the list of processes that have been started or stopped
            if delname in self.itemdict:
                self.endProcess(delname)
            if delname in self.processes:
                del self.processes[delname]
            if delname in self.itemdict:
                del self.itemdict[delname]
            category=self.model.data(self.model.index(index.row(),2))
            dic=self.props.get(category,{})
            del dic[delname]
            self.props.set(category,dic)
            #self.props.set(typ+'/'+name,{'script':script,'active':False})

            self.model.removeRow(index.row())



    def configureCurrent(self):
        selmodel=self.dataView.selectionModel()
        indexlist=selmodel.selectedRows()
        if indexlist != []:
            index=indexlist[0]
            name=self.model.data(index)
            category=self.model.data(self.model.index(index.row(),2))
            d = QDialog()
            d.setWindowTitle("Dialog")
            layout=QVBoxLayout()
            editor=PropEdit('/Analysis/'+category+'/'+name+'/')
            layout.addWidget(editor)
            d.setLayout(layout)
            d.exec_()

    def __del__(self):
        for name in self.processes:
            self.endProcess(name)

class addAnalysisWidget(QtWidgets.QWidget):

    def __init__(self,manager,parent=None):
        super().__init__(parent)
        self.manager=manager
        layout=QtWidgets.QHBoxLayout()
        addButton=QtWidgets.QPushButton('add')
        addButton.clicked.connect(self.addd)
        layout.addWidget(addButton)
        layout.addWidget(QtWidgets.QLabel('name'))
        self.nametext=QtWidgets.QLineEdit('')
        layout.addWidget(self.nametext)
        self.analysistype=QtWidgets.QComboBox()
        self.analysistype.addItem('Image')
        self.analysistype.addItem('Data')
        self.analysistype.currentTextChanged.connect(self.update_streamlist)
        layout.addWidget(self.analysistype)
        self.setLayout(layout)
        analysisdir=manager.analysisdir
        files = [f for f in os.listdir(analysisdir) if os.path.isfile(analysisdir+f) and f[0]!= '.' and
                ( f[-3:]=='.py' or f[-4:]=='.pyx')]
        self.analysisscript=QtWidgets.QComboBox()
        for f in files:
            self.analysisscript.addItem(f)
        layout.addWidget(self.analysisscript)
        self.streamlist=QComboBox()
        layout.addWidget(self.streamlist)
        self.update_streamlist('Image')
    def update_streamlist(self,category):

        for i in range(self.streamlist.count()): #remove all items first
            self.streamlist.removeItem(0)
        di= self.manager._props.get('/Servers/'+category+'Stream/active',{})
        for name,value in di.items():
            timedelta=time.time()-value['timestamp']
            self.streamlist.addItem(name+'[%i s]'%timedelta)
    def addd(self):
        name=self.nametext.text()
        category=self.analysistype.currentText()
        script=self.analysisscript.currentText()
        stream=self.streamlist.currentText()
        streams=[stream.split('[')[0]]
        self.manager._props.set(category+'/'+name,{'script':script,'active':False,category.lower()+'streams':streams})
        self.manager.tvw.addItem(self.manager.analysisdir,script,name,False,category,streams)
        #self.manager.initGUI()
class AnalysisManager(BWidget):
    ''' Manages the running of Analysis scripts
        Allows starting,stopping,configuring of the scripts properties
    '''


    def __init__(self,name='Analysis',parent=None):
        super().__init__(name,parent)
        self.analysisdir=self._props.get('analysisdir',tweezerpath+'/pytweezer/analysis/')
        self.initGUI()


    def initGUI(self):
        layout=QtW.QVBoxLayout()
        layout.addWidget(addAnalysisWidget(self))
        tvw=TreeViewWidget(self._props)
        self.tvw=tvw
        layout.addWidget(tvw)
        delButton=QPushButton('del')
        delButton.clicked.connect(self.delentry)
        layout.addWidget(delButton)
        configureButton=QPushButton('configure')
        configureButton.clicked.connect(self.configureFilter)
        layout.addWidget(configureButton)
        cmdlist=[]
        self.setLayout(layout)

        for category in ['Image','Data']:
            for name,param in sorted(self._props.get(category,{}).items(), key=lambda t:t[0].casefold()):
                try:
                    streams = self._props.get(category+'/'+name+'/imagestreams',['nostream'])
                except Exception as e:
                    print(' :( :( ANALYSISMANAGER: Script failed to start. ): ): ')
                try:
                    tvw.addItem(self.analysisdir,param['script'],name,param['active'],category,streams)
                except Exception as e:
                    print(' :( :( ANALYSISMANAGER: Script failed to start. ): ): ')
        self.setLayout(layout)
    def delentry(self):
        self.tvw.del_current()
    def configureFilter(self):
        self.tvw.configureCurrent()



def main():
    #def signal_handler(num, stack):
    #    print('Alarm in')

    #signal.signal(signal.SIGTERM, signal_handler)
    #signal.signal(signal.SIGINT, signal_handler)
    app = QtWidgets.QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile(icon_path + 'pytweezer_analysis_manager_icon.svg')
    app.setWindowIcon(icon)
    Win = AnalysisManager()
    Win.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
