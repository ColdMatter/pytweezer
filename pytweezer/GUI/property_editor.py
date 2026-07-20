from os.path import isfile

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QFileDialog, QListWidget, QListWidgetItem
import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from math import *
import numpy as np

from pytweezer.servers import Properties,tweezerpath,send_error,send_info
from pytweezer.GUI.table_parameter import *
from pytweezer.GUI.pytweezerQt import BMainWindow, BWidget
from PyQt5 import QtWidgets

import traceback
import json
import shutil
import signal
from datetime import datetime
from pytweezer.analysis.print_messages import print_error

app = QtWidgets.QApplication([])
## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
## test add/remove
## this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))

class TreeEdit():
    ''' Extract ionformation from Properties class and format accordig to treeview'''
    def __init__(self,name,parent,props=None, subtree='/'):
        self.parent=parent
        self.subtree=subtree
        self.props=Properties(name) if props==None else props

        #extract entries from property object and create parametergroup
        proptree = self.props.get('/') if self.subtree =='/' else self.props.get(self.subtree[:-1])

        # create pyqtGraph Parametergroup
        self.pGroup = Parameter.create(name='params', type='group', children=self._parsedict(proptree))
        self.pGroup.sigTreeStateChanged.connect(self.change)


    def _parsedict(self,dictionary):
        ''' format dictionary to be viewed in tree view

        :return :   list of dictionaries
        '''
        children=[]
        for key,value in dictionary.items():
            if type(value).__name__ in ['int','bool']:
                if key[-5:]=='color':
                    #color lists identified by name ending with color
                    #children.append({'name':key,'type':'color','value':tuple(value)})
                    children.append({'name':key,'type':'color',
                            'value':[int(value/256**i%256) for i in [2,1,0,3]]})
                else:
                    children.append({'name':key,'type':type(value).__name__,
                        'value':value,'decimals':6})
            elif type(value).__name__ in ['str']:
                if key[-18:]=='Region_of_Interest':
                    children.append({'name':key,'type':'list',
                        'value':value,'values':[value]+self.props.get('/Servers/roilist',[])})
                # elif key == 'Configuration_name':
                #     children.append({'name':key,'type':'list',
                #         'value':'foo','values':[value]+self.props.get('/Servers/configlist',[])})
                else:
                    children.append({'name':key,'type':type(value).__name__,
                        'value':repr(value),'decimals':6})
            elif type(value).__name__ in ['float']:
                children.append({'name':key,'type':type(value).__name__,
                                'value':value,
                                'step':10**floor(log10(abs(value/1000 if value !=0 else 1))),'decimals':10})
            elif type(value) ==list:
                    children.append({'name':key,'type':'str','value':repr(value)})
            elif type(value) == dict:
                if 'options' in value.keys():
                    children.append({'name':key,'type':'list',
                             'value':value['value'],'values':value['options']})
                else:
                    children.append({'name':key,'type':'group','children':self._parsedict(value)})
        return children

    ## If anything changes in the tree, this function is called
    def change(self,params,changes):
        ''' slot for any changes in the tree '''
        for param, change, data in changes:
            # print('set:')
            # print(param)
            # print(param.type())
            # print(param.value())
            # print(change)
            # print(data)

            path = self.pGroup.childPath(param)
            try:
                # if the change results from deleting the path is None
                if path is not None:
                    childName = '/'.join(path)
                    print('  parameter: %s'% childName)
                    print('  change:    %s'% change)
                    print('  data:      %s'% str(data))
                    print('  ----------')
                    if param.type() == 'str':
                        try:
                            edat=eval(data)
                            self.props.set(self.subtree+childName,edat)
                        except:
                            send_error('could not convert')
                            print_error('could not convert', 'error')
                            param.opts['data']=param.opts['default']
                    elif param.type() in ['int','bool','float']:
                        self.props.set(self.subtree+childName,data)

                    elif param.type() == 'color':
                        #print(data.rgba(),type(data.rgba()))
                        self.props.set(self.subtree+childName,data.rgba())
                    elif param.type() == 'list':
                        #print('property_editor.py', data)
                        self.props.set(self.subtree+childName,data)
                    # elif param.type() == 'dict':
                    #     #print('property_editor.py', data)
                    #     self.props.set(self.subtree+childName,data)


                    if False:
                        listname=childName[:childName.rfind('/')]
                        listindex=int(childName.split('/')[-1][5:])
                        #print('list modified', listindex)
                        l=self.props.get('/'+listname)
                        l[listindex]=data
                        self.props.set('/'+listname,l)
                        #print('setting',listname,'dd',l)

            except Exception as e:
                send_error('change error')
                print_error('property_editor change error {0}'.format(str(e)), 'error')
                print('sysinfo:',sys.exc_info()[0],path)
                print('sysinfo:',sys.exc_info()[1],path)
                traceback.print_tb(sys.exc_info()[2])



    def _update(self,path):
        """ update selected entry

        path: path in properties to updated entry
        """
        #find parameter
        par=self.pGroup
        foundpath=''
        for entry in path.split('/')[1:]:
            #print('ppath',entry)
            for child in par.children():
                if child.name()==entry:
                    par=child
                    foundpath=foundpath+'/'+entry
                    break
            else:
                break


        parent=par.parent()
        if parent is not None:
            parent.removeChild(par)
        else:
            parent=par
            foundpath='/'+path.split('/')[1:][0]
        #print('fount',par.name(),foundpath)
        #print('ppp',foundpath,'  ',path)
        subtree= self._parsedict({foundpath.split('/')[-1]:self.props.get(self.subtree+foundpath[1:])})[0]
        pp=Parameter.create(**subtree)
        parent.addChild(pp)

    def _update_whole_tree(self):
        subtree= self._parsedict({'irrelevant':self.props.get(self.subtree)})[0]
        pp=Parameter.create(**subtree)
        grp=self.pGroup
        for par in grp.children():
            par.remove()
        for par in pp.children():
            grp.addChild(par)
        self.parent.pTree.expandToDepth(0)

    def update(self):
        changes=self.props.changes(includeparent=False)
        for change in changes:
            if change==self.subtree:
                #print('update whole tree')
                self._update_whole_tree()
            elif len(change)> len(self.subtree) and change[:len(self.subtree)]==self.subtree:
                #print('change in subtree', self.subtree,change)
                self._update(change[len(self.subtree)-1:])

## Create tree of Parameter objects
class PropEdit(BWidget):
    def __init__(self,subtree='/',parent=None, ):
        """GUI Window for the property tree

        :subtree : (string)
            show only properties on this path
        """
        print('PropertyEditor starting')
        name='GUI/PropertyEditor'
        super().__init__(name,parent)
        if subtree[-1]!='/':
            subtree=subtree+'/'
        self.subtree=subtree
        #print('property_editor: subtree',subtree)
        self.props=Properties(name)

        # layout = QtGui.QGridLayout()  # deprecated
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        tree=TreeEdit(name,self,self.props,self.subtree)
        self.tree=tree
        paramTree = ParameterTree()
        paramTree.setParameters(tree.pGroup, showTop=False)
        # provide a better overview by collapsing the tree by default
        # and sorting it in ascending order
        paramTree.expandToDepth(0)
        paramTree.setSortingEnabled(True)
        paramTree.sortItems(0, QtCore.Qt.AscendingOrder)
        self.pTree=paramTree
        layout.addWidget(paramTree, 1, 0, 1, 1)

        # The update button is the first since it will be selected by default and one might press enter after editing.

        # Button to delete current property trre entry
        # upButton=QtGui.QPushButton('update')  # deprecated
        upButton = QtWidgets.QPushButton('update')
        upButton.clicked.connect(self.update)
        layout.addWidget(upButton)

        # Button to delete current property trre entry
        # delButton=QtGui.QPushButton('del')  # deprecated
        delButton = QtWidgets.QPushButton('del')
        delButton.clicked.connect(self.delete)
        layout.addWidget(delButton)

        # upButton=QtGui.QPushButton('save')  # deprecated
        upButton = QtWidgets.QPushButton('save')
        upButton.clicked.connect(self.save)
        layout.addWidget(upButton)

        # upButton=QtGui.QPushButton('load')  # deprecated
        upButton = QtWidgets.QPushButton('load')
        upButton.clicked.connect(self.load)
        layout.addWidget(upButton)
        #timer=QtCore.QTimer(self)
        #timer.timeout.connect(self.update)
        #timer.start(1000)
        #self.timer=timer

        self.show()
        self.resize(800,800)
        g = self.geometry()
        geo = self.props.get('Geometry', g.getRect())
        self.setGeometry(*geo)

    def update(self):
        self.tree.update()

    def save(self):
        ''' save the current properties to a file'''
        fname = QFileDialog.getSaveFileName(self, 'Save into file',
            tweezerpath+'/configuration/user',"json dictionaries (*.json)")[0]
        #ensure fname ends with .json
        if fname[-5:]!='.json':
            fname=fname+'.json'
        #print('saving into '+fname)
        #print(self.props.get('/'))
        try:
            with open(fname, "w") as outfile:
                json.dump(self.props.get(self.subtree), outfile, indent=4)
        except:
            send_error('saving failed '+fname)
            print_error('property_editor.py: saving file failed', 'error')

    def load(self):
        fname = QFileDialog.getOpenFileName(self,'Load Properties',tweezerpath+'/configuration/user','json dictionaries, (*.json)')[0]
        if fname[-5:]!='.json':
            fname=fname+'.json'
        #print(fname)
        try:
            with open(fname) as inputfile:
                prop=json.load(inputfile)
        except:
            send_error('error loading '+fname)
            print_error('error loading {0}'.format(fname), 'error')
            prop={}
        for k,v in prop.items():
            #print('setting: ',self.subtree+k)
            self.props.set(self.subtree+k,v)

    def delete(self):
        """ delete currently selected item from GUI tree and properties
        """
        selitems=self.pTree.selectedItems()
        for item in selitems:

            # find path to selected item
            param= item.param
            path=[]
            while param is not None:
                path.append(param.name())
                param=param.parent()
            keys=self.subtree+'/'.join(reversed(path[:-1]))

            #print('PROPERTY EDITOR DELETE',keys)
            # delete item from GUI
            item.param.remove()

            # delete item from properties
            #print('Deleting ',keys)
            self.props.delete(keys)

    def closeEvent(self, event):
        print('event...')
        g = self.geometry()
        self.props.set('Geometry', g.getRect())

        super().closeEvent(event)

class PropSelector(BWidget):
    def __init__(self, name, props, subtree='/', parent=None):
        """GUI Window for the property tree

        :subtree : (string)
            show only properties on this path
        """
        print('Property Selector starting')
        #name='GUI/PropertySelector'
        super().__init__(name,parent)
        if subtree[-1] != '/':
            subtree = subtree+'/'
        self.subtree = subtree
        #print('property_editor: subtree',subtree)
        self.props = Properties(name)
        #self.props.set('keyList',[])
        self.keyList = self.props.get('keyList', [''])

        # layout = QtGui.QGridLayout()  # deprecated
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        tree=TreeEdit(name, self, self.props, subtree=self.subtree)
        self.tree=tree
        paramTree = ParameterTree()
        paramTree.setParameters(tree.pGroup, showTop=False)
        # provide a better overview by collapsing the tree by default
        # and sorting it in ascending order
        paramTree.expandToDepth(0)
        paramTree.setSortingEnabled(True)
        paramTree.sortItems(0, QtCore.Qt.AscendingOrder)
        self.pTree=paramTree
        layout.addWidget(paramTree, 1, 0, 1, 1)

        # selButton=QtGui.QPushButton('select')  # deprecated
        selButton = QtWidgets.QPushButton('select')
        selButton.clicked.connect(self.select)
        layout.addWidget(selButton)

        self.qList=QListWidget()
        self.updateKeys()
        layout.addWidget(self.qList)

        # delButton=QtGui.QPushButton('del')  # deprecated
        delButton = QtWidgets.QPushButton('del')
        delButton.clicked.connect(self.delete_marked)
        layout.addWidget(delButton)


    def select(self):
        """ select currently selected item from GUI tree and properties
        """
        selitems = self.pTree.selectedItems()
        for item in selitems:
            # find path to selected item
            param = item.param
            path = []
            while param is not None:
                path.append(param.name())
                param = param.parent()
            keys = ''+'/'.join(reversed(path[:-1]))
            self.keyList.append(keys)
            self.props.set('keyList', self.keyList)
            self.updateKeys()

    def updateKeys(self):
        self.qList.clear()
        for key in self.keyList:
            QListWidgetItem(key, self.qList)
        #self.parent.keyList = self.keyList

    def delete_marked(self):
        row=self.qList.currentRow()
        if row >=0:
            keys=self.props.get('keyList',[''])
            del keys[row]
            self.props.set('keyList',keys)
            self.keyList = keys
        self.updateKeys()

# ## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys

#     pp=PropEdit(name='Property Editor')
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtWidgets.QApplication.instance().exec_()
#     else:
#         print('sys.flag.interactive ',sys.flag.interactive, 'hasattr(OtCore)',hasattr(OtCore,'PYQT_VERSION'))

def main():
    app = QApplication(sys.argv)
    Win = PropEdit()
    Win.show()
    app.exec_()


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
