from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import PyQt5.QtCore as Qt
from pytweezer.experiment.experiment import Experiment, NumberValue
from pytweezer.servers import Properties,PropertyAttribute
from pytweezer.servers import tweezerpath, icon_path
from pytweezer.GUI.pytweezerQt import BMainWindow
from pytweezer.analysis.print_messages import print_error


class DefaultExp(Experiment):
    ''' an experiment structure containing all attributes.

    '''
    def build(self):
        # self.setattr_argument("count", NumberValue(ndecimals=0, step=1,value=1))
        self.setattr_argument("count", NumberValue, ndecimals=0, step=1,value=1)
        groups={'ungrouped':[]}  #contains the grouping of the experiment
        hidden=[]
        for k,v in self._device_db.items():
            if v['type']=='attr':
                self.setattr_device(k)
            if v['type']=='generic_attr':
                if v['driver'] not in self.__dict__:  #load driver if not loaded
                    try:
                        self.setattr_device(v['driver'])
                    except:
                        print_error('no driver found: ' + str(v['driver']), 'error')
                self.setattr_device(k)
            #check for group
            if v['type']in ['attr','generic_attr']:
                if 'group' in v:
                    if v['group']!='hidden':

                        if v['group'] in groups:
                            groups[v['group']].append(k)
                        else:
                            groups[v['group']]=[k]
                    else:
                        hidden.append(k)
                else:
                    groups['ungrouped'].append(k)
        self.groups=groups
        self.hidden=hidden

    def run(self):
        pass

class BasicManager(QFrame):
    def __init__(self,props=None,parent=None,**kwargs):
        args_defaults={'unit':'','maxval':10,'minval':-10,'display_multiplier':1,'step':1,
                        'name':'','ndecimals':4,'default_value':0}
        self.__dict__.update(args_defaults)
        self.__dict__.update(kwargs)
        super().__init__(parent)
        self._props=props
        self.setStyleSheet("BasicManager {background-color: rgb(210,230,240);color:blue; margin:0px; border:2px solid rgb(220, 240, 255);} ")
        self._initGUI()

    def _initGUI(self):
        self.initSpin()
        if 'tooltip' in self.__dict__:
            self.setToolTip(self.tooltip)

        layout=QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)
        layout.addWidget(QLabel(self.parName))
        layout.addWidget(self.spin)
        self.setLayout(layout)

        timer=Qt.QTimer(self)
        if self._props:
            timer.timeout.connect(self.updateSpin)
        timer.start(1000)
        self.timer=timer

    def updateSpin(self):
        pass

class IntManager(BasicManager):
    '''Box for handling integers

    Args:
        props:(Properties)
            handle for the properties

        exp:(experiment)
            pointing to the experiment

        parname:(str
            name of the parameter

        maxval:  maximum value of spin
        minval:  minimum value of box
        display_multiplier:     multiply the value in the display with this value when setting the

        step:   stepsize  only relevant for floats

        tooltip: (str)
                Tooltip string

    '''
    def __init__(self,props=None,exp=None,parName='',parent=None,**kwargs):
        super().__init__(props,parent,parName=parName,exp=exp,**kwargs)






    def initSpin(self):
        self.spin=QSpinBox()
        default_val = self.__dict__.get('value', self.default_value)
        if self._props:
            val=self._props.get(self.parName,default_val)     #the values in properties are in SI units. Non SI only on disp
        else:
            val = default_val
        self.value=val
        self.spin.setValue(val/self._multiplier)
        self.updateValue(val/self.display_multiplier)
        self.spin.valueChanged.connect(self.updateValue)

    def updateValue(self,v):
        setattr(self.exp,self.parName,v*self.display_multiplier)
        self._props.set(self.parName,v*self.display_multiplier)
    def updateSpin(self):
        ''' updates the spin box if properties have changed '''
        val=self._props.get(self.parName,0)     #the values in properties are in SI units. Non SI only on disp
        if val != self.spin.value():
            self.spin.setValue(val/self.display_multiplier)



class BoolManager(QCheckBox):
    ''' GUI for Bool Values
    Args:
        props:(Properties)
            pointing to the properties

    KWArgs:
        exp:(experiment)
            handle for the experiment
        parName:(str)
            parameter Name
    '''
    def __init__(self,props=None,exp=None,parName='',parent=None,**kwargs):
        self.__dict__.update(kwargs)
        super().__init__(parName,parent)
        self._props=props
        self.exp=exp
        self.parName=parName
        self.name=parName
        self.stateChanged.connect(self.updateValue)
        default_val = bool(self.__dict__.get('value', False))
        if self._props:
            val=self._props.get(self.parName,default_val)     #the values in properties are in SI units. Non SI only on disp
        else:
            val = default_val
        self.setChecked(val)
        self.updateValue(val)

        timer=Qt.QTimer(self)
        timer.timeout.connect(self.updateCheckbox)
        timer.start(1000)
        self.timer=timer

    def updateValue(self,val):
        v=bool(val)
        setattr(self.exp,self.parName,v)
        self._props.set(self.parName,v)

    def updateCheckbox(self):
        ''' updates the spin box if properties have changed '''
        val=self._props.get(self.parName,False)     #the values in properties are in SI units. Non SI only on disp
        if val != self.isChecked():
            self.setChecked(val)


class FloatManager(IntManager):
    def initSpin(self):
        self.spin = QDoubleSpinBox()
        self.spin.setMaximum(self.maxval)
        self.spin.setMinimum(self.minval)
        self.spin.setDecimals(self.ndecimals)
        if self.unit:
            self.spin.setSuffix(' '+self.unit)
        self.spin.setSingleStep(self.step)
        default_val = self.__dict__.get('value', self.default_value)
        if self._props:
            print(self.parName)
            val=self._props.get(self.parName,default_val)     #the values in properties are in SI units. Non SI only on disp
        else:
            val = default_val
        self.spin.setValue(val/self.display_multiplier)
        self.updateValue(val/self.display_multiplier)
        self.spin.valueChanged.connect(self.updateValue)

class ComboManager(BasicManager):


    def __init__(self,props=None,exp=None,parName='',parent=None,argument=None,stringlist=None,**kwargs):
        ''' Manages Combo boxes
        either a list of string (stringlist) or an argument class containing a stringlist needs to be provided
        '''
        self.stringlist = stringlist if argument == None else argument.stringlist
        super().__init__(props,parent,parName=parName,**kwargs)



    def initSpin(self):
        self.spin = QComboBox()
        self.spin.InsertAtBottom
        for s in self.stringlist:
                self.spin.addItem(s)

        default_val = self.__dict__.get(
            'value', self.stringlist[0] if self.stringlist else ''
        )
        if self._props:
            current_val = self._props.get(self.parName, default_val)
        else:
            current_val = default_val

        current_index = self.spin.findText(current_val)
        if current_index < 0:
            current_index = 0
        if self.spin.count() > 0:
            self.spin.setCurrentIndex(current_index)
            self.updateValue(current_index)

        self.spin.activated.connect(self.updateValue)



    def updateValue(self,v):
        v = self.stringlist[v]
        setattr(self.exp,self.parName,v*self.display_multiplier)
        self._props.set(self.parName,v*self.display_multiplier)
    def updateSpin(self):
        ''' updates the spin box if properties have changed '''
        pass


class FrequencyManager(IntManager):
    """ Can handler frequency setting of DDS channels. The `SimpleDefaults` filters this out by name of the
    attributes."""

    def initSpin(self):
        self.spin = QDoubleSpinBox()
        self.spin.setMinimum(0)
        self.spin.setMaximum(499)
        self.spin.setDecimals(1)
        self.spin.setSuffix(' MHz')
        self.spin.setSingleStep(0.1)
        val=self._props.get(self.parName,0)
        self.spin.setValue(val)

    def updateValue(self,v):
        setattr(self.exp,self.parName,1e6*v)
        self._props.set(self.parName,1e6*v)
class MFrame(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)

class SimpleDefaults(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setStyleSheet("SimpleDefaults {background-color: rgb(200,220,225);color:blue; margin:0px; border:5px solid rgb(0, 0, 80);} ")

        self._props=Properties('Experiments/Defaults')
        self.exp=DefaultExp()
        exp=self.exp
        exp.build()
        mainlayout=QVBoxLayout()
        mainlayout.setSpacing(3)
        mainlayout.setContentsMargins(0,0,0,0)
        self.setLayout(mainlayout)
        pos=0
        positions=[(j,i) for j in range(30) for i in range(4)]
        for gk,gv in sorted(self.exp.groups.items())+[('hidden',[])]:
            layout=QGridLayout()
            layout.setSpacing(3)
            layout.setContentsMargins(0,0,0,0)
            frame=MFrame()
            frame.setStyleSheet("MFrame {background-color: rgb(180,200,225);color:red; margin:2px; border:3px solid rgb(0, 0, 80);} ")
            frame.setLayout(layout)
            mainlayout.addWidget(frame)
            layout.addWidget(QLabel(gk),0,0)
            pos=4
            #print(gk,gv)
            for typclass in [int,float,bool]:
                for k in sorted(gv):
                    v=self.exp._attributes[k]
                    #for k,v in sorted(self.exp._attributes.items()):
                    typ=type(getattr(exp,k))
                    if typ==typclass:
                        try:
                            if typ == int:
                                imgr=IntManager(self._props,exp,k)
                            elif typ == float:
                                    unit=type(exp).__dict__[k].display_unit
                                    maxval=type(exp).__dict__[k].maxval
                                    minval=type(exp).__dict__[k].minval
                                    multiplier=type(exp).__dict__[k].multiplier
                                    step=type(exp).__dict__[k].step
                                    imgr=FloatManager(self._props,exp,k,unit=unit,maxval=maxval,minval=minval,display_multiplier=multiplier,step=step)
                            elif typ == bool:
                                imgr=BoolManager(self._props,exp,k)
                            else:
                                send_error('Error: Unknown type' ,typ)
                            layout.addWidget(imgr,*positions[pos])
                        except Exception as e:
                            print_error('Defaults failed to load ' + k + ';' + str(e), 'error')
                            layout.addWidget(QLabel(k),*positions[pos])
                        pos=pos+1
                pos=pos-pos%4+4
        for pos,name in enumerate(self.exp.hidden):
            layout.addWidget(QLabel(name),*positions[pos+4])
        #self.setStyleSheet("background-color:#00004d;")

def main():
    qApp = QApplication(sys.argv)
    icon = QtGui.QIcon()
    icon.addFile(icon_path + 'pytweezer_simple_default_icon.svg')
    qApp.setWindowIcon(icon)
    Win = SimpleDefaults()
    Win.show()
    qApp.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
