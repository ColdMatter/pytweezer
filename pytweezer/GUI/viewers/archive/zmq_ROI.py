import pytweezer as bc
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *

import pyqtgraph as pg
from pytweezer.servers import Properties,PropertyAttribute

class zmq_ROI(pg.ROI):
    """ Region of Interest  which communicates its properties through BaLis central Property class

    Args:
        name (string): Unique name of the ROI.
            Non unique names will share their properties. You MUST ensure the name
            doesnt collide with any other name in the balis property dictionary

        kwargs (arguments):  see pyqtgraph ROI documentation

    """
    roilist=PropertyAttribute('/Servers/roilist',[])    
    p_pos=PropertyAttribute('pos',[0,0])
    p_size=PropertyAttribute('size',[30,30])
    color=PropertyAttribute('color',int(255*256**3+255))

    def __init__(self,name, **kwargs):
        #kw={key: value for key, value in kwargs.iteritems() 
        #        if not (key in ['dada'])}
        self.name=name
        self._props=Properties(name)
        roilist=self.roilist
        if '/'+self.name not in roilist:
            roilist.append('/'+self.name)
            self.roilist=roilist
        
        kwargs.pop('streamname',None)
        super(zmq_ROI,self).__init__(tuple(self.p_pos),size=tuple(self.p_size),**kwargs)
        self.update_from_properties() 
        timer=QtCore.QTimer(self)
        timer.timeout.connect(self._publish_values)
        timer.start(1000)
        self.timer=timer
        self.oldpos= [self.pos().x(),self.pos().y(),self.size().x(),self.size().y()]

    def update_from_properties(self):
        col=QColor(self.color)
        self.setPen(pg.mkPen(col, width=1))
        self.setPos(self.p_pos)
        self.setSize(self.p_size)

    def _publish_values(self):
        newpos= [self.pos().x(),self.pos().y(),self.size().x(),self.size().y()]
        if self.oldpos !=newpos:
            self.p_pos=newpos[:2]
            self.p_size=newpos[2:]
        elif '/'+self.name in self._props.changes():
            self.update_from_properties()
            newpos= [self.pos().x(),self.pos().y(),self.size().x(),self.size().y()]
        
        self.oldpos=newpos

