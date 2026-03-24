""" Monitor the content of streams """
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pytweezer.servers import DataClient,ImageClient,CommandClient
from pytweezer.servers.messageclient import MessageClient
from pytweezer.GUI.pytweezerQt import BWidget


class StreamMonitor(QWidget):
    def __init__(self,name,streamtype='Data',parent=None):
        super().__init__(parent)
        if streamtype=='Data':    
            self.stream=DataClient(name)
        elif streamtype=='Image':
            self.stream=ImageClient(name)
        elif streamtype =='Command':
            self.stream=CommandClient(name)
        elif streamtype =='Message':
            self.stream=MessageClient(name)
        self.stream.subscribe('')       #listen to all streams
        self.msglist=[]

        layout=QVBoxLayout()
        layout.addWidget(QLabel(name))
        self.text=QTextEdit()
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.resize(800,800)

        timer=QtCore.QTimer(self)
        timer.timeout.connect(self._update_list)
        timer.start(100)
        self.timer=timer

    def _update_list(self):
        
        while self.stream.has_new_data():
            msg=self.stream.recv()
            if msg != None:
                self.msglist=[msg[0]+repr(msg[1])[:80]]+self.msglist
                self.msglist=self.msglist[:40]
                self.text.setPlainText('\n'.join(self.msglist))



def main(name):
    qApp = QApplication(sys.argv)
    Win = QTabWidget()
    image = StreamMonitor(name,'Image')
    Win.addTab(image,"Image")
    image = StreamMonitor(name,'Data')
    Win.addTab(image,"Data")
    image = StreamMonitor(name,'Command')
    Win.addTab(image,"Command")
    image = StreamMonitor(name,'Message')
    Win.addTab(image,"Message")
    Win.show()
    qApp.exec_()




if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main('StreamMonitor')









