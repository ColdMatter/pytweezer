
from PyQt5.QtWidgets import *
from pytweezer.servers import Properties,balipath,PropertyAttribute
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeyEvent
from pytweezer.analysis.print_messages import print_error


class BWidget(QWidget):
    """ base class of pytweezer Widgets
        includes logging of move and resize events"""

    def __init__(self,name='Noname',parent=None, create_props=True):
        super().__init__(parent)
        if create_props:
            self._props = Properties(name)
        self._name = name
        settings=QtCore.QSettings("pytweezer", self._name)
        try:
            self.restoreGeometry(settings.value("geometry"))
        except:
            print_error('geometry not found', 'warning')

    def closeEvent(self,event):
        settings=QtCore.QSettings("pytweezer", self._name)
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


class BFrame(QFrame):
    """ base class of pytweezer Widgets
        includes logging of move and resize events"""

    def __init__(self,name='Noname',parent=None):
        super().__init__(parent)
        self._props=Properties(name)
        self._name = name
        settings=QtCore.QSettings("pytweezer", self._name)
        try:
            #print(settings.value('geometry'))
            self.restoreGeometry(settings.value("geometry"))
        except:
            print_error('geometry not found', 'warning')

    def closeEvent(self,event):
        settings=QtCore.QSettings("pytweezer", self._name)
        settings.setValue("geometry", self.saveGeometry())
        #print('dddd',self.saveGeometry().toHex())
        super().closeEvent(event)

class BMainWindow(QMainWindow):
    def __init__(self,name='Noname',parent=None):
        super().__init__(parent)
        self._props=Properties(name)
        self._name = name
        settings=QtCore.QSettings("pytweezer", self._name)
        try:
            self.restoreGeometry(settings.value("geometry"))
        except:
            print_error('geometry not found', 'warning')
        self.setWindowTitle(name)

    def closeEvent(self,event):
        settings=QtCore.QSettings("pytweezer", self._name)
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


class SearchComboBox(QComboBox):
    """
    QComboBox with the autocompleter QCompleter enabled.
    This adds an editable QLineEdit which allows the contents of the combobox to be searched.
    Filtered list appears as a popup below the search box.
    The full list can be accessed by click the drop-down arrow.
    match_flag: can be either 'contains' or 'begins to get the matchFlag to MatchContains or MatchStartsWith.
    Has a custom QLineEdit called SearchLineEdit.
    """
    def __init__(self, parent=None, match_flag='contains'):
        super().__init__(parent)
        self.setLineEdit(SearchLineEdit(self))
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.completer().setCompletionMode(QCompleter.PopupCompletion)
        if match_flag == 'contains':
            self.matchFlag = Qt.MatchContains
        elif match_flag == 'begins':
            self.matchFlag = Qt.MatchStartsWith
        else:
            print('SearchComboBox: invalid matchflag set. Must be either contains or begins')
            print('Setting to default: contains')
            self.matchFlag = Qt.MatchContains
        self.completer().setFilterMode(self.matchFlag)
        self.setDuplicatesEnabled(False)

    def add_if_new(self, text):
        """if it's in the box, add it. return the index of the item"""
        idx = self.findText(text)
        if idx < 0:  # findText returns -1 if the item isn't in the combobox
            self.addItem(text)
            idx = self.findText(text)
        return idx

    def find_or_add(self, text):
        """if it's in the box, set it, if not, add and set it"""
        idx = self.add_if_new(text)
        self.setCurrentIndex(idx)


class SearchLineEdit(QLineEdit):
    """
    Custom QLineEdit for the SearchComboBox class.
    On focus in (e.g. when clicked on for the first time) the full text is selected.
    On focus out or pressing enter, the current text is stored. When pressing escape, the stored text is applied.
    On escape, reverts the
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.setText(self.lastText)
        elif event.key() == Qt.Key_Enter:
            self.lastText = self.text()
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        QTimer.singleShot(0, self.selectAll)  # ensures other events are processed first. Prevents UI locking up.

    def focusOutEvent(self, event):
        super().focusInEvent(event)
        self.lastText = self.text()
