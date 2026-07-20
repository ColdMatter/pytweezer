from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QDateTime

"""
Here we define the models Queue and PrepStation tables.

A model tells a table what form its data will take (e.g. a dict or list), and how to present and manipulate that data.
For example, how to get, set, and delete data, and how the data should be sorted.

The model can both read from its data set, and edit the data set. So if you edit the data directly, the table will 
update to reflect the change (provided the layoutChanged signal is emitted). Or if you delete an element
from the table, it will be deleted from the data set.
"""


class _SyncSubstruct:
    """
    Defines methods for manipulating a table data dictionary.
    Based on the ARTIQ models. Copied+pasted without much adaptation.
    """
    def __init__(self, update_cb, ref):
        self.update_cb = update_cb
        self.ref = ref

    def append(self, x):
        self.ref.append(x)
        self.update_cb()

    def insert(self, i, x):
        self.ref.insert(i, x)
        self.update_cb()

    def pop(self, i=-1):
        self.ref.pop(i)
        self.update_cb()

    def __setitem__(self, key, value):
        self.ref[key] = value
        self.update_cb()

    def __delitem__(self, key):
        self.ref.__delitem__(key)
        self.update_cb()

    def __getitem__(self, key):
        return _SyncSubstruct(self.update_cb, self.ref[key])


class DictSyncModel(QtCore.QAbstractTableModel):
    """
    Base class for turning a dictionary (backing_store) into a table.
    Based on the ARTIQ models. Copied+pasted without much adaptation.
    """
    def __init__(self, headers, dataNames, init):
        self.headers = headers
        self.dataNames = dataNames
        self.backing_store = init  # the table data dictionary
        self.row_to_key = sorted(  # a list of the dictionary keys in the order in which they should be displayed
            self.backing_store.keys(),
            key=lambda k: self.sort_key(k, self.backing_store[k]))
        QtCore.QAbstractTableModel.__init__(self)

    def rowCount(self, parent=None):
        return len(self.backing_store)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        else:
            k = self.row_to_key[index.row()]
            return self.convert(k, self.backing_store[k], index.column())

    def setData(self, index, value, role=Qt.DisplayRole):
        if value == '':
            return False
        if role == Qt.EditRole:
            k = self.row_to_key[index.row()]
            col = index.column()
            if col == 7:
                value = value.toString()
            if col in (4, 5, 6):
                value = int(value)
            self.backing_store[k][self.dataNames[col]] = value
            return True

    def headerData(self, col, orientation, role=Qt.DisplayRole):
        if (orientation == Qt.Horizontal and
                role == Qt.DisplayRole):
            return self.headers[col]
        return None

    def _find_row(self, k, v):
        lo = 0
        hi = len(self.row_to_key)
        while lo < hi:
            mid = (lo + hi)//2
            if (self.sort_key(self.row_to_key[mid],
                              self.backing_store[self.row_to_key[mid]]) <
                    self.sort_key(k, v)):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def __setitem__(self, k, v):
        if k in self.backing_store:
            old_row = self.row_to_key.index(k)
            new_row = self._find_row(k, v)
            if old_row == new_row:
                self.dataChanged.emit(self.index(old_row, 0),
                                      self.index(old_row, len(self.headers)-1))
            else:
                self.beginMoveRows(QtCore.QModelIndex(), old_row, old_row,
                                   QtCore.QModelIndex(), new_row)
            self.backing_store[k] = v
            self.row_to_key[old_row], self.row_to_key[new_row] = \
                self.row_to_key[new_row], self.row_to_key[old_row]
            if old_row != new_row:
                self.endMoveRows()
        else:
            row = self._find_row(k, v)
            self.beginInsertRows(QtCore.QModelIndex(), row, row)
            self.backing_store[k] = v
            self.row_to_key.insert(row, k)
            self.endInsertRows()

    def __delitem__(self, k):
        row = self.row_to_key.index(k)
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        del self.row_to_key[row]
        del self.backing_store[k]
        self.endRemoveRows()

    def __getitem__(self, k):
        def update():
            self[k] = self.backing_store[k]
        return _SyncSubstruct(update, self.backing_store[k])

    def sort_key(self, k, v):
        raise NotImplementedError

    def convert(self, k, v, column):
        raise NotImplementedError

    def flags(self, index):
        raise NotImplementedError


class ScheduleModel(DictSyncModel):
    """
    This is the table model used by the ExperimentQ, with specific rules for the column headers etc.
    The data structure is a dictionary of dictionaries. Each dictionary is a single experiment "task".
    The keys of the main dictionary are a unique ID called the task number.
    Items are sorted first by their "priority" value, then by the task number.
    """
    def __init__(self, init):
        headers = ["taskNr", "label", "experiment", "status", "rep", "run", "priority", "due"]
        dataNames = ["task", "label", "expName", "status", "repetition", "run", "priority", "dueDateTime"]
        super().__init__(headers, dataNames, init)

    def sort_key(self, k, v):
        return -v["priority"], k

    def convert(self, k, v, column):
        """
        converts dictionary keys to columns
        """
        if column == 0:
            return k
        if column == 7:
            return QDateTime.fromString(v[self.dataNames[7]])
        elif column in range(1, 7):
            return v[self.dataNames[column]]
        else:
            raise ValueError

    def flags(self, index):
        if index.column() in (4, 5, 6, 7):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable


class ListSyncModel(QtCore.QAbstractTableModel):
    """
    Like DictSyncModel, but handles a list instead.
    """
    def __init__(self, headers, dataNames, init):
        super().__init__()
        self.backing_store = init
        self.headers = headers
        self.dataNames = dataNames

    def rowCount(self, parent=None):
        return len(self.backing_store)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        elif role == Qt.DisplayRole or role == Qt.EditRole:
            row = index.row()
            return self.convert(row, self.backing_store[row], index.column())
        else:
            return None

    def setData(self, index, value, role=Qt.DisplayRole):
        if value == '':
            return False
        if role == Qt.EditRole:
            k = index.row()
            col = index.column()
            if col == 7:
                value = value.toString()
            if self.backing_store[k]["status"] != 'Sleeping':
                if QDateTime.fromString(self.backing_store[k]["dueDateTime"]) <= QDateTime.currentDateTime():
                    self.backing_store[k]["status"] = 'Queued'
                else:
                    self.backing_store[k]["status"] = 'Waiting'
            self.backing_store[k][self.dataNames[col]] = value
            return True

    def headerData(self, col, orientation, role=Qt.DisplayRole):
        if (orientation == Qt.Horizontal and
                role == Qt.DisplayRole):
            return self.headers[col]
        return None

    def __delitem__(self, k):
        self.beginRemoveRows(QtCore.QModelIndex(), k, k)
        del self.backing_store[k]
        self.endRemoveRows()

    def __getitem__(self, k):
        return self.backing_store[k]

    def sort_key(self, k, v):
        raise NotImplementedError

    def convert(self, k, v, column):
        raise NotImplementedError

    def flags(self, index):
        raise NotImplementedError


class PrepModel(ListSyncModel):
    """
    This is the table model used by the PrepStation. Here we use a list, as the tasks in PrepStation don't need
    unique IDs and the order can be freely changed.
    """
    def __init__(self, init):
        headers = ["taskNr", "label", "experiment", "status", "nReps", "nRuns", "priority", "due"]
        dataNames = ["task", "label", "expName", "status", "nReps", "nRuns", "priority", "dueDateTime"]
        super().__init__(headers, dataNames, init)

    def convert(self, row, v, column):
        if column == 0:
            return row
        if column == 7:
            return QDateTime.fromString(v[self.dataNames[7]])
        elif column in range(1, 7):
            return v[self.dataNames[column]]
        else:
            raise ValueError

    def flags(self, index):
        if index.column() in (2, 4, 6, 7):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
