import weakref

from PyQt6 import QtCore

from pytweezer.GUI.background import BackgroundReader
from tests.helpers import settle


class Owner(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.reader = BackgroundReader(self)
        self.outcomes = []

    def read(self, read):
        # As the panels' callbacks do, this one refers to its owner.
        self.reader.start(
            "Test read", read, lambda outcome: self.outcomes.append(outcome)
        )


def unreadable():
    raise OSError("file gone")


def test_a_kept_exception_does_not_keep_its_owner_alive(qapp):
    # Otherwise the owner lives in a cycle through the reading thread's frames,
    # and the garbage collector may free its widgets off the GUI thread.
    owner = Owner()
    owner.read(unreadable)
    settle(qapp, "Test read")
    (error,) = owner.outcomes
    assert isinstance(error, OSError)

    gone = weakref.ref(owner)
    del owner, error
    assert gone() is None


def test_an_outcome_after_its_owner_is_destroyed_is_dropped(qapp):
    owner = Owner()
    reader, outcomes = owner.reader, owner.outcomes
    released = QtCore.QSemaphore()
    reader.start("Test read", lambda: released.acquire() or 1, outcomes.append)
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    released.release()
    settle(qapp, "Test read")
    assert outcomes == []
