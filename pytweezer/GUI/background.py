"""Run slow reads off the GUI thread and show their outcomes back on it."""

import threading
from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore


class BackgroundReader(QtCore.QObject):
    """Runs each read on a daemon thread and passes its outcome to a callback on the GUI thread.

    The outcome is the read's return value, or the exception it raised. The
    reader is deliberately not parented to ``owner``: a running read keeps it
    alive, so its thread never signals a deleted object, and outcomes that
    arrive after ``owner`` is destroyed are dropped.

    An exception's tracebacks are cleared once its callback returns. They hold
    the reading thread's frames, and through them the callback and its owner;
    an owner that kept the exception would then live in a reference cycle,
    which the garbage collector may break on any thread, deleting widgets off
    the GUI thread (a segfault).

    Args:
        owner: The widget whose callbacks are called; create the reader on the
            GUI thread.
    """

    _finished = QtCore.pyqtSignal(object, object)

    def __init__(self, owner: QtCore.QObject):
        super().__init__()
        self._owner_alive = True
        owner.destroyed.connect(self._owner_destroyed)
        self._finished.connect(self._deliver)

    def start(
        self, name: str, read: Callable[[], Any], callback: Callable[[Any], None]
    ):
        """Run ``read()`` on a thread called ``name``; then ``callback(outcome)`` here."""

        def run():
            try:
                outcome = read()
            except Exception as error:
                outcome = error
            self._finished.emit(callback, outcome)

        threading.Thread(target=run, daemon=True, name=name).start()

    def _owner_destroyed(self):
        self._owner_alive = False

    def _deliver(self, callback, outcome):
        try:
            if self._owner_alive:
                callback(outcome)
        finally:
            if isinstance(outcome, BaseException):
                _clear_tracebacks(outcome)


def _clear_tracebacks(error: BaseException) -> None:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        error.__traceback__ = None
        error = error.__cause__ or error.__context__
