"""Base class and entry-point helper for pytweezer *applets*.

An **applet** is a small standalone process that runs locally on a lab PC,
subscribes to one or more data/image streams on the messaging fabric, and
displays them live. Applets are started and managed by the Applet Launcher
(:mod:`pytweezer.GUI.applet_launcher`); each is launched as
``python <script> <name>`` where ``name`` is both the label shown in the
launcher and the applet's Properties namespace.

Rather than have every applet script re-implement the same boilerplate
(Properties connection, window title, geometry persistence, an update timer,
and the subscription/configure context-menu dialogs), applets subclass
:class:`Applet` and override a few hooks. See ``docs/applets.md`` for the full
writeup.
"""

import argparse
import logging
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout

from pytweezer.servers import Properties
from pytweezer.GUI.subscription_editor import SubscriptionEditor
from pytweezer.GUI.property_editor import PropEdit
from pytweezer.GUI.theme import apply_theme, curve_color
from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.GUI.applet")

#: Window/taskbar icon for every applet, so a screenful of applets is
#: distinguishable from the main GUI at a glance.
ICON_PATH = os.path.join(os.path.dirname(__file__), "icons", "pytweezers_viewer_icon.svg")


class Applet(QtWidgets.QWidget):
    """Base class every pytweezer applet is built on.

    Handles the machinery common to all applets so subclasses only describe
    *what* they display:

    * a :class:`~pytweezer.servers.properties.Properties` connection under
      ``name`` (exposed as both ``self.props`` and ``self._props`` — the latter
      is what :class:`~pytweezer.servers.PropertyAttribute` descriptors read);
    * the window title (set to ``name``);
    * geometry persistence via ``QSettings("pytweezer", name)``;
    * a repeating poll timer that calls :meth:`poll`;
    * the shared "subscriptions" and "configure" context-menu dialogs.

    Subclasses typically set :attr:`stream_category`, build their widgets in
    :meth:`init_gui`, refresh subscriptions in :meth:`update_subscriptions`, and
    pull fresh stream data in :meth:`poll`.
    """

    #: ``"Image"`` or ``"Data"`` — which stream catalog the subscription editor
    #: lists, and (lower-cased + ``"streams"``) the default Properties key the
    #: subscription list is stored under.
    stream_category = "Data"

    #: Milliseconds between :meth:`poll` calls. Set to ``0`` to disable the
    #: timer (e.g. an applet driven purely by Qt signals).
    poll_interval = 10

    #: Fallback window size, used the first time an applet runs (before any
    #: geometry has been saved). Override :meth:`sizeHint` for a different one.
    default_size = (760, 560)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self._props = Properties(name)
        self.props = self._props
        self.setWindowTitle(name)
        self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self._restore_geometry()

        self.init_gui()

        if self.poll_interval:
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.poll)
            self._timer.start(self.poll_interval)

    # -- hooks for subclasses ---------------------------------------------

    def init_gui(self):
        """Build the applet's widgets. Called once, after ``self._props`` and
        the window title are set up. Override in subclasses."""

    def poll(self):
        """Pull any new stream data and refresh the display. Called every
        :attr:`poll_interval` ms. Override in subclasses."""

    def update_subscriptions(self):
        """Re-apply the stream subscriptions from Properties.

        Called automatically after the subscription or configure dialog closes
        (property edits may have changed the stream list). Override to
        unsubscribe/re-subscribe the applet's stream clients."""

    def sizeHint(self):
        return QtCore.QSize(*self.default_size)

    # -- display helpers ---------------------------------------------------

    def stream_color(self, stream, index=0):
        """Colour to draw ``stream`` in, as a :class:`QColor`.

        Reads the per-stream ``<stream>/color`` property, defaulting to the
        ``index``-th colour of the shared palette — so several streams on one
        plot are told apart without any configuration, while a user who wants a
        specific colour can still set it. Because ``props.get`` writes its
        default back, the colour appears in the configure dialog ready to edit
        after the first draw.
        """
        return QtGui.QColor(self.props.get(f"{stream}/color", curve_color(index)))

    # -- shared context-menu dialogs --------------------------------------

    def open_subscription_editor(self, streamkey=None):
        """Pop up the stream-subscription editor, then re-apply subscriptions.

        ``streamkey`` overrides which Properties key the list is stored under
        (defaults to ``<stream_category>.lower() + "streams"``). Applets that
        subscribe to more than one kind of stream (e.g. images + masks) pass a
        different ``streamkey`` per menu action."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.name} — subscriptions")
        layout = QVBoxLayout()
        editor = SubscriptionEditor(
            self._props, self.stream_category, streamkey=streamkey
        )
        layout.addWidget(editor)
        dialog.setLayout(layout)
        dialog.exec_()
        self.update_subscriptions()

    def open_config_editor(self):
        """Pop up the property editor for this applet's subtree, then re-apply
        subscriptions."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.name} — configure")
        layout = QVBoxLayout()
        editor = PropEdit("/" + self.name + "/")
        layout.addWidget(editor)
        dialog.setLayout(layout)
        dialog.exec_()
        self.update_subscriptions()

    # -- geometry persistence ---------------------------------------------

    def _restore_geometry(self):
        settings = QtCore.QSettings("pytweezer", self.name)
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        settings = QtCore.QSettings("pytweezer", self.name)
        settings.setValue("geometry", self.saveGeometry())
        # run_applet() exits the process hard (os._exit) right after the Qt loop
        # ends, so force the geometry write to disk now or it never lands.
        settings.sync()
        super().closeEvent(event)


def run_applet(applet_cls, default_name=None):
    """Standard applet ``main``: parse the ``name`` CLI arg (as passed by the
    Applet Launcher), construct ``applet_cls(name)``, and run the Qt loop.

    Use from an applet script's ``__main__`` guard::

        if __name__ == "__main__":
            run_applet(ImageDisplay)
    """
    if default_name is None:
        default_name = applet_cls.__name__
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        nargs="?",
        default=default_name,
        help="name of this applet instance (its Properties namespace)",
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    # An applet is its own process, so it inherits nothing from the main GUI's
    # QApplication — without this it would open in the platform's default light
    # style next to a dark control window.
    apply_theme(app)
    applet = applet_cls(args.name)
    applet.show()
    app.exec_()

    # The window has closed. The applet's Properties connection starts
    # *non-daemon* ZMQ threads that loop forever and would hang interpreter
    # shutdown, leaving the process alive — so the Applet Launcher's poll would
    # never see it exit and keep showing it as running. Exit hard (as bin/gui.py
    # does) so the process actually dies and the launcher marks it stopped.
    logging.shutdown()
    os._exit(0)
