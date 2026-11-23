"""A running applet re-subscribes when its subscriptions are changed from its context menu."""

import pytest
from PyQt6 import QtCore

from pytweezer.GUI import applet, image_applet, live_plot_applet, zmq_roi
from pytweezer.GUI.viewers import data_sidebar, image_monitor, image_plot_monitor
from pytweezer.GUI.viewers.live_plot import LivePlot


class FakeProps:
    def __init__(self, name=""):
        self.values = {}

    def get(self, key, default=None):
        return self.values.setdefault(key, default)

    def set(self, key, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)

    def changes(self, includeparent=True):
        return []


class RecordingStream:
    """A stream client that remembers what it is subscribed to."""

    def __init__(self, name="", **kwargs):
        self.channels = set()

    def subscribe(self, channel):
        self.channels.update([channel] if isinstance(channel, str) else channel)

    def unsubscribe(self, channel=None):
        self.channels.clear()

    def has_new_data(self, timeout_ms=1):
        return False


class FakeFeed(QtCore.QObject):
    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.connected = False

    def snapshot(self):
        return {}

    def held_points(self):
        return {}

    def points(self, rid):
        return []

    def close(self):
        pass


@pytest.fixture
def hubs(qapp, monkeypatch):
    for module, name, fake in [
        (applet, "Properties", FakeProps),
        (zmq_roi, "Properties", FakeProps),
        (image_applet, "ImageClient", RecordingStream),
        (image_applet, "DataClient", RecordingStream),
        (data_sidebar, "DataClient", RecordingStream),
        (image_plot_monitor, "DataClient", RecordingStream),
        (live_plot_applet, "DataClient", RecordingStream),
        (live_plot_applet, "ExperimentFeed", FakeFeed),
    ]:
        monkeypatch.setattr(module, name, fake)
    return monkeypatch


def choose_in_dialog(monkeypatch, props, key, value):
    """Make the next dialog act as if the user set Properties ``key`` to ``value`` in it."""
    monkeypatch.setattr(applet.QDialog, "exec", lambda dialog: props.set(key, value))


def click(menu, text):
    (action,) = [action for action in menu.actions() if action.text() == text]
    action.trigger()


@pytest.mark.parametrize(
    ("item", "key", "client", "streams"),
    [
        ("subscriptions", "imagestreams", "imgstream", ["cam2"]),
        ("Image masks", "maskstreams", "maskstream", ["mask2"]),
        ("Overlays", "overlaystreams", "overlayclient", ["fit2"]),
    ],
)
def test_an_image_monitor_resubscribes_from_its_context_menu(
    hubs, item, key, client, streams
):
    monitor = image_monitor.ImageDisplay("Test Monitor")
    choose_in_dialog(hubs, monitor.props, key, streams)

    click(monitor.view.plot.getViewBox().menu, item)

    assert getattr(monitor, client).channels == set(streams)


def test_a_projection_monitor_resubscribes_its_projections_from_its_context_menu(
    hubs,
):
    monitor = image_plot_monitor.ImageProjectionDisplay("Test Projections")
    choose_in_dialog(hubs, monitor.props, "projectionstreams", ["cloud"])

    click(monitor.view.plot.getViewBox().menu, "Projections")

    assert monitor.projectionclient.channels == {"cloud_x", "cloud_y"}


def test_a_live_plot_resubscribes_from_its_context_menu(hubs):
    plot = LivePlot("Test Plot")
    choose_in_dialog(hubs, plot.props, "plot0/datastreams", ["tweezer_loading"])

    click(plot.panels["plot0"].view.getViewBox().menu, "subscriptions")

    assert plot.panels["plot0"].stream.channels == {"tweezer_loading"}
