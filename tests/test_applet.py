from pytweezer.GUI import applet


class FakeProps:
    def __init__(self, name=""):
        self.values = {}

    def get(self, key, default=None):
        return self.values.setdefault(key, default)


def test_an_applet_subscribes_once_after_init_gui_has_built_its_widgets(
    qapp, monkeypatch
):
    monkeypatch.setattr(applet, "Properties", FakeProps)
    calls = []

    class Viewer(applet.Applet):
        poll_interval = 0

        def init_gui(self):
            self.subscribed_streams = []
            calls.append("init_gui")

        def update_subscriptions(self):
            self.subscribed_streams.append("stream")
            calls.append("update_subscriptions")

    viewer = Viewer("Test Applet")

    assert calls == ["init_gui", "update_subscriptions"]
    assert viewer.subscribed_streams == ["stream"]
