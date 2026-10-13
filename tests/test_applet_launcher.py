"""The applet *list* is shared via Properties; *which applets are running* is
per-machine and lives in local QSettings. Launchers are built via ``__new__`` so
no Properties hub is contacted and no applet subprocesses are spawned.
"""

import pytest

from pytweezer.GUI.applet_launcher import AppletLauncher


class FakeProps:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, defaultvalue=None):
        return self.data.get(key, defaultvalue)

    def set(self, key, value):
        self.data[key] = value


class FakeProcess:
    def __init__(self, exited=False):
        self.terminated = exited

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0 if self.terminated else None


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    """Point QSettings at a temp dir so tests never touch real user settings."""
    from PyQt6.QtCore import QSettings

    previous = QSettings.defaultFormat()
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    yield tmp_path
    QSettings.setDefaultFormat(previous)


def make_launcher(name="TestLauncher", applets=None, active=None, props=None, qt=False):
    """``qt=True`` creates the underlying QWidget, which ``closeEvent`` needs."""
    launcher = AppletLauncher.__new__(AppletLauncher)
    if qt:
        from PyQt6 import QtWidgets

        QtWidgets.QWidget.__init__(launcher)
    launcher._name = name
    launcher._props = props if props is not None else FakeProps()
    launcher._applets = dict(applets or {})
    launcher._active = set(active or ())
    launcher._processes = {}
    launcher._rows = {}
    launcher._set_status = lambda *args, **kwargs: None
    return launcher


def saved_active(name="TestLauncher"):
    return make_launcher(name=name)._load_active()


def running_launcher(qt=False):
    """A launcher with "Image Monitor" running and persisted as active."""
    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}}, active={"Image Monitor"}, qt=qt
    )
    launcher._processes["Image Monitor"] = FakeProcess()
    launcher._save_active()
    return launcher


# QSettings collapses a one-element list to a bare string and an empty list to None.
@pytest.mark.parametrize(
    "active", [{"Live Plot", "Image Monitor"}, {"Image Monitor"}, set()]
)
def test_active_set_round_trips_through_local_settings(active):
    make_launcher(active=active)._save_active()
    assert saved_active() == active


def test_active_set_is_per_launcher_name():
    assert saved_active("NeverRunBefore") == set()
    make_launcher(name="LauncherA", active={"Image Monitor"})._save_active()
    make_launcher(name="LauncherB", active={"Live Plot"})._save_active()
    assert saved_active("LauncherA") == {"Image Monitor"}
    assert saved_active("LauncherB") == {"Live Plot"}


def test_set_active_persists_locally_and_not_to_properties():
    props = FakeProps()
    launcher = make_launcher(applets={"Image Monitor": {"script": "a.py"}}, props=props)

    launcher._set_active("Image Monitor", True)
    assert launcher._active == saved_active() == {"Image Monitor"}

    launcher._set_active("Image Monitor", False)
    assert launcher._active == saved_active() == set()

    assert props.data == {}


def test_start_active_applets_starts_local_ones_and_prunes_deleted():
    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}, "Live Plot": {"script": "b.py"}},
        active={"Live Plot", "Deleted On Another Client"},
    )
    started = []
    launcher._start_applet = started.append

    launcher._start_active_applets()

    assert started == ["Live Plot"]
    assert launcher._active == saved_active() == {"Live Plot"}


def test_shutdown_terminates_children_but_keeps_the_active_set(qapp):
    """Closing the GUI must not erase what was running, or nothing auto-starts."""
    from PyQt6.QtGui import QCloseEvent

    launcher = running_launcher(qt=True)
    process = launcher._processes["Image Monitor"]

    launcher.closeEvent(QCloseEvent())

    assert process.terminated
    assert launcher._processes == {}
    assert launcher._active == saved_active() == {"Image Monitor"}


def test_user_stop_clears_the_active_set():
    launcher = running_launcher()
    process = launcher._processes["Image Monitor"]

    launcher._stop_applet("Image Monitor")

    assert process.terminated
    assert launcher._active == saved_active() == set()


def test_applet_that_exits_on_its_own_is_forgotten():
    launcher = running_launcher()
    launcher._processes["Image Monitor"] = FakeProcess(exited=True)

    launcher.refresh_status()

    assert launcher._active == saved_active() == set()


def test_shared_applet_list_carries_no_running_state():
    legacy = FakeProps(
        {"Applets": {"Image Monitor": {"script": "a.py", "active": True}}}
    )
    assert make_launcher(props=legacy)._load_applets() == {
        "Image Monitor": {"script": "a.py"}
    }

    props = FakeProps()
    launcher = make_launcher(props=props)
    launcher._ensure_defaults()
    assert props.data["Applets"]
    for entry in props.data["Applets"].values():
        assert set(entry) == {"script", "description"}

    props = FakeProps()
    launcher = make_launcher(props=props)
    launcher._add_row = lambda *args, **kwargs: None
    launcher.add_applet("New Applet", "new.py")
    assert props.data["Applets"] == {"New Applet": {"script": "new.py"}}
    assert launcher._active == set()
