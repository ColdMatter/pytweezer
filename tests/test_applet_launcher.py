"""Applet Launcher tests: the shared applet list vs the local running-state.

The applet *list* is shared across clients via Properties; *which applets are
running* is per-machine and lives in local QSettings. These tests build the
launcher via ``__new__`` (as ``test_gui.py`` does) so no Properties hub
connection is opened and no applet subprocesses are spawned.
"""

import pytest

from pytweezer.GUI.applet_launcher import AppletLauncher


class FakeProps:
    """Minimal stand-in for a Properties connection, backed by a plain dict."""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, defaultvalue=None):
        return self.data.get(key, defaultvalue)

    def set(self, key, value):
        self.data[key] = value


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0 if self.terminated else None


@pytest.fixture
def settings_dir(tmp_path):
    """Point QSettings at a temp dir so tests never touch real user settings."""
    from PyQt6.QtCore import QSettings

    previous = QSettings.defaultFormat()
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    yield tmp_path
    QSettings.setDefaultFormat(previous)


def make_launcher(name="TestLauncher", applets=None, active=None, props=None, qt=False):
    """A launcher with its collaborators stubbed, skipping ``__init__``.

    ``qt=True`` additionally runs ``QWidget.__init__`` so the underlying Qt
    object exists — needed only by tests that call ``closeEvent``, which reaches
    ``BWidget.closeEvent``'s ``saveGeometry()``. Requires the ``qapp`` fixture.
    """
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
    # Status updates need real rows; irrelevant to the state logic under test.
    launcher._set_status = lambda *args, **kwargs: None
    return launcher


# --------------------------------------------------------------------------- #
# Local active-set persistence
# --------------------------------------------------------------------------- #

def test_active_set_round_trips_through_local_settings(settings_dir):
    launcher = make_launcher(active={"Live Plot", "Image Monitor"})
    launcher._save_active()

    reloaded = make_launcher()
    assert reloaded._load_active() == {"Live Plot", "Image Monitor"}


def test_single_active_applet_round_trips(settings_dir):
    """QSettings collapses a one-element list to a bare string on read."""
    launcher = make_launcher(active={"Image Monitor"})
    launcher._save_active()

    assert make_launcher()._load_active() == {"Image Monitor"}


def test_empty_active_set_round_trips(settings_dir):
    """An empty list comes back as None rather than []."""
    launcher = make_launcher(active=set())
    launcher._save_active()

    assert make_launcher()._load_active() == set()


def test_load_active_defaults_to_empty_for_a_fresh_machine(settings_dir):
    assert make_launcher(name="NeverRunBefore")._load_active() == set()


def test_active_set_is_per_launcher_name(settings_dir):
    """Different names are independent local namespaces."""
    make_launcher(name="LauncherA", active={"Image Monitor"})._save_active()
    make_launcher(name="LauncherB", active={"Live Plot"})._save_active()

    assert make_launcher(name="LauncherA")._load_active() == {"Image Monitor"}
    assert make_launcher(name="LauncherB")._load_active() == {"Live Plot"}


def test_set_active_persists_locally_and_not_to_properties(settings_dir):
    props = FakeProps()
    launcher = make_launcher(applets={"Image Monitor": {"script": "a.py"}}, props=props)

    launcher._set_active("Image Monitor", True)
    assert launcher._active == {"Image Monitor"}
    assert make_launcher()._load_active() == {"Image Monitor"}

    launcher._set_active("Image Monitor", False)
    assert launcher._active == set()
    assert make_launcher()._load_active() == set()

    # Running state must never reach the shared tree.
    assert props.data == {}


# --------------------------------------------------------------------------- #
# Startup: launch only what this machine was running
# --------------------------------------------------------------------------- #

def test_start_active_applets_starts_only_the_locally_active_ones(settings_dir):
    launcher = make_launcher(
        applets={
            "Image Monitor": {"script": "a.py"},
            "Live Plot": {"script": "b.py"},
        },
        active={"Live Plot"},
    )
    started = []
    launcher._start_applet = started.append

    launcher._start_active_applets()
    assert started == ["Live Plot"]


def test_start_active_applets_prunes_applets_deleted_elsewhere(settings_dir):
    """A name removed from the shared list is dropped from the local set."""
    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}},
        active={"Image Monitor", "Deleted On Another Client"},
    )
    started = []
    launcher._start_applet = started.append

    launcher._start_active_applets()

    assert started == ["Image Monitor"]
    assert launcher._active == {"Image Monitor"}
    assert make_launcher()._load_active() == {"Image Monitor"}


# --------------------------------------------------------------------------- #
# Stop vs shutdown
# --------------------------------------------------------------------------- #

def test_shutdown_terminates_children_but_keeps_the_active_set(settings_dir, qapp):
    """Closing the GUI must not erase what was running, or nothing auto-starts."""
    from PyQt6.QtGui import QCloseEvent

    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}},
        active={"Image Monitor"},
        qt=True,
    )
    process = FakeProcess()
    launcher._processes["Image Monitor"] = process
    launcher._save_active()

    launcher.closeEvent(QCloseEvent())

    assert process.terminated
    assert launcher._processes == {}
    assert launcher._active == {"Image Monitor"}
    assert make_launcher()._load_active() == {"Image Monitor"}


def test_user_stop_clears_the_active_set(settings_dir):
    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}}, active={"Image Monitor"}
    )
    process = FakeProcess()
    launcher._processes["Image Monitor"] = process
    launcher._save_active()

    launcher._stop_applet("Image Monitor")

    assert process.terminated
    assert launcher._active == set()
    assert make_launcher()._load_active() == set()


def test_applet_that_exits_on_its_own_is_forgotten(settings_dir):
    """Closing an applet's own window means don't start it next session."""
    launcher = make_launcher(
        applets={"Image Monitor": {"script": "a.py"}}, active={"Image Monitor"}
    )
    process = FakeProcess()
    process.terminated = True  # already exited, so poll() returns 0
    launcher._processes["Image Monitor"] = process
    launcher._save_active()

    launcher.refresh_status()

    assert launcher._active == set()
    assert make_launcher()._load_active() == set()


# --------------------------------------------------------------------------- #
# Shared list carries no running-state
# --------------------------------------------------------------------------- #

def test_load_applets_strips_a_legacy_active_flag():
    props = FakeProps(
        {"Applets": {"Image Monitor": {"script": "a.py", "active": True}}}
    )
    launcher = make_launcher(props=props)

    assert launcher._load_applets() == {"Image Monitor": {"script": "a.py"}}


def test_ensure_defaults_seeds_entries_without_active():
    props = FakeProps()
    launcher = make_launcher(props=props)

    launcher._ensure_defaults()

    seeded = props.data["Applets"]
    assert seeded
    for entry in seeded.values():
        assert "active" not in entry
        assert set(entry) == {"script", "description"}


def test_add_applet_does_not_write_active_or_start_anything(settings_dir):
    props = FakeProps()
    launcher = make_launcher(props=props)
    launcher._add_row = lambda *args, **kwargs: None

    launcher.add_applet("New Applet", "new.py")

    assert props.data["Applets"] == {"New Applet": {"script": "new.py"}}
    assert launcher._active == set()
