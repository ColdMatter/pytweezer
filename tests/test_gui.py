"""Headless GUI tests. ``__new__`` is used where a real panel's construction
would have side effects (a DevicesPanel spawns subprocesses for active devices).
"""

import pytest

from bin.managed_panel import DevicesPanel, ManagedRow


def test_devices_panel_check_host_is_case_and_whitespace_insensitive():
    panel = DevicesPanel.__new__(DevicesPanel)
    panel.host_addr = "PH-BEAST"
    assert panel.check_host("  ph-beast  ") is True
    assert panel.check_host("PH-BEAST2") is False
    assert panel.check_host(None) is False


def test_this_pc_address_ignores_hostname_case(monkeypatch):
    from pytweezer.configuration import config

    monkeypatch.setattr(config, "HOSTS", {"IC-CZC4287H3W": "10.59.3.2"})
    monkeypatch.setattr(config.socket, "gethostname", lambda: "ic-czc4287h3w")
    assert config.this_pc_address() == "10.59.3.2"
    monkeypatch.setattr(config.socket, "gethostname", lambda: "elsewhere")
    assert config.this_pc_address() is None


@pytest.fixture
def make_row(qapp):
    rows = []

    def make(**kwargs):
        rows.append(ManagedRow("Row", script="does_not_exist.py", **kwargs))
        return rows[-1]

    yield make
    for row in rows:
        if row.timer is not None:
            row.timer.stop()


def test_managed_row_toggle_reflects_controllability_and_state(make_row):
    inactive = make_row(active=False)
    assert inactive.toggleButton.text() == "Start"
    assert inactive.process is None

    assert make_row(controllable=False).toggleButton is None

    # A composite whose coordinator stood down is still a running process.
    degraded = make_row()
    degraded.apply_status("degraded")
    assert degraded.toggleButton.text() == "Stop"


def test_client_control_panel_marks_portless_servers_unprobeable(qapp, monkeypatch):
    from bin import managed_panel

    config = {"Servers": {"Datalogger": {"script": "x.py", "host": "10.0.0.1"}}}
    monkeypatch.setattr(managed_panel, "get_config", lambda: config)
    panel = managed_panel.ControlPanel("Servers", "Servers", controllable=False)

    row = panel._rows["Datalogger"]
    assert row.property("state") == "unprobeable"
    assert row.stateLabel.text() == "n/a"


_PANEL_CONFIG = {
    "Devices": {
        "Rig": {
            "host": "10.0.0.1",  # not this machine -> nothing is ever spawned
            "port": 1,
            "devices": {
                "Rig SLM": {"class": "pkg:Slm"},
                "Rig Cam": {"class": "pkg:Cam"},
            },
            "coordinator": "pkg:Coord",
        },
    }
}


class _FakeControl:
    def __init__(self):
        self.sent = []

    def request(self, action, device):
        self.sent.append((action, device))

    @property
    def requested(self):
        return self

    def connect(self, _slot):
        pass

    def close(self):
        pass


@pytest.fixture
def devices_panel(qapp, monkeypatch):
    """Build a DevicesPanel from ``config`` without binding any ZMQ socket."""
    from bin import managed_panel

    control = _FakeControl()
    # Skip the cross-PC status feed rather than bind a ZMQ socket.
    monkeypatch.setattr(
        managed_panel,
        "DeviceStatusClient",
        lambda *a, **k: (_ for _ in ()).throw(OSError),
    )
    monkeypatch.setattr(managed_panel, "DeviceControlChannel", lambda: control)

    def make(config):
        monkeypatch.setattr(managed_panel, "get_config", lambda: config)
        panel = managed_panel.DevicesPanel("Devices")
        return panel, control

    return make


def test_devices_panel_breaks_composites_into_sub_device_rows(devices_panel):
    panel, _control = devices_panel(_PANEL_CONFIG)

    assert set(panel._rows) == {"Rig", "Rig Cam", "Rig SLM"}
    # Sub-devices share the rig's process, so only the rig is startable.
    assert panel._rows["Rig Cam"].toggleButton is None
    assert panel._rows["Rig Cam"].property("tier") == "child"
    assert panel._rows["Rig"].property("tier") is None

    panel._apply_snapshot(
        {
            "Rig": {"state": "degraded", "host": "10.0.0.1", "port": 1},
            "Rig Cam": {"state": "up", "host": "10.0.0.1", "port": 1},
            "Rig SLM": {"state": "failed", "host": "10.0.0.1", "port": 1},
        }
    )
    assert [
        panel._rows[n].property("state") for n in ("Rig", "Rig Cam", "Rig SLM")
    ] == [
        "degraded",
        "up",
        "failed",
    ]


def test_remote_device_toggle_sends_requests_and_waits_for_the_feed(devices_panel):
    panel, control = devices_panel(_PANEL_CONFIG)
    rig = panel._rows["Rig"]
    assert rig.process is None

    rig.toggleButton.click()
    assert control.sent == [("start", "Rig")]
    assert rig.process is None
    rig.apply_status("down")
    assert rig.property("state") == "starting"
    rig.apply_status("up")
    assert rig.toggleButton.text() == "Stop"

    rig.toggleButton.click()
    assert control.sent[-1] == ("stop", "Rig")
    # The feed reads "up" until the host has acted on the stop.
    rig.apply_status("up")
    assert rig.property("state") == "stopping"
    rig.apply_status("down")
    assert rig.property("state") == "down"


def test_devices_panel_carries_out_requests_only_for_its_own_devices(
    devices_panel, monkeypatch
):
    from bin import managed_panel

    monkeypatch.setattr(managed_panel, "this_pc_address", lambda: None)  # -> 127.0.0.1
    config = {
        "Devices": {
            "Local Cam": {"host": "127.0.0.1", "port": 1},
            "Remote Cam": {"host": "10.0.0.1", "port": 2},
        }
    }
    panel, _control = devices_panel(config)
    calls = []
    for name, row in panel._rows.items():
        monkeypatch.setattr(
            row, "startProcess", lambda n=name: calls.append(("start", n))
        )
        monkeypatch.setattr(
            row, "stopProcess", lambda n=name: calls.append(("stop", n))
        )

    panel._handle_request("start", "Remote Cam", "PC-A")
    panel._handle_request("stop", "Local Cam", "PC-A")  # already stopped
    assert calls == []

    panel._handle_request("start", "Local Cam", "PC-A")
    assert calls == [("start", "Local Cam")]

    panel._rows["Local Cam"].apply_status("up")
    panel._handle_request("start", "Local Cam", "PC-A")  # already running
    panel._handle_request("stop", "Local Cam", "PC-A")
    assert calls == [("start", "Local Cam"), ("stop", "Local Cam")]


def test_stopping_an_unowned_device_hands_it_to_stop_orphan(make_row):
    orphans = []
    row = make_row(stop_orphan=orphans.append)
    row.apply_status("up")  # running, but not spawned by this row
    row.toggleButton.click()
    assert orphans == ["Row"]
    assert row.property("state") == "stopping"

    # Closing a panel terminates rows; that must not hunt for orphans.
    other = make_row(stop_orphan=orphans.append)
    other.apply_status("up")
    other.terminateProcess()
    assert orphans == ["Row"]
