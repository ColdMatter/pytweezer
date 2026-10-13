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


def test_bwidget_constructs_headless_without_property_hub(qapp):
    from pytweezer.GUI.pytweezerQt import BWidget

    w = BWidget("HeadlessTest", create_props=False)
    assert w._name == "HeadlessTest"
    assert not hasattr(w, "_props")


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


def test_devices_panel_breaks_composites_into_sub_device_rows(qapp, monkeypatch):
    from bin import managed_panel

    monkeypatch.setattr(managed_panel, "get_config", lambda: _PANEL_CONFIG)
    # Skip the cross-PC status feed rather than bind a ZMQ socket.
    monkeypatch.setattr(
        managed_panel,
        "DeviceStatusClient",
        lambda *a, **k: (_ for _ in ()).throw(OSError),
    )
    panel = managed_panel.DevicesPanel("Devices")

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
