"""Headless GUI tests. These construct real PyQt5 widgets under the offscreen
platform (set in conftest.py). Pure-logic methods that don't need a live widget
tree are exercised via ``__new__`` to avoid the panels' construction side
effects (a real DevicesPanel spawns device subprocesses for active devices).
"""

from bin.managed_panel import DevicesPanel, ManagedRow


# --------------------------------------------------------------------------- #
# Host filtering (pure logic; no Qt construction)
# --------------------------------------------------------------------------- #

def test_devices_panel_check_host_is_case_and_whitespace_insensitive():
    panel = DevicesPanel.__new__(DevicesPanel)
    panel.host_addr = "PH-BEAST"
    assert panel.check_host("PH-BEAST") is True
    assert panel.check_host("  ph-beast  ") is True


def test_devices_panel_check_host_rejects_other_hosts():
    panel = DevicesPanel.__new__(DevicesPanel)
    panel.host_addr = "192.168.0.10"
    assert panel.check_host("192.168.0.11") is False


def test_devices_panel_check_host_none_is_false():
    panel = DevicesPanel.__new__(DevicesPanel)
    panel.host_addr = "192.168.0.10"
    assert panel.check_host(None) is False


# --------------------------------------------------------------------------- #
# Real widget construction, headless (needs the QApplication fixture)
# --------------------------------------------------------------------------- #

def test_bwidget_constructs_headless_without_property_hub(qapp):
    from pytweezer.GUI.pytweezerQt import BWidget

    w = BWidget("HeadlessTest", create_props=False)
    assert w._name == "HeadlessTest"
    # create_props=False must skip the Properties() hub connection.
    assert not hasattr(w, "_props")


def test_managed_row_inactive_builds_and_does_not_spawn(qapp):
    row = ManagedRow("RowLabel", script="does_not_exist.py", active=False)
    try:
        assert row.toggleButton.text() == "Start"
        # active=False -> no subprocess launched.
        assert row.process is None
    finally:
        if row.timer is not None:
            row.timer.stop()


def test_managed_row_uncontrollable_has_no_toggle(qapp):
    row = ManagedRow("RowLabel", script="does_not_exist.py", controllable=False)
    try:
        assert row.toggleButton is None
        assert row.process is None
    finally:
        if row.timer is not None:
            row.timer.stop()


def test_managed_row_indent_marks_the_row_as_a_child(qapp):
    row = ManagedRow("Sub Device", controllable=False, indent=18)
    assert row.property("tier") == "child"
    assert row.layout().contentsMargins().left() == 28


def test_managed_row_degraded_offers_stop(qapp):
    # A composite whose coordinator stood down is still a running process.
    row = ManagedRow("Rig", script="does_not_exist.py")
    try:
        row.apply_status("degraded")
        assert row.toggleButton.text() == "Stop"
    finally:
        if row.timer is not None:
            row.timer.stop()


# --------------------------------------------------------------------------- #
# Composite devices are broken out into per-sub-device rows
# --------------------------------------------------------------------------- #

_PANEL_CONFIG = {
    "Devices": {
        "Rig": {
            "host": "10.0.0.1",  # not this machine -> nothing is ever spawned
            "port": 1,
            "devices": {"Rig SLM": {"class": "pkg:Slm"}, "Rig Cam": {"class": "pkg:Cam"}},
            "coordinator": "pkg:Coord",
        },
    }
}


def _devices_panel(qapp, monkeypatch):
    import bin.managed_panel as managed_panel

    monkeypatch.setattr(
        managed_panel.ConfigReader, "getConfiguration", staticmethod(lambda: _PANEL_CONFIG)
    )
    # The panel subscribes to the cross-PC status feed on construction; without a
    # hub there is nothing to receive, so skip it rather than bind a ZMQ socket.
    monkeypatch.setattr(
        managed_panel, "DeviceStatusClient", lambda *a, **k: (_ for _ in ()).throw(OSError)
    )
    return managed_panel.DevicesPanel("Devices")


def test_devices_panel_lists_composite_sub_devices(qapp, monkeypatch):
    panel = _devices_panel(qapp, monkeypatch)
    assert set(panel._rows) == {"Rig", "Rig Cam", "Rig SLM"}
    # Sub-devices share the rig's process, so only the rig is startable.
    assert panel._rows["Rig Cam"].toggleButton is None
    assert panel._rows["Rig Cam"].property("tier") == "child"
    assert panel._rows["Rig"].property("tier") is None


def test_devices_panel_applies_status_to_sub_device_rows(qapp, monkeypatch):
    panel = _devices_panel(qapp, monkeypatch)
    panel._apply_snapshot(
        {
            "Rig": {"state": "degraded", "host": "10.0.0.1", "port": 1},
            "Rig Cam": {"state": "up", "host": "10.0.0.1", "port": 1},
            "Rig SLM": {"state": "failed", "host": "10.0.0.1", "port": 1},
        }
    )
    assert panel._rows["Rig"].property("state") == "degraded"
    assert panel._rows["Rig Cam"].property("state") == "up"
    assert panel._rows["Rig SLM"].property("state") == "failed"
    assert panel._rows["Rig SLM"].stateLabel.text() == "Failed"
