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
