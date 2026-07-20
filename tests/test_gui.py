"""Headless GUI tests. These construct real PyQt5 widgets under the offscreen
platform (set in conftest.py). Pure-logic methods that don't need a live widget
tree are exercised via ``__new__`` to avoid the panels' construction side
effects (a real DeviceManager spawns device subprocesses for active devices).
"""

import pytest

from bin.process_manager import DeviceManager, ProcessManager


# --------------------------------------------------------------------------- #
# Host filtering (pure logic; no Qt construction)
# --------------------------------------------------------------------------- #

def test_base_process_manager_accepts_every_host():
    pm = ProcessManager.__new__(ProcessManager)
    assert pm.check_host("anything") is True
    assert pm.check_host(None) is True


def test_device_manager_check_host_is_case_and_whitespace_insensitive():
    dm = DeviceManager.__new__(DeviceManager)
    dm.host_addr = "PH-BEAST"
    assert dm.check_host("PH-BEAST") is True
    assert dm.check_host("  PH-BEAST  ") is True


def test_device_manager_check_host_rejects_other_hosts():
    dm = DeviceManager.__new__(DeviceManager)
    dm.host_addr = "192.168.0.10"
    assert dm.check_host("192.168.0.11") is False


def test_device_manager_check_host_none_is_false():
    dm = DeviceManager.__new__(DeviceManager)
    dm.host_addr = "192.168.0.10"
    assert dm.check_host(None) is False


# --------------------------------------------------------------------------- #
# Real widget construction, headless (needs the QApplication fixture)
# --------------------------------------------------------------------------- #

def test_bwidget_constructs_headless_without_property_hub(qapp):
    from pytweezer.GUI.pytweezerQt import BWidget

    w = BWidget("HeadlessTest", create_props=False)
    assert w._name == "HeadlessTest"
    # create_props=False must skip the Properties() hub connection.
    assert not hasattr(w, "_props")


def test_process_tile_inactive_builds_and_does_not_spawn(qapp):
    from bin.process_tile_base import ProcessTile

    tile = ProcessTile(script="does_not_exist.py", name="TileLabel", active=False)
    try:
        assert tile.startButton.text() == "TileLabel"
        # active=False -> no subprocess launched.
        assert tile.process is None
    finally:
        tile.timer.stop()
