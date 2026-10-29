"""The NI Cards tab, driven by a fake client in place of the NICards device."""

import time

import pytest
from PyQt6 import QtCore
from PyQt6.QtTest import QTest

from pytweezer.GUI.ni_cards import panel as panel_module
from pytweezer.GUI.ni_cards.cells import last_set_text
from pytweezer.GUI.ni_cards.client import CardsClient
from pytweezer.GUI.ni_cards.flow_grid import CELL_WIDTH, FlowGrid
from pytweezer.GUI.ni_cards.panel import NICardsPanel, ni_card_devices

DO = "PXI1Slot4/port3/line5"
AI = "PXI1Slot2/ai0"
AO = "PXI1Slot2/ao0"


def channel(physical, kind, aliases, range_=None, terminal=None):
    return {
        "physical": physical,
        "kind": kind,
        "aliases": aliases,
        "range": range_,
        "terminal": terminal,
    }


LAYOUT = {
    "error": None,
    "cards": [
        {
            "board": "digitalPattern2",
            "address": "/PXI1Slot4",
            "model": "PXI-6535",
            "error": None,
            "channels": [channel(DO, "do", ["MOT1Shutter"])],
            "unaliased": [channel("PXI1Slot4/port0/line2", "do", [])],
        },
        {
            "board": "analogPattern",
            "address": "/PXI1Slot2",
            "model": "PXIe-6738",
            "error": None,
            "channels": [
                channel(AI, "ai", ["v00PD"], terminal="Rse"),
                channel(AO, "ao", ["v21Lock"], range_=[0.0, 1.0]),
                channel("PXI1Slot2/port0/line1", "di", ["bXLockBlockFlag"]),
            ],
            "unaliased": [],
        },
    ],
}


def state(holds=None, card_holds=None, last_set=None, notices=None):
    return {
        "holds": holds or {},
        "card_holds": card_holds or {},
        "last_set": last_set or {},
        "notices": notices or [],
    }


class FakeClient(QtCore.QObject):
    layout_received = QtCore.pyqtSignal(dict)
    state_received = QtCore.pyqtSignal(dict)
    values_received = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str, str)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.calls = []

    def fetch_layout(self):
        self.calls.append(("fetch_layout",))

    def fetch_state(self):
        self.calls.append(("fetch_state",))

    def write_do(self, physical, high):
        self.calls.append(("write_do", physical, high))

    def write_ao(self, physical, volts):
        self.calls.append(("write_ao", physical, volts))

    def read(self, physicals):
        self.calls.append(("read", list(physicals)))

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def view(qapp, client):
    panel = NICardsPanel(devices=["CaF NI Cards"], make_client=lambda name: client)
    panel.show()
    client.connection_changed.emit(True)
    client.layout_received.emit(LAYOUT)
    yield panel.views["CaF NI Cards"]
    panel.close()


def test_ni_card_devices_picks_entries_by_class():
    devices = {
        "CaF NI Cards": {"class": "pytweezer.drivers.ni_cards:NICards"},
        "CaF ThorCam": {"class": "pytweezer.drivers.thorcam:ThorCam"},
        "Rb NI Cards": {"class": "pytweezer.drivers.ni_cards:NICards"},
    }
    assert ni_card_devices(devices) == ["CaF NI Cards", "Rb NI Cards"]


@pytest.mark.parametrize(
    "entry, kind, expected",
    [
        (None, "ao", "unknown"),
        ({"value": 1.25, "time": 1793300000.0}, "ao", "last set 1.25 V at {}"),
        ({"value": True, "time": 1793300000.0}, "do", "last set high at {}"),
        (
            {
                "value": 1.25,
                "time": 1793300000.0,
                "source": "motmaster",
                "script": "LoadTweezer",
            },
            "ao",
            "end of LoadTweezer: 1.25 V at {}",
        ),
    ],
)
def test_last_set_text(entry, kind, expected):
    if entry is not None:
        expected = expected.format(
            time.strftime("%H:%M", time.localtime(entry["time"]))
        )
    assert last_set_text(entry, kind) == expected


@pytest.mark.parametrize(
    "width, columns", [(CELL_WIDTH, 1), (3 * CELL_WIDTH, 3), (1000, 5)]
)
def test_flow_grid_column_count_follows_width(qapp, width, columns):
    from PyQt6.QtWidgets import QLabel

    grid = FlowGrid()
    for i in range(12):
        grid.add(QLabel(str(i)))
    grid.show()
    grid.resize(width, 300)
    qapp.processEvents()
    assert grid.column_count() == columns
    grid.close()


def test_panel_builds_offscreen_with_a_layout(view):
    assert set(view.cells) == {
        DO,
        "PXI1Slot4/port0/line2",
        AI,
        AO,
        "PXI1Slot2/port0/line1",
    }
    assert view.unaliased_sections["digitalPattern2"].is_collapsed()


def test_state_marks_held_cells_and_card_banner(view, client):
    notice = "CaF MotMaster is not reachable: its runs are not greyed out"
    assert view.notices.isHidden()
    client.state_received.emit(
        state(
            holds={AO: "Wavemeter Lock"},
            card_holds={"/PXI1Slot4": "Hardware Control"},
            notices=[notice],
        )
    )
    assert not view.notices.isHidden()
    assert view.notices.text() == notice
    client.state_received.emit({k: v for k, v in state().items() if k != "notices"})
    assert view.notices.isHidden()
    client.state_received.emit(
        state(
            holds={AO: "Wavemeter Lock"},
            card_holds={"/PXI1Slot4": "Hardware Control"},
        )
    )
    assert not view.cells[AO].isEnabled()
    assert "held by Wavemeter Lock" in view.cells[AO].detail_text()
    banner = view.banners["/PXI1Slot4"]
    assert not banner.isHidden()
    assert "Hardware Control" in banner.text()


def test_channel_becoming_held_stops_being_watched(view, client):
    QTest.mouseClick(view.cells[AI], QtCore.Qt.MouseButton.LeftButton)
    assert view.watched == {AI}
    client.state_received.emit(state(holds={AI: "Hardware Control"}))
    client.calls.clear()
    view.poll_readings()
    assert view.watched == set()
    assert not [call for call in client.calls if call[0] == "read"]


def test_unreachable_device_stops_reads_and_says_so(view, client):
    QTest.mouseClick(view.cells[AI], QtCore.Qt.MouseButton.LeftButton)
    assert view.read_timer.isActive()
    client.connection_changed.emit(False)
    assert not view.read_timer.isActive()
    assert not view.message.isHidden()
    assert "CaF NI Cards is not reachable" in view.message.text()


def test_other_gui_write_shows_as_last_set(view, client):
    entry = {"value": True, "time": 1793300000.0}
    client.state_received.emit(state(last_set={DO: entry}))
    assert view.cells[DO].detail_text() == last_set_text(entry, "do")


def test_do_click_requests_write_of_opposite_state(view, client):
    client.state_received.emit(state(last_set={DO: {"value": True, "time": 0.0}}))
    QTest.mouseClick(view.cells[DO], QtCore.Qt.MouseButton.LeftButton)
    assert ("write_do", DO, False) in client.calls


def test_ao_value_box_never_invents_or_keeps_a_stale_value(view, client):
    spin = view.cells[AO].spin
    assert spin.text() == "—"
    spin.setFocus()
    QTest.keyClick(spin, QtCore.Qt.Key.Key_Return)
    assert not [call for call in client.calls if call[0] == "write_ao"]
    client.state_received.emit(state(last_set={AO: {"value": 0.5, "time": 0.0}}))
    assert spin.text() == "0.5 V"


def test_tab_recovers_when_the_device_server_comes_up_late(qapp, monkeypatch):
    monkeypatch.setattr(panel_module, "RECONNECT_MS", 50)

    class Device:
        def layout(self):
            return LAYOUT

        def state(self):
            return state()

    attempts = []

    def connect():
        attempts.append(None)
        if len(attempts) == 1:
            raise ConnectionRefusedError("not listening yet")
        return Device()

    panel = NICardsPanel(
        devices=["CaF NI Cards"], make_client=lambda name: CardsClient(name, connect)
    )
    panel.show()
    view = panel.views["CaF NI Cards"]
    deadline = time.monotonic() + 5
    while not view.cells and time.monotonic() < deadline:
        QTest.qWait(20)
    panel.close()
    assert AO in view.cells
    assert view.message.isHidden()
