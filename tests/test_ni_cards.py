"""The NI test-panel driver: layout, reservations, protection and last-set values."""

import time

import pytest

from pytweezer.drivers.ni_cards import (
    PROTECTED,
    CardInfo,
    ChannelHeld,
    Holds,
    NICards,
    Reservations,
)

HARDWARE = """
string dio = "/PXI1Slot4";
Boards.Add("digitalPattern2", dio);
string analog = "/PXI1Slot2";
Boards.Add("analogPattern", analog);
string tcl2 = "/PXI1Slot8";
Boards.Add("tclBoard2", tcl2);
string usb = "/Dev3";
Boards.Add("usbBoard2", usb);

AddDigitalOutputChannel("MOT3Shutter", dio, 3, 7);
AddDigitalOutputChannel("rbOPShutter", dio, 3, 7);
AddDigitalOutputChannel("MOT1Shutter", dio, 3, 5);
AddAnalogOutputChannel("v21Lock", analog + "/ao0", 0.0, 1.0);
AddAnalogOutputChannel("MOTCoilsCurrent", analog + "/ao1");
AddAnalogInputChannel("v00PD", analog + "/ai0", AITerminalConfiguration.Rse);
AddDigitalInputChannel("bXLockBlockFlag", analog, 0, 1);
AddAnalogInputChannel("sourceTemp", tcl2 + "/ai4", AITerminalConfiguration.Rse);
AddAnalogOutputChannel("v10Lock", usb + "/ao1", 0, 5);
"""


class FakeIO:
    """Cards as NI-DAQmx would report them; records what reaches the hardware."""

    def __init__(self):
        self.calls = []
        self.cards = {
            "PXI1Slot4": CardInfo(
                "PXI-6535",
                ao=[],
                ai=[],
                lines=[
                    "PXI1Slot4/port0/line10",
                    "PXI1Slot4/port3/line7",
                    "PXI1Slot4/port0/line2",
                    "PXI1Slot4/port3/line5",
                ],
                ao_range=(0.0, 0.0),
            ),
            "PXI1Slot2": CardInfo(
                "PXIe-6738",
                ao=["PXI1Slot2/ao0", "PXI1Slot2/ao1"],
                ai=["PXI1Slot2/ai0", "PXI1Slot2/ai1"],
                lines=["PXI1Slot2/port0/line1"],
                ao_range=(-10.0, 10.0),
            ),
            "PXI1Slot8": CardInfo(
                "PXI-6221",
                ao=["PXI1Slot8/ao0"],
                ai=["PXI1Slot8/ai4", "PXI1Slot8/ai9"],
                lines=["PXI1Slot8/port0/line0"],
                ao_range=(-10.0, 10.0),
            ),
        }

    def card(self, device):
        card = self.cards.get(device)
        if isinstance(card, Exception):
            raise card
        return card

    def write_do(self, physical, high):
        self.calls.append(("write_do", physical, high))

    def write_ao(self, physical, volts, low, high):
        self.calls.append(("write_ao", physical, volts, low, high))

    def read_ai(self, physicals, terminals):
        self.calls.append(("read_ai", list(physicals), list(terminals)))
        return [1.5] * len(physicals)

    def read_di(self, physicals):
        self.calls.append(("read_di", list(physicals)))
        return [True] * len(physicals)


@pytest.fixture
def hardware_file(tmp_path):
    path = tmp_path / "MoleculeMOTHardware.cs"
    path.write_text(HARDWARE)
    return str(path)


@pytest.fixture
def io():
    return FakeIO()


def make_cards(hardware_file, io, reserved=None, protected=None):
    return NICards(hardware_file, reserved or {}, protected or [], io=io)


def card(layout, board):
    return next(c for c in layout["cards"] if c["board"] == board)


def test_layout_merges_shared_aliases_and_lists_unaliased_lines(hardware_file, io):
    dio = card(make_cards(hardware_file, io).layout(), "digitalPattern2")
    assert [c["aliases"] for c in dio["channels"]] == [
        ["MOT1Shutter"],
        ["MOT3Shutter", "rbOPShutter"],
    ]
    assert [(c["physical"], c["kind"]) for c in dio["unaliased"]] == [
        ("PXI1Slot4/port0/line2", "do"),
        ("PXI1Slot4/port0/line10", "do"),
    ]


def test_missing_hardware_file_reports_error_instead_of_raising(tmp_path, io):
    missing = str(tmp_path / "nowhere" / "MoleculeMOTHardware.cs")
    layout = make_cards(missing, io).layout()
    assert missing in layout["error"]
    assert layout["cards"] == []


@pytest.mark.parametrize(
    "reported, error",
    [(None, "not found on this PC"), (OSError("-200220"), "unreadable by NI-DAQmx")],
)
def test_card_not_on_this_pc_is_listed_and_rejects_writes(
    hardware_file, io, reported, error
):
    io.cards["Dev3"] = reported
    cards = make_cards(hardware_file, io)
    usb = card(cards.layout(), "usbBoard2")
    assert usb["error"].startswith(error)
    assert [c["aliases"] for c in usb["channels"]] == [["v10Lock"]]
    with pytest.raises(ValueError, match=error):
        cards.write_ao("Dev3/ao1", 1.0)
    assert io.calls == []


@pytest.mark.parametrize(
    "kwargs, typo",
    [
        ({"reserved": {"Wavemeter Lock": {"channels": ["v21Lok"]}}}, "v21Lok"),
        ({"reserved": {"Hardware Control": {"cards": ["tclBord2"]}}}, "tclBord2"),
        ({"protected": ["MOTCoilCurrent"]}, "MOTCoilCurrent"),
        ({"reserved": {"Wavemeter Lock": {"channel": ["v21Lock"]}}}, "channel"),
    ],
)
def test_unknown_reserved_or_protected_name_fails_construction(
    hardware_file, io, kwargs, typo
):
    with pytest.raises(ValueError, match=typo):
        make_cards(hardware_file, io, **kwargs)


def test_reserved_card_holds_every_channel_including_unaliased(hardware_file, io):
    cards = make_cards(
        hardware_file, io, reserved={"Hardware Control": {"cards": ["tclBoard2"]}}
    )
    state = cards.state()
    assert state["card_holds"] == {"/PXI1Slot8": "Hardware Control"}
    on_card = {p for p in state["holds"] if p.startswith("PXI1Slot8/")}
    assert on_card == {
        "PXI1Slot8/ai4",
        "PXI1Slot8/ai9",
        "PXI1Slot8/ao0",
        "PXI1Slot8/port0/line0",
    }
    assert {state["holds"][p] for p in on_card} == {"Hardware Control"}


def test_earlier_source_wins_for_channels_and_cards():
    first = Holds({"A/ao0": "first"}, {"/A": "first"})
    second = Holds(
        {"A/ao0": "second", "B/ao0": "second"}, {"/A": "second", "/B": "second"}
    )
    merged = Reservations([lambda: first, lambda: second]).current()
    assert merged.channels == {"A/ao0": "first", "B/ao0": "second"}
    assert merged.cards == {"/A": "first", "/B": "second"}


@pytest.mark.parametrize("operation", ["write", "read"])
@pytest.mark.parametrize(
    "physical, holder",
    [("PXI1Slot2/ao0", "Wavemeter Lock"), ("PXI1Slot2/ao1", PROTECTED)],
)
def test_held_and_protected_channels_refuse_writes_and_reads(
    hardware_file, io, operation, physical, holder
):
    cards = make_cards(
        hardware_file,
        io,
        reserved={"Wavemeter Lock": {"channels": ["v21Lock"]}},
        protected=["MOTCoilsCurrent"],
    )
    with pytest.raises(ChannelHeld, match=holder):
        if operation == "write":
            cards.write_ao(physical, 0.1)
        else:
            cards.read([physical])
    assert io.calls == []


def test_ao_write_outside_declared_range_is_rejected(hardware_file, io):
    cards = make_cards(hardware_file, io)
    with pytest.raises(ValueError):
        cards.write_ao("PXI1Slot2/ao0", 1.5)
    assert io.calls == []
    cards.write_ao("PXI1Slot2/ao0", 0.5)
    assert io.calls == [("write_ao", "PXI1Slot2/ao0", 0.5, 0.0, 1.0)]


def test_last_set_records_value_and_time_of_each_write(hardware_file, io):
    cards = make_cards(hardware_file, io)
    cards.write_do("PXI1Slot4/port3/line5", True)
    entry = cards.state()["last_set"]["PXI1Slot4/port3/line5"]
    assert entry["value"] is True
    assert abs(entry["time"] - time.time()) < 1.0


def test_read_returns_values_keyed_by_physical(hardware_file, io):
    values = make_cards(hardware_file, io).read(
        ["PXI1Slot2/ai0", "PXI1Slot2/ai1", "PXI1Slot2/port0/line1"]
    )
    assert values == {
        "PXI1Slot2/ai0": 1.5,
        "PXI1Slot2/ai1": 1.5,
        "PXI1Slot2/port0/line1": True,
    }
    assert ("read_ai", ["PXI1Slot2/ai0", "PXI1Slot2/ai1"], ["Rse", None]) in io.calls
