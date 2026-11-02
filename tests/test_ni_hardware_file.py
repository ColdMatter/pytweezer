"""Reading channel aliases out of MOTMaster hardware files (``EDMSuite/DAQ/*Hardware.cs``)."""

import pytest

from pytweezer.drivers.ni_hardware_file import Alias, parse_hardware_file


def test_board_address_resolves_through_string_variable():
    hardware = parse_hardware_file(
        """
        string a = "/PXI1Slot2";
        Boards.Add("analogPattern", a);
        AddAnalogOutputChannel("xShim", a + "/ao17");
        """
    )
    assert hardware.boards["analogPattern"].address == "/PXI1Slot2"
    assert hardware.aliases == [
        Alias("xShim", "ao", "PXI1Slot2/ao17", "analogPattern", None, None)
    ]


def test_board_address_resolves_through_boards_lookup():
    hardware = parse_hardware_file(
        """
        Boards.Add("pg", "/PXI1Slot5");
        string pgBoard = (string)Boards["pg"];
        AddDigitalOutputChannel("tCoolSwitch", pgBoard, 0, 0);
        AddAnalogInputChannel("pd", (string)Boards["pg"] + "/ai2", AITerminalConfiguration.Nrse);
        """
    )
    switch, pd = hardware.aliases
    assert (switch.physical, switch.board) == ("PXI1Slot5/port0/line0", "pg")
    assert (pd.physical, pd.board) == ("PXI1Slot5/ai2", "pg")


@pytest.mark.parametrize(
    "commented",
    [
        '// AddDigitalOutputChannel("gone", b, 0, 1);',
        """/*
        AddDigitalOutputChannel("gone", b, 0, 1);
        */""",
        '/*AddDigitalOutputChannel("gone", b, 0, 1);\n AddDigitalOutputChannel("gone2", b, 0, 2);*/',
    ],
)
def test_commented_out_declarations_are_ignored(commented):
    hardware = parse_hardware_file(
        f"""
        string b = "/Dev1";
        Boards.Add("board", b);
        {commented}
        """
    )
    assert hardware.aliases == []


def test_alias_on_commented_out_board_is_skipped():
    hardware = parse_hardware_file(
        """
        //string daqBoard = (string)Boards["daq"];
        AddAnalogInputChannel("iodine", daqBoard + "/ai2", AITerminalConfiguration.Nrse);
        """
    )
    assert hardware.aliases == []


def test_ao_range_and_ai_terminal_are_read():
    hardware = parse_hardware_file(
        """
        string b = "/Dev3";
        Boards.Add("usb", b);
        AddAnalogOutputChannel("v21Lock", b + "/ao0", 0.0, 1.0);
        AddAnalogOutputChannel("bXLock", b + "/ao2");
        AddAnalogInputChannel("v00PD", b + "/ai0", AITerminalConfiguration.Rse);
        """
    )
    ranged, unranged, pd = hardware.aliases
    assert ranged.ao_range == (0.0, 1.0)
    assert unranged.ao_range is None
    assert pd.terminal == "Rse"


def test_digital_input_is_parsed_as_di():
    hardware = parse_hardware_file(
        """
        string b = "/PXI1Slot3";
        Boards.Add("tcl", b);
        AddDigitalInputChannel("bXLockBlockFlag", b, 0, 1);
        """
    )
    (flag,) = hardware.aliases
    assert (flag.kind, flag.physical) == ("di", "PXI1Slot3/port0/line1")


def test_unrelated_calls_are_ignored():
    hardware = parse_hardware_file(
        """
        string b = "/PXI1Slot8";
        Boards.Add("tcl2", b);
        Info.Add("ToFPMTSignal", b + "/ai3");
        AddCounterChannel("counter", b + "/ctr0");
        AddAnalogInputThermocoupleChannel("tc", b + "/ai1", AITerminalConfiguration.Rse, AIThermocoupleType.K);
        wmlConfig.AddSlaveLaser("v0", "v00Lock", 1);
        """
    )
    assert hardware.aliases == []


def test_two_aliases_on_one_line_are_both_kept():
    hardware = parse_hardware_file(
        """
        string b = "/PXI1Slot4";
        Boards.Add("digitalPattern2", b);
        AddDigitalOutputChannel("MOT3Shutter", b, 3, 7);
        AddDigitalOutputChannel("rbOPShutter", b, 3, 7);
        """
    )
    first, second = hardware.aliases
    assert [first.name, second.name] == ["MOT3Shutter", "rbOPShutter"]
    assert first.physical == second.physical == "PXI1Slot4/port3/line7"


@pytest.mark.parametrize(
    ("declarations", "expected"),
    [
        (
            """
            Info.Add("PatternGeneratorBoard", dp);
            Dictionary<string, string> additionalPatternBoards = new Dictionary<string, string>();
            additionalPatternBoards.Add(dp2, dp2);
            Info.Add("AdditionalPatternGeneratorBoards", additionalPatternBoards);
            Dictionary<string, string> analogBoards = new Dictionary<string, string>();
            analogBoards.Add("AO", ap);
            Info.Add("AnalogBoards", analogBoards);
            """,
            ["/Dev1", "/PXI1Slot6", "/PXI1Slot2"],
        ),
        ('Info.Add("PGType", "dedicated");', []),
    ],
)
def test_pattern_boards_are_read_from_info(declarations, expected):
    hardware = parse_hardware_file(
        f"""
        string dp = "/Dev1"; string dp2 = "/PXI1Slot6"; string ap = "/PXI1Slot2";
        {declarations}
        """
    )
    assert hardware.pattern_boards == expected


def test_safe_state_values_are_read_from_info():
    hardware = parse_hardware_file(
        """
        Info.Add(DAQ.Analog.DAQMxAnalogSafeState.InfoKey, new Dictionary<string, double>
        {
            { "MOTCoilsCurrent", 0.0 },
            { "transferCoils", 1.5 }, // amps, not volts
        });
        Info.Add("PGType", "dedicated");
        """
    )
    assert hardware.safe_states == {"MOTCoilsCurrent": 0.0, "transferCoils": 1.5}
