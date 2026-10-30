"""The NI Cards driver while MOTMaster runs, or a queued task reserves it."""

import re
import time

import pytest

from pytweezer.drivers.ni_cards import ChannelHeld, NICards
from pytweezer.drivers.ni_cards_sources import (
    LastRun,
    MotMasterNotAnswering,
    MotMasterTooOld,
)
from tests.helpers import HARDWARE, FakeIO

PATTERN_BOARDS = """
Info.Add("PatternGeneratorBoard", dio);
Dictionary<string, string> analogBoards = new Dictionary<string, string>();
analogBoards.Add("AO", analog);
Info.Add("AnalogBoards", analogBoards);
"""

SAFE_STATE = """
Info.Add(DAQ.Analog.DAQMxAnalogSafeState.InfoKey, new Dictionary<string, double>
{
    { "MOTCoilsCurrent", 0.0 },
});
"""

RUNNING = "CaF MotMaster (running a sequence)"
UNREACHABLE = "CaF MotMaster is not reachable: its runs are not greyed out"
SETUP_ERROR = (
    "Cannot connect to CaF MotMaster (setup error, see log): "
    "its runs are not greyed out"
)
TOO_OLD = (
    "CaF MotMaster does not report end-of-run values: it needs a newer MOTMaster build"
)

V21_LOCK = "PXI1Slot2/ao0"
MOT1_SHUTTER = "PXI1Slot4/port3/line5"
TCL_AO = "PXI1Slot8/ao0"
MOT_COILS = "PXI1Slot2/ao1"


class FakeStatus:
    """MOTMaster's answers; an exception instance is raised instead of returned."""

    def __init__(self):
        self.running = False
        self.last = None
        self.setup_error = False

    @staticmethod
    def _answer(value):
        if isinstance(value, Exception):
            raise value
        return value

    def is_running(self):
        return self._answer(self.running)

    def last_run(self):
        return self._answer(self.last)


class FakeQueue:
    def __init__(self):
        self.devices = {}

    def running_devices(self):
        return self.devices

    def close(self):
        pass


@pytest.fixture
def hardware_file(tmp_path):
    path = tmp_path / "MoleculeMOTHardware.cs"
    path.write_text(HARDWARE + PATTERN_BOARDS)
    return str(path)


@pytest.fixture
def io():
    return FakeIO()


@pytest.fixture
def status():
    return FakeStatus()


@pytest.fixture
def queue():
    return FakeQueue()


def make_cards(hardware_file, io, status, queue):
    return NICards(
        hardware_file,
        {},
        [],
        "CaF MotMaster",
        io=io,
        motmaster_status=status,
        queue=queue,
    )


@pytest.fixture
def cards(hardware_file, io, status, queue):
    return make_cards(hardware_file, io, status, queue)


def run(number, finished, values=None, outputs_known=True):
    return LastRun(number, "LoadTweezer", finished, outputs_known, values or {})


def test_running_motmaster_holds_its_pattern_boards_and_refuses_writes(
    cards, io, status
):
    status.running = True
    state = cards.state()
    assert state["card_holds"] == {"/PXI1Slot4": RUNNING, "/PXI1Slot2": RUNNING}
    assert state["holds"]["PXI1Slot4/port0/line2"] == RUNNING
    assert not [p for p in state["holds"] if p.startswith("PXI1Slot8/")]
    with pytest.raises(ChannelHeld, match="running a sequence"):
        cards.write_ao(V21_LOCK, 0.5)
    assert io.calls == []


def test_write_checks_motmaster_fresh_not_from_the_last_poll(cards, io, status):
    cards.state()
    status.running = True
    with pytest.raises(ChannelHeld):
        cards.write_do(MOT1_SHUTTER, True)
    assert io.calls == []


@pytest.mark.parametrize(
    "error, holder",
    [
        (MotMasterNotAnswering("no answer in 2.0 s"), "CaF MotMaster (not answering)"),
        (RuntimeError("bad proxy"), "CaF MotMaster (status error, see log)"),
    ],
)
def test_not_answering_motmaster_holds_and_refuses(
    cards, io, status, error, holder, caplog
):
    status.running = error
    cards.state()
    state = cards.state()
    assert state["card_holds"] == {"/PXI1Slot4": holder, "/PXI1Slot2": holder}
    with pytest.raises(ChannelHeld, match=re.escape(holder)):
        cards.read(["PXI1Slot2/ai0"])
    assert io.calls == []
    logged = [r for r in caplog.records if "bad proxy" in str(r.exc_info)]
    assert len(logged) == (1 if isinstance(error, RuntimeError) else 0)


@pytest.mark.parametrize(
    "setup_error, notice", [(False, UNREACHABLE), (True, SETUP_ERROR)]
)
def test_unreachable_motmaster_holds_nothing_and_says_so(
    cards, io, status, setup_error, notice
):
    status.running = None
    status.setup_error = setup_error
    state = cards.state()
    assert [n for n in state["notices"] if n in (UNREACHABLE, SETUP_ERROR)] == [notice]
    assert state["card_holds"] == {}
    cards.write_ao(V21_LOCK, 0.5)
    assert io.calls == [("write_ao", V21_LOCK, 0.5, 0.0, 1.0)]


@pytest.mark.parametrize(
    "devices, holder",
    [
        ({"CaF MotMaster": 42, "Rb MotMaster": 42}, "CaF MotMaster (queued run 42)"),
        ({"Rb MotMaster": 7}, None),
    ],
)
def test_queued_run_holds_while_task_reserves_this_motmaster(
    cards, queue, devices, holder
):
    queue.devices = devices
    state = cards.state()
    assert state["card_holds"].get("/PXI1Slot4") == holder
    assert state["holds"].get(MOT1_SHUTTER) == holder


def test_end_of_run_sets_analogue_values_and_resets_the_rest(cards, status):
    cards.write_do(MOT1_SHUTTER, False)
    cards.write_ao(V21_LOCK, 0.9)
    cards.write_ao(TCL_AO, 2.0)
    before = cards.state()["last_set"][TCL_AO]
    status.last = run(
        1, time.time() + 1.0, values={"v21Lock": 0.4, "MOT1Shutter": True}
    )
    last_set = cards.state()["last_set"]
    assert (
        last_set[V21_LOCK].items()
        >= {
            "value": 0.4,
            "source": "motmaster",
            "script": "LoadTweezer",
        }.items()
    )
    assert MOT1_SHUTTER not in last_set
    assert last_set[TCL_AO] == before


def test_protected_channel_shows_its_end_value(hardware_file, io, status, queue):
    cards = NICards(
        hardware_file,
        {},
        ["v21Lock"],
        "CaF MotMaster",
        io=io,
        motmaster_status=status,
        queue=queue,
    )
    status.last = run(1, time.time() + 1.0, values={"v21Lock": 0.4})
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.4


@pytest.mark.parametrize("outputs_known", [True, False])
def test_safe_state_channel_shows_its_safe_value_after_any_run(
    tmp_path, io, status, queue, outputs_known
):
    path = tmp_path / "MoleculeMOTHardware.cs"
    path.write_text(HARDWARE + PATTERN_BOARDS + SAFE_STATE)
    cards = make_cards(str(path), io, status, queue)
    status.last = run(
        1,
        time.time() + 1.0,
        values={"MOTCoilsCurrent": 0.8},
        outputs_known=outputs_known,
    )
    assert cards.state()["last_set"][MOT_COILS]["value"] == 0.0


def test_panel_write_newer_than_the_run_survives(cards, status):
    cards.write_ao(V21_LOCK, 0.7)
    status.last = run(1, time.time() - 60.0, values={"v21Lock": 0.4})
    assert (
        cards.state()["last_set"][V21_LOCK].items()
        >= {
            "value": 0.7,
            "source": "panel",
        }.items()
    )


def test_unknown_outputs_reset_pattern_channels(cards, status):
    cards.write_do(MOT1_SHUTTER, True)
    cards.write_do("PXI1Slot4/port0/line2", True)
    cards.write_ao(V21_LOCK, 0.5)
    cards.write_ao(TCL_AO, 2.0)
    status.last = run(1, time.time() + 1.0, outputs_known=False)
    assert list(cards.state()["last_set"]) == [TCL_AO]


def test_restarts_neither_show_stale_values_nor_roll_them_back(
    hardware_file, io, status, queue
):
    finished = time.time()
    status.last = run(5, finished, values={"v21Lock": 0.4})
    cards = make_cards(hardware_file, io, status, queue)
    assert V21_LOCK not in cards.state()["last_set"]
    status.last = run(6, finished + 60.0, values={"v21Lock": 0.6})
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.6
    status.last = run(1, finished - 600.0, values={"v21Lock": 0.1})
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.6
    status.last = None
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.6


def test_too_old_build_says_so_but_still_holds_while_running(cards, status):
    status.last = MotMasterTooOld("Controller has no GetLastRunOutputs")
    status.running = True
    state = cards.state()
    assert TOO_OLD in state["notices"]
    assert state["card_holds"]["/PXI1Slot4"] == RUNNING
