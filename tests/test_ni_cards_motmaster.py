"""The NI Cards driver while MOTMaster runs, or a queued task reserves it."""

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

RUNNING = "CaF MotMaster (running a sequence)"
UNREACHABLE = "CaF MotMaster is not reachable: its runs are not greyed out"
TOO_OLD = (
    "CaF MotMaster does not report end-of-run values: it needs a newer MOTMaster build"
)

V21_LOCK = "PXI1Slot2/ao0"
MOT1_SHUTTER = "PXI1Slot4/port3/line5"
TCL_AO = "PXI1Slot8/ao0"


class FakeStatus:
    """MOTMaster's answers; an exception instance is raised instead of returned."""

    def __init__(self):
        self.running = False
        self.last = None

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
    "error", [MotMasterNotAnswering("no answer in 2.0 s"), RuntimeError("bad proxy")]
)
def test_not_answering_motmaster_holds_and_refuses(cards, io, status, error, caplog):
    status.running = error
    cards.state()
    state = cards.state()
    holder = "CaF MotMaster (not answering)"
    assert state["card_holds"] == {"/PXI1Slot4": holder, "/PXI1Slot2": holder}
    with pytest.raises(ChannelHeld, match="not answering"):
        cards.read(["PXI1Slot2/ai0"])
    assert io.calls == []
    logged = [r for r in caplog.records if "bad proxy" in str(r.exc_info)]
    assert len(logged) == (1 if isinstance(error, RuntimeError) else 0)


def test_unreachable_motmaster_holds_nothing_and_says_so(cards, io, status):
    status.running = None
    state = cards.state()
    assert UNREACHABLE in state["notices"]
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


def test_restarted_motmaster_does_not_roll_values_back(
    hardware_file, io, status, queue
):
    finished = time.time()
    status.last = run(5, finished, values={"v21Lock": 0.4})
    cards = make_cards(hardware_file, io, status, queue)
    status.last = run(1, finished - 600.0, values={"v21Lock": 0.1})
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.4
    status.last = None
    assert cards.state()["last_set"][V21_LOCK]["value"] == 0.4


def test_too_old_build_says_so_but_still_holds_while_running(cards, status):
    status.last = MotMasterTooOld("Controller has no GetLastRunOutputs")
    status.running = True
    state = cards.state()
    assert TOO_OLD in state["notices"]
    assert state["card_holds"]["/PXI1Slot4"] == RUNNING
