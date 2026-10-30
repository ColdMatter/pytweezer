"""The NI cards on one lab PC, by their MOTMaster aliases, for the NI Cards tab.

``NICards`` reads the PC's MOTMaster hardware file for boards and aliases, asks
NI-DAQmx what each card physically has, and does single static writes and
on-demand reads. Channels another program owns are *reserved* (by alias or by
whole card) and protected channels are never touched; both are enforced here,
not only in the GUI. The last value written to each output is remembered,
because most of these cards cannot read an output back.

With ``motmaster`` set, MOTMaster's pattern boards are held whole while that
MOTMaster runs a sequence or a running queued task reserves it, and after each
run the analogue outputs show the values its pattern ended on. The run
MOTMaster reports as its last when the driver starts is not shown.

Config entry shape (``CONFIG["Devices"][name]``)::

    {
        "class": "pytweezer.drivers.ni_cards:NICards",
        "sim_class": "pytweezer.drivers.ni_cards:SimulatedNICards",
        "hardware_file": r"C:\\ControlPrograms\\EDMSuite\\DAQ\\MoleculeMOTHardware.cs",
        "reserved": {"Wavemeter Lock": {"channels": ["v00Lock"]},
                     "Hardware Control": {"cards": ["tclBoard2"]}},
        "protected": ["MOTCoilsCurrent"],
        "motmaster": "CaF MotMaster",
        "teardown": "close",
        ...
    }
"""

import math
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from pytweezer.configuration.paths import tweezerpath
from pytweezer.drivers.ni_cards_sources import (
    LastRun,
    ManagerQueue,
    MotMasterNotAnswering,
    MotMasterTooOld,
    RemotingMotMasterStatus,
    SimulatedMotMasterStatus,
)
from pytweezer.drivers.ni_hardware_file import HardwareFile, read_hardware_file
from pytweezer.logging_utils import get_logger
from pytweezer.servers.simulated_device import simulate

logger = get_logger("pytweezer.drivers.ni_cards")

PROTECTED = "Protected"
NOT_FOUND = "not found on this PC"
NO_PATTERN_BOARDS = (
    "The hardware file names no MOTMaster pattern boards: "
    "MOTMaster runs are not greyed out"
)
_KIND_ORDER = {"ai": 0, "ao": 1, "di": 2, "do": 3}

# TODO: check on the rig whether the PCI-6534 and PXI-6535 digital lines hold
# the pattern's last sample once the pattern task stops; if they do, set this
# to True. Until then a run leaves every digital pattern line "unknown".
SHOW_DIGITAL_END_VALUES = False


@dataclass
class CardInfo:
    """What NI-DAQmx reports for one device."""

    model: str
    ao: list[str]
    ai: list[str]
    lines: list[str]
    ao_range: tuple[float, float]
    input_lines: list[str] = field(default_factory=list)


class MotMasterStatus(Protocol):
    setup_error: bool

    def is_running(self) -> bool | None: ...
    def last_run(self) -> LastRun | None: ...


class Queue(Protocol):
    def running_devices(self) -> dict[str, int] | None: ...
    def close(self) -> None: ...


class CardIO(Protocol):
    def card(self, device: str) -> CardInfo | None: ...
    def write_do(self, physical: str, high: bool) -> None: ...
    def write_ao(
        self, physical: str, volts: float, low: float, high: float
    ) -> None: ...
    def read_ai(
        self, physicals: list[str], terminals: list[str | None]
    ) -> list[float]: ...
    def read_di(self, physicals: list[str]) -> list[bool]: ...


class ChannelHeld(Exception):
    """A write or read on a channel that is reserved or protected."""


@dataclass(frozen=True)
class Holds:
    """What is held: channels by physical name, whole cards by address.

    A source that holds a card also lists every physical on it in ``channels``.
    """

    channels: dict[str, str] = field(default_factory=dict)
    cards: dict[str, str] = field(default_factory=dict)


class Reservations:
    """Who holds what, merged from several sources; earlier sources win."""

    def __init__(self, sources: list[Callable[[], Holds]]):
        self.sources = sources

    def current(self) -> Holds:
        channels: dict[str, str] = {}
        cards: dict[str, str] = {}
        for source in self.sources:
            held = source()
            for physical, holder in held.channels.items():
                channels.setdefault(physical, holder)
            for address, holder in held.cards.items():
                cards.setdefault(address, holder)
        return Holds(channels, cards)


def _ao_range(declared: tuple | None, info: CardInfo | None) -> list[float]:
    """The declared range, clipped to what the card can output."""
    card = info.ao_range if info else (0.0, 0.0)
    if declared is None:
        return list(card)
    if card == (0.0, 0.0):
        return list(declared)
    return [max(declared[0], card[0]), min(declared[1], card[1])]


def _natural_key(physical: str):
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", physical)
    ]


def _channel_key(channel: dict):
    return _KIND_ORDER[channel["kind"]], _natural_key(channel["physical"])


class NICards:
    """Drive and read the NI cards declared in one MOTMaster hardware file.

    Args:
        hardware_file: Path to the PC's ``*Hardware.cs``.
        reserved: ``{holder: {"channels": [alias, ...], "cards": [board, ...]}}``.
        protected: Aliases that are never written or read.
        motmaster: The ``CONFIG["Devices"]`` name of the MOTMaster that drives
            this PC's pattern boards; ``None`` for no MOTMaster holds.
        io: NI-DAQmx access; the real hardware when omitted.
        motmaster_status: How to ask MOTMaster whether it is running and what
            its last run ended on; over .NET remoting when omitted.
        queue: Which devices the Experiment Manager's running tasks hold; a
            mirror of the manager's state when omitted.

    ``reserved`` and ``protected`` have no defaults, so a misspelt config key
    fails construction instead of leaving channels unprotected.
    """

    def __init__(
        self,
        hardware_file: str,
        reserved: dict,
        protected: list[str],
        motmaster: str | None = None,
        *,
        io: CardIO | None = None,
        motmaster_status: MotMasterStatus | None = None,
        queue: Queue | None = None,
    ):
        if io is None:
            io = DaqmxIO()
        self._io = io
        self._error = None
        self._cards: list[dict] = []
        self._channels: dict[str, dict] = {}
        self._card_of: dict[str, dict] = {}
        self._last_set: dict[str, dict] = {}
        try:
            hardware = read_hardware_file(hardware_file)
        except OSError as error:
            self._error = f"Cannot read {hardware_file}: {error}"
            logger.error(self._error)
            hardware = HardwareFile({}, [])
        self._build_layout(hardware)
        static = self._static_holds(hardware, reserved, protected)
        addresses = {card["address"] for card in self._cards}
        self._pattern_boards = [a for a in hardware.pattern_boards if a in addresses]
        self._pattern_channels = {
            p for a in self._pattern_boards for p in self._card_channels(a)
        }
        self._pattern_aliases = {
            alias: self._channels[p]
            for p in self._pattern_channels
            for alias in self._channels[p]["aliases"]
        }
        self._safe_volts = {
            p: hardware.safe_states[alias]
            for p, channel in self._channels.items()
            if channel["kind"] == "ao"
            for alias in channel["aliases"]
            if alias in hardware.safe_states
        }
        self._motmaster = motmaster
        self._status = None
        self._queue = queue
        self._closed = False
        self._motmaster_reachable = True
        self._motmaster_setup_error = False
        self._manager_reachable = True
        self._motmaster_too_old = False
        self._last_seen_finished = -math.inf
        self._logged_errors: set[str] = set()
        sources = [lambda: static]
        if motmaster is not None and self._pattern_boards:
            self._status = (
                motmaster_status
                if motmaster_status is not None
                else RemotingMotMasterStatus(motmaster)
            )
            if self._queue is None:
                self._queue = ManagerQueue()
            sources += [self._running_holds, self._queued_holds]
        self.reservations = Reservations(sources)
        # Panel writes since MOTMaster's last run went with the previous server,
        # so that run's values may be stale: pattern channels stay unknown until
        # the next run finishes.
        self._sync_end_of_run(apply=False)

    def _build_layout(self, hardware: HardwareFile) -> None:
        for board in hardware.boards.values():
            try:
                info = self._io.card(board.device)
                error = None if info else NOT_FOUND
            except Exception as exc:
                logger.exception("Cannot query %s (%s)", board.name, board.address)
                info, error = None, f"unreadable by NI-DAQmx: {exc}"
            aliased: dict[str, dict] = {}
            for alias in hardware.aliases:
                if alias.board != board.name:
                    continue
                channel = aliased.setdefault(
                    alias.physical,
                    {
                        "physical": alias.physical,
                        "kind": alias.kind,
                        "aliases": [],
                        "range": None,
                        "terminal": alias.terminal,
                    },
                )
                channel["aliases"].append(alias.name)
                if alias.kind == "ao":
                    channel["range"] = _ao_range(alias.ao_range, info)
            unaliased = []
            if info is not None:
                for kind, physicals in (
                    ("ao", info.ao),
                    ("ai", info.ai),
                    ("do", info.lines),
                    ("di", info.input_lines),
                ):
                    for physical in physicals:
                        if physical not in aliased:
                            unaliased.append(
                                {
                                    "physical": physical,
                                    "kind": kind,
                                    "aliases": [],
                                    "range": list(info.ao_range)
                                    if kind == "ao"
                                    else None,
                                    "terminal": None,
                                }
                            )
            card = {
                "board": board.name,
                "address": board.address,
                "model": info.model if info else None,
                "error": error,
                "channels": sorted(aliased.values(), key=_channel_key),
                "unaliased": sorted(unaliased, key=_channel_key),
            }
            self._cards.append(card)
            for channel in card["channels"] + card["unaliased"]:
                self._channels[channel["physical"]] = channel
                self._card_of[channel["physical"]] = card

    def _static_holds(
        self, hardware: HardwareFile, reserved: dict, protected: list[str]
    ) -> Holds:
        for holder, held in reserved.items():
            if unknown := set(held) - {"channels", "cards"}:
                raise ValueError(
                    f"{holder}'s reservation has unknown keys {sorted(unknown)}; "
                    "expected 'channels' and/or 'cards'"
                )
        if self._error:
            return Holds()
        by_alias = {alias.name: alias.physical for alias in hardware.aliases}
        channels: dict[str, str] = {}
        cards: dict[str, str] = {}
        for name in protected:
            if name not in by_alias:
                raise ValueError(
                    f"Protected channel {name!r} is not in the hardware file"
                )
            channels[by_alias[name]] = PROTECTED
        for holder, held in reserved.items():
            for name in held.get("channels", []):
                if name not in by_alias:
                    raise ValueError(
                        f"{holder} reserves {name!r}, which is not in the hardware file"
                    )
                channels.setdefault(by_alias[name], holder)
            for board_name in held.get("cards", []):
                board = hardware.boards.get(board_name)
                if board is None:
                    raise ValueError(
                        f"{holder} reserves card {board_name!r}, which is not in the hardware file"
                    )
                cards.setdefault(board.address, holder)
                for physical in self._card_channels(board.address):
                    channels.setdefault(physical, holder)
        return Holds(channels, cards)

    def _card_channels(self, address: str) -> list[str]:
        """The physicals of the card at ``address``, aliased and unaliased."""
        return [
            channel["physical"]
            for card in self._cards
            if card["address"] == address
            for channel in card["channels"] + card["unaliased"]
        ]

    def _hold_pattern_boards(self, why: str) -> Holds:
        holder = f"{self._motmaster} ({why})"
        return Holds(
            dict.fromkeys(self._pattern_channels, holder),
            dict.fromkeys(self._pattern_boards, holder),
        )

    def _log_once(self, call: str, error: Exception) -> None:
        """Log an unexpected failure of ``call`` the first time its message is seen."""
        message = f"{call}: {type(error).__name__}: {error}"
        if message not in self._logged_errors:
            self._logged_errors.add(message)
            logger.exception("%s failed unexpectedly", call)

    def _running_holds(self) -> Holds:
        self._motmaster_reachable = True
        try:
            running = self._status.is_running()
        except MotMasterNotAnswering:
            return self._hold_pattern_boards("not answering")
        except Exception as error:
            self._log_once(f"{self._motmaster} is_running", error)
            return self._hold_pattern_boards("status error, see log")
        if running is None:
            self._motmaster_reachable = False
            self._motmaster_setup_error = self._status.setup_error
        if running:
            return self._hold_pattern_boards("running a sequence")
        return Holds()

    def _queued_holds(self) -> Holds:
        try:
            devices = self._queue.running_devices()
        except Exception as error:
            self._log_once("Experiment Manager queue", error)
            devices = None
        self._manager_reachable = devices is not None
        if devices and self._motmaster in devices:
            return self._hold_pattern_boards(f"queued run {devices[self._motmaster]}")
        return Holds()

    def _sync_end_of_run(self, apply: bool = True) -> None:
        """Apply MOTMaster's last run if it finished after the last one seen.

        Args:
            apply: If false, only mark the run as seen, leaving its values
                unapplied.
        """
        if self._status is None:
            return
        try:
            last = self._status.last_run()
        except MotMasterTooOld:
            self._motmaster_too_old = True
            return
        except MotMasterNotAnswering:
            return
        except Exception as error:
            self._log_once(f"{self._motmaster} last_run", error)
            return
        self._motmaster_too_old = False
        # A restarted MOTMaster numbers its runs from 1 again, so only a later
        # finish time counts as a new run.
        if last is None or last.finished <= self._last_seen_finished:
            return
        self._last_seen_finished = last.finished
        if not apply:
            return
        for physical in self._pattern_channels & self._last_set.keys():
            if self._last_set[physical]["time"] < last.finished:
                del self._last_set[physical]
        end_values = {}
        if last.outputs_known:
            shown = ("ao", "do") if SHOW_DIGITAL_END_VALUES else ("ao",)
            for name, value in last.values.items():
                channel = self._pattern_aliases.get(name)
                if channel is not None and channel["kind"] in shown:
                    end_values[channel["physical"]] = (
                        float(value) if channel["kind"] == "ao" else bool(value)
                    )
        # MOTMaster sets its safe state after every run, however the run ended.
        end_values.update(self._safe_volts)
        for physical, value in end_values.items():
            newer = self._last_set.get(physical, {}).get("time", -math.inf)
            if newer >= last.finished:
                continue
            self._last_set[physical] = {
                "value": value,
                "time": last.finished,
                "source": "motmaster",
                "script": last.script,
            }

    def _notices(self) -> list[str]:
        if self._motmaster is None:
            return []
        if not self._pattern_boards:
            if self._error:
                return []
            return [NO_PATTERN_BOARDS]
        notices = []
        if not self._motmaster_reachable and self._motmaster_setup_error:
            notices.append(
                f"Cannot connect to {self._motmaster} (setup error, see log): "
                "its runs are not greyed out"
            )
        elif not self._motmaster_reachable:
            notices.append(
                f"{self._motmaster} is not reachable: its runs are not greyed out"
            )
        if self._motmaster_too_old:
            notices.append(
                f"{self._motmaster} does not report end-of-run values: "
                "it needs a newer MOTMaster build"
            )
        if not self._manager_reachable:
            notices.append(
                "The Experiment Manager is not reachable: queued runs are not greyed out"
            )
        return notices

    def close(self) -> None:
        """Stop mirroring the Experiment Manager's queue."""
        if self._queue is not None and not self._closed:
            self._closed = True
            self._queue.close()

    def layout(self) -> dict:
        """Every card with its aliased and unaliased channels, or the file error."""
        return {"error": self._error, "cards": self._cards}

    def state(self) -> dict:
        """What is held and by whom, the last value of each output, and notices.

        A ``last_set`` entry has ``"value"``, ``"time"`` and ``"source"``
        (``"panel"`` or ``"motmaster"``); a MOTMaster entry also has the
        ``"script"`` whose run ended on that value.
        """
        held = self.reservations.current()
        self._sync_end_of_run()
        return {
            "holds": held.channels,
            "card_holds": held.cards,
            "last_set": dict(self._last_set),
            "notices": self._notices(),
        }

    def _usable(self, physical: str, kinds: tuple[str, ...], held: Holds) -> dict:
        holder = held.channels.get(physical)
        if holder is not None:
            raise ChannelHeld(f"{physical} is held by {holder}")
        channel = self._channels.get(physical)
        if channel is None:
            raise ValueError(f"{physical} is not a channel on this PC")
        card = self._card_of[physical]
        if card["error"]:
            raise ValueError(f"{card['board']} ({card['address']}) is {card['error']}")
        if channel["kind"] not in kinds:
            raise ValueError(
                f"{physical} is {channel['kind'].upper()}, not {'/'.join(kinds).upper()}"
            )
        return channel

    def write_do(self, physical: str, high: bool) -> None:
        """Set a digital output line high or low."""
        self._usable(physical, ("do",), self.reservations.current())
        self._io.write_do(physical, bool(high))
        self._last_set[physical] = {
            "value": bool(high),
            "time": time.time(),
            "source": "panel",
        }

    def write_ao(self, physical: str, volts: float) -> None:
        """Set an analogue output, within its declared range or the card's."""
        channel = self._usable(physical, ("ao",), self.reservations.current())
        low, high = channel["range"]
        if not low <= volts <= high:
            raise ValueError(f"{volts} V is outside {physical}'s range {low}…{high} V")
        self._io.write_ao(physical, float(volts), low, high)
        self._last_set[physical] = {
            "value": float(volts),
            "time": time.time(),
            "source": "panel",
        }

    def read(self, physicals: list[str]) -> dict[str, float | bool]:
        """One reading from each analogue or digital input."""
        held = self.reservations.current()
        channels = [self._usable(p, ("ai", "di"), held) for p in physicals]
        ai = [c for c in channels if c["kind"] == "ai"]
        di = [c for c in channels if c["kind"] == "di"]
        values: dict[str, float | bool] = {}
        if ai:
            readings = self._io.read_ai(
                [c["physical"] for c in ai], [c["terminal"] for c in ai]
            )
            values.update(zip([c["physical"] for c in ai], readings))
        if di:
            readings = self._io.read_di([c["physical"] for c in di])
            values.update(zip([c["physical"] for c in di], readings))
        return {p: values[p] for p in physicals}


_TERMINALS = {
    "Rse": "RSE",
    "Nrse": "NRSE",
    "Differential": "DIFF",
    "PseudoDifferential": "PSEUDO_DIFF",
}


def _by_device(physicals: list[str]) -> dict[str, list[int]]:
    """Indices of ``physicals`` grouped by device, in first-seen order."""
    groups: dict[str, list[int]] = {}
    for index, physical in enumerate(physicals):
        groups.setdefault(physical.split("/", 1)[0], []).append(index)
    return groups


def _as_list(reading) -> list:
    # nidaqmx returns a bare value for a single channel, else a list.
    return list(reading) if isinstance(reading, (list, tuple)) else [reading]


def _widest(voltages: list[float]) -> tuple[float, float]:
    """The widest ``(low, high)`` of NI-DAQmx's flat list of range pairs."""
    pairs = list(zip(voltages[::2], voltages[1::2]))
    low, high = max(pairs, key=lambda p: p[1] - p[0]) if pairs else (0.0, 0.0)
    return float(low), float(high)


class DaqmxIO:
    """NI-DAQmx on this PC. Every call opens a task and closes it before returning."""

    def __init__(self):
        self._ai_ranges: dict[str, tuple[float, float]] = {}

    def card(self, device: str) -> CardInfo | None:
        import nidaqmx.system

        # TODO: an unplugged USB card that is still configured in MAX is listed
        # here, so it shows as present and fails only on a write. Fine for the
        # PXIe cards this is for; check presence properly if USB cards matter.
        if device not in nidaqmx.system.System.local().devices.device_names:
            return None
        info = nidaqmx.system.Device(device)
        ao = list(info.ao_physical_chans.channel_names)
        ai = list(info.ai_physical_chans.channel_names)
        outputs = list(info.do_lines.channel_names)
        if ai:
            self._ai_ranges[device] = _widest(list(info.ai_voltage_rngs))
        return CardInfo(
            model=info.product_type,
            ao=ao,
            ai=ai,
            lines=outputs,
            ao_range=_widest(list(info.ao_voltage_rngs)) if ao else (0.0, 0.0),
            input_lines=[
                line for line in info.di_lines.channel_names if line not in outputs
            ],
        )

    def write_do(self, physical: str, high: bool) -> None:
        import nidaqmx

        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(physical)
            task.write(high)

    def write_ao(self, physical: str, volts: float, low: float, high: float) -> None:
        import nidaqmx

        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(physical, min_val=low, max_val=high)
            task.write(volts)

    def read_ai(self, physicals: list[str], terminals: list[str | None]) -> list[float]:
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration

        values: list[float] = [0.0] * len(physicals)
        for device, indices in _by_device(physicals).items():
            # Without explicit limits DAQmx reads ±5 V and saturates above it.
            low, high = self._ai_ranges.get(device, (-10.0, 10.0))
            with nidaqmx.Task() as task:
                for i in indices:
                    terminal = _TERMINALS.get(terminals[i] or "", "DEFAULT")
                    task.ai_channels.add_ai_voltage_chan(
                        physicals[i],
                        terminal_config=TerminalConfiguration[terminal],
                        min_val=low,
                        max_val=high,
                    )
                for i, value in zip(indices, _as_list(task.read())):
                    values[i] = float(value)
        return values

    def read_di(self, physicals: list[str]) -> list[bool]:
        import nidaqmx

        values: list[bool] = [False] * len(physicals)
        for indices in _by_device(physicals).values():
            with nidaqmx.Task() as task:
                for i in indices:
                    task.di_channels.add_di_chan(physicals[i])
                for i, value in zip(indices, _as_list(task.read())):
                    values[i] = bool(value)
        return values


class SimulatedIO:
    """Every board present, with a full set of channels and plausible readings."""

    def __init__(self, extra: list[str] = ()):
        self._extra = list(extra)
        self._written: dict[str, float | bool] = {}
        self._t0 = time.time()

    def card(self, device: str) -> CardInfo:
        def mine(prefix):
            return [p for p in self._extra if p.startswith(f"{device}/{prefix}")]

        def merged(standard, prefix):
            return list(dict.fromkeys(standard + mine(prefix)))

        return CardInfo(
            model="Simulated",
            ao=merged([f"{device}/ao{i}" for i in range(8)], "ao"),
            ai=merged([f"{device}/ai{i}" for i in range(8)], "ai"),
            lines=merged(
                [f"{device}/port{p}/line{n}" for p in range(4) for n in range(8)],
                "port",
            ),
            ao_range=(-10.0, 10.0),
        )

    def write_do(self, physical: str, high: bool) -> None:
        self._written[physical] = high

    def write_ao(self, physical: str, volts: float, low: float, high: float) -> None:
        self._written[physical] = volts

    def read_ai(self, physicals: list[str], terminals: list[str | None]) -> list[float]:
        t = time.time() - self._t0
        return [
            math.sin(2 * math.pi * 0.1 * t + index) + random.gauss(0.0, 0.01)
            for index, _ in enumerate(physicals)
        ]

    def read_di(self, physicals: list[str]) -> list[bool]:
        return [bool(self._written.get(p, False)) for p in physicals]


@simulate(NICards)
class SimulatedNICards(NICards):
    """``NICards`` on simulated cards.

    A lab-PC path to the hardware file that does not exist here falls back to the
    same file in a sibling EDMSuite checkout (``<repo>/../EDMSuite/DAQ/``).
    MOTMaster never runs in simulation; the Experiment Manager's queue is real.
    """

    def __init__(
        self,
        hardware_file: str,
        reserved: dict,
        protected: list[str],
        motmaster: str | None = None,
    ):
        if not os.path.exists(hardware_file):
            name = re.split(r"[\\/]", hardware_file)[-1]
            sibling = os.path.join(tweezerpath, "..", "EDMSuite", "DAQ", name)
            if os.path.exists(sibling):
                hardware_file = os.path.realpath(sibling)
        try:
            aliased = [a.physical for a in read_hardware_file(hardware_file).aliases]
        except OSError:
            aliased = []
        super().__init__(
            hardware_file,
            reserved,
            protected,
            motmaster,
            io=SimulatedIO(aliased),
            motmaster_status=SimulatedMotMasterStatus(),
            queue=ManagerQueue() if motmaster is not None else None,
        )
