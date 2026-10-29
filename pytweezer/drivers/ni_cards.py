"""The NI cards on one lab PC, by their MOTMaster aliases, for the NI Cards tab.

``NICards`` reads the PC's MOTMaster hardware file for boards and aliases, asks
NI-DAQmx what each card physically has, and does single static writes and
on-demand reads. Channels another program owns are *reserved* (by alias or by
whole card) and protected channels are never touched; both are enforced here,
not only in the GUI. The last value written to each output is remembered,
because most of these cards cannot read an output back.

Config entry shape (``CONFIG["Devices"][name]``)::

    {
        "class": "pytweezer.drivers.ni_cards:NICards",
        "sim_class": "pytweezer.drivers.ni_cards:SimulatedNICards",
        "hardware_file": r"C:\\ControlPrograms\\EDMSuite\\DAQ\\MoleculeMOTHardware.cs",
        "reserved": {"Wavemeter Lock": {"channels": ["v00Lock"]},
                     "Hardware Control": {"cards": ["tclBoard2"]}},
        "protected": ["MOTCoilsCurrent"],
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
from pytweezer.drivers.ni_hardware_file import HardwareFile, read_hardware_file
from pytweezer.logging_utils import get_logger
from pytweezer.servers.simulated_device import simulate

logger = get_logger("pytweezer.drivers.ni_cards")

PROTECTED = "Protected"
NOT_FOUND = "not found on this PC"
_KIND_ORDER = {"ai": 0, "ao": 1, "di": 2, "do": 3}


@dataclass
class CardInfo:
    """What NI-DAQmx reports for one device."""

    model: str
    ao: list[str]
    ai: list[str]
    lines: list[str]
    ao_range: tuple[float, float]
    input_lines: list[str] = field(default_factory=list)


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


class Reservations:
    """Who holds which physical channel, merged from several sources.

    Each source returns ``{physical: holder}``; earlier sources win.
    """

    def __init__(self, sources: list[Callable[[], dict[str, str]]]):
        self.sources = sources

    def holds(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for source in self.sources:
            for physical, holder in source().items():
                merged.setdefault(physical, holder)
        return merged


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

    ``reserved`` and ``protected`` have no defaults, so a misspelt config key
    fails construction instead of leaving channels unprotected.
        io: NI-DAQmx access; the real hardware when omitted.
    """

    def __init__(
        self,
        hardware_file: str,
        reserved: dict,
        protected: list[str],
        *,
        io: CardIO | None = None,
    ):
        if io is None:
            io = DaqmxIO()
        self._io = io
        self._error = None
        self._cards: list[dict] = []
        self._channels: dict[str, dict] = {}
        self._card_of: dict[str, dict] = {}
        self._last_set: dict[str, dict] = {}
        self._card_holds: dict[str, str] = {}
        try:
            hardware = read_hardware_file(hardware_file)
        except OSError as error:
            self._error = f"Cannot read {hardware_file}: {error}"
            logger.error(self._error)
            hardware = HardwareFile({}, [])
        self._build_layout(hardware)
        static = self._static_holds(hardware, reserved, protected)
        self.reservations = Reservations([lambda: static])

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
    ) -> dict[str, str]:
        for holder, held in reserved.items():
            if unknown := set(held) - {"channels", "cards"}:
                raise ValueError(
                    f"{holder}'s reservation has unknown keys {sorted(unknown)}; "
                    "expected 'channels' and/or 'cards'"
                )
        if self._error:
            return {}
        by_alias = {alias.name: alias.physical for alias in hardware.aliases}
        holds = {}
        for name in protected:
            if name not in by_alias:
                raise ValueError(
                    f"Protected channel {name!r} is not in the hardware file"
                )
            holds[by_alias[name]] = PROTECTED
        for holder, held in reserved.items():
            for name in held.get("channels", []):
                if name not in by_alias:
                    raise ValueError(
                        f"{holder} reserves {name!r}, which is not in the hardware file"
                    )
                holds.setdefault(by_alias[name], holder)
            for board_name in held.get("cards", []):
                board = hardware.boards.get(board_name)
                if board is None:
                    raise ValueError(
                        f"{holder} reserves card {board_name!r}, which is not in the hardware file"
                    )
                self._card_holds[board.address] = holder
                for physical, card in self._card_of.items():
                    if card["address"] == board.address:
                        holds.setdefault(physical, holder)
        return holds

    def layout(self) -> dict:
        """Every card with its aliased and unaliased channels, or the file error."""
        return {"error": self._error, "cards": self._cards}

    def state(self) -> dict:
        """What is held and by whom, and the last value written to each output."""
        return {
            "holds": self.reservations.holds(),
            "card_holds": dict(self._card_holds),
            "last_set": dict(self._last_set),
        }

    def _usable(self, physical: str, kinds: tuple[str, ...]) -> dict:
        holder = self.reservations.holds().get(physical)
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
        self._usable(physical, ("do",))
        self._io.write_do(physical, bool(high))
        self._last_set[physical] = {"value": bool(high), "time": time.time()}

    def write_ao(self, physical: str, volts: float) -> None:
        """Set an analogue output, within its declared range or the card's."""
        low, high = self._usable(physical, ("ao",))["range"]
        if not low <= volts <= high:
            raise ValueError(f"{volts} V is outside {physical}'s range {low}…{high} V")
        self._io.write_ao(physical, float(volts), low, high)
        self._last_set[physical] = {"value": float(volts), "time": time.time()}

    def read(self, physicals: list[str]) -> dict[str, float | bool]:
        """One reading from each analogue or digital input."""
        channels = [self._usable(p, ("ai", "di")) for p in physicals]
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
    """

    def __init__(
        self,
        hardware_file: str,
        reserved: dict,
        protected: list[str],
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
        super().__init__(hardware_file, reserved, protected, io=SimulatedIO(aliased))
