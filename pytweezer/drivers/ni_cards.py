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

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pytweezer.drivers.ni_hardware_file import HardwareFile, read_hardware_file
from pytweezer.logging_utils import get_logger

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
        io: NI-DAQmx access; the real hardware when omitted.
    """

    def __init__(
        self,
        hardware_file: str,
        reserved: dict | None = None,
        protected: list[str] | None = None,
        *,
        io: CardIO | None = None,
    ):
        if io is None:
            raise NotImplementedError("NI-DAQmx access is not written yet")
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
        static = self._static_holds(hardware, reserved or {}, protected or [])
        self.reservations = Reservations([lambda: static])

    def _build_layout(self, hardware: HardwareFile) -> None:
        for board in hardware.boards.values():
            info = self._io.card(board.device)
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
                    channel["range"] = list(
                        alias.ao_range or (info.ao_range if info else (0.0, 0.0))
                    )
            unaliased = []
            if info is not None:
                for kind, physicals in (
                    ("ao", info.ao),
                    ("ai", info.ai),
                    ("do", info.lines),
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
                "error": None if info else NOT_FOUND,
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
