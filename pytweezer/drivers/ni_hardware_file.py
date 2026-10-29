"""Channel aliases read from a MOTMaster hardware file (``EDMSuite/DAQ/*Hardware.cs``).

The C# constructor of a ``DAQ.HAL.Hardware`` subclass declares its boards with
``Boards.Add(name, address)`` and its channels with ``AddAnalogOutputChannel``,
``AddAnalogInputChannel``, ``AddDigitalOutputChannel`` and
``AddDigitalInputChannel``. Only those declarations are read; everything else in
the file is ignored. Board addresses are usually bound to a local string first
(``string aoBoard = (string)Boards["aoBoard"];``), so string variables and
``Boards[...]`` lookups are followed in source order.

The boards MOTMaster drives as pattern generators are named by three ``Info``
entries: ``PatternGeneratorBoard``, ``AdditionalPatternGeneratorBoards`` and
``AnalogBoards``. The last two are dictionaries built with ``<variable>.Add``.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.drivers.ni_hardware_file")

Kind = Literal["ao", "ai", "do", "di"]


@dataclass(frozen=True)
class Board:
    """A card, by its name in the hardware file and its NI-DAQmx address."""

    name: str
    address: str

    @property
    def device(self) -> str:
        """The NI-DAQmx device name, e.g. ``"PXI1Slot4"`` for ``"/PXI1Slot4"``."""
        return self.address.strip("/")


@dataclass(frozen=True)
class Alias:
    """One declared channel."""

    name: str
    kind: Kind
    physical: str
    board: str
    ao_range: tuple[float, float] | None
    terminal: str | None


@dataclass(frozen=True)
class HardwareFile:
    """The boards (in ``Boards.Add`` order) and aliases (in declaration order).

    ``pattern_boards`` holds the addresses of the boards MOTMaster drives as
    pattern generators, without duplicates.
    """

    boards: dict[str, Board]
    aliases: list[Alias]
    pattern_boards: list[str] = field(default_factory=list)


_KINDS = {
    ("Analog", "Output"): "ao",
    ("Analog", "Input"): "ai",
    ("Digital", "Output"): "do",
    ("Digital", "Input"): "di",
}

_PATTERN_BOARD_KEYS = (
    "PatternGeneratorBoard",
    "AdditionalPatternGeneratorBoards",
    "AnalogBoards",
)

_STATEMENT = re.compile(
    r"\b[Ss]tring\s+(?P<var>\w+)\s*=\s*(?P<value>[^;]*);"
    r"|\bBoards\.Add\("
    r"|\b(?P<target>\w+)\.Add\("
    r"|\bAdd(?P<signal>Analog|Digital)(?P<direction>Input|Output)Channel\("
)
_CHAR_LITERAL = re.compile(r"'(?:\\.|[^'\\])'")
_LOOKUP = re.compile(r'\(\s*string\s*\)\s*Boards\s*\[\s*"([^"]*)"\s*\]')


def _number(literal: str) -> float:
    """A C# numeric literal, with any ``f``/``d``/``m`` suffix."""
    return float(literal.rstrip("fFdDmM"))


def strip_comments(source: str) -> str:
    """``source`` with ``//`` and ``/* */`` comments removed, string literals intact."""
    out = []
    i, n = 0, len(source)
    while i < n:
        if source.startswith('@"', i):
            end = i + 2
            while end < n and (source[end] != '"' or source.startswith('""', end)):
                end += 2 if source[end] == '"' else 1
            out.append(source[i : end + 1])
            i = end + 1
        elif source[i] == '"':
            end = i + 1
            while end < n and source[end] != '"':
                end += 2 if source[end] == "\\" else 1
            out.append(source[i : end + 1])
            i = end + 1
        elif char := _CHAR_LITERAL.match(source, i):
            out.append(char.group())
            i = char.end()
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def _call_arguments(source: str, start: int) -> tuple[list[str], int]:
    """Split the arguments of a call whose ``(`` ends just before ``start``."""
    args, depth, current, i = [], 0, [], start
    while i < len(source):
        char = source[i]
        if char == '"':
            end = source.find('"', i + 1)
            if end < 0:
                end = len(source) - 1
            current.append(source[i : end + 1])
            i = end + 1
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            if depth == 0:
                args.append("".join(current).strip())
                return args, i + 1
            depth -= 1
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(char)
        i += 1
    return args, i


def parse_hardware_file(source: str) -> HardwareFile:
    """Read the boards and channel aliases a hardware file declares."""
    source = strip_comments(source)
    variables: dict[str, str] = {}
    boards: dict[str, Board] = {}
    aliases: list[Alias] = []
    dictionary_values: dict[str, list[str]] = {}
    pattern_expressions: dict[str, str] = {}

    def evaluate(expression: str) -> str | None:
        parts = []
        for term in expression.split("+"):
            term = term.strip()
            if lookup := _LOOKUP.fullmatch(term):
                board = boards.get(lookup.group(1))
                value = board.address if board else None
            elif term.startswith('"') and term.endswith('"'):
                value = term[1:-1]
            else:
                value = variables.get(term)
            if value is None:
                return None
            parts.append(value)
        return "".join(parts)

    def board_at(address: str) -> Board | None:
        return next((b for b in boards.values() if b.address == address), None)

    position = 0
    while match := _STATEMENT.search(source, position):
        position = match.end()
        if match.group("var"):
            value = evaluate(match.group("value"))
            if value is not None:
                variables[match.group("var")] = value
            continue
        args, position = _call_arguments(source, match.end())
        if match.group("target"):
            if len(args) != 2:
                continue
            if match.group("target") != "Info":
                if (value := evaluate(args[1])) is not None:
                    dictionary_values.setdefault(match.group("target"), []).append(
                        value
                    )
            elif (key := evaluate(args[0])) in _PATTERN_BOARD_KEYS:
                pattern_expressions[key] = args[1]
            continue
        if not match.group("signal"):
            if len(args) == 2:
                name, address = evaluate(args[0]), evaluate(args[1])
                if name is not None and address is not None:
                    boards[name] = Board(name, address)
            continue

        kind = _KINDS[match.group("signal"), match.group("direction")]
        alias = evaluate(args[0]) if args else None
        if alias is None:
            continue
        if kind in ("do", "di"):
            if len(args) != 4 or not (args[2].isdigit() and args[3].isdigit()):
                logger.warning("Skipping %r: its port and line are not literals", alias)
                continue
            address = evaluate(args[1])
            board = board_at(address) if address else None
            physical = board and f"{board.device}/port{args[2]}/line{args[3]}"
        else:
            full = evaluate(args[1]) if len(args) >= 2 else None
            address, _, channel = (full or "").rpartition("/")
            board = board_at(address) if full else None
            physical = board and f"{board.device}/{channel}"
        if board is None:
            logger.debug("Skipping %r: its board is not declared", alias)
            continue

        ao_range = None
        terminal = None
        if kind == "ao" and len(args) == 4:
            try:
                ao_range = (_number(args[2]), _number(args[3]))
            except ValueError:
                logger.warning("%r: range %s, %s is not literal", alias, *args[2:])
        if kind == "ai" and len(args) >= 3:
            terminal = args[2].rpartition(".")[2]
        aliases.append(Alias(alias, kind, physical, board.name, ao_range, terminal))

    pattern_boards: list[str] = []
    for key in _PATTERN_BOARD_KEYS:
        if (expression := pattern_expressions.get(key)) is None:
            continue
        if key == "PatternGeneratorBoard":
            addresses = [evaluate(expression)]
        else:
            addresses = dictionary_values.get(expression, [])
        for address in addresses:
            if address is None:
                logger.debug("Skipping %s: %s is not a known address", key, expression)
            elif address not in pattern_boards:
                pattern_boards.append(address)

    return HardwareFile(boards, aliases, pattern_boards)


def read_hardware_file(path: str) -> HardwareFile:
    """Parse the hardware file at ``path``; raises ``OSError`` if it cannot be read."""
    with open(path, encoding="utf-8", errors="replace") as file:
        return parse_hardware_file(file.read())
