"""Saved parameters: named calibrated values shared through the Experiment Manager.

A saved parameter is a single float, int, bool or string kept under a key.
:class:`SavedParameterBook` holds the rules for creating and changing them,
:class:`SavedParameterStore` persists the book as JSON and
:class:`SavedParameterLog` appends every change to a JSONL file, newest read first.
"""

import difflib
import math
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pytweezer.experiment.queue import ModelStore, now

Value = float | int | bool | str


class SavedParameterError(ValueError):
    """A saved-parameter request that can't be carried out."""


class SavedParameter(BaseModel):
    """One saved value; ``updated_by`` is whoever last changed it."""

    key: str
    value: Value
    group: str = ""
    unit: str = ""
    updated_at: datetime
    updated_by: str


class SavedParameterChange(BaseModel):
    """One entry in the change log; ``old`` is ``None`` for a created key."""

    time: datetime
    who: str
    key: str
    old: Value | None
    new: Value
    reason: str
    rid: int | None = None


class SavedParameterState(BaseModel):
    schema_version: int = 1
    parameters: list[SavedParameter] = Field(default_factory=list)


def _checked(key: str, value: Value, existing: Value | None) -> Value:
    """``value`` as it is to be stored, or :class:`SavedParameterError`.

    A key keeps the type of its current value; an int is accepted for a float
    key and stored as a float.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise SavedParameterError(f"{key!r}: {value} is not a finite number")
    if existing is None:
        if type(value) not in (float, int, bool, str):
            raise SavedParameterError(
                f"{key!r}: {value!r} is not a float, int, bool or string"
            )
        return value
    kind = type(existing)
    if kind is float and type(value) is int:
        return float(value)
    if type(value) is not kind:
        raise SavedParameterError(
            f"{key!r} holds a {kind.__name__}; {value!r} is a {type(value).__name__}"
        )
    return value


class SavedParameterBook:
    """Every saved parameter, and the rules for changing them."""

    def __init__(self, state: SavedParameterState | None = None) -> None:
        self.state = state or SavedParameterState()

    def get(self, key: str) -> SavedParameter:
        key = key.strip()
        for parameter in self.state.parameters:
            if parameter.key == key:
                return parameter
        raise self._unknown(key)

    def _unknown(self, key: str) -> SavedParameterError:
        close = difflib.get_close_matches(
            key, [parameter.key for parameter in self.state.parameters], n=3
        )
        hint = f"; did you mean {', '.join(map(repr, close))}?" if close else ""
        return SavedParameterError(f"no saved parameter {key!r}{hint}")

    def all(self) -> list[SavedParameter]:
        return sorted(self.state.parameters, key=lambda p: (p.group, p.key))

    def set(
        self,
        values: dict[str, Value],
        *,
        reason: str,
        who: str,
        create: bool = False,
        group: str = "",
        unit: str = "",
        rid: int | None = None,
    ) -> list[SavedParameterChange]:
        """Set every value in ``values`` or, if any is refused, none of them.

        Keys that don't exist are refused unless ``create``; ``group`` and
        ``unit`` apply only to keys created here. Returns a change for each
        value that differs from the stored one.
        """
        existing = {parameter.key: parameter for parameter in self.state.parameters}
        planned: list[tuple[str, Value, SavedParameter | None]] = []
        for raw_key, value in values.items():
            key = raw_key.strip()
            if not key:
                raise SavedParameterError("a saved parameter needs a key")
            if any(key == planned_key for planned_key, _, _ in planned):
                raise SavedParameterError(f"{key!r} is given more than once")
            current = existing.get(key)
            if current is None and not create:
                raise self._unknown(key)
            stored = _checked(key, value, None if current is None else current.value)
            planned.append((key, stored, current))

        time = now()
        changes = []
        for key, value, current in planned:
            old = None if current is None else current.value
            if current is None:
                self.state.parameters.append(
                    SavedParameter(
                        key=key,
                        value=value,
                        group=group,
                        unit=unit,
                        updated_at=time,
                        updated_by=who,
                    )
                )
            elif old == value:
                continue
            else:
                current.value = value
                current.updated_at = time
                current.updated_by = who
            changes.append(
                SavedParameterChange(
                    time=time,
                    who=who,
                    key=key,
                    old=old,
                    new=value,
                    reason=reason,
                    rid=rid,
                )
            )
        return changes


class SavedParameterStore(ModelStore):
    """Persists a :class:`SavedParameterState`."""

    model = SavedParameterState


class SavedParameterLog:
    """Append-only JSONL record of every change to a saved parameter."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, changes: list[SavedParameterChange]) -> None:
        if not changes:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as file:
            file.writelines(change.model_dump_json() + "\n" for change in changes)

    def read(
        self, key: str | None = None, *, limit: int | None = None
    ) -> list[SavedParameterChange]:
        """Changes to ``key`` (every key if ``None``), newest first, at most ``limit``."""
        try:
            lines = self.path.read_text().splitlines()
        except FileNotFoundError:
            return []
        changes = [SavedParameterChange.model_validate_json(line) for line in lines]
        matching = [c for c in reversed(changes) if key in (None, c.key)]
        return matching if limit is None else matching[:limit]
