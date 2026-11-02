"""Measurement files, read where they are or fetched from the Experiment Manager.

The manager serves its data root to other PCs as the sipyco RPC target
:data:`RPC_TARGET` on ``results_port``, so no PC needs a share of its disk.
:class:`ResultsArchive` reads a data root directly; :class:`RemoteArchive` is
the same interface over RPC. :func:`default_archive` picks one for this PC.

Paths are relative to the data root, ``YYYY/MM/DD/{rid:06d}_{Class}.h5`` as in
``Task.h5_path``; days are ``YYYY/MM/DD``.
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from sipyco.pc_rpc import Client

from pytweezer.experiment.client import results_address
from pytweezer.experiment.scan import Scan
from pytweezer.experiment.storage import (
    Measurement,
    data_root,
    find_measurement,
    is_this_pc,
    load_measurement,
    read_header,
    read_planned_points,
)
from pytweezer.servers.sipyco_patches import apply_patches

# Large replies (whole measurements) are slow to read without it.
apply_patches()

RPC_TARGET = "results"
#: Header statuses a file can still move on from; any other is read once.
_UNSETTLED = {"running", "unreadable", ""}


class ArchiveUnavailable(ConnectionError):
    """The Experiment Manager's archive could not be reached."""


class ResultsArchive:
    """The measurement files under one data root."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.location = str(self.root)
        self._headers: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def days(self) -> list[str]:
        """Days holding at least one measurement, newest first."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} not found")
        days = [
            day.relative_to(self.root).as_posix()
            for day in self.root.glob("*/*/*")
            if day.is_dir() and any(day.glob("*.h5"))
        ]
        return sorted(days, reverse=True)

    def runs(self, day: str) -> list[dict[str, str]]:
        """``{"path", "status", "label"}`` of every measurement of ``day``, newest first."""
        runs = []
        for path in sorted(self._inside(day).glob("*.h5"), reverse=True):
            relpath = path.relative_to(self.root).as_posix()
            with self._lock:
                summary = self._headers.get(relpath)
            if summary is None:
                summary = _header_summary(path)
                if summary[0] not in _UNSETTLED:
                    with self._lock:
                        self._headers[relpath] = summary
            status, label = summary
            runs.append({"path": relpath, "status": status, "label": label})
        return runs

    def find(self, rid: int) -> str:
        """Path of the measurement of task ``rid``."""
        return find_measurement(rid, self.root).relative_to(self.root).as_posix()

    def load(self, path: str, results: list[str] | None = None) -> Measurement:
        """The measurement at ``path``; ``results`` limits which results are loaded."""
        return load_measurement(self._file(path), results)

    def planned_points(self, path: str) -> dict[str, np.ndarray]:
        """Every planned point's ``/points`` columns; readable while it runs."""
        return read_planned_points(self._file(path))

    def _file(self, path: str) -> Path:
        file = self._inside(path)
        if file.suffix != ".h5":
            raise ValueError(f"{path!r} is not a measurement file")
        return file

    def _inside(self, path: str) -> Path:
        # RPC callers name paths, so keep them from reading outside the root.
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError(f"{path!r} is outside the data root")
        return resolved


def _header_summary(path: Path) -> tuple[str, str]:
    try:
        header = read_header(path)
    except OSError:
        # Also what a file being created looks like; it is read again next time.
        return "unreadable", ""
    return str(header.get("status", "?")), str(header.get("label", ""))


class ArchiveTarget:
    """RPC target serving a :class:`ResultsArchive`.

    Reads run in a thread, so a slow one never holds up the event loop it is
    served from (the manager's, which also publishes the queue).
    """

    def __init__(self, archive: ResultsArchive):
        self._archive = archive

    async def days(self) -> list[str]:
        return await asyncio.to_thread(self._archive.days)

    async def runs(self, day: str) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._archive.runs, day)

    async def find(self, rid: int) -> str:
        return await asyncio.to_thread(self._archive.find, rid)

    async def load(self, path: str, results: list[str] | None = None) -> dict:
        measurement = await asyncio.to_thread(self._archive.load, path, results)
        return _measurement_to_wire(measurement)

    async def planned_points(self, path: str) -> dict[str, Any]:
        columns = await asyncio.to_thread(self._archive.planned_points, path)
        return _to_wire(columns)


class RemoteArchive:
    """A :class:`ResultsArchive` on the Experiment Manager's PC, over sipyco RPC.

    Connects on first use and again after a failure, so it outlives manager
    restarts. Any call that cannot reach the manager raises
    :class:`ArchiveUnavailable`; errors reading a file are re-raised as the
    manager saw them.

    Args:
        address: ``(host, port)`` of the archive; CONFIG by default.
        timeout: Seconds to wait for a reply.
    """

    def __init__(self, address: tuple[str, int] | None = None, timeout: float = 30.0):
        self.host, self.port = address or results_address()
        self.timeout = timeout
        self.location = f"the Experiment Manager ({self.host}:{self.port})"
        self._client: Client | None = None
        self._lock = threading.Lock()

    def days(self) -> list[str]:
        return self._call("days")

    def runs(self, day: str) -> list[dict[str, str]]:
        return self._call("runs", day)

    def find(self, rid: int) -> str:
        return self._call("find", int(rid))

    def load(self, path: str, results: list[str] | None = None) -> Measurement:
        wire = self._call("load", path, None if results is None else list(results))
        return _measurement_from_wire(wire)

    def planned_points(self, path: str) -> dict[str, np.ndarray]:
        return _from_wire(self._call("planned_points", path))

    def close(self) -> None:
        with self._lock:
            self._disconnect()

    def _call(self, name: str, *args: Any) -> Any:
        with self._lock:
            try:
                if self._client is None:
                    self._client = Client(
                        self.host, self.port, RPC_TARGET, timeout=self.timeout
                    )
                method = getattr(self._client, name)
            except (OSError, EOFError) as error:
                self._disconnect()
                raise ArchiveUnavailable(f"cannot reach {self.location}") from error
            try:
                return method(*args)
            except (ConnectionError, EOFError, TimeoutError) as error:
                # A file error from the manager arrives as the builtin it was
                # (FileNotFoundError, OSError, ...); these mean the link itself broke.
                self._disconnect()
                raise ArchiveUnavailable(f"lost {self.location}") from error

    def _disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close_rpc()
            except OSError:
                pass
            self._client = None


def default_archive() -> ResultsArchive | RemoteArchive:
    """Local files on the manager's PC or with ``PYTWEEZER_DATA_DIR`` set; RPC elsewhere."""
    from pytweezer.configuration.config import get_config

    conf = get_config().get("Servers", {}).get("Experiment Manager", {})
    if os.environ.get("PYTWEEZER_DATA_DIR") or is_this_pc(conf.get("host")):
        return ResultsArchive(data_root())
    return RemoteArchive()


def fetch_measurement(
    source: int | str, results: list[str] | None = None
) -> Measurement:
    """The measurement of rid ``source``, or at data-root-relative path ``source``, from any PC."""
    archive = default_archive()
    path = archive.find(source) if isinstance(source, (int, np.integer)) else source
    return archive.load(path, results)


# sipyco encodes numpy arrays except those of dtype object, which is how text
# columns and results come out of h5py; they travel as flat lists plus shape.
_OBJECT_ARRAY = "__object_array__"


def _to_wire(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return {_OBJECT_ARRAY: value.ravel().tolist(), "shape": value.shape}
    if isinstance(value, dict):
        return {key: _to_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    return value


def _from_wire(value: Any) -> Any:
    if isinstance(value, dict):
        if _OBJECT_ARRAY in value:
            flat = np.empty(len(value[_OBJECT_ARRAY]), dtype=object)
            flat[:] = value[_OBJECT_ARRAY]
            return flat.reshape(value["shape"])
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


def _measurement_to_wire(measurement: Measurement) -> dict:
    return _to_wire(
        {
            "attrs": measurement.attrs,
            "arguments": measurement.arguments,
            "argument_schema": measurement.argument_schema,
            "scan": measurement.scan.model_dump_json(),
            "points": measurement.points,
            "results": measurement.results,
            "recorded": measurement.recorded,
            "constants": measurement.constants,
            "units": measurement.units,
            "result_shapes": measurement.result_shapes,
            "result_kinds": measurement.result_kinds,
            "source": measurement.source,
        }
    )


def _measurement_from_wire(wire: dict) -> Measurement:
    fields = _from_wire(wire)
    fields["scan"] = Scan.model_validate_json(fields["scan"])
    fields["result_shapes"] = {
        name: tuple(shape) for name, shape in fields["result_shapes"].items()
    }
    return Measurement(**fields)
