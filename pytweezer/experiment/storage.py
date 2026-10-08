"""One HDF5 file per measurement, written point by point.

Layout of ``{data_root}/YYYY/MM/DD/{rid:06d}_{ClassName}.h5``::

    /                     attrs: schema_version, rid, experiment, class_name, label,
                                 submitter, host, status, error, t_submit, t_start,
                                 t_end, n_points, n_done, git_commit, git_dirty
    /arguments            attrs: effective value of every argument;
                                 ``__schema__`` = JSON of the argument declarations
    /scan                 attrs: ``spec`` = Scan JSON
    /points/<column>      index, repetition, <scanned argument>..., t_start, t_end
    /results/<name>       (n_points, *shape), row i = point i
    /recorded/<name>      (n_points,) bool, whether point i recorded <name>
    /constants/<name>     recorded from prepare()/finish()
    /source/<module>      experiment source text, attr sha256

Every column under ``/points`` and ``/results`` has one row per *planned*
point, in execution order, so the first ``n_done`` rows are the ones that ran.
The file is created with mode ``"x"`` (never overwritten) and flushed after
every point; readers open it with ``locking=False`` so a file still being
written can be inspected.
"""

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pytweezer.configuration.paths import tweezerpath
from pytweezer.experiment.scan import Point, Scan

SCHEMA_VERSION = 1
_STR = h5py.string_dtype()


class RecordError(ValueError):
    """A recorded value doesn't fit what was recorded under that name before."""


def data_root() -> Path:
    """Root directory of the measurement files on this PC.

    ``PYTWEEZER_DATA_DIR`` wins, so a client PC can point at the server's
    share; otherwise the Experiment Manager's ``data_root`` from CONFIG. When
    the manager is simulating, files go in a ``simulated/`` subdirectory, so
    simulated runs never mix with real data or use up its rids.
    """
    from pytweezer.configuration.config import get_config

    conf = get_config().get("Servers", {}).get("Experiment Manager", {})
    env = os.environ.get("PYTWEEZER_DATA_DIR")
    root = Path(env or conf.get("data_root") or Path(tweezerpath) / "data")
    return root / "simulated" if conf.get("simulate") else root


def measurement_relpath(rid: int, class_name: str, when: datetime) -> Path:
    return Path(f"{when:%Y}", f"{when:%m}", f"{when:%d}", f"{rid:06d}_{class_name}.h5")


def highest_rid(root: Path) -> int:
    """Largest rid among measurement files under ``root``, or 0."""
    highest = 0
    for path in Path(root).glob("*/*/*/*.h5"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            highest = max(highest, int(prefix))
    return highest


def git_provenance() -> dict[str, Any]:
    """Commit and dirty flag of the checkout, best effort (git may not be on PATH)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tweezerpath,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=tweezerpath,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "", "git_dirty": False}
    return {"git_commit": commit, "git_dirty": bool(dirty)}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _attr_value(value: Any) -> Any:
    return "" if value is None else value


class MeasurementWriter:
    """Writes one measurement file. ``path=None`` keeps it in memory only."""

    def __init__(
        self,
        path: Path | str | None,
        *,
        header: dict[str, Any],
        argument_values: dict[str, Any],
        argument_schema: dict[str, Any],
        scan: Scan,
        points: list[Point],
        sources: dict[str, str],
    ) -> None:
        if path is None:
            self.file = h5py.File(
                f"memory-{id(self)}.h5", "w", driver="core", backing_store=False
            )
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = h5py.File(path, "x", locking=False)
        self.path = path
        self.n_points = len(points)
        self._point: Point | None = None
        self._recorded_this_point: set[str] = set()
        self.point_scalars: dict[str, float] = {}
        self.point_t_start: float | None = None
        self.point_t_end: float | None = None

        f = self.file
        f.attrs.update({key: _attr_value(value) for key, value in header.items()})
        f.attrs.update(
            schema_version=SCHEMA_VERSION,
            host=socket.gethostname(),
            status="running",
            error="",
            t_start=_now_iso(),
            t_end="",
            n_points=self.n_points,
            n_done=0,
        )

        arguments = f.create_group("arguments")
        arguments.attrs.update(argument_values)
        arguments.attrs["__schema__"] = json.dumps(argument_schema)

        f.create_group("scan").attrs["spec"] = scan.model_dump_json()

        columns = f.create_group("points")
        columns["index"] = np.array([p.index for p in points], dtype=np.int64)
        columns["repetition"] = np.array([p.repetition for p in points], dtype=np.int64)
        for name in points[0].values if points else ():
            values = [p.values[name] for p in points]
            if all(isinstance(v, str) for v in values):
                columns.create_dataset(name, data=values, dtype=_STR)
            else:
                columns[name] = np.asarray(values)
        for column in ("t_start", "t_end"):
            columns.create_dataset(
                column, shape=(self.n_points,), dtype=np.float64, fillvalue=np.nan
            )

        f.create_group("results")
        f.create_group("recorded")
        f.create_group("constants")
        source = f.create_group("source")
        for module, text in sources.items():
            source.create_dataset(module, data=text, dtype=_STR)
            source[module].attrs["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        f.flush()

    def begin_point(self, point: Point) -> None:
        self._point = point
        self._recorded_this_point = set()
        self.point_scalars = {}
        self.point_t_start = datetime.now().timestamp()
        self.file["points/t_start"][point.index] = self.point_t_start

    def end_point(self) -> None:
        point = self._point
        self.point_t_end = datetime.now().timestamp()
        self.file["points/t_end"][point.index] = self.point_t_end
        self.file.attrs["n_done"] = point.index + 1
        self._point = None
        self.file.flush()

    def abandon_point(self) -> None:
        """Leave the current point unfinished, so later records become constants."""
        self._point = None

    def record(self, name: str, value: Any, unit: str = "") -> None:
        """Record into the current point, or as a constant outside a point."""
        _check_name(name)
        if self._point is None:
            self._record_constant(name, value, unit)
        else:
            self._record_result(name, value, unit)

    def _record_constant(self, name: str, value: Any, unit: str) -> None:
        group = self.file["constants"]
        if name in group:
            raise RecordError(f"constant {name!r} was already recorded")
        data, dtype, encoding = _encode(name, value)
        group.create_dataset(name, data=data, dtype=dtype)
        group[name].attrs.update(unit=unit, encoding=encoding)

    def _record_result(self, name: str, value: Any, unit: str) -> None:
        if name in self._recorded_this_point:
            raise RecordError(
                f"{name!r} was recorded twice in point {self._point.index}"
            )
        data, dtype, encoding = _encode(name, value)
        results = self.file["results"]
        if name not in results:
            shape = np.shape(data)
            fill = np.nan if np.dtype(dtype).kind in "fc" else None
            results.create_dataset(
                name,
                shape=(self.n_points, *shape),
                dtype=dtype,
                chunks=(1, *shape) if shape else None,
                fillvalue=fill,
            )
            results[name].attrs.update(unit=unit, encoding=encoding)
            self.file["recorded"].create_dataset(
                name, shape=(self.n_points,), dtype=bool
            )
        dataset = results[name]
        _check_fits(name, dataset, data, dtype)
        dataset[self._point.index] = data
        self.file["recorded"][name][self._point.index] = True
        self._recorded_this_point.add(name)
        if np.ndim(data) == 0 and np.dtype(dtype).kind in "biuf":
            self.point_scalars[name] = float(data)

    def finalise(self, status: str, error: str | None = None) -> None:
        self.file.attrs.update(status=str(status), error=error or "", t_end=_now_iso())
        self.file.flush()

    def close(self) -> None:
        if self.file.id.valid:
            self.file.close()


def _check_name(name: str) -> None:
    if not name or "/" in name or name.startswith("__"):
        raise RecordError(
            f"invalid record name {name!r}: must be non-empty, without '/', "
            "and not start with '__'"
        )


def _encode(name: str, value: Any) -> tuple[Any, Any, str]:
    """Return ``(data, h5 dtype, encoding)`` for a recorded value."""
    if isinstance(value, str):
        return value, _STR, ""
    if isinstance(value, dict):
        return json.dumps(value), _STR, "json"
    data = np.asarray(value)
    if data.dtype.kind not in "biufc":
        raise RecordError(
            f"cannot record {name!r}: unsupported type {type(value).__name__} "
            f"(dtype {data.dtype})"
        )
    return data, data.dtype, ""


def _check_fits(name: str, dataset: h5py.Dataset, data: Any, dtype: Any) -> None:
    if dataset.dtype.kind == "O" or h5py.check_string_dtype(dataset.dtype):
        if dtype is not _STR:
            raise RecordError(f"{name!r} was recorded as text, now got {type(data)}")
        return
    if dtype is _STR or not np.can_cast(dtype, dataset.dtype, casting="safe"):
        raise RecordError(
            f"{name!r} was first recorded with dtype {dataset.dtype}, "
            f"now got {'text' if dtype is _STR else np.dtype(dtype)}"
        )
    if np.shape(data) != dataset.shape[1:]:
        raise RecordError(
            f"{name!r} was first recorded with shape {dataset.shape[1:]}, "
            f"now got shape {np.shape(data)}"
        )


@dataclass
class Measurement:
    """A measurement file loaded into memory, trimmed to the points that ran."""

    attrs: dict[str, Any]
    arguments: dict[str, Any]
    argument_schema: dict[str, Any]
    scan: Scan
    points: dict[str, np.ndarray]
    results: dict[str, np.ndarray]
    recorded: dict[str, np.ndarray]
    constants: dict[str, Any]
    units: dict[str, str] = field(default_factory=dict)
    #: Per-point shape of every result, including those not loaded.
    result_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: numpy dtype kind of every result ("O" for text), including those not loaded.
    result_kinds: dict[str, str] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @property
    def rid(self) -> int:
        return int(self.attrs["rid"])

    @property
    def status(self) -> str:
        return str(self.attrs["status"])

    @property
    def n_done(self) -> int:
        return int(self.attrs["n_done"])


def load_measurement(
    source: Path | str | h5py.File, results: Iterable[str] | None = None
) -> Measurement:
    """Read a measurement file; ``results`` limits which results are loaded (default all)."""
    wanted = None if results is None else set(results)
    if isinstance(source, h5py.File):
        return _read(source, None, wanted)
    with h5py.File(source, "r", locking=False) as f:
        return _read(f, Path(source), wanted)


def read_header(path: Path | str) -> dict[str, Any]:
    """Root attributes only, for listing files cheaply."""
    with h5py.File(path, "r", locking=False) as f:
        return {key: _plain(value) for key, value in f.attrs.items()}


def read_arguments(path: Path | str) -> dict[str, Any]:
    """The effective argument values only, without loading points or results."""
    with h5py.File(path, "r", locking=False) as f:
        attrs = f["arguments"].attrs
        return {key: _plain(attrs[key]) for key in attrs if key != "__schema__"}


def _read(f: h5py.File, path: Path | None, wanted: set[str] | None) -> Measurement:
    n_done = int(f.attrs["n_done"])
    units = {}
    results = {}
    shapes = {}
    kinds = {}
    for name, dataset in f["results"].items():
        shapes[name] = dataset.shape[1:]
        kinds[name] = (
            "O" if h5py.check_string_dtype(dataset.dtype) else dataset.dtype.kind
        )
        units[name] = str(dataset.attrs.get("unit", ""))
        if wanted is None or name in wanted:
            results[name] = _decode(dataset, dataset[:n_done])
    constants = {}
    for name, dataset in f["constants"].items():
        constants[name] = _decode(dataset, dataset[()])
        units[name] = str(dataset.attrs.get("unit", ""))
    arguments = dict(f["arguments"].attrs)
    schema = json.loads(arguments.pop("__schema__"))
    return Measurement(
        attrs={key: _plain(value) for key, value in f.attrs.items()},
        arguments={key: _plain(value) for key, value in arguments.items()},
        argument_schema=schema,
        scan=Scan.model_validate_json(f["scan"].attrs["spec"]),
        points={
            name: _decode(dataset, dataset[:n_done])
            for name, dataset in f["points"].items()
        },
        results=results,
        recorded={name: ds[:n_done] for name, ds in f["recorded"].items()},
        constants=constants,
        units=units,
        result_shapes=shapes,
        result_kinds=kinds,
        source={name: ds[()].decode() for name, ds in f["source"].items()},
        path=path,
    )


def _decode(dataset: h5py.Dataset, data: Any) -> Any:
    if h5py.check_string_dtype(dataset.dtype):
        if isinstance(data, bytes):
            text = data.decode()
            return json.loads(text) if dataset.attrs.get("encoding") == "json" else text
        texts = [item.decode() for item in np.ravel(data)]
        if dataset.attrs.get("encoding") == "json":
            return [json.loads(text) if text else None for text in texts]
        return np.array(texts, dtype=object).reshape(np.shape(data))
    return data


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value
