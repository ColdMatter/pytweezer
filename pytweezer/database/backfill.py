"""Load measurement files into the ``runs`` and ``points`` tables.

The Experiment Manager records runs as they happen; this fills in whatever it
missed (runs from before the database existed, or written while it was down).
Every row is an upsert, so running it again is harmless::

    poetry run pytweezer-db-backfill            # the configured data root
    poetry run pytweezer-db-backfill --root D:/data
"""

import argparse
from pathlib import Path

from pytweezer.database.writer import DBWriter
from pytweezer.experiment.storage import Measurement, data_root, load_measurement
from pytweezer.logging_utils import get_logger

logger = get_logger("Database")

_POINT_BOOKKEEPING = {"index", "repetition", "t_start", "t_end"}


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def scalar_result_names(measurement: Measurement) -> list[str]:
    """Results stored as one number per point, as the manager sends them live."""
    return [
        name
        for name, shape in measurement.result_shapes.items()
        if shape == () and measurement.result_kinds.get(name, "O") in "biuf"
    ]


def load_with_scalars(path: Path | str) -> Measurement:
    """A measurement with only its scalar results loaded (images are skipped)."""
    names = scalar_result_names(load_measurement(path, results=()))
    return load_measurement(path, results=names)


def point_rows(measurement: Measurement) -> list[dict]:
    """One dict per point that ran, shaped like :meth:`DBWriter.record_point`'s args."""
    points = measurement.points
    scanned = [name for name in points if name not in _POINT_BOOKKEEPING]
    scalars = [
        name for name in scalar_result_names(measurement) if name in measurement.results
    ]
    rows = []
    for i in range(measurement.n_done):
        rows.append(
            {
                "rid": measurement.rid,
                "point_index": int(points["index"][i]),
                "t_start": float(points["t_start"][i]),
                "t_end": float(points["t_end"][i]),
                "scan_values": {name: _plain(points[name][i]) for name in scanned},
                "scalars": {
                    name: float(measurement.results[name][i])
                    for name in scalars
                    if measurement.recorded[name][i]
                },
            }
        )
    return rows


def run_row(measurement: Measurement, root: Path) -> dict:
    """The ``runs`` row for a measurement file, as the manager would have written it."""
    attrs = measurement.attrs
    path = measurement.path
    return {
        "rid": measurement.rid,
        "experiment": attrs.get("experiment"),
        "class_name": attrs.get("class_name"),
        "label": attrs.get("label"),
        "submitter": attrs.get("submitter"),
        "arguments": measurement.arguments,
        "scan": measurement.scan.model_dump(mode="json"),
        "status": attrs.get("status"),
        "error": attrs.get("error") or None,
        "t_submit": attrs.get("t_submit"),
        "t_start": attrs.get("t_start"),
        "t_end": attrs.get("t_end"),
        "points_done": measurement.n_done,
        "points_total": int(attrs.get("n_points", 0)),
        "h5_path": path.relative_to(root).as_posix() if path else None,
        "simulated": bool(attrs.get("simulated", False)),
    }


def backfill(root: Path | str | None = None, writer=None) -> int:
    """Upsert every queued-run measurement file under ``root``; returns how many."""
    root = Path(root) if root is not None else data_root()
    owns_writer = writer is None
    writer = writer or DBWriter()
    count = 0
    for path in sorted(root.rglob("*.h5")):
        try:
            measurement = load_with_scalars(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if measurement.rid < 0:
            continue  # run_local(): never queued, so it has no rid
        writer.record_run(run_row(measurement, root))
        for row in point_rows(measurement):
            writer.record_point(**row)
        count += 1
    if owns_writer:
        writer.close(timeout=60)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="data root to scan (default: configured)")
    args = parser.parse_args()
    count = backfill(args.root)
    print(f"Queued {count} runs for the database")


if __name__ == "__main__":
    main()
