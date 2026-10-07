"""Matching scan points to readings, and turning measurement files into rows."""

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from pytweezer.database import analysis
from pytweezer.database.analysis import aggregate_by_point, points_with_readings
from pytweezer.database.backfill import backfill, point_rows, run_row
from pytweezer.experiment import Experiment, ListAxis, Number, Scan, load_measurement
from pytweezer.experiment.runner import open_writer


class Sweep(Experiment):
    x = Number(0.0, unit="V")

    def run_point(self):
        pass


def write_run(path, rid=7):
    """A completed 3-point run; point 1 records no counts."""
    scan = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])])
    _, points, writer = open_writer(path, Sweep, {}, scan, {"rid": rid, "label": "t"})
    for point in points:
        writer.begin_point(point)
        if point.index != 1:
            writer.record("counts", 10 * point.index)
        writer.record("image", np.zeros((2, 2)))
        writer.end_point()
    writer.finalise("completed")
    writer.close()
    return path


def test_each_window_is_aggregated_and_empty_windows_take_the_last_reading():
    times = [0.0, 1.0, 2.0, 3.0, 10.0]
    values = [5.0, 1.0, 3.0, 8.0, 9.0]
    out = aggregate_by_point([0.5, 4.0, -5.0], [2.5, 6.0, -4.0], times, values)
    assert out[0] == 2.0  # mean of the readings at 1 and 2
    assert out[1] == 8.0  # nothing in [4, 6]: last reading before 6
    assert math.isnan(out[2])  # nothing before -4 at all


def test_point_rows_hold_scanned_values_scalars_and_times(tmp_path):
    rows = point_rows(load_measurement(write_run(tmp_path / "m.h5")))

    assert [row["point_index"] for row in rows] == [0, 1, 2]
    assert [row["scan_values"] for row in rows] == [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    assert [row["scalars"] for row in rows] == [{"counts": 0.0}, {}, {"counts": 20.0}]
    assert all(row["t_start"] <= row["t_end"] for row in rows)


def test_run_row_from_file(tmp_path):
    path = write_run(tmp_path / "2026" / "m.h5")
    row = run_row(load_measurement(path), tmp_path)
    assert row["rid"] == 7 and row["class_name"] == "Sweep"
    assert row["status"] == "completed" and row["points_done"] == 3
    assert row["arguments"] == {"x": 0.0}
    assert row["h5_path"] == "2026/m.h5"


def test_backfill_skips_local_runs_and_unreadable_files(tmp_path, recording_db):
    write_run(tmp_path / "a.h5", rid=7)
    write_run(tmp_path / "b.h5", rid=-1)
    (tmp_path / "broken.h5").write_bytes(b"not hdf5")

    assert backfill(tmp_path, writer=recording_db) == 1
    assert [run["rid"] for run in recording_db.runs] == [7]
    assert len(recording_db.points) == 3


def test_points_with_readings_joins_by_point_window(tmp_path, monkeypatch):
    path = write_run(tmp_path / "m.h5")
    t_start = load_measurement(path).points["t_start"]
    fake = pd.DataFrame(
        {
            "time": [datetime.fromtimestamp(t_start[0] - 100, UTC)],
            "field": ["ai0"],
            "value": [4.5],
        }
    )
    monkeypatch.setattr(analysis, "_long_readings", lambda *a: fake)

    table = points_with_readings(path, "ni_adc")
    assert list(table.index) == [0, 1, 2]
    assert list(table["x"]) == [1.0, 2.0, 3.0]
    assert list(table["ni_adc/ai0"]) == [4.5, 4.5, 4.5]
    assert table["counts"].isna().tolist() == [False, True, False]
    assert "image" not in table


def test_unknown_rid_is_a_clear_error(monkeypatch):
    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args):
            return self

        def fetchone(self):
            return None

    monkeypatch.setattr(analysis, "_connect", lambda dsn=None: Conn())
    with pytest.raises(LookupError, match="rid 99"):
        points_with_readings(99, "ni_adc")
