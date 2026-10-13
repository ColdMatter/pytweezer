"""Matching scan points to readings, and turning measurement files into rows."""

import math
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pytweezer.database import analysis
from pytweezer.database.analysis import aggregate_by_point, points_with_readings
from pytweezer.database.backfill import backfill
from pytweezer.experiment import load_measurement
from tests.helpers import write_run


def test_each_window_is_aggregated_and_empty_windows_take_the_last_reading():
    times = [0.0, 1.0, 2.0, 3.0, 10.0]
    values = [5.0, 1.0, 3.0, 8.0, 9.0]
    out = aggregate_by_point([0.5, 4.0, -5.0], [2.5, 6.0, -4.0], times, values)
    assert out[0] == 2.0
    assert out[1] == 8.0
    assert math.isnan(out[2])


def test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files(
    tmp_path, recording_db
):
    write_run(tmp_path / "2026" / "a.h5", rid=7)
    write_run(tmp_path / "b.h5", rid=-1)
    (tmp_path / "broken.h5").write_bytes(b"not hdf5")

    assert backfill(tmp_path, writer=recording_db) == 1
    (run,) = recording_db.runs
    assert run["rid"] == 7 and run["class_name"] == "Sweep"
    assert run["status"] == "completed" and run["points_done"] == 3
    assert run["arguments"] == {"x": 0.0}
    assert run["h5_path"] == "2026/a.h5"

    points = recording_db.points
    assert [p["point_index"] for p in points] == [0, 1, 2]
    assert [p["scan_values"] for p in points] == [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    assert [p["scalars"] for p in points] == [{"counts": 0.0}, {}, {"counts": 20.0}]


def test_points_with_readings_joins_by_point_window(tmp_path, monkeypatch):
    path = write_run(tmp_path / "m.h5")
    t_start = load_measurement(path).points["t_start"]
    long_readings = pd.DataFrame(
        {
            "time": [datetime.fromtimestamp(t_start[0] - 100, UTC)],
            "field": ["ai0"],
            "value": [4.5],
        }
    )
    monkeypatch.setattr(analysis, "_long_readings", lambda *a: long_readings)

    table = points_with_readings(path, "ni_adc")
    assert list(table.index) == [0, 1, 2]
    assert list(table["x"]) == [1.0, 2.0, 3.0]
    assert list(table["ni_adc/ai0"]) == [4.5, 4.5, 4.5]
    assert table["counts"].isna().tolist() == [False, True, False]
    assert "image" not in table


def test_unknown_rid_is_a_lookup_error(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(analysis, "_connect", lambda dsn=None: conn)
    with pytest.raises(LookupError):
        points_with_readings(99, "ni_adc")
