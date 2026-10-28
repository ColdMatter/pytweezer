"""Matching scan points to readings, and turning measurement files into rows."""

import math
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pytest

from pytweezer.database import analysis
from pytweezer.database.analysis import aggregate_by_point, points_with_readings
from pytweezer.database.backfill import backfill
from pytweezer.experiment import load_measurement
from tests.helpers import write_run

# Readings whose value equals their time, at t = 0, 1, ..., 9.
TIMES = np.arange(10.0)


@pytest.mark.parametrize(
    "agg, expected",
    [("mean", 3.5), ("median", 3.5), ("max", 5.0), (np.sum, 14.0)],
    ids=["mean", "median", "max", "callable"],
)
def test_readings_in_a_window_are_reduced_with_both_edges_included(agg, expected):
    assert aggregate_by_point([2.0], [5.0], TIMES, TIMES, agg) == [expected]


def test_a_window_without_readings_takes_the_last_reading_before_it():
    out = aggregate_by_point([4.0], [5.0], [0.0, 10.0], [1.0, 2.0])
    assert out[0] == 1.0


def test_a_window_before_the_first_reading_is_nan():
    out = aggregate_by_point([0.0], [1.0], [10.0], [1.0])
    assert math.isnan(out[0])


class FakeReadingsDatabase:
    """The database at the network edge: answers the readings query with given rows."""

    def __init__(self, rows):
        self.rows = sorted(rows)

    @contextmanager
    def __call__(self, dsn=None):
        yield self

    def execute(self, sql, params):
        return self

    def fetchall(self):
        return self.rows


def test_each_point_gets_the_readings_inside_its_own_window(tmp_path, monkeypatch):
    run = load_measurement(write_run(tmp_path / "m.h5"))
    t0 = 1_800_000_000.0
    run.points["t_start"] = t0 + np.array([0.0, 10.0, 20.0])
    run.points["t_end"] = t0 + np.array([5.0, 15.0, 25.0])
    # Two readings on each window's edges, so their mean is neither of them.
    rows = []
    for i in range(3):
        for edge, value in ((0.0, 10 * i), (5.0, 10 * i + 2)):
            when = datetime.fromtimestamp(t0 + 10 * i + edge, UTC)
            rows.append((when, "ai0", float(value)))
    monkeypatch.setattr(analysis, "_connect", FakeReadingsDatabase(rows))

    table = points_with_readings(run, "ni_adc")
    assert list(table.index) == [0, 1, 2]
    assert list(table["ni_adc/ai0"]) == [1.0, 11.0, 21.0]
    assert list(table["x"]) == [1.0, 2.0, 3.0]
    assert table["counts"].isna().tolist() == [False, True, False]
    assert "image" not in table
    assert table["t_start"].iloc[1] == datetime.fromtimestamp(t0 + 10, UTC)


def test_backfill_writes_one_point_row_per_point_that_ran(tmp_path, recording_db):
    path = write_run(tmp_path / "2026/10/21/000007_Sweep.h5", points_run=2)
    backfill(tmp_path, writer=recording_db)
    points = load_measurement(path).points

    rows = recording_db.points
    assert [row["point_index"] for row in rows] == [0, 1]
    assert [row["scan_values"] for row in rows] == [{"x": 1.0}, {"x": 2.0}]
    assert [row["t_start"] for row in rows] == list(points["t_start"])
    assert [row["t_end"] for row in rows] == list(points["t_end"])


def test_backfill_leaves_out_a_scalar_a_point_did_not_record(tmp_path, recording_db):
    write_run(tmp_path / "2026/10/21/000007_Sweep.h5")
    backfill(tmp_path, writer=recording_db)
    scalars = [row["scalars"] for row in recording_db.points]
    assert scalars == [{"counts": 0.0}, {}, {"counts": 20.0}]


def test_backfill_maps_the_file_header_to_the_runs_columns(tmp_path, recording_db):
    write_run(tmp_path / "2026/10/21/000007_Sweep.h5", points_run=2)
    backfill(tmp_path, writer=recording_db)
    (run,) = recording_db.runs
    assert run["rid"] == 7 and run["class_name"] == "Sweep"
    assert run["h5_path"] == "2026/10/21/000007_Sweep.h5"
    assert run["status"] == "terminated"
    assert (run["points_done"], run["points_total"]) == (2, 3)
    assert run["arguments"] == {"x": 0.0}
    assert run["error"] is None


def test_backfill_skips_local_runs_and_unreadable_files(tmp_path, recording_db):
    write_run(tmp_path / "2026/10/21/000007_Sweep.h5", rid=7)
    write_run(tmp_path / "2026/10/21/local.h5", rid=-1)
    (tmp_path / "broken.h5").write_bytes(b"not hdf5")

    assert backfill(tmp_path, writer=recording_db) == 1
    assert [run["rid"] for run in recording_db.runs] == [7]


def test_backfill_skips_simulated_runs_that_share_a_real_rid(tmp_path, recording_db):
    real = "2026/10/21/000005_Sweep.h5"
    write_run(tmp_path / real, rid=5)
    write_run(tmp_path / "simulated" / real, rid=5)

    assert backfill(tmp_path, writer=recording_db) == 1
    assert [run["h5_path"] for run in recording_db.runs] == [real]
    assert len(recording_db.points) == 3
