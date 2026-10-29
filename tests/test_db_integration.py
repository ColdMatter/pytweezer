"""Against a real Postgres: set PYTWEEZER_TEST_DSN to a scratch database.

Its tables are emptied first. Skipped otherwise.
"""

import os
from datetime import UTC, datetime, timedelta

import pytest

from tests.helpers import write_run

DSN = os.environ.get("PYTWEEZER_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PYTWEEZER_TEST_DSN not set")


@pytest.fixture
def conn():
    import psycopg

    from pytweezer.database.schema import ensure_schema

    with psycopg.connect(DSN, autocommit=True) as conn:
        ensure_schema(conn)
        ensure_schema(conn)  # idempotent
        conn.execute("TRUNCATE readings, runs, points, measurements")
        yield conn


@pytest.fixture
def writer():
    from pytweezer.database import DBWriter

    writer = DBWriter(DSN, batch_s=0.01)
    yield writer
    writer.close()


def test_readings_round_trip(conn, writer):
    from pytweezer.database.analysis import readings

    t0 = datetime(2026, 10, 7, 12, tzinfo=UTC)
    for i in range(3):
        writer.write("laser", {"power": i}, {"system": "Rb"}, t0 + timedelta(seconds=i))
    writer.write("laser", {"power": 99.0}, {"system": "CaF"}, t0)
    assert writer.flush(10)

    table = readings("laser", ["power"], start=t0, tags={"system": "Rb"}, dsn=DSN)
    assert list(table["power"]) == [0.0, 1.0, 2.0]


def test_a_run_recorded_at_start_and_finish_is_one_row_with_its_points(conn, writer):
    writer.record_run({"rid": 1, "status": "running", "points_done": 0})
    for index in range(3):
        writer.record_point(1, index, 10.0 * index, 10.0 * index + 5, {"x": index}, {})
    writer.record_run({"rid": 1, "status": "completed", "points_done": 3})
    assert writer.flush(10)

    runs = conn.execute("SELECT rid, status, points_done FROM runs").fetchall()
    assert runs == [(1, "completed", 3)]
    points = conn.execute(
        "SELECT point_index, scan_values->>'x' FROM points WHERE rid = 1 "
        "ORDER BY point_index"
    ).fetchall()
    assert points == [(0, "0"), (1, "1"), (2, "2")]


def test_a_measurement_registered_twice_is_one_row_with_the_latest_limits(conn, writer):
    writer.record_measurement("ni_adc", "NI ADC Logger", 1.0, {"ai0": [0, None]}, True)
    writer.record_measurement("ni_adc", "NI ADC Logger", 1.0, {"ai0": [0, 5]}, False)
    assert writer.flush(10)

    limits = conn.execute(
        "SELECT m.active, l.key, (l.value->>0)::float, (l.value->>1)::float "
        "FROM measurements m, jsonb_each(m.limits) l"
    ).fetchall()
    assert limits == [(False, "ai0", 0.0, 5.0)]


def test_non_finite_json_values_are_stored_as_null_without_blocking_writes(
    conn, writer
):
    scalars = {"width": float("nan"), "peak": [1.0, float("inf")], "n": 3.0}
    writer.record_point(1, 0, 0.0, 1.0, {}, scalars)
    writer.write("laser", {"power": 1.0})
    assert writer.flush(3)

    stored = conn.execute("SELECT scalars FROM points").fetchone()
    assert stored == ({"width": None, "peak": [1.0, None], "n": 3.0},)
    assert conn.execute("SELECT count(*) FROM readings").fetchone() == (1,)


def test_a_row_the_database_rejects_is_dropped_and_the_rest_are_written(conn, writer):
    writer.write("laser", {"power": 1.0})
    writer.record_run({"rid": 1, "label": "NUL \x00 is not valid text"})
    writer.write("laser", {"power": 2.0})
    assert writer.flush(3)

    powers = conn.execute("SELECT value FROM readings ORDER BY value").fetchall()
    assert powers == [(1.0,), (2.0,)]
    assert conn.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_a_backfilled_run_joins_each_point_with_its_own_readings(
    conn, writer, tmp_path, monkeypatch
):
    from pytweezer.database import analysis
    from pytweezer.database.backfill import backfill
    from pytweezer.experiment.storage import load_measurement

    path = write_run(tmp_path / "m.h5", point_s=0.02)
    assert backfill(tmp_path, writer=writer) == 1
    points = load_measurement(path).points
    for i, (start, end) in enumerate(zip(points["t_start"], points["t_end"])):
        middle = datetime.fromtimestamp((start + end) / 2, UTC)
        writer.write("ni_adc", {"ai0": float(i)}, time=middle)
    assert writer.flush(10)

    monkeypatch.setattr(analysis, "data_root", lambda: tmp_path)
    table = analysis.points_with_readings(7, "ni_adc", dsn=DSN)
    assert list(table["ni_adc/ai0"]) == [0.0, 1.0, 2.0]


def test_backfilling_twice_leaves_one_row_per_run_and_point(conn, writer, tmp_path):
    from pytweezer.database.backfill import backfill

    write_run(tmp_path / "m.h5")
    for _ in range(2):
        backfill(tmp_path, writer=writer)
    assert writer.flush(10)

    assert conn.execute("SELECT count(*) FROM runs").fetchone() == (1,)
    assert conn.execute("SELECT count(*) FROM points WHERE rid = 7").fetchone() == (3,)
