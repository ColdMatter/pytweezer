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


def test_upserts_keep_one_row_per_key(conn, writer):
    writer.record_run({"rid": 1, "status": "running", "arguments": {"a": 1}})
    writer.record_run({"rid": 1, "status": "completed", "arguments": {"a": 2}})
    writer.record_measurement("ni_adc", "NI ADC Logger", 1.0, {"ai0": [0, None]}, True)
    writer.record_measurement("ni_adc", "NI ADC Logger", 1.0, {"ai0": [0, 5]}, False)
    assert writer.flush(10)

    runs = conn.execute("SELECT status, arguments->>'a' FROM runs").fetchall()
    assert runs == [("completed", "2")]
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


def test_backfilled_run_joins_with_readings(conn, writer, tmp_path, monkeypatch):
    from pytweezer.database import analysis
    from pytweezer.database.backfill import backfill
    from pytweezer.experiment.storage import load_measurement

    path = write_run(tmp_path / "m.h5")
    assert backfill(tmp_path, writer=writer) == 1
    t_start = load_measurement(path).points["t_start"]
    writer.write(
        "ni_adc", {"ai0": 1.25}, time=datetime.fromtimestamp(t_start[0] - 1, UTC)
    )
    assert writer.flush(10)

    monkeypatch.setattr(analysis, "data_root", lambda: tmp_path)
    table = analysis.points_with_readings(7, "ni_adc", dsn=DSN)
    assert list(table["ni_adc/ai0"]) == [1.25, 1.25, 1.25]
    assert conn.execute("SELECT count(*) FROM points WHERE rid = 7").fetchone() == (3,)
