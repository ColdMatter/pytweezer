"""Against a real Postgres: set PYTWEEZER_TEST_DSN to a scratch database.

Its readings, runs and points tables are emptied first. Skipped otherwise.
"""

import os
from datetime import UTC, datetime, timedelta

import pytest

from tests.test_db_analysis import write_run

DSN = os.environ.get("PYTWEEZER_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PYTWEEZER_TEST_DSN not set")


@pytest.fixture
def conn():
    import psycopg

    from pytweezer.database.schema import ensure_schema

    with psycopg.connect(DSN, autocommit=True) as conn:
        ensure_schema(conn)
        ensure_schema(conn)  # idempotent
        conn.execute("TRUNCATE readings, runs, points")
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
        writer.write(
            "laser",
            {"power": i, "locked": True},
            {"system": "Rb"},
            t0 + timedelta(seconds=i),
        )
    writer.write("laser", {"power": 99.0}, {"system": "CaF"}, t0)
    assert writer.flush(10)

    table = readings("laser", ["power"], start=t0, tags={"system": "Rb"}, dsn=DSN)
    assert list(table["power"]) == [0.0, 1.0, 2.0]


def test_run_upsert_keeps_one_row(conn, writer):
    writer.record_run({"rid": 1, "status": "running", "arguments": {"a": 1}})
    writer.record_run({"rid": 1, "status": "completed", "arguments": {"a": 2}})
    assert writer.flush(10)

    rows = conn.execute("SELECT status, arguments->>'a' FROM runs").fetchall()
    assert rows == [("completed", "2")]


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
