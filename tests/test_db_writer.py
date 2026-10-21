"""DBWriter queues rows and writes them from a background thread, never raising."""

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pytest

from pytweezer.database import writer as writer_module
from pytweezer.database.schema import TIMESCALE_SQL, ensure_schema
from pytweezer.database.writer import DBWriter


class FakeConnection:
    """Postgres with TimescaleDB installed; records executemany() rows per table.

    Unless TimescaleDB is preloaded, CREATE EXTENSION timescaledb is FATAL: the
    server drops the connection rather than raising an ordinary error.
    """

    def __init__(self, sent=None, preloaded=True):
        self.sent = {} if sent is None else sent
        self.preloaded = preloaded
        self.executed = []
        self.closed = False

    def execute(self, sql, params=None):
        if self.closed:
            raise ConnectionError("the connection is lost")
        self.executed.append(sql)
        if "CREATE EXTENSION" in sql and not self.preloaded:
            self.closed = True
            raise ConnectionError('extension "timescaledb" must be preloaded')
        return self

    def fetchone(self):
        return (1,) if self.preloaded else None

    @contextmanager
    def transaction(self):
        yield

    @contextmanager
    def cursor(self):
        yield self

    def executemany(self, sql, rows):
        table = sql.split()[2]
        self.sent.setdefault(table, []).extend(rows)

    def close(self):
        self.closed = True


class FakeDatabase:
    """A connect() callable that can be switched between up and down."""

    def __init__(self, up=True):
        self.up = up
        self.sent = {}

    def __call__(self, dsn):
        if not self.up:
            raise ConnectionError("database down")
        return FakeConnection(self.sent)


@pytest.fixture
def db():
    return FakeDatabase()


@pytest.fixture
def writer(db):
    writer = DBWriter("postgresql://u:secret@db:5432/x", connect=db, batch_s=0.01)
    yield writer
    writer.close(timeout=5)


@pytest.mark.parametrize("preloaded", [True, False])
def test_schema_reports_timescale_and_keeps_the_connection(preloaded):
    conn = FakeConnection(preloaded=preloaded)
    assert ensure_schema(conn) is preloaded
    assert (TIMESCALE_SQL in conn.executed) is preloaded
    assert not conn.closed


def test_write_makes_a_row_per_numeric_field_sharing_one_timestamp(db, writer):
    writer.write(
        "laser",
        {"power": 1.5, "locked": True, "name": "x", "none": None, "list": [1]},
        tags={"n": 2},
    )
    assert writer.flush(2)

    rows = db.sent["readings"]
    assert [(r[1], r[2], r[3]) for r in rows] == [
        ("laser", "power", 1.5),
        ("laser", "locked", 1.0),
    ]
    assert rows[0][0] == rows[1][0]
    assert json.loads(rows[0][4]) == {"n": "2"}


@pytest.mark.parametrize(
    "value, stored",
    [(np.bool_(True), 1.0), (np.float32(0.5), 0.5), (np.int64(3), 3.0)],
    ids=["bool", "float32", "int64"],
)
def test_numpy_scalar_fields_are_written(db, writer, value, stored):
    writer.write("laser", {"field": value})
    assert writer.flush(2)
    assert [row[3] for row in db.sent["readings"]] == [stored]


def test_records_convert_times_and_json(db, writer):
    writer.record_run(
        {
            "rid": 7,
            "class_name": "Scan",
            "arguments": {"detuning": -3.0},
            "t_start": "2026-10-07T12:00:00+01:00",
            "t_end": None,
            "simulated": False,
        }
    )
    writer.record_point(3, 0, 1_700_000_000.0, 1_700_000_001.5, {"x": 1}, {"n": 4.0})
    writer.record_measurement("ni_adc", "NI ADC", 1.0, {"ai0": [0.0, None]}, True)
    assert writer.flush(2)

    (run,) = db.sent["runs"]
    assert run[0] == 7 and run[2] == "Scan"
    assert json.loads(run[5]) == {"detuning": -3.0}
    assert run[10] == datetime.fromisoformat("2026-10-07T12:00:00+01:00")
    assert run[11] is None and run[15] is False

    (point,) = db.sent["points"]
    assert point[:3] == (3, 0, datetime.fromtimestamp(1_700_000_000.0, UTC))
    assert json.loads(point[5]) == {"n": 4.0}

    (measurement,) = db.sent["measurements"]
    assert measurement[:3] == ("ni_adc", "NI ADC", 1.0)
    assert json.loads(measurement[3]) == {"ai0": [0.0, None]}
    assert measurement[4] is True


def test_write_does_not_wait_for_an_unreachable_database():
    writer = DBWriter("postgresql://u:p@127.0.0.1:1/none", batch_s=0.01)
    start = time.monotonic()
    for _ in range(100):
        writer.write("m", {"x": 1.0})
    assert time.monotonic() - start < 0.1
    assert not writer.flush(0.5)
    writer.close(timeout=5)


def test_rows_are_kept_through_an_outage_and_written_once_it_ends(db, writer):
    db.up = False
    writer.write("m", {"x": 1.0})
    assert not writer.flush(0.3)
    assert "readings" not in db.sent

    db.up = True
    assert writer.flush(2)
    assert len(db.sent["readings"]) == 1


def test_backlog_beyond_max_pending_drops_the_oldest():
    writer = DBWriter(connect=FakeDatabase(up=False), max_pending=3, batch_s=60)
    writer.write("m", {f"f{i}": float(i) for i in range(5)})

    with writer._cond:
        kept = [row[2] for _table, row in writer._pending]
    assert kept == ["f2", "f3", "f4"]
    writer.close(timeout=5)


def test_log_messages_hide_the_password(writer):
    assert writer.dsn_host == "postgresql://db:5432/x"


def test_a_failed_schema_setup_is_retried_on_a_fresh_connection(
    monkeypatch, db, writer
):
    attempts = []

    def flaky_schema(conn):
        attempts.append(conn)
        if len(attempts) == 1:
            raise RuntimeError("permission denied")

    monkeypatch.setattr(writer_module, "ensure_schema", flaky_schema)
    writer.write("m", {"x": 1.0})
    assert writer.flush(5)

    assert attempts[0].closed and not attempts[1].closed
    assert len(db.sent["readings"]) == 1
