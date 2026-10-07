"""DBWriter queues rows and writes them from a background thread, never raising."""

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from pytweezer.database.writer import DBWriter, _coerce_fields


class FakeConnection:
    """Records executemany() calls per table; ensure_schema's execute() is a no-op."""

    def __init__(self, sent):
        self.sent = sent
        self.closed = False

    def execute(self, sql, params=None):
        pass

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
        self.connects = 0

    def __call__(self, dsn):
        self.connects += 1
        if not self.up:
            raise ConnectionError("database down")
        return FakeConnection(self.sent)


def make_writer(db, **kwargs):
    kwargs.setdefault("batch_s", 0.01)
    return DBWriter("postgresql://u:secret@db:5432/x", connect=db, **kwargs)


def test_coerce_fields_keeps_numbers_and_bools_only():
    clean = _coerce_fields({"a": 1, "b": 2.5, "c": True, "d": "x", "e": None, "f": [1]})
    assert clean == {"a": 1.0, "b": 2.5, "c": True}


def test_write_makes_a_row_per_field_sharing_one_timestamp():
    db = FakeDatabase()
    writer = make_writer(db)
    writer.write("laser", {"power": 1.5, "locked": True, "name": "x"}, tags={"n": 2})
    assert writer.flush(2)

    rows = db.sent["readings"]
    assert [(r[1], r[2], r[3]) for r in rows] == [
        ("laser", "power", 1.5),
        ("laser", "locked", 1.0),
    ]
    assert rows[0][0] == rows[1][0]
    assert json.loads(rows[0][4]) == {"n": "2"}
    writer.close()


def test_record_run_maps_columns_and_times():
    db = FakeDatabase()
    writer = make_writer(db)
    writer.record_run(
        {
            "rid": 7,
            "class_name": "Scan",
            "arguments": {"detuning": -3.0},
            "status": "running",
            "t_start": "2026-10-07T12:00:00+01:00",
            "t_end": None,
            "simulated": False,
        }
    )
    assert writer.flush(2)

    (row,) = db.sent["runs"]
    assert row[0] == 7 and row[2] == "Scan"
    assert json.loads(row[5]) == {"detuning": -3.0}
    assert row[10] == datetime.fromisoformat("2026-10-07T12:00:00+01:00")
    assert row[11] is None
    assert row[15] is False
    writer.close()


def test_record_point_converts_epoch_times():
    db = FakeDatabase()
    writer = make_writer(db)
    writer.record_point(3, 0, 1_700_000_000.0, 1_700_000_001.5, {"x": 1}, {"n": 4.0})
    assert writer.flush(2)

    (row,) = db.sent["points"]
    assert row[:2] == (3, 0)
    assert row[2] == datetime.fromtimestamp(1_700_000_000.0, UTC)
    assert json.loads(row[5]) == {"n": 4.0}
    writer.close()


def test_write_does_not_wait_for_an_unreachable_database():
    writer = DBWriter("postgresql://u:p@127.0.0.1:1/none", batch_s=0.01)
    start = time.monotonic()
    for _ in range(100):
        writer.write("m", {"x": 1.0})
    assert time.monotonic() - start < 0.1
    assert not writer.flush(0.5)
    writer.close(timeout=5)


def test_rows_are_kept_through_an_outage_and_written_once_it_ends():
    db = FakeDatabase(up=False)
    writer = make_writer(db)
    writer.write("m", {"x": 1.0})
    assert not writer.flush(0.3)
    assert "readings" not in db.sent

    db.up = True
    assert writer.flush(2)
    assert len(db.sent["readings"]) == 1
    writer.close()


def test_backlog_beyond_max_pending_drops_the_oldest():
    db = FakeDatabase(up=False)
    writer = make_writer(db, max_pending=3, batch_s=60)
    writer.write("m", {f"f{i}": float(i) for i in range(5)})

    with writer._cond:
        kept = [row[2] for _table, row in writer._pending]
    assert kept == ["f2", "f3", "f4"]
    writer.close(timeout=5)


def test_log_messages_hide_the_password():
    assert make_writer(FakeDatabase()).dsn_host == "postgresql://db:5432/x"


def test_a_failed_schema_setup_is_retried_on_a_fresh_connection(monkeypatch):
    from pytweezer.database import writer as writer_module

    attempts = []

    def flaky_schema(conn):
        attempts.append(conn)
        if len(attempts) == 1:
            raise RuntimeError("permission denied")

    monkeypatch.setattr(writer_module, "ensure_schema", flaky_schema)
    db = FakeDatabase()
    writer = make_writer(db)
    writer.write("m", {"x": 1.0})
    assert writer.flush(5)

    assert attempts[0].closed and not attempts[1].closed
    assert len(db.sent["readings"]) == 1
    writer.close()
