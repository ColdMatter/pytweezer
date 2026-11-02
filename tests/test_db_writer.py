"""DBWriter queues rows and writes them from a background thread, never raising."""

import json
import threading
from datetime import UTC, datetime

import numpy as np
import pytest

from pytweezer.database.writer import POINT_COLUMNS, RUN_COLUMNS, DBWriter
from tests.helpers import FakeDatabase


@pytest.fixture
def db():
    return FakeDatabase()


@pytest.fixture
def writer(db):
    writer = DBWriter("postgresql://u:secret@db:5432/x", connect=db, batch_s=0.01)
    yield writer
    writer.close(timeout=5)


def fields_written(db):
    return [row[2] for row in db.sent.get("readings", [])]


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


INSTANT = datetime(2026, 10, 7, 11, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "when",
    [INSTANT, INSTANT.timestamp(), "2026-10-07T12:00:00+01:00"],
    ids=["datetime", "epoch", "iso-with-offset"],
)
def test_run_and_point_times_store_the_same_instant_however_given(db, writer, when):
    writer.record_run({"rid": 7, "t_start": when})
    writer.record_point(7, 0, when, when, {}, {})
    assert writer.flush(2)

    (run,) = db.sent["runs"]
    (point,) = db.sent["points"]
    assert run[RUN_COLUMNS.index("t_start")] == INSTANT
    assert point[POINT_COLUMNS.index("t_start")] == INSTANT


def test_numpy_values_in_json_columns_are_stored_as_plain_json(db, writer):
    scalars = {"n": np.int64(3), "width": np.float32(0.5), "trace": np.arange(3)}
    writer.record_point(7, 0, 0.0, 1.0, {"x": np.float64(1.5)}, scalars)
    assert writer.flush(2)

    (point,) = db.sent["points"]
    assert json.loads(point[POINT_COLUMNS.index("scan_values")]) == {"x": 1.5}
    assert json.loads(point[POINT_COLUMNS.index("scalars")]) == {
        "n": 3,
        "width": 0.5,
        "trace": [0, 1, 2],
    }


def test_writes_return_while_the_database_is_still_connecting(db):
    connecting, connected = threading.Event(), threading.Event()

    def slow_connect(dsn):
        connecting.set()
        connected.wait()
        return db(dsn)

    writer = DBWriter(connect=slow_connect, batch_s=0.01)
    writer.write("m", {"x": 0.0})
    assert connecting.wait(2)
    more = threading.Thread(
        target=lambda: [writer.write("m", {"x": float(i)}) for i in range(100)]
    )
    more.start()
    more.join(timeout=1)
    finished_before_connecting = not more.is_alive()
    connected.set()
    assert finished_before_connecting, "write() waited for the database"
    assert writer.flush(2)
    assert len(db.sent["readings"]) == 101
    writer.close(timeout=5)


def test_rows_queued_through_an_outage_are_all_written_in_order(db, writer):
    db.up = False
    for name in ("a", "b", "c"):
        writer.write("m", {name: 1.0})
    assert not writer.flush(0.3)
    assert fields_written(db) == []

    db.up = True
    assert writer.flush(5)
    assert fields_written(db) == ["a", "b", "c"]


def test_a_backlog_beyond_max_pending_keeps_the_newest_rows():
    db = FakeDatabase(up=False)
    writer = DBWriter(connect=db, max_pending=3, batch_s=0.01)
    writer.write("m", {f"f{i}": float(i) for i in range(5)})
    assert not writer.flush(0.3)

    db.up = True
    assert writer.flush(5)
    assert fields_written(db) == ["f2", "f3", "f4"]
    writer.close(timeout=5)


def test_close_writes_every_queued_row_before_returning():
    db = FakeDatabase()
    writer = DBWriter(connect=db, batch_s=60)
    for name in ("a", "b", "c"):
        writer.write("m", {name: 1.0})
    writer.close(timeout=5)
    assert fields_written(db) == ["a", "b", "c"]


def test_a_server_without_timescaledb_preloaded_still_takes_writes():
    db = FakeDatabase(preloaded=False)
    writer = DBWriter(connect=db, batch_s=0.01)
    writer.write("m", {"x": 1.0})
    assert writer.flush(5)
    assert fields_written(db) == ["x"]
    writer.close(timeout=5)


def test_rows_survive_a_failed_schema_setup():
    db = FakeDatabase(schema_failures=1)
    writer = DBWriter(connect=db, batch_s=0.01)
    writer.write("m", {"x": 1.0})
    writer.flush(0.2)  # the failed attempt; the next flush cuts its backoff short
    assert writer.flush(5)
    assert fields_written(db) == ["x"]
    writer.close(timeout=5)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:secret@db:5432/x",
        "host=db port=5432 password=secret dbname=x",
        "host=db password='secret' dbname=x",
    ],
    ids=["url", "key-value", "quoted"],
)
def test_log_messages_hide_the_password(dsn):
    dsn_host = DBWriter(dsn).dsn_host
    assert "secret" not in dsn_host
    assert "db" in dsn_host
