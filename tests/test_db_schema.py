"""ensure_schema creates the tables, and the Timescale part only when it can."""

from contextlib import contextmanager

import pytest

from pytweezer.database.schema import TIMESCALE_SQL, ensure_schema


class FakeServer:
    """Mimics Postgres with TimescaleDB installed, preloaded or not.

    Without the preload, CREATE EXTENSION timescaledb is FATAL: the server
    drops the connection rather than raising an ordinary error.
    """

    def __init__(self, preloaded):
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
        rows = [(1,)] if "pg_settings" in sql and self.preloaded else []
        return FakeResult(rows)

    @contextmanager
    def transaction(self):
        yield


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


@pytest.mark.parametrize("preloaded", [True, False])
def test_reports_whether_timescale_is_active(preloaded):
    server = FakeServer(preloaded)
    assert ensure_schema(server) is preloaded
    assert (TIMESCALE_SQL in server.executed) is preloaded


def test_connection_survives_timescale_not_preloaded():
    server = FakeServer(preloaded=False)
    ensure_schema(server)
    assert not server.closed
