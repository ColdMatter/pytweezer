"""Helpers shared by several test modules (fixtures live in conftest.py)."""

import time
from contextlib import contextmanager

import numpy as np
from PyQt6 import QtCore

from pytweezer.experiment import Experiment, ListAxis, Number, Scan
from pytweezer.experiment.runner import open_writer


def until(condition, timeout=5.0, qapp=None):
    """Poll ``condition`` until it holds or ``timeout`` passes; return its last value.

    Pass ``qapp`` to keep processing Qt events while waiting.
    """
    deadline = time.monotonic() + timeout
    while not (result := condition()) and time.monotonic() < deadline:
        if qapp is not None:
            qapp.processEvents()
        time.sleep(0.01)
    return result


class RecordingDB:
    """Stand-in for :class:`~pytweezer.database.writer.DBWriter` that records rows."""

    def __init__(self):
        self.runs = []
        self.points = []
        self.writes = []
        self.measurements = []
        self.closed = False

    def record_run(self, run):
        self.runs.append(dict(run))

    def record_point(self, rid, point_index, t_start, t_end, scan_values, scalars):
        self.points.append(
            {
                "rid": rid,
                "point_index": point_index,
                "t_start": t_start,
                "t_end": t_end,
                "scan_values": scan_values,
                "scalars": scalars,
            }
        )

    def write(self, measurement, fields, tags=None, time=None):
        self.writes.append((measurement, fields, tags, time))

    def record_measurement(self, measurement, logger, interval_s, limits, active):
        self.measurements.append((measurement, logger, interval_s, limits, active))

    def close(self):
        self.closed = True


class FakeFeed(QtCore.QObject):
    """Stands in for the GUI's live queue/points feed from the Experiment Manager."""

    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.connected = True
        self.rows = {}

    def points(self, rid):
        return list(self.rows.get(rid, []))

    def close(self):
        pass


class Sweep(Experiment):
    x = Number(0.0, unit="V")

    def run_point(self):
        pass


def write_run(path, rid=7, points_run=3, point_s=0.0):
    """A 3-point run of which the first ``points_run`` ran; point 1 records no counts.

    Each point lasts at least ``point_s``, so a reading can be placed inside it.
    """
    scan = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])])
    _, points, writer = open_writer(path, Sweep, {}, scan, {"rid": rid, "label": "t"})
    for point in points[:points_run]:
        writer.begin_point(point)
        if point.index != 1:
            writer.record("counts", 10 * point.index)
        writer.record("image", np.zeros((2, 2)))
        time.sleep(point_s)
        writer.end_point()
    writer.finalise("completed" if points_run == len(points) else "terminated")
    writer.close()
    return path


class FakeConnection:
    """Postgres at the network edge; records executemany() rows per table.

    Unless TimescaleDB is preloaded, CREATE EXTENSION timescaledb is FATAL: the
    server drops the connection rather than raising an ordinary error.
    """

    def __init__(self, sent, preloaded=True, schema_fails=False):
        self.sent = sent
        self.preloaded = preloaded
        self.schema_fails = schema_fails
        self.closed = False

    def execute(self, sql, params=None):
        if self.closed:
            raise ConnectionError("the connection is lost")
        if self.schema_fails:
            raise RuntimeError("permission denied for schema public")
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

    def __init__(self, up=True, preloaded=True, schema_failures=0):
        self.up = up
        self.preloaded = preloaded
        self.schema_failures = schema_failures
        self.sent = {}

    def __call__(self, dsn):
        if not self.up:
            raise ConnectionError("database down")
        schema_fails = self.schema_failures > 0
        self.schema_failures -= 1
        return FakeConnection(self.sent, self.preloaded, schema_fails)
