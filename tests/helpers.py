"""Helpers shared by several test modules (fixtures live in conftest.py)."""

import time

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
