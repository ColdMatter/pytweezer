"""Unit tests for the InfluxDB loggers in ``pytweezer/loggers/``.

A logger's ``__init__`` calls ``setup()`` -- which opens its data source -- and
builds an :class:`InfluxWriter`. Neither is wanted in a test, so :func:`build`
constructs the logger with ``simulate`` forced on and the writer replaced by
:class:`RecordingWriter`. What's left under test is ``read()``: the mapping from
a raw source reading to the points that reach InfluxDB, which is the part worth
testing.

:func:`assert_storable` is the guard worth applying to every logger. InfluxDB
fields must be scalar, and ``InfluxWriter`` silently drops the ones that aren't
-- so a logger that returns a string or an array loses that value with no error
anywhere. A test is the only place that mistake surfaces cheaply.
"""

import pytest

from pytweezer.loggers import base as logger_base
from pytweezer.servers.influx_client import _coerce_fields


class RecordingWriter:
    """Stand-in for :class:`InfluxWriter` that records instead of connecting."""

    def __init__(self, *_args, **_kwargs):
        self.points = []
        self.closed = False

    def write(self, measurement, fields, tags=None, time=None):
        self.points.append((measurement, fields, tags))

    def close(self):
        self.closed = True


def build(cls, conf=None, name=None):
    """Return a logger instance with InfluxDB stubbed and simulation forced on.

    ``conf`` is the logger's ``CONFIG["Loggers"]`` entry (only the keys its
    ``setup()``/``read()`` actually read need to be present). The recording
    writer is reachable afterwards as ``instance.writer``.
    """
    conf = dict(conf or {})
    conf.setdefault("simulate", True)

    writer = RecordingWriter()
    real_cls = logger_base.InfluxWriter
    logger_base.InfluxWriter = lambda *a, **k: writer
    try:
        return cls(name or cls.__name__, conf)
    finally:
        logger_base.InfluxWriter = real_cls


def assert_storable(points):
    """Fail if any field would be silently dropped on the way into InfluxDB."""
    assert points, "read() produced no points"
    for point in points:
        assert 2 <= len(point) <= 3, (
            f"expected (measurement, fields[, tags]), got {point!r}"
        )
        measurement, fields = point[0], point[1]
        assert isinstance(measurement, str) and measurement
        assert isinstance(fields, dict)
        dropped = sorted(set(fields) - set(_coerce_fields(fields)))
        assert not dropped, (
            f"{measurement!r} fields {dropped} are not numeric; InfluxWriter "
            f"drops them without raising, so they never reach InfluxDB"
        )


# --------------------------------------------------------------------------- #
# ni_adc_logger
# --------------------------------------------------------------------------- #

NI_CONF = {
    "channels": ["Dev1/ai0", "Dev1/ai1"],
    "measurement": "ni_adc",
    "tags": {"system": "Rb"},
    "interval": 0.5,
}


def _ni_logger(conf=None):
    from pytweezer.loggers.ni_adc_logger import NIADCLogger

    return build(NIADCLogger, conf if conf is not None else NI_CONF)


def test_ni_adc_reads_one_point_per_cycle_with_a_field_per_channel():
    points = _ni_logger().read()

    assert len(points) == 1
    measurement, fields, tags = points[0]
    assert measurement == "ni_adc"
    assert set(fields) == {"ai0", "ai1"}          # device prefix stripped
    assert tags == {"system": "Rb"}


def test_ni_adc_fields_survive_the_trip_into_influx():
    assert_storable(_ni_logger().read())


def test_ni_adc_without_channels_reads_nothing_rather_than_failing():
    # A misconfigured logger should idle, not crash the process on every cycle.
    assert _ni_logger({"simulate": True}).read() is None


def test_ni_adc_close_releases_the_writer():
    logger = _ni_logger()
    logger.close()
    assert logger.writer.closed, "close() must call super().close()"


def test_interval_comes_from_config():
    assert _ni_logger().interval == 0.5


# --------------------------------------------------------------------------- #
# base loop
# --------------------------------------------------------------------------- #

def test_write_points_accepts_both_two_and_three_element_points():
    logger = _ni_logger()
    logger._write_points([
        ("a", {"x": 1.0}),
        ("b", {"y": 2.0}, {"system": "CaF"}),
        None,                                   # skipped, not an error
    ])

    assert [p[0] for p in logger.writer.points] == ["a", "b"]
    assert logger.writer.points[0][2] is None
    assert logger.writer.points[1][2] == {"system": "CaF"}


def test_read_failures_do_not_stop_the_run_loop():
    # run() catches read() errors and keeps polling, so a flaky sensor doesn't
    # take the logger down. Assert on the base class contract directly.
    logger = _ni_logger()

    def boom():
        raise RuntimeError("sensor unplugged")

    logger.read = boom
    logger.interval = 0.01

    import threading
    thread = threading.Thread(target=logger.run, daemon=True)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "run() died on a read() exception"
    logger.stop()
    thread.join(timeout=2)
