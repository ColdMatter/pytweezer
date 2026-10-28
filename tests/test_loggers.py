"""Database loggers in ``pytweezer/loggers/``, built with simulation on and the
DBWriter replaced by :class:`RecordingDB`."""

import threading

import pytest

from pytweezer.database.writer import _coerce_fields
from pytweezer.loggers import base as logger_base
from pytweezer.loggers.ni_adc_logger import NIADCLogger
from tests.helpers import RecordingDB

NI_CONF = {
    "channels": ["Dev1/ai0", "Dev1/ai1"],
    "measurement": "ni_adc",
    "tags": {"system": "Rb"},
    "interval": 0.5,
}


@pytest.fixture
def build(monkeypatch):
    def build(cls, conf, name=None):
        writer = RecordingDB()
        monkeypatch.setattr(logger_base, "DBWriter", lambda *a, **k: writer)
        return cls(name or cls.__name__, {"simulate": True, **conf})

    return build


@pytest.fixture
def ni_logger(build):
    return build(NIADCLogger, NI_CONF)


def assert_storable(points):
    """Fail if any field would be silently dropped by DBWriter (non-numeric)."""
    assert points
    for measurement, fields, *_tags in points:
        assert isinstance(measurement, str) and measurement
        dropped = set(fields) - set(_coerce_fields(fields))
        assert not dropped, f"{measurement!r} fields {dropped} are not numeric"


def test_ni_adc_reads_one_storable_point_per_cycle(ni_logger):
    points = ni_logger.read()

    assert_storable(points)
    ((measurement, fields, tags),) = points
    assert measurement == "ni_adc"
    assert set(fields) == {"ai0", "ai1"}
    assert tags == {"system": "Rb"}


def test_ni_adc_channels_sharing_a_short_name_keep_their_full_names(build):
    conf = {**NI_CONF, "channels": ["Dev1/ai0", "Dev2/ai0", "Dev1/ai1"]}
    ((_measurement, fields, _tags),) = build(NIADCLogger, conf).read()
    assert set(fields) == {"Dev1/ai0", "Dev2/ai0", "ai1"}


def test_write_points_accepts_two_and_three_element_points_at_one_time(ni_logger):
    ni_logger._write_points(
        [
            ("a", {"x": 1.0}),
            ("b", {"y": 2.0}, {"system": "CaF"}),
            None,
        ]
    )

    written = ni_logger.writer.writes
    assert [(p[0], p[2]) for p in written] == [("a", None), ("b", {"system": "CaF"})]
    times = {p[3] for p in written}
    assert len(times) == 1 and None not in times


def test_each_measurement_is_registered_once_with_its_limits(build):
    logger = build(
        logger_base.Logger,
        {"interval": 2.0, "limits": {"x": [0, 5], "y": [None, 1]}},
        name="Probe",
    )
    logger._write_points([("a", {"x": 1.0}), ("b", {"y": 2.0})])
    logger._write_points([("a", {"x": 1.0})])

    limits = {"x": [0.0, 5.0], "y": [None, 1.0]}
    assert logger.writer.measurements == [
        ("a", "Probe", 2.0, limits, True),
        ("b", "Probe", 2.0, limits, True),
    ]


def test_close_marks_measurements_inactive_and_closes_the_writer(ni_logger):
    ni_logger._write_points([("a", {"x": 1.0})])
    ni_logger.close()

    assert ni_logger.writer.measurements[-1] == ("a", "NIADCLogger", 0.5, {}, False)
    assert ni_logger.writer.closed


def test_malformed_limits_fail_at_startup(build):
    with pytest.raises(ValueError, match="x"):
        build(logger_base.Logger, {"limits": {"x": 5}})


def test_read_failures_do_not_stop_the_run_loop(ni_logger):
    def boom():
        raise RuntimeError("sensor unplugged")

    ni_logger.read = boom
    ni_logger.interval = 0.01

    thread = threading.Thread(target=ni_logger.run, daemon=True)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "run() died on a read() exception"
    ni_logger.stop()
    thread.join(timeout=2)
