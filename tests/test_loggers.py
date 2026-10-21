"""Database loggers in ``pytweezer/loggers/``, with the DBWriter replaced by
:class:`RecordingDB` unless a test needs the real one."""

import sys
import threading
from types import SimpleNamespace

import pytest

from pytweezer.database.writer import DBWriter, _coerce_fields
from pytweezer.loggers import base as logger_base
from pytweezer.loggers.ni_adc_logger import NIADCLogger
from tests.helpers import FakeDatabase, RecordingDB

NI_CONF = {
    "channels": ["Dev1/ai0", "Dev1/ai1"],
    "measurement": "ni_adc",
    "tags": {"system": "Rb"},
    "interval": 0.5,
}


@pytest.fixture
def build(monkeypatch):
    def build(cls, conf, name=None, writer=None):
        writer = writer or RecordingDB()
        monkeypatch.setattr(logger_base, "DBWriter", lambda *a, **k: writer)
        return cls(name or cls.__name__, {"simulate": True, **conf})

    return build


@pytest.fixture
def ni_logger(build):
    return build(NIADCLogger, NI_CONF)


class Scripted(logger_base.Logger):
    """Returns the configured points, one list per ``read()``, then stops."""

    def setup(self):
        self.script = list(self.conf["script"])

    def read(self):
        points = self.script.pop(0)
        if not self.script:
            self.stop()
        return points


def run_to_end(logger):
    """Run the logger's loop, as the logger server does, until it stops itself."""
    thread = threading.Thread(target=logger.run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "run() did not stop"


def test_points_from_one_read_are_written_with_one_shared_timestamp(build):
    script = [[("a", {"x": 1.0}), ("b", {"y": 2.0}, {"system": "CaF"}), None]]
    logger = build(Scripted, {"script": script, "interval": 0.01})
    run_to_end(logger)

    written = logger.writer.writes
    assert [(w[0], w[1], w[2]) for w in written] == [
        ("a", {"x": 1.0}, None),
        ("b", {"y": 2.0}, {"system": "CaF"}),
    ]
    times = {w[3] for w in written}
    assert len(times) == 1 and None not in times


def test_stopping_the_loop_writes_the_rows_already_queued(build):
    db = FakeDatabase()
    writer = DBWriter(connect=db, batch_s=60)
    script = [[("a", {"x": 1.0})]]
    run_to_end(build(Scripted, {"script": script, "interval": 0.01}, writer=writer))
    assert [row[2] for row in db.sent["readings"]] == ["x"]


LIMITS_CONF = {"interval": 0.01, "limits": {"x": [0, 5], "y": [None, 1]}}
TWO_READS = [[("a", {"x": 1.0}), ("b", {"y": 2.0})], [("a", {"x": 1.0})]]


def test_each_measurement_is_registered_once_with_its_limits(build):
    logger = build(Scripted, {**LIMITS_CONF, "script": TWO_READS}, name="Probe")
    run_to_end(logger)

    limits = {"x": [0.0, 5.0], "y": [None, 1.0]}
    active = [m for m in logger.writer.measurements if m[4]]
    assert active == [
        ("a", "Probe", 0.01, limits, True),
        ("b", "Probe", 0.01, limits, True),
    ]


def test_a_clean_stop_marks_every_measurement_inactive(build):
    logger = build(Scripted, {**LIMITS_CONF, "script": TWO_READS}, name="Probe")
    run_to_end(logger)

    inactive = [m for m in logger.writer.measurements if not m[4]]
    assert sorted(m[0] for m in inactive) == ["a", "b"]


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


@pytest.fixture
def daq(monkeypatch):
    """nidaqmx at the hardware edge: one task that returns ``daq.voltages``."""
    daq = SimpleNamespace(channels=[], voltages=None)
    task = SimpleNamespace(
        ai_channels=SimpleNamespace(add_ai_voltage_chan=daq.channels.append),
        read=lambda: daq.voltages,
        close=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "nidaqmx", SimpleNamespace(Task=lambda: task))
    return daq


def test_ni_adc_stores_each_channel_voltage_under_its_own_name(build, daq):
    channels = ["Dev1/ai0", "Dev1/ai1", "Dev2/ai3"]
    logger = build(NIADCLogger, {"channels": channels, "simulate": False})
    daq.voltages = [0.1, 0.2, 0.3]

    ((measurement, fields, _tags),) = logger.read()
    assert daq.channels == channels
    assert measurement == "ni_adc"
    assert list(fields.items()) == [("ai0", 0.1), ("ai1", 0.2), ("ai3", 0.3)]


def test_ni_adc_stores_a_single_channel_bare_float(build, daq):
    logger = build(NIADCLogger, {"channels": ["Dev1/ai0"], "simulate": False})
    daq.voltages = 0.5  # nidaqmx returns a float, not a list, for one channel

    ((_measurement, fields, _tags),) = logger.read()
    assert fields == {"ai0": 0.5}


def test_ni_adc_channels_sharing_a_short_name_keep_their_full_names(build):
    conf = {**NI_CONF, "channels": ["Dev1/ai0", "Dev2/ai0", "Dev1/ai1"]}
    ((_measurement, fields, _tags),) = build(NIADCLogger, conf).read()
    assert set(fields) == {"Dev1/ai0", "Dev2/ai0", "ai1"}


def test_simulated_ni_adc_gives_a_storable_field_per_channel(ni_logger):
    ((measurement, fields, tags),) = ni_logger.read()
    assert measurement == "ni_adc"
    assert set(fields) == {"ai0", "ai1"}
    assert set(_coerce_fields(fields)) == set(fields), "non-numeric fields"
    assert tags == {"system": "Rb"}
