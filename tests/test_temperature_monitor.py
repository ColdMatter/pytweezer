"""The camera temperature monitor, with the DBWriter replaced by RecordingDB."""

import pytest

from pytweezer.database.writer import _coerce_fields
from pytweezer.drivers import temperature_monitor
from pytweezer.drivers.temperature_monitor import FIELD, TemperatureMonitor
from tests.helpers import until


@pytest.fixture
def monitor(monkeypatch, recording_db):
    monitors = []

    def start(read, max_c=None):
        monkeypatch.setattr(temperature_monitor, "DBWriter", lambda: recording_db)
        monitors.append(TemperatureMonitor(read, "cam_temperature", 0.01, max_c))
        return monitors[-1]

    yield start
    for started in monitors:
        started.stop()


def test_each_reading_is_written_as_a_storable_field(monitor, recording_db):
    readings = iter([-8.0, -7.5])
    monitor(lambda: next(readings, -7.5))

    assert until(lambda: len(recording_db.writes) >= 2)
    (measurement, fields, *_), second = recording_db.writes[:2]
    assert measurement == "cam_temperature"
    assert _coerce_fields(fields) == {FIELD: -8.0}
    assert second[1] == {FIELD: -7.5}


def test_the_limit_reaches_the_measurements_table(monitor, recording_db):
    monitor(lambda: -8.0, max_c=0.0)

    (_measurement, _logger, interval_s, limits, active), *_ = recording_db.measurements
    assert limits == {FIELD: [None, 0.0]}
    assert interval_s == 0.01
    assert active


def test_a_failed_read_does_not_stop_the_readings_after_it(monitor, recording_db):
    readings = iter([RuntimeError("DCAMERR_TEMPERATURE_TROUBLE")])

    def read():
        reading = next(readings, 66.0)
        if isinstance(reading, Exception):
            raise reading
        return reading

    monitor(read)

    assert until(lambda: recording_db.writes)
    assert recording_db.writes[0][1] == {FIELD: 66.0}


def test_stopping_marks_the_measurement_inactive(recording_db, monkeypatch):
    monkeypatch.setattr(temperature_monitor, "DBWriter", lambda: recording_db)
    TemperatureMonitor(lambda: -8.0, "cam_temperature", 0.01).stop()

    assert recording_db.measurements[-1][-1] is False
    assert recording_db.closed
