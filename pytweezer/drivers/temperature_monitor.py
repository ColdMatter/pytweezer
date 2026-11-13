"""Polling a camera's sensor temperature into the database."""

import threading
from collections.abc import Callable

from pytweezer.database.writer import DBWriter
from pytweezer.logging_utils import get_logger

LOGGER = get_logger("temperature monitor")

FIELD = "sensor_temperature_c"


class TemperatureMonitor:
    """Write ``read_temperature()`` to ``measurement`` every ``interval_s``.

    The reads run on a daemon thread from construction until :meth:`stop`. A
    failed read is logged once and retried on the next interval. Crossing
    ``max_c`` upwards logs a warning, and coming back below it logs that too;
    ``max_c`` is also the upper limit Grafana's *Reading out of range* alert
    uses, and the measurement's interval drives its *Logger stale* alert.

    Args:
        read_temperature: Returns the temperature in °C.
        measurement: Database measurement the temperature is written to, as
            field ``sensor_temperature_c``.
        interval_s: Seconds between reads.
        max_c: Highest acceptable temperature in °C, or None for no limit.
    """

    def __init__(
        self,
        read_temperature: Callable[[], float],
        measurement: str,
        interval_s: float,
        max_c: float | None = None,
    ):
        self._read = read_temperature
        self.measurement = measurement
        self.interval_s = float(interval_s)
        self.max_c = max_c
        self._writer = DBWriter()
        self._stop = threading.Event()
        self._register(active=True)
        self._thread = threading.Thread(
            target=self._run, name=f"{measurement} monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop polling, waiting for a read in progress, and close the writer."""
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5.0)
        self._register(active=False)
        self._writer.close()

    def _register(self, active: bool) -> None:
        limits = {} if self.max_c is None else {FIELD: [None, float(self.max_c)]}
        self._writer.record_measurement(
            self.measurement, self.measurement, self.interval_s, limits, active
        )

    def _run(self) -> None:
        failing = False
        too_hot = False
        while True:
            try:
                temperature = float(self._read())
            except Exception:
                if not failing:
                    LOGGER.exception("Could not read %s", self.measurement)
                failing = True
            else:
                if failing:
                    LOGGER.info("Reading %s again", self.measurement)
                failing = False
                self._writer.write(self.measurement, {FIELD: temperature})
                hot = self.max_c is not None and temperature > self.max_c
                if hot and not too_hot:
                    LOGGER.warning(
                        "%s is %.1f °C, above its %.1f °C limit: check the cooling",
                        self.measurement,
                        temperature,
                        self.max_c,
                    )
                elif too_hot and not hot:
                    LOGGER.info("%s is back to %.1f °C", self.measurement, temperature)
                too_hot = hot
            if self._stop.wait(self.interval_s):
                return
