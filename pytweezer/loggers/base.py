"""Generic :class:`Logger` base class for database metric loggers.

A *Logger* is a small background worker whose only job is to read some data
source and write values into the pytweezer database. Concrete loggers subclass this and override
:meth:`setup` (open connections) and :meth:`read` (return the current values);
the base :meth:`run` loop handles the polling cadence, writing, and teardown.

For a source that pushes data rather than being polled (e.g. a ZMQ subscription),
override :meth:`run` directly instead of :meth:`read`.

Loggers are launched exactly like devices — see
``pytweezer/servers/logger_server.py`` and the ``CONFIG["Loggers"]`` config
category. Nothing reaches the database unless a Logger (or an explicit
:class:`~pytweezer.database.writer.DBWriter` call) puts it there.
"""

import signal
import time
from datetime import UTC, datetime

from pytweezer.database.writer import DBWriter
from pytweezer.logging_utils import get_logger

logger = get_logger("Logger")


class Logger:
    """Base class for background database loggers.

    Subclasses typically override :meth:`setup` and :meth:`read`. Config values
    live in the logger's ``CONFIG["Loggers"][name]`` entry, available as
    ``self.conf``. ``self.writer`` is a ready-to-use
    :class:`~pytweezer.database.writer.DBWriter`.
    """

    def __init__(self, name, conf):
        self.name = name
        self.conf = conf or {}
        self.interval = float(self.conf.get("interval", 1.0))
        self.limits = _parse_limits(self.conf.get("limits", {}))
        self.writer = DBWriter()
        self._running = False
        self._registered = set()
        self.setup()

    # ---- overridable hooks --------------------------------------------- #

    def setup(self):
        """Open connections / prepare state. Override as needed (default: no-op)."""

    def read(self):
        """Return an iterable of ``(measurement, fields, tags)`` tuples, or ``None``.

        Called every ``interval`` seconds by :meth:`run`. Override this in a
        polling logger. ``tags`` may be ``None``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override read() or run()"
        )

    # ---- driver loop --------------------------------------------------- #

    def _write_points(self, points):
        read_time = datetime.now(UTC)
        for point in points:
            if point is None:
                continue
            if len(point) == 2:
                measurement, fields = point
                tags = None
            else:
                measurement, fields, tags = point
            if measurement not in self._registered:
                self._register(measurement, active=True)
                self._registered.add(measurement)
            self.writer.write(measurement, fields, tags=tags, time=read_time)

    def _register(self, measurement, active):
        self.writer.record_measurement(
            measurement, self.name, self.interval, self.limits, active
        )

    def run(self):
        """Poll :meth:`read` every ``interval`` seconds and write the results.

        Blocks until interrupted (Ctrl-C / SIGTERM). Override for push-driven
        loggers, but call :meth:`close` on exit.
        """
        self._running = True

        def _stop(_signo, _frame):
            self._running = False

        try:
            signal.signal(signal.SIGTERM, _stop)
        except (ValueError, OSError):
            # Not on the main thread — rely on KeyboardInterrupt / stop().
            pass

        logger.info("Logger %r started (interval=%.2fs)", self.name, self.interval)
        try:
            while self._running:
                try:
                    points = self.read()
                except Exception:
                    logger.exception("Logger %r read() failed", self.name)
                    points = None
                if points:
                    self._write_points(points)
                # Sleep in slices so SIGTERM/stop() takes effect promptly.
                waited = 0.0
                while self._running and waited < self.interval:
                    time.sleep(min(0.1, self.interval))
                    waited += 0.1
        except KeyboardInterrupt:
            logger.info("Logger %r interrupted, shutting down.", self.name)
        finally:
            self._running = False
            self.close()

    def stop(self):
        self._running = False

    def close(self):
        """Release resources. Override to add teardown, but call ``super().close()``."""
        for measurement in self._registered:
            self._register(measurement, active=False)
        self.writer.close()


def _parse_limits(limits):
    """``{field: [low, high]}`` with float or ``None`` bounds; raises on anything else."""
    parsed = {}
    for field, bounds in limits.items():
        try:
            low, high = bounds
            parsed[field] = [None if b is None else float(b) for b in (low, high)]
        except (TypeError, ValueError):
            raise ValueError(
                f"limits[{field!r}] must be [low, high] (either may be None), "
                f"got {bounds!r}"
            ) from None
    return parsed
