"""Generic :class:`Logger` base class for InfluxDB metric loggers.

A *Logger* is a small background worker whose only job is to read some data
source and push values into InfluxDB. Concrete loggers subclass this and override
:meth:`setup` (open connections) and :meth:`read` (return the current values);
the base :meth:`run` loop handles the polling cadence, writing, and teardown.

For a source that pushes data rather than being polled (e.g. a ZMQ subscription),
override :meth:`run` directly instead of :meth:`read`.

Loggers are launched exactly like devices — see
``pytweezer/servers/logger_server.py`` and the ``CONFIG["Loggers"]`` config
category. Nothing reaches InfluxDB unless a Logger (or an explicit
:class:`~pytweezer.servers.influx_client.InfluxWriter` call) puts it there.
"""

import signal
import time

from pytweezer.servers.influx_client import InfluxWriter
from pytweezer.logging_utils import get_logger

logger = get_logger("Logger")


class Logger:
    """Base class for background InfluxDB loggers.

    Subclasses typically override :meth:`setup` and :meth:`read`. Config values
    live in the logger's ``CONFIG["Loggers"][name]`` entry, available as
    ``self.conf``. ``self.writer`` is a ready-to-use
    :class:`~pytweezer.servers.influx_client.InfluxWriter`.
    """

    def __init__(self, name, conf):
        self.name = name
        self.conf = conf or {}
        self.interval = float(self.conf.get("interval", 1.0))
        self.writer = InfluxWriter()
        self._running = False
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
        for point in points:
            if point is None:
                continue
            if len(point) == 2:
                measurement, fields = point
                tags = None
            else:
                measurement, fields, tags = point
            self.writer.write(measurement, fields, tags=tags)

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
        self.writer.close()
