"""InfluxDB write helper — the single write path into the time-series store.

Everything that pushes values into InfluxDB goes through :class:`InfluxWriter`:
the background :mod:`pytweezer.loggers` and any ad-hoc code (a device driver, a
Jupyter notebook). Connection details (URL/token/org/bucket) live in one place —
the ``INFLUXDB`` block of ``pytweezer/configuration/config.py``, overridable via
environment variables — so callers never have to know them.

Notebook / REPL usage (requirement: "convenient way to push from a notebook")::

    from pytweezer.servers.influx_client import log
    log("laser", power=1.23, wavelength=780, tags={"system": "Rb"})

Custom / driver usage (own writer instance)::

    from pytweezer.servers.influx_client import InfluxWriter
    writer = InfluxWriter()
    writer.write("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})
    writer.close()

Design note: writes **never raise**. A missing/unreachable InfluxDB logs a warning
and the call returns — so a notebook cell or a logger loop is never taken down by
the database being offline.
"""

from datetime import datetime, timezone
from numbers import Real

from pytweezer.configuration.config import INFLUXDB
from pytweezer.logging_utils import get_logger

logger = get_logger("InfluxDB")


def _coerce_fields(fields):
    """Keep only real numeric (and bool) field values; drop everything else.

    InfluxDB fields must be scalar. Arrays/strings/None are silently skipped so a
    caller can hand over a whole reading dict without pre-filtering.
    """
    clean = {}
    for key, value in fields.items():
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, Real):
            clean[key] = float(value)
        else:
            logger.debug("Skipping non-numeric field %r=%r", key, value)
    return clean


class InfluxWriter:
    """Thin wrapper over ``influxdb_client`` bound to the configured bucket.

    Lazily connects on first write so constructing a writer never fails even if
    InfluxDB is down. Reusable and cheap; one instance per logger/notebook is fine.
    """

    def __init__(self, url=None, token=None, org=None, bucket=None):
        self.url = url or INFLUXDB["url"]
        self.token = token or INFLUXDB["token"]
        self.org = org or INFLUXDB["org"]
        self.bucket = bucket or INFLUXDB["bucket"]
        self._client = None
        self._write_api = None
        self._failed = False  # set True after a connect failure to throttle log spam

    def _ensure_connected(self):
        """Build the client/write API on demand. Returns True if usable."""
        if self._write_api is not None:
            return True
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self._client = InfluxDBClient(
                url=self.url, token=self.token, org=self.org
            )
            # SYNCHRONOUS keeps ordering simple and surfaces errors on write();
            # the payloads here are low-rate, so batching buys little.
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._failed = False
            logger.debug("Connected to InfluxDB at %s (org=%s, bucket=%s)",
                         self.url, self.org, self.bucket)
            return True
        except Exception as exc:
            if not self._failed:
                logger.warning("Cannot connect to InfluxDB at %s: %s", self.url, exc)
                self._failed = True
            return False

    def write(self, measurement, fields, tags=None, time=None):
        """Write one point. Never raises — logs a warning on failure and returns.

        Args:
            measurement (str): InfluxDB measurement name.
            fields (dict): field name -> value; non-numeric values are dropped.
            tags (dict, optional): indexed string tags (e.g. system/device).
            time (datetime, optional): timestamp; defaults to now (UTC).
        """
        clean = _coerce_fields(fields)
        if not clean:
            logger.debug("No numeric fields to write for measurement %r", measurement)
            return
        if not self._ensure_connected():
            return
        try:
            from influxdb_client import Point

            point = Point(measurement)
            for tag_key, tag_value in (tags or {}).items():
                point = point.tag(tag_key, str(tag_value))
            for field_key, field_value in clean.items():
                point = point.field(field_key, field_value)
            point = point.time(time or datetime.now(timezone.utc))

            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            self._failed = False  # recovered — let the next outage log again
        except Exception as exc:
            if not self._failed:
                logger.warning("InfluxDB write failed (measurement=%r): %s",
                               measurement, exc)
                self._failed = True

    def close(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        finally:
            self._client = None
            self._write_api = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


# --------------------------------------------------------------------------- #
# Notebook convenience: module-level default writer + log()
# --------------------------------------------------------------------------- #

_default_writer = None


def get_default_writer():
    """Return a lazily-created process-wide :class:`InfluxWriter`."""
    global _default_writer
    if _default_writer is None:
        _default_writer = InfluxWriter()
    return _default_writer


def log(measurement, fields=None, tags=None, time=None, **field_kwargs):
    """Push one point to InfluxDB using the shared default writer.

    Convenience for notebooks/REPL — connection details are hidden in config.
    Fields may be passed as a dict and/or as keyword arguments::

        log("laser", power=1.23, wavelength=780)
        log("laser", {"power": 1.23}, tags={"system": "Rb"})
    """
    merged = dict(fields or {})
    merged.update(field_kwargs)
    get_default_writer().write(measurement, merged, tags=tags, time=time)
