"""The single write path into the pytweezer database.

Loggers, device drivers, notebooks and the Experiment Manager all write through
:class:`DBWriter`. The connection string lives in the ``DATABASE`` block of
``pytweezer/configuration/config.py`` (overridable with ``PYTWEEZER_DB_DSN``).

Notebook / REPL usage::

    from pytweezer.database import log
    log("laser", power=1.23, wavelength=780, tags={"system": "Rb"})

Driver usage (own writer instance)::

    from pytweezer.database import DBWriter
    writer = DBWriter()
    writer.write("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})
    writer.close()

Writes **never raise and never block**: each call queues rows for a background
thread, which batches them into the database. While the database is unreachable
rows are held (oldest dropped beyond ``max_pending``) and retried with backoff,
so neither a notebook cell nor the Experiment Manager's loop waits on it.
"""

import atexit
import json
import threading
from collections import deque
from datetime import UTC, datetime
from numbers import Real

from pytweezer.configuration.config import DATABASE
from pytweezer.database.schema import ensure_schema
from pytweezer.logging_utils import get_logger

logger = get_logger("Database")

RUN_COLUMNS = (
    "rid",
    "experiment",
    "class_name",
    "label",
    "submitter",
    "arguments",
    "scan",
    "status",
    "error",
    "t_submit",
    "t_start",
    "t_end",
    "points_done",
    "points_total",
    "h5_path",
    "simulated",
)
POINT_COLUMNS = ("rid", "point_index", "t_start", "t_end", "scan_values", "scalars")
_JSON_COLUMNS = {"arguments", "scan", "scan_values", "scalars"}


def _upsert(table, columns, key):
    placeholders = ", ".join(
        "%s::jsonb" if c in _JSON_COLUMNS else "%s" for c in columns
    )
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in key)
    return (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}"
    )


SQL = {
    "readings": "INSERT INTO readings (time, measurement, field, value, tags) "
    "VALUES (%s, %s, %s, %s, %s::jsonb)",
    "runs": _upsert("runs", RUN_COLUMNS, ("rid",)),
    "points": _upsert("points", POINT_COLUMNS, ("rid", "point_index")),
}


def _coerce_fields(fields):
    """Keep only real numeric (and bool) field values; drop everything else.

    A caller can hand over a whole reading dict without pre-filtering: arrays,
    strings and ``None`` are skipped.
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


def _json(value):
    return json.dumps(value or {}, default=str)


def _timestamp(value):
    """A datetime from a datetime, an epoch float, or an ISO string; else None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, Real):
        return datetime.fromtimestamp(value, UTC)
    return datetime.fromisoformat(value)


def _connect_psycopg(dsn):
    import psycopg

    return psycopg.connect(dsn, autocommit=True, connect_timeout=3)


class DBWriter:
    """Queues rows and writes them from a background thread.

    Constructing one never touches the network; the thread and connection start
    on the first write. One instance per logger, driver or notebook is fine.

    Args:
        dsn: libpq connection string; defaults to ``DATABASE["dsn"]``.
        connect: ``connect(dsn)`` returning an autocommit connection; injectable
            for tests.
        max_pending: rows held while the database is unreachable before the
            oldest are dropped.
        batch_s: how long the thread gathers rows before sending a batch.
    """

    def __init__(self, dsn=None, *, connect=None, max_pending=1_000_000, batch_s=0.5):
        self.dsn = dsn or DATABASE["dsn"]
        self._connect = connect or _connect_psycopg
        self.max_pending = max_pending
        self.batch_s = batch_s
        self._pending = deque()
        self._cond = threading.Condition()
        self._thread = None
        self._closing = False
        self._hurry = False
        self._in_flight = 0
        self._conn = None
        self._failed = False
        self._dropped = 0

    # ---- public API ---------------------------------------------------- #

    def write(self, measurement, fields, tags=None, time=None):
        """Queue one reading: a row per numeric field, all sharing one timestamp.

        Args:
            measurement (str): what was measured, e.g. ``"ni_adc"``.
            fields (dict): field name -> value; non-numeric values are dropped.
            tags (dict, optional): string labels such as system or device.
            time (datetime, optional): timestamp; defaults to now (UTC).
        """
        clean = _coerce_fields(fields)
        if not clean:
            logger.debug("No numeric fields to write for measurement %r", measurement)
            return
        time = time or datetime.now(UTC)
        tags_json = _json({k: str(v) for k, v in (tags or {}).items()})
        self._enqueue(
            ("readings", (time, measurement, field, float(value), tags_json))
            for field, value in clean.items()
        )

    def record_run(self, run):
        """Queue an insert-or-update of one ``runs`` row, keyed on ``rid``.

        ``run`` maps column names (:data:`RUN_COLUMNS`) to values; missing ones
        are stored as NULL. Times may be datetimes, epoch floats or ISO strings.
        """
        row = []
        for column in RUN_COLUMNS:
            value = run.get(column)
            if column in _JSON_COLUMNS:
                value = _json(value)
            elif column.startswith("t_"):
                value = _timestamp(value)
            row.append(value)
        self._enqueue([("runs", tuple(row))])

    def record_point(self, rid, point_index, t_start, t_end, scan_values, scalars):
        """Queue an insert-or-update of one ``points`` row (times as epoch floats)."""
        row = (
            int(rid),
            int(point_index),
            _timestamp(t_start),
            _timestamp(t_end),
            _json(scan_values),
            _json(scalars),
        )
        self._enqueue([("points", row)])

    def flush(self, timeout=5.0):
        """Wait until every queued row is written; returns False on timeout."""
        with self._cond:
            if self._thread is None:
                return not self._pending
            self._hurry = True
            self._cond.notify_all()
            return self._cond.wait_for(
                lambda: not self._pending and not self._in_flight, timeout
            )

    def close(self, timeout=5.0):
        """Write what is queued (waiting at most ``timeout``) and disconnect."""
        with self._cond:
            thread = self._thread
            self._closing = True
            self._cond.notify_all()
        if thread is not None:
            thread.join(timeout)
        with self._cond:
            if self._pending:
                logger.warning(
                    "Closing with %d rows not written to the database",
                    len(self._pending),
                )
                self._pending.clear()
            self._thread = None
            self._closing = False
        self._disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    # ---- background thread --------------------------------------------- #

    def _enqueue(self, rows):
        with self._cond:
            self._pending.extend(rows)
            self._trim()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="DBWriter", daemon=True
                )
                self._thread.start()
                atexit.register(self.close)
            self._cond.notify_all()

    def _trim(self):
        excess = len(self._pending) - self.max_pending
        if excess > 0:
            for _ in range(excess):
                self._pending.popleft()
            if not self._dropped:
                logger.warning(
                    "Database backlog over %d rows; dropping the oldest",
                    self.max_pending,
                )
            self._dropped += excess

    def _run(self):
        backoff = 0.0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending or self._closing)
                if not self._closing:
                    self._cond.wait_for(
                        lambda: self._closing or self._hurry,
                        backoff or self.batch_s,
                    )
                batch = list(self._pending)
                self._pending.clear()
                self._in_flight = len(batch)
                closing = self._closing
            ok = self._send(batch) if batch else True
            with self._cond:
                if not ok:
                    self._pending.extendleft(reversed(batch))
                    self._trim()
                self._in_flight = 0
                self._hurry = False
                self._cond.notify_all()
            backoff = 0.0 if ok else min(max(1.0, 2 * backoff), 30.0)
            if closing:
                return

    def _ensure_connected(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        self._conn = self._connect(self.dsn)
        ensure_schema(self._conn)
        return self._conn

    def _send(self, batch):
        by_table = {}
        for table, row in batch:
            by_table.setdefault(table, []).append(row)
        try:
            conn = self._ensure_connected()
            with conn.transaction(), conn.cursor() as cursor:
                for table, rows in by_table.items():
                    cursor.executemany(SQL[table], rows)
        except Exception as exc:
            if not self._failed:
                logger.warning("Database write failed (%s): %s", self.dsn_host, exc)
                self._failed = True
            self._disconnect()
            return False
        if self._failed:
            logger.info("Database writes resumed (%s)", self.dsn_host)
            self._failed = False
        return True

    def _disconnect(self):
        conn, self._conn = self._conn, None
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    @property
    def dsn_host(self):
        """The DSN with any password removed, for log messages."""
        if "@" in self.dsn:
            scheme, _, rest = self.dsn.partition("://")
            return f"{scheme}://{rest.rpartition('@')[2]}"
        return self.dsn


_default_writer = None


def get_default_writer():
    """Return a lazily created process-wide :class:`DBWriter`."""
    global _default_writer
    if _default_writer is None:
        _default_writer = DBWriter()
    return _default_writer


def log(measurement, fields=None, tags=None, time=None, **field_kwargs):
    """Write one reading using the shared default writer.

    Fields may be passed as a dict and/or as keyword arguments::

        log("laser", power=1.23, wavelength=780)
        log("laser", {"power": 1.23}, tags={"system": "Rb"})
    """
    merged = dict(fields or {})
    merged.update(field_kwargs)
    get_default_writer().write(measurement, merged, tags=tags, time=time)
