"""Create the pytweezer tables on a fresh database, idempotently.

Every writer runs :func:`ensure_schema` on each new connection, so a new
database needs no manual setup beyond installing the ``timescaledb`` extension.
Without the extension the plain tables still work, uncompressed.
"""

from importlib.resources import files

from pytweezer.logging_utils import get_logger

logger = get_logger("Database")

CORE_SQL = files(__package__).joinpath("schema.sql").read_text()
TIMESCALE_SQL = files(__package__).joinpath("timescale.sql").read_text()

# Serialises schema creation between processes connecting at the same moment;
# concurrent CREATE TABLE IF NOT EXISTS can otherwise collide in pg_type.
_SCHEMA_LOCK_KEY = 72780001

_warned_no_timescale = False


def ensure_schema(conn) -> bool:
    """Create any missing tables; returns whether TimescaleDB is active.

    ``conn`` must be in autocommit mode so each ``transaction()`` is a real one.
    """
    global _warned_no_timescale
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
        conn.execute(CORE_SQL)
    try:
        if not _timescale_preloaded(conn):
            raise RuntimeError(
                "timescaledb is not in the server's shared_preload_libraries"
            )
        with conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
            conn.execute(TIMESCALE_SQL)
    except Exception as exc:
        if not _warned_no_timescale:
            logger.warning(
                "TimescaleDB unavailable, readings stay an uncompressed table: %s",
                exc,
            )
            _warned_no_timescale = True
        return False
    return True


def _timescale_preloaded(conn) -> bool:
    # Without the preload, CREATE EXTENSION timescaledb is FATAL and drops the
    # connection. shared_preload_libraries is superuser-only, but the loader's
    # own settings are visible to any role exactly when it is preloaded.
    row = conn.execute(
        "SELECT 1 FROM pg_settings WHERE name = 'timescaledb.disable_load'"
    ).fetchone()
    return row is not None
