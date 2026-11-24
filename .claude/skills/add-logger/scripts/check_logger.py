"""Static check for a logger entry in CONFIG["Loggers"].

Builds the logger the same way ``logger_server.py`` does in production, but
forced into simulation and with the database replaced by a recording stand-in, so no
hardware is opened and no points are written. Then it calls ``read()`` once and
inspects what would have reached the database.

The failures it exists to catch are all silent at runtime:

* a ``"class"`` that is missing, doesn't import or isn't a ``Logger`` subclass
  (the process dies at startup with an error nobody sees, and the GUI tile
  just says Crashed)
* config keys the logger class never reads, so it silently runs on a default
* field values the database cannot store: ``DBWriter`` drops every non-numeric
  field without raising, so a string or array reading vanishes with no error
* a ``read()`` return shape the base loop can't unpack

Usage:
    uv run python check_logger.py "NI ADC Logger"
    uv run python check_logger.py --all
"""

import argparse
import inspect
import os
import sys

# Keep this check's log records off a running lab's Message hub.
os.environ.setdefault("PYTWEEZER_LOG_PUBLISH", "0")

from pytweezer.configuration.config import get_config, load_class
from pytweezer.servers import logger_server

#: Keys the framework itself consumes; never reported as unread by the class.
FRAMEWORK_KEYS = {
    "run_at_startup",
    "class",
    "host",
    "port",
    "interval",
    "limits",
    "simulate",
    "tooltip",
    "description",
}

problems = []
notes = []


def problem(msg):
    problems.append(msg)
    print(f"  FAIL  {msg}")


def note(msg):
    notes.append(msg)
    print(f"  warn  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


class RecordingWriter:
    """Stand-in for :class:`DBWriter` that records instead of connecting."""

    def __init__(self, *_args, **_kwargs):
        self.points = []
        self.measurements = []
        self.closed = False

    def write(self, measurement, fields, tags=None, time=None):
        self.points.append((measurement, fields, tags))

    def record_measurement(self, measurement, logger, interval_s, limits, active):
        self.measurements.append((measurement, logger, interval_s, limits, active))

    def close(self):
        self.closed = True


def check_class(conf):
    """``"class"`` must import as a :class:`Logger` subclass."""
    from pytweezer.loggers.base import Logger

    path = conf.get("class")
    if not path:
        problem("entry has no 'class'; expected a 'module.path:ClassName' string")
        return False
    try:
        cls = load_class(path)
    except Exception as exc:
        problem(f"'class' {path!r} does not import: {type(exc).__name__}: {exc}")
        return False
    if not (isinstance(cls, type) and issubclass(cls, Logger)):
        problem(f"'class' {path!r} is not a Logger subclass")
        return False
    ok(f"'class' imports {cls.__name__}")
    if "script" in conf:
        note("'script' is unnecessary - every logger runs logger_server.py")
    return True


def check_unread_keys(conf, instance):
    """Warn about config keys whose name never appears in the class's module.

    Logger config is read through ``self.conf.get(...)`` rather than matched
    against a constructor signature, so this is a text scan rather than a
    signature check -- a heuristic, hence a warning.
    """
    try:
        source = inspect.getsource(inspect.getmodule(type(instance)))
    except (OSError, TypeError):
        note("could not read the logger module's source to check config keys")
        return

    unread = [
        key
        for key in conf
        if key not in FRAMEWORK_KEYS and f"{key!r}"[1:-1] not in source
    ]
    for key in unread:
        note(
            f"config key {key!r} never appears in "
            f"{type(instance).__module__} - nothing reads it, so it silently "
            f"does nothing"
        )
    if not unread:
        ok("every config key is referenced by the logger class")


def check_points(instance, writer):
    """Call read() once and validate what it would write to the database."""
    from pytweezer.database.writer import _coerce_fields

    try:
        points = instance.read()
    except NotImplementedError:
        note(
            "read() is not implemented - assuming a push-driven logger that "
            "overrides run(); the field checks below are skipped"
        )
        return
    except Exception as exc:
        problem(f"read() raised {type(exc).__name__}: {exc}")
        return

    if points is None:
        note("read() returned None (no points this cycle) - nothing to check")
        return

    try:
        points = list(points)
    except TypeError:
        problem(
            f"read() returned {type(points).__name__}, which is not iterable; "
            f"it must yield (measurement, fields[, tags]) tuples or None"
        )
        return

    if not points:
        note("read() returned an empty sequence - nothing would be written")
        return

    malformed = False
    for index, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or not 2 <= len(point) <= 3:
            malformed = True
            problem(
                f"read()[{index}] is {point!r}; expected a "
                f"(measurement, fields) or (measurement, fields, tags) tuple"
            )
            continue
        measurement, fields = point[0], point[1]
        tags = point[2] if len(point) == 3 else None

        if not isinstance(measurement, str) or not measurement:
            problem(f"read()[{index}] measurement {measurement!r} is not a name")
        if not isinstance(fields, dict):
            malformed = True
            problem(f"read()[{index}] fields is {type(fields).__name__}, not a dict")
            continue
        if tags is not None and not isinstance(tags, dict):
            problem(f"read()[{index}] tags is {type(tags).__name__}, not a dict/None")

        kept = _coerce_fields(fields)
        dropped = sorted(set(fields) - set(kept))
        for key in dropped:
            problem(
                f"field {key!r}={fields[key]!r} is not numeric - DBWriter "
                f"drops it without raising, so this value never reaches "
                f"the database"
            )
        if not kept:
            problem(
                f"read()[{index}] has no storable fields, so the whole point "
                f"is discarded"
            )
        elif not dropped:
            ok(
                f"point {measurement!r}: fields "
                f"{', '.join(sorted(kept))}" + (f"; tags {tags}" if tags else "")
            )

    if malformed:
        return

    # Push them through the base loop the way run() would, to prove the
    # unpacking works and to show exactly what lands in the database.
    try:
        instance._write_points(points)
    except Exception as exc:
        problem(
            f"_write_points() failed on read()'s output: {type(exc).__name__}: {exc}"
        )
        return
    ok(f"one cycle would write {len(writer.points)} point(s) to the database")


def check_logger(name):
    print(f"\n=== {name} ===")
    before = len(problems)

    canonical, conf = logger_server.resolve_logger(name)
    if canonical != name:
        note(f"config key is actually {canonical!r}")

    if not check_class(conf):
        return

    if not conf.get("run_at_startup", False):
        note("'run_at_startup' is False - the tile will not auto-start with the GUI")

    # Build in simulation with the database stubbed out: same code path as
    # production, no hardware opened and no points written.
    from pytweezer.loggers import base as logger_base

    writer = RecordingWriter()
    real_writer_cls = logger_base.DBWriter
    logger_base.DBWriter = lambda *a, **k: writer
    try:
        sim_conf = dict(conf, simulate=True)
        try:
            instance = logger_server.build_logger(canonical, sim_conf)
        except Exception as exc:
            problem(
                f"build failed in simulation: {type(exc).__name__}: {exc} "
                f"(setup() runs inside __init__, so this is what the process "
                f"would do at startup)"
            )
            return
        ok(
            f"builds in simulation: {type(instance).__name__} "
            f"(interval={instance.interval}s)"
        )

        check_unread_keys(conf, instance)
        check_points(instance, writer)

        try:
            instance.close()
        except Exception as exc:
            problem(f"close() raised {type(exc).__name__}: {exc}")
        else:
            if not writer.closed:
                note("close() did not close the writer - call super().close()")
            else:
                ok("close() releases the source and the writer")
    finally:
        logger_base.DBWriter = real_writer_cls

    if len(problems) == before:
        print("  --> no problems found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name", nargs="?", help="logger name (key in CONFIG['Loggers'])"
    )
    parser.add_argument("--all", action="store_true", help="check every logger")
    args = parser.parse_args()

    if args.all:
        names = list(get_config().get("Loggers", {}))
    elif args.name:
        names = [args.name]
    else:
        parser.error("give a logger name or --all")

    for name in names:
        try:
            check_logger(name)
        except KeyError as exc:
            problem(f"{name}: {exc}")
        except Exception as exc:
            problem(f"{name}: unexpected {type(exc).__name__}: {exc}")

    print(f"\n{len(problems)} problem(s), {len(notes)} warning(s)")
    sys.stdout.flush()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
