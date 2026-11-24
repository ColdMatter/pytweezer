"""Generic logger launcher — the ``CONFIG["Loggers"]`` counterpart to
``device_server.py``.

Every logger runs this one launcher: ``pytweezer-logger <logger_name>``, or the
Loggers tab, which runs ``python logger_server.py <name>``. It imports the
:class:`~pytweezer.loggers.base.Logger` subclass named by the entry's
``"class"`` (a ``"module.path:ClassName"`` string) and runs it, so a new logger
type is a new class and a new instance is a new config entry; nothing is
registered here.
"""

import argparse
import signal

from pytweezer.configuration.config import get_config, load_class
from pytweezer.logging_utils import get_logger

logger = get_logger("logger server")


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #


def _normalize(s):
    """Collapse whitespace and lowercase, for lenient name matching."""
    return "".join(s.split()).lower()


def resolve_logger(name):
    """Return ``(canonical_name, conf)`` for ``name`` in ``CONFIG["Loggers"]``.

    Matches the config key exactly first, then whitespace-/case-insensitively so
    command-line callers can pass ``rblaserlogger`` for ``"Rb Laser Logger"``.
    Raises ``KeyError`` listing the available loggers if nothing matches.
    """
    loggers = get_config().get("Loggers", {})
    if name in loggers:
        return name, loggers[name]

    target = _normalize(name)
    for key, conf in loggers.items():
        if _normalize(key) == target:
            logger.debug("Resolved logger name %r -> %r", name, key)
            return key, conf

    available = ", ".join(sorted(loggers)) or "(none)"
    raise KeyError(
        f"Logger {name!r} not found in config. Available loggers: {available}"
    )


def build_logger(name, conf=None):
    """Return the :class:`Logger` instance for the logger named ``name``."""
    if conf is None:
        name, conf = resolve_logger(name)

    path = conf.get("class")
    if not path:
        raise KeyError(
            f"Logger {name!r} has no 'class'; expected a 'module.path:ClassName' "
            "string naming its Logger subclass"
        )
    return load_class(path)(name, conf)


def run_logger(name):
    """Build and run the logger named ``name`` (blocks until interrupted)."""
    name, conf = resolve_logger(name)
    instance = build_logger(name, conf)
    logger.info("Running logger %r (%s)", name, conf["class"])
    instance.run()


def main():
    parser = argparse.ArgumentParser(description="Start a configured database logger")
    parser.add_argument("name", help="logger name (key in CONFIG['Loggers'])")
    args, _unknown = parser.parse_known_args()

    def _stop(_signo, _frame):
        # Let run()'s loop unwind so close()/teardown runs.
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stop)
    except (ValueError, OSError):
        pass

    run_logger(args.name)


if __name__ == "__main__":
    main()
