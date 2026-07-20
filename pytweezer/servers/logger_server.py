"""Generic logger launcher — the ``CONFIG["Loggers"]`` counterpart to
``device_server.py``.

Each logger in ``CONFIG["Loggers"]`` is a background worker that reads a data
source and pushes values into InfluxDB. Rather than give every logger its own
``argparse`` + config-reading boilerplate, this module provides a single
launcher:

* ``pytweezer-logger <logger_name>`` (console script) — start one logger.
* The **logger manager** launches the same thing: each logger's ``script`` in the
  config points here, so ``ProcessTile`` runs ``python logger_server.py <name>``.

Adding a *new logger instance* means only editing ``config.py``. Adding a new
*logger type* means writing a :class:`~pytweezer.loggers.base.Logger` subclass and
adding one factory to :data:`LOGGER_REGISTRY`. Subclasses are imported lazily
inside their factory so a broken/optional logger never breaks importing this
launcher.
"""

import argparse
import signal

from pytweezer.servers.configreader import ConfigReader
from pytweezer.logging_utils import get_logger

logger = get_logger("logger server")


# --------------------------------------------------------------------------- #
# Logger-type factories: (name, conf) -> Logger instance
# --------------------------------------------------------------------------- #

def _make_ni_adc(name, conf):
    from pytweezer.loggers.ni_adc_logger import NIADCLogger

    return NIADCLogger(name, conf)


#: Maps a logger's ``"logger"`` config key to the factory that builds it.
LOGGER_REGISTRY = {
    "ni_adc": _make_ni_adc,
}


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
    loggers = ConfigReader.getConfiguration().get("Loggers", {})
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
    """Return the :class:`Logger` instance for the logger named ``name``.

    The logger's ``"logger"`` key selects the factory in :data:`LOGGER_REGISTRY`.
    """
    if conf is None:
        name, conf = resolve_logger(name)

    logger_type = conf.get("logger")
    if logger_type is None:
        raise KeyError(
            f"Logger {name!r} has no 'logger' key; expected one of "
            f"{sorted(LOGGER_REGISTRY)}"
        )
    try:
        factory = LOGGER_REGISTRY[logger_type]
    except KeyError:
        raise KeyError(
            f"Unknown logger type {logger_type!r} for {name!r}; "
            f"registered types: {sorted(LOGGER_REGISTRY)}"
        ) from None

    return factory(name, conf)


def run_logger(name):
    """Build and run the logger named ``name`` (blocks until interrupted)."""
    name, conf = resolve_logger(name)
    instance = build_logger(name, conf)
    logger.info("Running logger %r (type=%s)", name, conf.get("logger"))
    instance.run()


def main():
    parser = argparse.ArgumentParser(
        description="Start a configured InfluxDB logger"
    )
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
