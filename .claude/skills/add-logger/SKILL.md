---
name: add-logger
description: Write a new InfluxDB logger for pytweezer — a Logger subclass in pytweezer/loggers/ that owns its data source, plus its LOGGER_REGISTRY factory and CONFIG["Loggers"] entry. Use this whenever the user wants to add, create, write, or register a logger, or wants a value recorded/tracked/monitored/trended over time in InfluxDB — a laser power, temperature, pressure, lock error, ADC voltage, flow rate, magnetic field — or asks how to get a reading into InfluxDB, Grafana, or the Loggers tab, or to add an entry to CONFIG["Loggers"]. Applies even when the user only names the sensor and the quantity ("log the chamber pressure every 10 seconds") without saying "logger" or "InfluxDB". This is for durable time-series metrics, not live pub/sub stream processing.
---

# Adding an InfluxDB logger to pytweezer

A logger is a small background process that reads a source it owns and pushes
numbers into InfluxDB on an interval. `pytweezer/loggers/base.py` owns the
polling loop, the writing and the teardown, so writing a logger means **one
class with a `setup` and a `read` method**, plus two lines of wiring.

Three edits, always the same three:

1. a `Logger` subclass in `pytweezer/loggers/<name>_logger.py`
2. a factory in `LOGGER_REGISTRY` (`pytweezer/servers/logger_server.py`)
3. an entry in `CONFIG["Loggers"]` (`pytweezer/configuration/config.py`)

A logger **owns its data source** — it opens the DAQ, serial port or socket
itself. Reading a value off a device that already has a driver and an RPC server
is a different job with a different answer (`InfluxWriter` inside that driver);
see `docs/influx_logging.md` §6 and don't build a Logger for it.

## How the framework runs your logger

Worth reading before writing code, because two of these steps decide where your
mistakes will surface:

1. `run_logger(name)` resolves the config entry and calls the factory that
   `"logger"` names in `LOGGER_REGISTRY`.
2. `Logger.__init__` stores `conf`, reads `interval`, builds an `InfluxWriter`,
   and **calls your `setup()`** — so opening the hardware happens at process
   start. A `setup()` that raises kills the logger immediately; the GUI tile
   shows Crashed and the traceback only exists in the log.
3. `run()` loops: `read()` → `_write_points(...)` → sleep `interval`.
4. Ctrl-C or SIGTERM unwinds the loop and calls `close()`.

`read()` raising is *not* fatal — `run()` catches it, logs it, and polls again
next cycle, so a sensor that drops out doesn't take the process down. Neither
does InfluxDB: `InfluxWriter.write()` never raises, it logs a warning and
returns. That resilience is deliberate, and it's also why a broken logger looks
exactly like a working one from the outside — nothing crashes, values just never
arrive. Verify with the bundled checker rather than by watching the tile.

The sleep is spent in 0.1 s slices so a stop signal takes effect promptly, which
makes the interval accurate to about that. Loggers are for monitoring cadences
(seconds to minutes), not acquisition.

## Writing the class

```python
class ChamberPressureLogger(Logger):
    """One-line summary of what this logs and off what hardware."""

    def setup(self):
        self.measurement = self.conf.get("measurement", "chamber")
        self.tags = self.conf.get("tags")
        self.simulate = bool(self.conf.get("simulate", False))
        self._port = None
        if self.simulate:
            logger.warning("Logger %r running in SIMULATION MODE", self.name)
            return
        import serial              # lazy: keep the import off dev machines
        self._port = serial.Serial(self.conf["device"], timeout=1)

    def read(self):
        if self.simulate:
            return [(self.measurement, {"pressure": 2.0e-9}, self.tags)]
        raw = self._port.readline()
        return [(self.measurement, {"pressure": float(raw)}, self.tags)]

    def close(self):
        if self._port is not None:
            self._port.close()
        super().close()
```

`pytweezer/loggers/ni_adc_logger.py` is the worked example to follow.

The conventions behind that shape:

- **Everything configurable comes from `self.conf`, with a default.** The config
  entry is the only knob a labmate has; a value hardcoded in `setup()` means
  editing Python to retune a channel. Nothing validates the keys you read, so
  the checker's unread-key warning is how a typo gets caught.
- **Import the vendor library inside `setup()`, not at module top.** Every logger
  module is importable on every machine that way — including the dev laptop
  running the tests, and `logger_server.py` itself.
- **Handle `simulate` and return plausible fake readings.** `SIMULATING` in the
  config makes the whole system runnable with no hardware; a logger that can't
  simulate is a logger nobody can develop against. Fake values should look like
  the real signal (a slow drift, a little noise) — a constant makes a broken
  plot indistinguishable from a working one.
- **`close()` ends with `super().close()`**, which closes the writer. Releasing
  a serial port or DAQ task but leaking the Influx client is the usual slip.
- **Return `None` from `read()` when there's nothing to report** — a
  misconfigured or idle logger should sit quietly rather than write garbage.

For a source that *pushes* (a ZMQ subscription, a callback-driven SDK), override
`run()` instead of `read()`, and call `close()` on the way out — you're
replacing the loop, so the teardown is yours to preserve.

## What `read()` returns, and what InfluxDB will keep

An iterable of `(measurement, fields)` or `(measurement, fields, tags)`, or
`None`. One point per measurement per cycle; several tuples if you're logging
into more than one measurement.

- **Fields must be numeric or bool.** `InfluxWriter` drops everything else
  silently — a string status, `None`, a numpy array — and if that leaves no
  fields, the whole point is discarded. No exception, no warning above debug
  level. This is the single most common way a logger appears to work and stores
  nothing. Encode a state as a number (`{"locked": 1}`) or put it in a tag.
- **Tags are indexed strings, so keep them static and low-cardinality** —
  `{"system": "Rb"}`, `{"channel": "ai0"}`. A tag that changes every cycle (a
  timestamp, a reading) creates a new series each time and will bloat the
  database.
- **Field names are the schema.** Renaming a field later leaves the old series
  orphaned, so pick names that will still read well in a year: `ai0`, not
  `value1`.
- Timestamps are added for you at write time. Pass one explicitly only if the
  reading carries its own clock.

## Register the type

In `pytweezer/servers/logger_server.py`, add a factory and one registry entry:

```python
def _make_chamber_pressure(name, conf):
    from pytweezer.loggers.chamber_pressure_logger import ChamberPressureLogger

    return ChamberPressureLogger(name, conf)


LOGGER_REGISTRY = {
    "ni_adc": _make_ni_adc,
    "chamber_pressure": _make_chamber_pressure,
}
```

The import lives inside the factory so that a logger whose dependency isn't
installed can't break importing the launcher — which every other logger needs.

A *new instance* of an existing logger type needs no code at all: a second
config entry pointing at the same `"logger"` key is enough.

## The config entry

Under `CONFIG["Loggers"]` in `pytweezer/configuration/config.py`:

```python
"Chamber Pressure Logger": {
    "active": False,
    "script": "../pytweezer/servers/logger_server.py",
    "logger": "chamber_pressure",
    "host": SERVER_HOST,
    "interval": 10.0,
    "simulate": SIMULATING,
    "device": "COM4",
    "measurement": "chamber",
    "tags": {"system": "Rb"},
},
```

- **`script`** is that exact path for every logger, and it is not optional:
  the Loggers tab indexes `params["script"]` with no default, so omitting it
  raises `KeyError` while the tab is being built and takes out *every* logger
  row, not just yours.
- **`logger`** selects the registry factory; it must match the key you added.
- **`host`** is `SERVER_HOST` — loggers run on the server PC, next to InfluxDB.
- **No `port`.** A logger binds nothing and serves nothing, so don't copy
  `get_next_port()` over from a device entry; its row is statused by polling the
  subprocess, not by probing a socket.
- **`active: True`** starts the logger automatically when the server GUI opens.
  Leave it `False` until the hardware is actually there.
- **`simulate`** is the module-level `SIMULATING` flag, not a hardcoded bool.
- Remaining keys are yours, read by `setup()`/`read()` through `self.conf`.

Connection details (URL, token, org, bucket) are not your concern — they live in
the `INFLUXDB` block and `InfluxWriter` reads them itself.

## Verify it

Run the bundled checker. It builds the logger exactly as production would, with
simulation forced on and InfluxDB replaced by a recorder, then does one dry
`read()` and reports what would actually have been stored:

```bash
poetry run python .claude/skills/add-logger/scripts/check_logger.py "Chamber Pressure Logger"
```

It flags the silent ones — an unregistered `"logger"` type, a missing `script`,
config keys nothing reads, fields InfluxDB would drop, a `read()` shape the loop
can't unpack, a `close()` that forgets the writer.

Then add a case to `tests/test_loggers.py`, copying the bundled starter first if
that file doesn't exist yet:

```bash
cp .claude/skills/add-logger/assets/test_loggers.py tests/test_loggers.py
```

`build(cls, conf)` constructs the logger with the writer stubbed and `simulate`
on, so a test is a plain function call with no hardware and no database.
`assert_storable(points)` is the guard worth applying to every logger — it fails
on exactly the fields InfluxDB would have thrown away:

```python
def test_chamber_pressure_reads_one_point():
    from pytweezer.loggers.chamber_pressure_logger import ChamberPressureLogger

    points = build(ChamberPressureLogger, {"measurement": "chamber"}).read()

    assert_storable(points)
    assert points[0][1]["pressure"] > 0
```

Test what actually bites: the reading-to-fields mapping for a known input, the
degenerate case (no channels configured, sensor returning nothing) that must
return `None` rather than raise, and `close()` releasing both the source and the
writer. Run:

```bash
poetry run pytest tests/ -q
```

## Turning it on

From the GUI: `pytweezer-server` → **Loggers** tab → Start. Standalone:

```bash
poetry run pytweezer-logger "Chamber Pressure Logger"
```

Values land in the `devices` bucket; the Influx UI is at
<http://localhost:8086> on the server PC. If InfluxDB isn't running yet,
`docs/influx_logging.md` §1 has the one-command Docker setup.

A green checker and a passing test prove the transform and the wiring, not that
the serial port speaks what you assumed or that the numbers mean the right
physics. Say which of those you actually verified rather than implying the
logger is confirmed end to end.
