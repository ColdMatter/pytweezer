---
name: add-logger
description: Write a new InfluxDB logger for pytweezer — a Logger subclass in pytweezer/loggers/ that owns its data source, plus its LOGGER_REGISTRY factory and CONFIG["Loggers"] entry. Use this whenever the user wants to add, create, write, or register a logger, or wants a value recorded/tracked/monitored/trended over time in InfluxDB — a laser power, temperature, pressure, lock error, ADC voltage, flow rate, magnetic field — or asks how to get a reading into InfluxDB, Grafana, or the Loggers tab, or to add an entry to CONFIG["Loggers"]. Applies even when the user only names the sensor and the quantity ("log the chamber pressure every 10 seconds") without saying "logger" or "InfluxDB". This is for durable time-series metrics, not live pub/sub stream processing.
---

# Adding an InfluxDB logger to pytweezer

A logger is a small background process that reads a source it owns and pushes
numbers into InfluxDB on an interval. `pytweezer/loggers/base.py` owns the
polling loop, the writing and the teardown, so writing one means **a class with
`setup` and `read`**, plus two lines of wiring:

1. a `Logger` subclass in `pytweezer/loggers/<name>_logger.py`
2. a factory in `LOGGER_REGISTRY` (`pytweezer/servers/logger_server.py`)
3. an entry in `CONFIG["Loggers"]` (`pytweezer/configuration/config.py`)

**A logger owns its data source** — it opens the DAQ, serial port or socket
itself. To log a value off a device that already has a driver and an RPC server,
don't build a logger: give that driver its own `InfluxWriter` and write from
inside it, where the value is already in hand. `InfluxWriter.write()` never
raises, so that is safe even in a hot path.

## What the base class does to you

`Logger.__init__` calls your `setup()`, so hardware opens at process start and a
`setup()` that raises kills the logger immediately — the GUI tile shows Crashed
and the traceback exists only in the log. After that `run()` loops `read()` →
write → sleep, and `close()` runs on Ctrl-C or SIGTERM.

Nothing else is fatal: `read()` raising is caught and retried next cycle, and
`InfluxWriter.write()` swallows a dead database. That resilience is deliberate,
and it is why **a broken logger looks exactly like a working one** — nothing
crashes, values just never arrive. Verify with the checker, not by watching the
tile.

The sleep runs in 0.1 s slices so a stop signal lands promptly, which also sets
how accurate `interval` is. Loggers are for monitoring cadences (seconds to
minutes), not acquisition.

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
        import serial  # lazy: keep the import off dev machines

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

`pytweezer/loggers/ni_adc_logger.py` is the worked example to follow. The
conventions behind that shape:

- **Everything configurable comes from `self.conf`, with a default** — the config
  entry is the only knob a labmate has. Nothing validates the keys you read, so
  the checker's unread-key warning is how a typo gets caught.
- **Import the vendor library inside `setup()`**, so every logger module stays
  importable on a machine without that hardware.
- **Handle `simulate`, and make the fake look like the real signal** (a slow
  drift, a little noise). A constant makes a broken plot indistinguishable from
  a working one.
- **`close()` ends with `super().close()`** — releasing the port but leaking the
  Influx client is the usual slip.
- **Return `None` from `read()`** when there is nothing to report.

For a source that *pushes* (a ZMQ subscription, a callback-driven SDK), override
`run()` instead of `read()` and call `close()` on the way out — you are replacing
the loop, so its teardown becomes yours.

## What `read()` returns, and what InfluxDB keeps

An iterable of `(measurement, fields)` or `(measurement, fields, tags)`, or
`None`; several tuples if you log into more than one measurement.

- **Fields must be numeric or bool.** Anything else — a string status, `None`, a
  numpy array — is dropped silently, and if that empties the point, the point
  goes too. No exception, no warning above debug level. This is the most common
  way a logger appears to work and stores nothing. Encode states as numbers
  (`{"locked": 1}`) or put them in a tag.
- **Tags are indexed strings: keep them static and low-cardinality**
  (`{"system": "Rb"}`). A tag that changes every cycle creates a new series each
  time and bloats the database.
- **Field names are the schema** — renaming one later orphans the old series, so
  pick `ai0`, not `value1`.
- Timestamps are added at write time; pass one only if the reading carries its
  own clock.

## Wiring it up

In `pytweezer/servers/logger_server.py`, a factory plus one registry entry — the
import goes *inside* the factory so a missing dependency can't break importing
the launcher every other logger needs:

```python
def _make_chamber_pressure(name, conf):
    from pytweezer.loggers.chamber_pressure_logger import ChamberPressureLogger

    return ChamberPressureLogger(name, conf)


LOGGER_REGISTRY = {"ni_adc": _make_ni_adc, "chamber_pressure": _make_chamber_pressure}
```

Then the config entry under `CONFIG["Loggers"]`:

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

- **`script`** is that exact path for every logger and is **not optional** — the
  Loggers tab indexes it with no default, so omitting it raises `KeyError` while
  the tab builds and takes out *every* logger row, not just yours.
- **`logger`** must match the registry key; **`host`** is `SERVER_HOST` (loggers
  run next to InfluxDB); **`simulate`** is the `SIMULATING` flag, not a literal.
- **No `port`** — a logger binds nothing; its row is statused by polling the
  subprocess, not by probing a socket.
- `active: True` auto-starts it with the server GUI; leave it `False` until the
  hardware is there. Remaining keys are yours, read through `self.conf`.

Connection details (URL, token, org, bucket) live in the `INFLUXDB` block;
`InfluxWriter` reads them itself.

A *new instance* of an existing logger type needs no code — just a second config
entry pointing at the same `"logger"` key.

## Verify it

The bundled checker builds the logger as production would, with simulation
forced on and InfluxDB replaced by a recorder, does one dry `read()`, and reports
what would actually have been stored — flagging the silent failures: an
unregistered `"logger"`, a missing `script`, config keys nothing reads, fields
InfluxDB would drop, a `read()` shape the loop can't unpack, a `close()` that
forgets the writer.

```bash
poetry run python .claude/skills/add-logger/scripts/check_logger.py "Chamber Pressure Logger"
```

Then add a case to `tests/test_loggers.py` (copy the starter first if that file
doesn't exist: `cp .claude/skills/add-logger/assets/test_loggers.py tests/`). Its
`build(cls, conf)` constructs the logger with the writer stubbed and `simulate`
on, and `assert_storable(points)` fails on exactly the fields InfluxDB would have
discarded — worth applying in every logger's test:

```python
def test_chamber_pressure_reads_one_point():
    points = build(ChamberPressureLogger, {"measurement": "chamber"}).read()

    assert_storable(points)
    assert points[0][1]["pressure"] > 0
```

Test what bites: the reading-to-fields mapping, the degenerate case (no channels,
sensor returning nothing) returning `None` rather than raising, and `close()`
releasing both the source and the writer. Then `poetry run pytest tests/ -q`.

## Turning it on

`pytweezer-server` → **Loggers** tab → Start, or standalone:

```bash
poetry run pytweezer-logger "Chamber Pressure Logger"
```

Values land in the `devices` bucket; the Influx UI is at
<http://localhost:8086> on the server PC. If InfluxDB isn't running there yet,
2.7 OSS is one self-initialising container:

```bash
docker run -d --name influxdb -p 8086:8086 \
  -v influxdb-data:/var/lib/influxdb2 \
  -e DOCKER_INFLUXDB_INIT_MODE=setup \
  -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
  -e DOCKER_INFLUXDB_INIT_PASSWORD=changeme-please \
  -e DOCKER_INFLUXDB_INIT_ORG=pytweezer \
  -e DOCKER_INFLUXDB_INIT_BUCKET=devices \
  -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=pytweezer-token \
  influxdb:2.7
```

Those values match the `INFLUXDB` defaults, so a fresh checkout works untouched;
data persists in the `influxdb-data` volume, and every value is overridable by
env var (`INFLUXDB_URL`, `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_BUCKET`) so
a real deployment need not commit a token.

A green checker and a passing test prove the transform and the wiring — not that
the serial port speaks what you assumed, or that the numbers mean the right
physics. Say which you actually verified.
