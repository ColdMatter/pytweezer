# InfluxDB logging

Durable time-series logging of experiment values (laser powers, temperatures,
lock errors, MOT numbers, …) into a **self-hosted InfluxDB 2.x**. Two ways in:

1. **Loggers** — background workers that read a source and push to InfluxDB on an
   interval. Configured like devices, started from the GUI's **Loggers** tab.
2. **`InfluxWriter` / `log()`** — a direct write path for device drivers or a
   Jupyter notebook.

Logging is **opt-in**: nothing reaches InfluxDB unless a Logger or an explicit
`InfluxWriter.write()` / `log()` call puts it there. The existing pub/sub loggers
(`datalogger`, `imagelogger`, `propertylogger`) are unrelated and are **not**
forwarded to InfluxDB.

## 1. Self-hosting InfluxDB (one command, no DB admin)

InfluxDB 2.7 OSS is a single container that initialises itself — org, bucket, and
API token are all created from environment variables on first start. Run once on
the server PC:

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

The org (`pytweezer`), bucket (`devices`), and token (`pytweezer-token`) match the
defaults in `pytweezer/configuration/config.py`, so a fresh checkout works with no
extra configuration. The admin UI is at <http://localhost:8086>.

For a real deployment, set a strong token and export it instead of hardcoding
(see below). Data persists in the `influxdb-data` volume across restarts, so
day-to-day maintenance is just keeping the container running.

## 2. Connection config

All connection details live in one place — the `INFLUXDB` block in
`pytweezer/configuration/config.py` — and every value can be overridden by an
environment variable, so tokens need not be committed:

| Env var           | Default                     |
|-------------------|-----------------------------|
| `INFLUXDB_URL`    | `http://<SERVER_HOST>:8086` |
| `INFLUXDB_TOKEN`  | `pytweezer-token`           |
| `INFLUXDB_ORG`    | `pytweezer`                 |
| `INFLUXDB_BUCKET` | `devices`                   |

Nothing else in the codebase needs to know these — both the loggers and the
notebook helper read them from here.

## 3. Pushing values from a notebook

```python
from pytweezer.servers.influx_client import log

log("laser", power=1.23, wavelength=780)                 # fields as kwargs
log("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})  # dict + tags
```

`log(measurement, fields=None, tags=None, **field_kwargs)` uses a shared default
writer. Non-numeric fields are dropped automatically, and a write **never raises**
— if InfluxDB is unreachable it logs a warning and returns, so a cell is never
taken down by the database being offline.

For a driver or a longer-lived object, hold your own writer:

```python
from pytweezer.servers.influx_client import InfluxWriter

writer = InfluxWriter()
writer.write("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})
writer.close()   # or use it as a context manager
```

## 4. Writing a new Logger

A **Logger owns its own data source** — a real or virtual device that belongs
directly to it (an NI DAQ, a serial sensor, a socket feed), which it opens itself.
It is *not* a way to scrape values off an existing device RPC server (for that,
see §6).

A Logger subclasses `pytweezer.loggers.base.Logger` and overrides `setup()`
(open the device) and `read()` (return the current values). The base `run()` loop
handles the polling cadence, writing, and teardown.

The worked example is `NIADCLogger` (`pytweezer/loggers/ni_adc_logger.py`): it
creates its own `nidaqmx.Task`, reads the configured analog-input channels, and
logs them as one measurement — with a `simulate` fallback so it runs without NI
hardware.

```python
# pytweezer/loggers/my_logger.py
from pytweezer.loggers.base import Logger

class MyLogger(Logger):
    def setup(self):
        self.source = ...            # open YOUR device/handle, using self.conf
        self.simulate = bool(self.conf.get("simulate", False))

    def read(self):
        # return an iterable of (measurement, fields[, tags]) tuples, or None
        return [("my_measurement", {"value": self.source.read()}, {"system": "Rb"})]

    def close(self):
        self.source.close()          # release the device
        super().close()
```

Register the class by adding a factory to `LOGGER_REGISTRY` in
`pytweezer/servers/logger_server.py`:

```python
def _make_my_logger(name, conf):
    from pytweezer.loggers.my_logger import MyLogger
    return MyLogger(name, conf)

LOGGER_REGISTRY = {
    "ni_adc": _make_ni_adc,
    "my_logger": _make_my_logger,
}
```

Then add a config entry under `CONFIG["Loggers"]` in `config.py`:

```python
"My Logger": {
    "active": True,
    "script": "../pytweezer/servers/logger_server.py",
    "logger": "my_logger",     # selects the factory above
    "host": SERVER_HOST,
    "interval": 2.0,
    "simulate": SIMULATING,
    # ...any logger-specific keys read by your setup()/read()...
},
```

## 5. Running a logger

From the GUI: `pytweezer-server` → **Loggers** tab → start/stop the tile.

Standalone (dual-mode `main()`, same as devices):

```bash
poetry run pytweezer-logger "NI ADC Logger"
# or
poetry run python pytweezer/servers/logger_server.py "NI ADC Logger"
```

Loggers run on the server PC alongside InfluxDB.

## 6. Logging from existing devices (cameras, MotMasters, …)

Loggers own *their own* devices. To log values off a device that already has a
driver and RPC server (e.g. a camera temperature), **don't** write a Logger for
it — instead give that driver its own `InfluxWriter` and write from inside the
driver where the value is already in hand:

```python
# inside a device driver
from pytweezer.servers.influx_client import InfluxWriter

class MyCamera:
    def __init__(self, ...):
        self._influx = InfluxWriter()

    def _on_frame(self, ...):
        self._influx.write("my_camera", {"temperature": self.get_temperature()},
                           tags={"system": "Rb"})
```

`InfluxWriter.write()` never raises, so this is safe to sprinkle into hot paths.

## 7. Future ideas

- **Poll an existing device's RPC methods.** A generic logger that connects with
  `get_device(conf["device"])` (`pytweezer/servers/device_client.py`) and reads a
  configured list of RPC attributes/methods each interval — a low-effort way to
  log from any device without touching its driver. Deferred in favour of the
  owns-its-device model above; add it as a `Logger` subclass + `LOGGER_REGISTRY`
  entry if wanted.
