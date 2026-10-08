# Database: monitor readings and experiment runs

Slow lab monitoring (laser powers, temperatures, lock errors, wavemeter
frequencies) and a record of every experiment run go into one
**PostgreSQL + TimescaleDB** database on the server PC. Grafana reads it for
dashboards. Because both kinds of data are in one place, "what were the
conditions during this run?" is a single SQL query.

Three tables:

| Table      | One row per                     | Written by                         |
|------------|---------------------------------|------------------------------------|
| `readings` | field of a monitor reading      | Loggers, drivers, `log()`          |
| `runs`     | queued experiment run           | Experiment Manager (start, finish) |
| `points`   | scan point that ran             | Experiment Manager (each point)    |

Bulk data (images, traces) stays in the per-run h5 files. `runs.h5_path` points
at the file, relative to the data root. Fast raw data (kHz traces, scope
waveforms) belongs in h5 files too, not in `readings`.

Logging is **opt-in**: nothing reaches `readings` unless a Logger or an explicit
`DBWriter.write()` / `log()` call puts it there.

## 1. Setting up the server PC (Windows, native)

Do this once on PH-BEAST. Check the TimescaleDB install docs for which Postgres
versions its current Windows build supports before choosing the version.

1. **PostgreSQL**: install with the EDB Windows installer (postgresql.org). It
   runs as a Windows service and asks for a password for the `postgres`
   superuser.
2. **TimescaleDB**: download the Windows build for that Postgres version, unzip
   it, and run its `setup.exe` as administrator. Let it run `timescaledb-tune`,
   which adds `timescaledb` to `shared_preload_libraries`. Restart the
   PostgreSQL service.
3. **Database and roles**: open `psql -U postgres` and run:

   ```sql
   CREATE ROLE pytweezer LOGIN PASSWORD '<writer password>';
   CREATE DATABASE pytweezer OWNER pytweezer;
   \c pytweezer
   CREATE EXTENSION IF NOT EXISTS timescaledb;

   -- read-only role for Grafana
   CREATE ROLE grafana LOGIN PASSWORD '<grafana password>';
   GRANT CONNECT ON DATABASE pytweezer TO grafana;
   GRANT USAGE ON SCHEMA public TO grafana;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
   ALTER DEFAULT PRIVILEGES FOR ROLE pytweezer IN SCHEMA public
       GRANT SELECT ON TABLES TO grafana;
   ```

   The tables themselves are created by the first pytweezer process that
   connects (`pytweezer/database/schema.sql` and `timescale.sql`). You can also
   apply them by hand with `psql -U pytweezer -d pytweezer -f <file>`.
4. **Lab network access**: add this line to `pg_hba.conf` (in the data
   directory):

   ```
   host  pytweezer  pytweezer  10.59.3.0/24  scram-sha-256
   ```

   Check that `listen_addresses = '*'` is set in `postgresql.conf`, restart the
   service, and open the port to the lab subnet in an administrator PowerShell:

   ```powershell
   New-NetFirewallRule -DisplayName "PostgreSQL (lab)" -Direction Inbound `
     -Protocol TCP -LocalPort 5432 -RemoteAddress 10.59.3.0/24 -Action Allow
   ```

5. **Every lab PC**: set the connection string, so the password stays out of
   the repository:

   ```bat
   setx PYTWEEZER_DB_DSN "postgresql://pytweezer:<writer password>@10.59.3.1:5432/pytweezer"
   ```

6. **Grafana**: install with the Windows installer (grafana.com). It runs as a
   service on port 3000. Then:
   1. Copy the three files under `deploy/grafana/provisioning/`
      (`datasources\pytweezer.yaml`, `dashboards\pytweezer.yaml` and
      `alerting\pytweezer.yaml`) into the matching folders under
      `<Grafana>\conf\provisioning\`.
   2. In the dashboards file, set `path` to `deploy\grafana\dashboards` in this
      PC's pytweezer checkout, so a `git pull` updates the dashboards.
   3. Set a system environment variable `GRAFANA_DB_PASSWORD` to the `grafana`
      role's password.
   4. In `<Grafana>\conf\custom.ini`, set an admin password and let anyone on
      the lab network view dashboards without logging in, so the GUI's
      **Open in Grafana** buttons go straight to the run:

      ```ini
      [security]
      admin_password = <admin password>

      [auth.anonymous]
      enabled = true
      org_role = Viewer
      ```

      Anonymous viewing is only acceptable because Grafana is reachable from
      the lab subnet alone. Don't expose port 3000 any wider.
   5. Open the port to the lab subnet, then restart the Grafana service:

      ```powershell
      New-NetFirewallRule -DisplayName "Grafana (lab)" -Direction Inbound `
        -Protocol TCP -LocalPort 3000 -RemoteAddress 10.59.3.0/24 -Action Allow
      ```

   The GUI expects Grafana at `http://<server PC>:3000`. If it lives
   elsewhere, set `PYTWEEZER_GRAFANA_URL` on each lab PC.

   Three dashboards, cross-linked at the top of each:
   - **pytweezer overview**: readings by measurement and field, filtered by
     the `system` tag (Rb/CaF). Each run is drawn as a shaded region. Below
     are a per-point result over time and a table of runs whose rids link to
     the run dashboard.
   - **pytweezer run**: one run, chosen by rid. Shows its arguments and scan,
     each numeric result against the scanned argument and against time, the
     logger readings during the run, and its points. **Open in Grafana** in
     the Experiments queue and the Results tab opens this dashboard on the
     selected run.
   - **pytweezer lab health**: the latest reading of every field against its
     limits, each logger's state (ok, stale or stopped), the running runs
     and the firing alerts.

   Two alert rules, in the `pytweezer` folder, evaluate every minute:
   - **Reading out of range**: a field's latest reading (within 10 min) has
     been outside its `limits` for 1 min.
   - **Logger stale**: a running logger has written nothing for 10 intervals
     (at least 5 min).

   No contact point is set up, so alerts only show on Grafana's Alerting page
   and the lab health dashboard. To be notified, add a contact point (email,
   Teams or Slack) in Grafana and route the `source=pytweezer` label to it.

**Without TimescaleDB** the code still works: `readings` stays a plain,
uncompressed table, and each process logs one warning when it connects.
**When simulating**, the default connection string points at `127.0.0.1`.
Without a local Postgres, writes are dropped after one warning.

### A local stack for simulation (Linux)

`pytweezer-server` on any PC but PH-BEAST simulates. Every process it starts
then expects Postgres on `127.0.0.1:5432` (user and password `pytweezer`) and
Grafana on `127.0.0.1:3000`. To get the database, dashboards and **Open in
Grafana** working while simulating, run both locally on those ports. None of
this needs root; everything lives in one directory, e.g.
`~/.local/share/pytweezer-dev`:

1. **Postgres + TimescaleDB**: install `postgresql=17` and `cmake` from
   conda-forge into that directory with
   [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
   conda-forge has no TimescaleDB, so build the latest release from source
   against that Postgres:
   `./bootstrap -DREGRESS_CHECKS=OFF -DTAP_CHECKS=OFF -DPG_CONFIG=<env>/bin/pg_config`,
   then `make install` in `build/`.
2. **Cluster**: `initdb` with `--auth=scram-sha-256`. In `postgresql.conf` set
   `listen_addresses = 'localhost'` and `shared_preload_libraries = 'timescaledb'`.
   Then create the `pytweezer` role (password `pytweezer`), the database and the
   `grafana` role as in step 3 above.
3. **Grafana**: unpack the Linux tarball. Give it a `grafana.ini` with
   `http_addr = 127.0.0.1` and anonymous Viewer access, and provision it as in
   step 6, with the dashboards `path` pointing at your checkout.
4. **Services**: run `postgres -D <data dir>` and `grafana server
   --homepath=<grafana> --config=<grafana.ini>` as systemd user services, so
   they start at login. Pass `GRAFANA_DB_PASSWORD` to Grafana through an
   `EnvironmentFile`.

Simulated runs then go to this local database, and their files to
`data/simulated/`, so they never mix with real ones.

## 2. Connection config

`DATABASE["dsn"]` in `pytweezer/configuration/config.py` is the only
connection setting. It defaults to
`postgresql://pytweezer:pytweezer@<SERVER_HOST>:5432/pytweezer`, and
`PYTWEEZER_DB_DSN` overrides it.

## 3. Writing readings

From a notebook:

```python
from pytweezer.database import log

log("laser", power=1.23, wavelength=780)  # fields as kwargs
log("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})  # dict + tags
```

From a driver or any longer-lived object, hold your own writer:

```python
from pytweezer.database import DBWriter

writer = DBWriter()
writer.write("chamber", {"pressure": 2.1e-9}, tags={"system": "CaF"})
writer.close()  # or use it as a context manager
```

Each numeric field becomes one row: `(time, measurement, field, value, tags)`.
Bools are stored as 0/1, and non-numeric fields are dropped. Tags are stored as
JSON, with their values converted to strings.

Writes **never raise and never block**. They are queued for a background
thread, which sends them in batches about every 0.5 s. While the database is
unreachable, rows are kept and retried; past 10⁶ rows the oldest are dropped,
and a warning is logged. `close()` sends whatever is still queued, waiting up to
5 s.

To log a value a device driver already has (e.g. a camera temperature), give
the driver its own `DBWriter` and write from inside the driver. Don't write a
Logger for it.

## 4. Writing a Logger

A **Logger owns its own data source**, such as an NI DAQ, a serial sensor or a
socket feed, and opens it itself. It subclasses `pytweezer.loggers.base.Logger`
and overrides `setup()` and `read()`. The base `run()` loop polls `read()` every
`interval`, stamps each cycle with one timestamp and writes it. The worked
example is `NIADCLogger` (`pytweezer/loggers/ni_adc_logger.py`).

The `add-logger` skill (`.claude/skills/add-logger/`) walks through the class,
the `LOGGER_REGISTRY` factory in `pytweezer/servers/logger_server.py` and the
`CONFIG["Loggers"]` entry. Start a logger from the GUI's **Loggers** tab, or
standalone with `poetry run pytweezer-logger "NI ADC Logger"`.

An entry may set per-field `limits`, either bound `None`:

```python
"limits": {"ai0": [0.0, 5.0], "ai1": [None, 3.0]},
```

The first time a logger writes a measurement, it upserts a row into the
`measurements` table, holding the logger's name, its `interval`, these
`limits` and `active = true`. A clean stop sets `active = false`. Grafana's
alert rules and the lab health dashboard read this table, so a crashed or hung
logger shows as stale, but one stopped from the GUI doesn't.

## 5. Experiment runs and points

The Experiment Manager writes rows into `runs` and `points`:

- **`runs`**: a row when a task starts, updated when it finishes. The final
  row's `arguments` are the effective values read back from the h5 file,
  defaults included.
- **`points`**: a row per scan point. Each row holds that point's `t_start`,
  `t_end`, the scanned argument values (`scan_values`) and its 0-d numeric
  results (`scalars`).

Runs from before the database existed, or written while it was down, can be
loaded from the files at any time. Every row is an upsert, so it is safe to
rerun:

```bash
poetry run pytweezer-db-backfill            # the configured data root
poetry run pytweezer-db-backfill --root D:/data
```

## 6. Reading it back

In a notebook:

```python
from pytweezer.database.analysis import readings, points_with_readings

readings("ni_adc", ["ai0"], start="2026-10-07 09:00")  # time index, column per field

table = points_with_readings(1234, "ni_adc", ["ai0"])  # rid, h5 path or Measurement
table.plot.scatter("ni_adc/ai0", "atom_number")
```

`points_with_readings` gives one row per point: its scanned values, scalar
results and the readings during that point (`agg="mean"` by default). Points
are often shorter than a logger's interval and so contain no reading; those
points take the last reading before they ended.

Directly in SQL:

```sql
-- mean chamber temperature and 780 power during each run of one experiment
SELECT r.rid, r.arguments->>'detuning' AS detuning,
       avg(m.value) FILTER (WHERE m.field = 'chamber_temp') AS temp,
       avg(m.value) FILTER (WHERE m.field = 'p780')         AS p780
FROM runs r
JOIN readings m ON m.time BETWEEN r.t_start AND r.t_end
WHERE r.class_name = 'MOTLoadScan'
GROUP BY r.rid, detuning;

-- runs during which the 780 lock error went above 0.1
SELECT DISTINCT r.rid, r.class_name, r.t_start
FROM runs r
JOIN readings m ON m.time BETWEEN r.t_start AND r.t_end
WHERE m.measurement = 'lock' AND m.field = 'err780' AND abs(m.value) > 0.1;

-- each point's result next to the latest temperature reading before it
SELECT p.rid, p.point_index, (p.scalars->>'atom_number')::float AS atoms, t.value AS temp
FROM points p
CROSS JOIN LATERAL (
    SELECT value FROM readings
    WHERE measurement = 'chamber' AND field = 'temp' AND time <= p.t_end
    ORDER BY time DESC LIMIT 1
) t
WHERE p.t_start > now() - interval '30 days';
```

## 7. Storage

`readings` is a TimescaleDB hypertable in 7-day chunks. Chunks older than
7 days are compressed (segmented by measurement and field), which brings
regular float series down to a few bytes per row. Nothing is deleted. To expire
old readings, add a retention policy:

```sql
SELECT add_retention_policy('readings', INTERVAL '2 years');
```

If dashboards spanning months get slow, a continuous aggregate (for example
1-minute means) is the next step. It isn't set up yet.

## 8. Future ideas

- **Poll an existing device's RPC methods.** A generic logger would connect with
  `get_device(conf["device"])` (`pytweezer/servers/device_client.py`) and read a
  configured list of RPC attributes or methods each interval. That would log
  from any device without touching its driver. It was deferred in favour of the
  owns-its-device model above. If wanted, add it as a `Logger` subclass plus a
  `LOGGER_REGISTRY` entry.
