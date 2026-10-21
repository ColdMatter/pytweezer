# Stage 1: Data recording

## pytweezer/database/analysis.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/analysis#1 | `points_with_readings` on a measurement file plus readings in the test database returns one row per point that ran, indexed by `point_index`, with its scanned values, scalar results, UTC `t_start`/`t_end` and a `"<measurement>/<field>"` column holding that point's readings. | 2 | integration | quiet: monitor values are attached to the wrong point, or columns go missing, and a correlation plot shows a false (or no) dependence | write a constant, distinct reading inside each point's window; each point's column must equal its own constant |
| data-recording/analysis#2 | `aggregate_by_point` reduces the readings inside each closed `[t_start, t_end]` window with the requested `agg` (parametrised over `"mean"`, `"median"`, `"max"` and a callable). | 3 | unit | quiet: every attached monitor value is subtly wrong (off-by-one at a window edge, wrong reduction) | readings with value equal to time at t = 0..9; window [2, 5] gives mean 3.5, median 3.5, max 5 (both edges included) |
| data-recording/analysis#3 | A window that contains no readings takes the last reading before it, not a neighbour after it. | 3 | unit | quiet: short points get a reading from the future or from the wrong point | step function: readings 1.0 at t=0 and 2.0 at t=10; window [4, 5] must give 1.0 |
| data-recording/analysis#4 | A window that ends before the first reading gives NaN rather than a borrowed value. | 3 | unit | quiet: a point before monitoring started shows a fabricated reading | readings start at t=10; window [0, 1] must give NaN |
| data-recording/analysis#5 | `readings(..., tags=...)` returns only readings carrying all the given tags, so two systems logging the same measurement name are kept apart. | 4 | integration | quiet: Rb and CaF readings are mixed into one column | — |
| data-recording/analysis#6 | Naive `start`/`end` strings are taken as this PC's local time, so `start="09:00"` during BST excludes readings from 08:00–09:00 local. | 4 | unit | quiet: query window is shifted by the UTC offset and silently returns the wrong hour | an aware `Europe/London` time and its naive equivalent select the same rows |

## pytweezer/database/backfill.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/backfill#1 | `point_rows` gives one row per point that ran (the first `n_done`), with its `index`, `t_start`, `t_end` and scanned values, and leaves the bookkeeping columns out of `scan_values`. | 2 | unit | quiet: database holds planned-but-unrun points, or points carry wrong scan values | — |
| data-recording/backfill#2 | A scalar result a point did not record is left out of that point's `scalars` rather than stored as NaN or 0. | 2 | unit | quiet: an unrecorded integer result appears in the database as a measured 0 | — |
| data-recording/backfill#3 | `run_row` maps the file header to the `runs` columns as the manager writes them: `h5_path` relative to the root in POSIX form, empty `error`/`git_commit` as NULL, `n_points` as `points_total`. | 2 | unit | quiet: `points_with_readings(rid)` can no longer find the file; provenance reads as an empty commit | — |
| data-recording/backfill#4 | Backfilling the same root twice into the test database leaves exactly one `runs` row and one `points` row per point that ran, matching the file. | 2 | integration | quiet: duplicate or missing rows in the run history | — |
| data-recording/backfill#5 | Files under the `simulated/` subdirectory are not loaded into the real runs, since their rids are a separate sequence. | 2 | unit | quiet: simulated runs overwrite real runs rows that share their rid (see Suspected bugs) | — |
| data-recording/backfill#6 | `backfill` skips `run_local` files (negative rid) and unreadable h5 files, and counts only the runs it recorded. | 4 | unit | quiet: junk rows with rid -1, or one bad file aborting the whole backfill | — |

## pytweezer/database/__init__.py

No entries: it only re-exports `DBWriter`, `get_default_writer` and `log`, which are covered under `writer.py`.

## pytweezer/database/schema.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/schema#1 | Running `ensure_schema` on an already initialised database succeeds and leaves existing rows untouched (every new writer connection does this). | 4 | integration | quiet: every writer logs one warning and then holds and eventually drops its rows | — |
| data-recording/schema#2 | An older `runs` table without `git_commit`/`git_dirty` gains those columns, so runs rows carrying provenance can still be written. | 4 | integration | quiet: every runs upsert fails and takes its batch with it | — |
| data-recording/schema#3 | On a server without TimescaleDB preloaded, `ensure_schema` reports it, keeps the connection open, and writes still succeed. *Added from stage 2; covered by `tests/test_db_writer.py::test_schema_reports_timescale_and_keeps_the_connection`.* | 4 | unit | quiet: every write fails behind one warning | — |

## pytweezer/database/writer.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/writer#1 | `write()` produces one `readings` row per numeric field, all with the call's measurement, one shared timestamp and the tags as string-valued JSON, and the value as a float. | 2 | unit | quiet: readings stored under the wrong field, time or tags | — |
| data-recording/writer#2 | Non-numeric fields (str, None, arrays) are dropped while every numeric one in the same call is kept, including Python bools and numpy scalars (parametrised over `np.float32`, `np.int64`, `np.bool_`). | 2 | unit | quiet: a field silently never reaches the database | — |
| data-recording/writer#3 | Rows queued while the database is unreachable are all written, in order, once it becomes reachable again. | 2 | unit | quiet: readings from a database outage are lost | — |
| data-recording/writer#4 | `close()` writes every row already queued before it returns. | 2 | unit | quiet: the last readings of a logger, or the finish row of a run, are lost at process exit | — |
| data-recording/writer#5 | Beyond `max_pending` queued rows the oldest are dropped and the newest kept. | 2 | unit | quiet: after a long outage the recent data is lost instead of the old | — |
| data-recording/writer#6 | `record_run`/`record_point` times given as a datetime, an epoch float or an ISO string with offset all store the same instant (parametrised). | 2 | unit | quiet: runs and points shifted by the UTC offset, so readings line up with the wrong points | one instant expressed three ways must give equal timestamps |
| data-recording/writer#7 | JSON columns holding numpy scalars and arrays (as read back from h5 files) are stored as plain JSON numbers and lists. | 2 | unit | quiet: values stored as their `str()` repr and unusable in SQL | — |
| data-recording/writer#8 | A run recorded at start and again at finish, with its points, reads back from the test database as one `runs` row with the finish values and one `points` row per point (normal case of writer -> Postgres). | 2 | integration | quiet: run history shows runs stuck as "running", or duplicates | — |
| data-recording/writer#9 | A point whose scalar result is NaN does not stop rows queued after it from being written. | 2 | integration | quiet: one failed fit silently stops all later database writes from that process (see Suspected bugs) | — |
| data-recording/writer#10 | A failure in `ensure_schema` closes that connection, and the queued rows are written later on a fresh one. *Added from stage 2; covered by `tests/test_db_writer.py::test_a_failed_schema_setup_is_retried_on_a_fresh_connection`.* | 4 | unit | quiet: rows are lost, or the writer stays on a half-set-up connection | — |
| data-recording/writer#11 | A row the database rejects (text containing NUL, say) is dropped with a warning, and the rows queued around it are written. *Added with the fix in PR #10; covered by `tests/test_db_integration.py::test_a_row_the_database_rejects_is_dropped_and_the_rest_are_written`.* | 2 | integration | quiet: one bad value stops all database logging until a restart | — |

## pytweezer/experiment/storage.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/storage#1 | Every recordable kind of value recorded in a point reads back identical, with its dtype and shape, through `load_measurement` (parametrised over float, int, bool, complex, a `uint16` image, str, dict). | 2 | unit | quiet: results truncated, recast or decoded wrongly | — |
| data-recording/storage#2 | The scanned value of each point (numeric and string columns), with its `index` and `repetition`, reads back from `/points` in execution order. | 2 | unit | quiet: results paired with the wrong scan value | — |
| data-recording/storage#3 | A run stopped part-way loads only its first `n_done` points, across points, results and recorded flags alike. | 2 | unit | quiet: unrun points appear as fill values (NaN, 0) in the analysis | — |
| data-recording/storage#4 | A point that does not record a float result reads back NaN, with `recorded` False for that point. | 2 | unit | quiet: a missing measurement looks like a real number | — |
| data-recording/storage#5 | A later value that cannot be stored without change under a name raises `RecordError` rather than being cast or reinterpreted (parametrised: float into an int result, a different shape, text into a numeric result, a str into a dict result and the reverse). | 2 | unit | quiet: values truncated or reinterpreted, or the file becomes unloadable (see Suspected bugs) | — |
| data-recording/storage#6 | Recording the same name twice within one point raises instead of overwriting the first value. | 2 | unit | quiet: the first measurement of the point is lost | — |
| data-recording/storage#7 | A `record()` outside a point is stored under `/constants` and reads back identical, not as a result row. | 2 | unit | quiet: a background image or calibration lands in a result row or is lost | — |
| data-recording/storage#8 | A `MeasurementWriter` at a path that already exists raises and leaves the existing file unchanged. | 2 | unit | quiet: an earlier measurement is overwritten | — |
| data-recording/storage#9 | The effective argument values and the header (with None stored as `""`) read back as plain Python values, through both `load_measurement` and `read_arguments` (which leaves out `__schema__`). | 2 | unit | quiet: "Resubmit" or the analysis uses arguments that differ from what ran | — |
| data-recording/storage#10 | `data_root` puts a simulating manager's files under `<root>/simulated/`, apart from real data and its rid sequence, and on a PC other than the manager's uses `client_data_root` when it is set. *Added from stage 2; covered by `tests/test_experiment_simulation.py::test_data_root_resolution`.* | 2 | unit | quiet: simulated files mix with real data and use up its rids, or a client PC reads the wrong folder | — |
| data-recording/storage#11 | `highest_rid` returns the largest rid among files in the `YYYY/MM/DD` layout (0 for an empty root) and ignores files without a numeric prefix, so the manager never reuses a rid. *Added from stage 2; covered by `tests/test_experiment_storage.py::test_paths_and_rid_scan`.* | 2 | unit | quiet: a reused rid overwrites the `runs` row through the upsert | — |
| data-recording/storage#12 | `Measurement.result_shapes` and `result_kinds` describe every result, including those not loaded with `results=`. *Added from stage 2; covered by `tests/test_experiment_storage.py::test_load_only_selected_results_but_know_all_shapes`.* | 4 | unit | quiet: `scalar_result_names` sends the wrong results to the database | — |

## pytweezer/loggers/base.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/base#1 | Every point returned by one `read()` cycle is written with a single shared timestamp, in either the `(measurement, fields)` or `(measurement, fields, tags)` form (parametrised over the shape). | 2 | unit | quiet: fields from one reading no longer line up in time, or tags are lost | — |
| data-recording/base#2 | Stopping `run()` closes the writer, so readings queued before the stop are written. | 2 | unit | quiet: the last readings before a stop are lost | — |
| data-recording/base#3 | The first write of each measurement registers it once in `measurements` with the logger's name, interval and parsed limits, marked active. | 4 | unit | quiet: Grafana's out-of-range and stale alerts never fire for that measurement | — |
| data-recording/base#4 | A clean stop marks every registered measurement inactive. | 4 | unit | quiet: stopping a logger from the GUI raises a false "Logger stale" alert | — |
| data-recording/base#5 | `limits` bounds are normalised to float or None (numeric strings accepted), and a malformed entry raises `ValueError` naming the field (parametrised). | 4 | unit | quiet: alert thresholds differ from the configured ones | — |
| data-recording/base#6 | A `read()` that raises is logged, and the run loop carries on at the next interval. *Added from stage 2; covered by `tests/test_loggers.py::test_read_failures_do_not_stop_the_run_loop`.* | 4 | unit | quiet: one transient sensor error leaves a gap in the record until someone notices | — |

## pytweezer/loggers/__init__.py

No entries: it only re-exports `Logger`, which is covered under `base.py`.

## pytweezer/loggers/ni_adc_logger.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| data-recording/ni_adc_logger#1 | With several channels, each voltage returned by the (faked) `nidaqmx` task is stored under its own short channel name, in channel order. | 2 | unit | quiet: voltages stored under the wrong channel name | — |
| data-recording/ni_adc_logger#2 | With a single channel, the bare float `nidaqmx` returns is stored as that channel's field. | 2 | unit | quiet: a one-channel logger stores nothing | — |
| data-recording/ni_adc_logger#3 | With no channels configured, `read()` returns None and no DAQ task is opened. | 4 | unit | loud: a crash on start or a reserved DAQ | — |
| data-recording/ni_adc_logger#4 | In simulation every configured channel gives a storable numeric field. | 4 | unit | quiet: the simulated logger looks alive and stores nothing | — |
| data-recording/ni_adc_logger#5 | `close()` releases the DAQ task and still closes the writer when `task.close()` raises. | 4 | unit | quiet: final readings lost; loud on the next start (DAQ resource reserved) | — |
| data-recording/ni_adc_logger#6 | Channels on different devices that share a short name (`Dev1/ai0`, `Dev2/ai0`) are stored under their full names; the others keep their short names. *Added with the fix in PR #10; covered by `tests/test_loggers.py::test_ni_adc_channels_sharing_a_short_name_keep_their_full_names`.* | 2 | unit | quiet: one device's voltage is lost | — |

## Suspected bugs

- pytweezer/database/writer.py: `_json` uses `json.dumps`, which writes NaN and infinity as the bare tokens `NaN`/`Infinity`. Postgres `jsonb` does not accept them, so the whole batch's transaction fails. `_run` then puts the batch back at the head of the queue and every retry fails the same way. From then on the writer stores nothing (later runs, points and readings included) until `max_pending` or `close()` drops the rows. This is inferred from the code and was not run against Postgres. The same happens with any one row the database rejects, because nothing ever separates a poison row from the rest of its batch. Smallest input: `w = DBWriter(); w.record_point(1, 0, 0.0, 1.0, {}, {"width": float("nan")}); w.write("m", {"a": 1.0})`, and neither row is ever stored.
- pytweezer/database/writer.py: `_coerce_fields` silently drops numpy booleans, because `np.bool_` is neither a `bool` nor a `numbers.Real` (confirmed). Smallest input: `DBWriter().write("lock", {"locked": np.bool_(True)})` queues no rows.
- pytweezer/database/writer.py: `dsn_host` only removes the password from URL-style DSNs, so a key-value DSN is logged in full with its password. Smallest input: `DBWriter("host=db password=secret").dsn_host` returns `"host=db password=secret"`.
- pytweezer/experiment/storage.py: `_check_fits` accepts any text value into any text dataset and ignores the `encoding` attribute that sets how a stored string is decoded. Confirmed in memory: after a dict in point 0 and a str in point 1, `load_measurement` raises `JSONDecodeError`, so the whole file cannot be loaded. The reverse order quietly returns the dict as its JSON text. Smallest input: `record("x", {"a": 1})` in point 0, then `record("x", "hello")` in point 1, then `load_measurement(...)`.
- pytweezer/database/backfill.py: `root.rglob("*.h5")` also picks up `<data_root>/simulated/`. Simulated runs number their rids separately, so they share rids with real runs. Because `sorted()` puts `simulated/` after the `YYYY/` directories, the simulated file's upsert wins: it overwrites the real `runs` row and merges its points into the real run's points. The same collision may happen live if a simulating Experiment Manager can reach the database, but that is outside this area. Smallest input: a root holding `2026/10/21/000005_A.h5` and `simulated/2026/10/21/000005_B.h5`, then `backfill(root)`, leaves the rid 5 `runs` row as class B with `simulated=True`.
- pytweezer/loggers/ni_adc_logger.py: field names are only the last path segment, so two channels with the same name on different devices collapse into one field and the first device's voltage is lost without a warning. Smallest input: `channels=["Dev1/ai0", "Dev2/ai0"]` makes `read()` return `{"ai0": <Dev2 value>}`.
