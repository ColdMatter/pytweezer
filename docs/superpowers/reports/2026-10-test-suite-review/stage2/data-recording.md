# Stage 2: data-recording

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| data-recording/analysis#1 | partly | `tests/test_db_analysis.py::test_points_with_readings_joins_by_point_window`, `tests/test_db_integration.py::test_backfilled_run_joins_with_readings` |
| data-recording/analysis#2 | partly | `tests/test_db_analysis.py::test_each_window_is_aggregated_and_empty_windows_take_the_last_reading` |
| data-recording/analysis#3 | covered | `tests/test_db_analysis.py::test_each_window_is_aggregated_and_empty_windows_take_the_last_reading` |
| data-recording/analysis#4 | covered | `tests/test_db_analysis.py::test_each_window_is_aggregated_and_empty_windows_take_the_last_reading` |
| data-recording/analysis#5 | covered | `tests/test_db_integration.py::test_readings_round_trip` |
| data-recording/analysis#6 | not | — |
| data-recording/backfill#1 | partly | `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` |
| data-recording/backfill#2 | covered | `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` |
| data-recording/backfill#3 | partly | `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` |
| data-recording/backfill#4 | partly | `tests/test_db_integration.py::test_backfilled_run_joins_with_readings` |
| data-recording/backfill#5 | not | — |
| data-recording/backfill#6 | covered | `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` |
| data-recording/schema#1 | partly | `tests/test_db_integration.py::test_readings_round_trip`, `tests/test_db_integration.py::test_upserts_keep_one_row_per_key`, `tests/test_db_integration.py::test_backfilled_run_joins_with_readings` |
| data-recording/schema#2 | not | — |
| data-recording/writer#1 | covered | `tests/test_db_writer.py::test_write_makes_a_row_per_numeric_field_sharing_one_timestamp` |
| data-recording/writer#2 | partly | `tests/test_db_writer.py::test_write_makes_a_row_per_numeric_field_sharing_one_timestamp` |
| data-recording/writer#3 | partly | `tests/test_db_writer.py::test_rows_are_kept_through_an_outage_and_written_once_it_ends` |
| data-recording/writer#4 | not | — |
| data-recording/writer#5 | covered | `tests/test_db_writer.py::test_backlog_beyond_max_pending_drops_the_oldest` |
| data-recording/writer#6 | partly | `tests/test_db_writer.py::test_records_convert_times_and_json` |
| data-recording/writer#7 | not | — |
| data-recording/writer#8 | partly | `tests/test_db_integration.py::test_upserts_keep_one_row_per_key` |
| data-recording/writer#9 | not | — |
| data-recording/storage#1 | partly | `tests/test_experiment_storage.py::test_strings_dicts_and_constants`, `tests/test_experiment_storage.py::test_missing_values_are_flagged_and_float_gaps_are_nan`, `tests/test_experiment_storage.py::test_a_result_keeps_its_first_shape_and_dtype`, `tests/test_experiment_simulation.py::test_run_local_simulated_experiment` |
| data-recording/storage#2 | partly | `tests/test_experiment_storage.py::test_partial_file_is_readable_mid_run`, `tests/test_experiment_storage.py::test_header_and_planned_points_and_no_overwrite` |
| data-recording/storage#3 | partly | `tests/test_experiment_storage.py::test_partial_file_is_readable_mid_run` |
| data-recording/storage#4 | covered | `tests/test_experiment_storage.py::test_missing_values_are_flagged_and_float_gaps_are_nan` |
| data-recording/storage#5 | partly | `tests/test_experiment_storage.py::test_a_result_keeps_its_first_shape_and_dtype` |
| data-recording/storage#6 | covered | `tests/test_experiment_storage.py::test_a_result_keeps_its_first_shape_and_dtype` |
| data-recording/storage#7 | covered | `tests/test_experiment_storage.py::test_strings_dicts_and_constants` |
| data-recording/storage#8 | partly | `tests/test_experiment_storage.py::test_header_and_planned_points_and_no_overwrite` |
| data-recording/storage#9 | partly | `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` |
| data-recording/base#1 | covered | `tests/test_loggers.py::test_write_points_accepts_two_and_three_element_points_at_one_time` |
| data-recording/base#2 | partly | `tests/test_loggers.py::test_close_marks_measurements_inactive_and_closes_the_writer` |
| data-recording/base#3 | covered | `tests/test_loggers.py::test_each_measurement_is_registered_once_with_its_limits` |
| data-recording/base#4 | covered | `tests/test_loggers.py::test_close_marks_measurements_inactive_and_closes_the_writer` |
| data-recording/base#5 | partly | `tests/test_loggers.py::test_each_measurement_is_registered_once_with_its_limits`, `tests/test_loggers.py::test_malformed_limits_fail_at_startup` |
| data-recording/ni_adc_logger#1 | not | — |
| data-recording/ni_adc_logger#2 | not | — |
| data-recording/ni_adc_logger#3 | partly | `tests/test_loggers.py::test_ni_adc_without_channels_reads_nothing_rather_than_failing` |
| data-recording/ni_adc_logger#4 | covered | `tests/test_loggers.py::test_ni_adc_reads_one_storable_point_per_cycle` |
| data-recording/ni_adc_logger#5 | not | — |

Notes on the partial entries:
- analysis#1: both tests attach one reading from before the run, so every point gets the same value through the fallback. Neither checks that readings land in the right point's window. With `PYTWEEZER_TEST_DSN` set, both still passed when the in-window aggregation branch of `aggregate_by_point` was removed, and again when its window end was made exclusive.
- analysis#2: the test checks only `"mean"`, and no reading sits on a window edge. Making `t_end` exclusive went undetected.
- backfill#1/#3: `t_start`/`t_end`, the "first `n_done` only" trimming, `error`/`git_commit` as NULL and `points_total` are not checked.
- backfill#4: the test backfills once. Idempotence under a second backfill is not tested.
- schema#1: the `conn` fixture calls `ensure_schema` twice but truncates the tables straight after, so "rows left untouched" is not checked.
- writer#2: numpy scalars are not tried (np.bool_ is the suspected bug). writer#3: only one row goes through the outage, so order and "all of them" are not checked. writer#6: three different instants are checked one by one, never one instant written three ways. writer#8: no `points` rows.
- storage#1: bool and complex results are not tried, and the image's values are never read back. storage#2: no string scan column and no `repetition`. storage#3: the mid-run test does not check that `recorded` is trimmed, and with the trimming removed it still passed. storage#5: text versus numeric and str versus dict are not tried. storage#8: "existing file unchanged" is not checked. storage#9: only `arguments` gets an indirect check, through backfill. `read_arguments` and the header's None-as-`""` are not checked.
- base#2: `close()` is called directly. Stopping `run()` is never shown to close the writer. base#5: numeric-string bounds are not tried.
- The three `test_db_integration.py` tests skip unless `PYTWEEZER_TEST_DSN` is set, so a plain `pytest` run gives analysis#5 no coverage at all.

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_db_analysis.py::test_each_window_is_aggregated_and_empty_windows_take_the_last_reading` | rewrite | Guards three behaviours in one test (aggregation, fallback, NaN); the name needs "and". Split them, parametrise over `agg`, and put a reading on each window edge. |
| `tests/test_db_analysis.py::test_backfill_records_queued_runs_and_skips_local_runs_and_unreadable_files` | rewrite | Wanted behaviours (point rows, unrecorded scalars, `run_row`, skipping) guarded in one test whose name needs two "and"s; split per behaviour. |
| `tests/test_db_analysis.py::test_points_with_readings_joins_by_point_window` | rewrite | Wanted (priority 2), but it patches the private `_long_readings` and uses one shared reading, so it cannot fail if readings go to the wrong point. Mutations confirmed that. Use a distinct reading per window. |
| `tests/test_db_analysis.py::test_unknown_rid_is_a_lookup_error` | delete | A loud failure (an exception), checked through a MagicMock connection; loud failures rank last. |
| `tests/test_db_integration.py::test_readings_round_trip` | keep | One integration test for the writer -> Postgres readings path, normal case, also catching tag filtering. It failed when the tag filter was removed. |
| `tests/test_db_integration.py::test_upserts_keep_one_row_per_key` | rewrite | Two upserts (runs and measurements) in one test. Make it the single run-lifecycle integration test, with start and finish plus points rows. |
| `tests/test_db_integration.py::test_backfilled_run_joins_with_readings` | rewrite | The right cross-process path, but the single shared reading means it passes with per-point windowing broken (mutation confirmed). Give each point its own reading. |
| `tests/test_db_writer.py::test_schema_reports_timescale_and_keeps_the_connection` | rewrite | Wanted (no TimescaleDB preload must not stop writes), but it checks SQL strings recorded by a fake. Assert instead that a writer on a non-preloaded fake still delivers its rows. |
| `tests/test_db_writer.py::test_write_makes_a_row_per_numeric_field_sharing_one_timestamp` | keep | Priority 2 (writer#1). It checks what reaches the database edge. It failed when tags were not stringified. |
| `tests/test_db_writer.py::test_records_convert_times_and_json` | rewrite | Three methods and two conversions in one test, with positional `run[10]`/`run[15]` indices that break if the columns are reordered. Parametrise one instant written three ways. |
| `tests/test_db_writer.py::test_write_does_not_wait_for_an_unreachable_database` | rewrite | "Writes never block" is wanted, but this uses a real socket and a wall-clock threshold and takes over 0.5 s. Use a fake `connect` that blocks on an Event. |
| `tests/test_db_writer.py::test_rows_are_kept_through_an_outage_and_written_once_it_ends` | keep | Priority 2 (writer#3), with a stand-in only at the database edge. It failed when the re-queue was removed. More than one row would also check the order. |
| `tests/test_db_writer.py::test_backlog_beyond_max_pending_drops_the_oldest` | rewrite | Wanted (writer#5), but it reads the private `_pending`/`_cond`. Bring the fake database up and check which rows arrive. |
| `tests/test_db_writer.py::test_log_messages_hide_the_password` | keep | A quiet failure (password in log files), checked through the public property. It failed when the stripping was removed. |
| `tests/test_db_writer.py::test_a_failed_schema_setup_is_retried_on_a_fresh_connection` | rewrite | Rows surviving a schema failure are worth guarding. A schema error escaping `_send` would kill the thread quietly. Asserting `.closed` on fake connections checks the fake, though; assert only that the row arrives. |
| `tests/test_experiment_storage.py::test_header_and_planned_points_and_no_overwrite` | rewrite | Header, planned points, source and no-overwrite in one test, read through raw h5py, not `load_measurement`/`read_header` as callers do. |
| `tests/test_experiment_storage.py::test_partial_file_is_readable_mid_run` | keep | Priority 2: a file mid-run reads back only its finished points, through the public readers. It failed when results were not trimmed to `n_done`. It should also assert `recorded`. |
| `tests/test_experiment_storage.py::test_missing_values_are_flagged_and_float_gaps_are_nan` | keep | Priority 2 (storage#4): one behaviour, independent check. It failed when the float fill became 0. |
| `tests/test_experiment_storage.py::test_a_result_keeps_its_first_shape_and_dtype` | rewrite | Refusing a recast (storage#5) and refusing a double record (storage#6) in one test. Split them and parametrise the refusals, adding text versus numeric and str versus dict. |
| `tests/test_experiment_storage.py::test_strings_dicts_and_constants` | rewrite | Round trip of str/dict, constants, and refusing to overwrite a constant: three behaviours. Fold the round trip into a parametrised storage#1 test. |
| `tests/test_experiment_storage.py::test_bad_records_are_refused` | delete | Bad names and unsupported types raise straight away; without the check h5py or the loader still fails loudly. Loud failures rank last. |
| `tests/test_experiment_storage.py::test_paths_and_rid_scan` | rewrite | `highest_rid` is worth guarding (a reused rid quietly overwrites a `runs` row), but it is mixed with `measurement_relpath` in one test. Split it, and add `find_measurement` as the green trial did. |
| `tests/test_experiment_storage.py::test_load_only_selected_results_but_know_all_shapes` | keep | One behaviour backfill relies on (shapes and kinds of unloaded results). It only covers float kinds: setting every kind to `"f"` went undetected. A text result would close that gap. |
| `tests/test_loggers.py::test_ni_adc_reads_one_storable_point_per_cycle` | keep | ni_adc_logger#4: the simulated logger gives storable fields. It failed when the simulated values became strings. It leans on the private `_coerce_fields`, which is a weakness. |
| `tests/test_loggers.py::test_ni_adc_without_channels_reads_nothing_rather_than_failing` | delete | A start-up edge case that is loud or harmless (stage 1 marks it loud). It does not check the quiet half (no DAQ task opened). |
| `tests/test_loggers.py::test_write_points_accepts_two_and_three_element_points_at_one_time` | rewrite | Wanted (base#1), but it drives the private `_write_points`, which only `run()` calls. Drive it through `read()`/`run()` and parametrise over the point shape. |
| `tests/test_loggers.py::test_each_measurement_is_registered_once_with_its_limits` | rewrite | Wanted (base#3), but it goes through the private `_write_points`. Drive it through `run()`. |
| `tests/test_loggers.py::test_close_marks_measurements_inactive_and_closes_the_writer` | rewrite | Two behaviours ("and"). "Closes the writer" is just `RecordingDB.closed`, a check on a fake. Split it, and show that stopping `run()` writes the queued rows. |
| `tests/test_loggers.py::test_malformed_limits_fail_at_startup` | keep | Guards against malformed limits being silently ignored, so alerts would never fire. It failed when bad entries were skipped. |
| `tests/test_loggers.py::test_read_failures_do_not_stop_the_run_loop` | keep | One behaviour through the public `run()`/`stop()`: a sensor glitch must not quietly end logging. It failed when exceptions were let through. |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_experiment_simulation.py::test_data_root_resolution` | Simulated runs are written under `simulated/`, apart from real data and its rid sequence, and a client PC finds the data on the share | yes: `data_root` lives in `storage.py`, and mixing simulated files into real data fails quietly. Stage 1 even built backfill#5 on that layout without listing the guarantee itself. |
| `tests/test_experiment_storage.py::test_paths_and_rid_scan` | `highest_rid` finds the largest rid in the `YYYY/MM/DD` layout, so the manager never reuses one | yes: a reused rid quietly overwrites a `runs` row through the upsert. It is in `storage.py` and has no stage-1 entry. |
| `tests/test_experiment_storage.py::test_load_only_selected_results_but_know_all_shapes` | `result_shapes`/`result_kinds` describe results that were not loaded | yes, at priority 4: `scalar_result_names` (backfill and `points_with_readings`) decides from these which results reach the database. |
| `tests/test_experiment_storage.py::test_partial_file_is_readable_mid_run` | `read_planned_points` returns every planned point while the run is going | no: storage#3 covers trimming to the finished points, and the planned points feed the GUI more than recorded data. |
| `tests/test_loggers.py::test_read_failures_do_not_stop_the_run_loop` | One failing `read()` does not end a logger's run loop | yes, at priority 4: a transient sensor error would quietly leave a gap in the record until someone noticed. |
| `tests/test_db_writer.py::test_a_failed_schema_setup_is_retried_on_a_fresh_connection` | Rows survive a failure in `ensure_schema` on a fresh connection | partly: writer#3 covers outages in general, but a schema error is a separate path through `_send`. |
| `tests/test_db_writer.py::test_schema_reports_timescale_and_keeps_the_connection` | A server without TimescaleDB preloaded still accepts writes | yes, at priority 4: schema#1/#2 cover upgrades but not this path, where every write would fail quietly behind one warning. |
| `tests/test_db_writer.py::test_log_messages_hide_the_password` | The DSN in log messages has its password removed | no: stage 1 dealt with it under Suspected bugs (key-value DSNs). It is security hygiene, not data recording. |

Merged into the stage-1 lists on 2026-10-21: every row above marked "yes" (and the "partly" schema-retry row) is now a stage-1 entry that names this test as its cover.

## Verdict
**rework**: 10 kept as is / 30 decided (17 rewrite, 3 delete). That is right on the one-third line, so this is not a start-again. The kept tests already cover the core of h5 storage (gaps, mid-run trimming), the writer's row shape and outage retention, and the logger loop, and each failed under a deliberate fault in a scratch copy (run with `PYTHONPATH` pointing at the copy). Most rewrites are mechanical: split tests that guard several behaviours, parametrise the cases, and stop going through private `_write_points`/`_pending`/`_long_readings`. Two rewrites matter more than the rest. The `points_with_readings` tests, unit and integration, give every point one shared reading, so they cannot catch readings attached to the wrong point; both still passed with per-point windowing broken. That is the data-recording behaviour analysts rely on most. Nine stage-1 entries have no coverage at all. The ones that matter most are writer#4 (close flushes), writer#7 (numpy in JSON), writer#9 (a NaN poisons the queue), backfill#5 (simulated rids collide with real ones), and the real-hardware `nidaqmx` channel mapping (ni_adc_logger#1/#2). Three of these also come with suspected bugs, so their tests should land with the fixes. The integration file only runs with `PYTWEEZER_TEST_DSN` set, so CI should set it, or the one writer -> Postgres test per path is never exercised.
