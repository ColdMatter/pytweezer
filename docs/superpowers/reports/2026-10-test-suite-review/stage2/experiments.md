# Stage 2: experiments

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| experiments/arguments#1 | covered | `tests/test_experiment_arguments.py::test_coerce_arguments_fills_defaults_and_rejects_unknown_names`, `tests/test_motmaster_experiment.py::test_dotted_names_that_are_not_searchable_are_rejected_at_submit` |
| experiments/arguments#2 | covered | `tests/test_experiment_arguments.py::test_coerce_arguments_fills_defaults_and_rejects_unknown_names` |
| experiments/arguments#3 | partly | `tests/test_experiment_arguments.py::test_invalid_values_are_rejected`, `tests/test_experiment_arguments.py::test_values_are_coerced_to_the_declared_type` (no below-`min` case for `Number`; `== -1.0` does not check that a Python float comes back; no numpy floats) |
| experiments/arguments#4 | covered | `tests/test_experiment_arguments.py::test_values_are_coerced_to_the_declared_type`, `tests/test_experiment_arguments.py::test_invalid_values_are_rejected` |
| experiments/arguments#5 | partly | `tests/test_experiment_arguments.py::test_invalid_values_are_rejected`, `tests/test_experiment_arguments.py::test_values_are_coerced_to_the_declared_type` (`1` and `"yes"` only; no `2`, no numpy bool, so the stage-1 bug goes unseen) |
| experiments/arguments#6 | covered | `tests/test_experiment_arguments.py::test_invalid_values_are_rejected`, `tests/test_experiment_arguments.py::test_values_are_coerced_to_the_declared_type` |
| experiments/arguments#7 | partly | `tests/test_experiment_arguments.py::test_schema_is_read_from_the_class_without_instantiating` (base first and redeclared default; a plain value removing an argument is not tested) |
| experiments/arguments#8 | covered | `tests/test_experiment_arguments.py::test_schema_is_read_from_the_class_without_instantiating`, `tests/test_experiment_gui.py::test_form_shows_display_units_and_returns_si` |
| experiments/catalogue#1 | covered | `tests/test_experiment_e2e.py::test_catalogue_lists_runnable_experiments` |
| experiments/catalogue#2 | not | — |
| experiments/catalogue#3 | not | — |
| experiments/catalogue#4 | not | — (the manager tests use a static `BROKEN` entry and never run a failing import) |
| experiments/client#1 | partly | `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements`, `tests/test_experiment_manager.py::test_notebook_recipes_round_trip` (tasks run with the given args, but the module/qualname from a class object and the shuffle seed are never checked) |
| experiments/client#2 | covered | `tests/test_experiment_manager.py::test_notebook_calls_check_arguments_before_sending` |
| experiments/client#3 | covered | `tests/test_experiment_manager.py::test_notebook_recipes_round_trip` |
| experiments/client#4 | not | — |
| experiments/client#5 | partly | `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements` (returns a finished task; the `TimeoutError` is not tested) |
| experiments/dataset#1 | not | — |
| experiments/dataset#2 | not | — |
| experiments/dataset#3 | not | — |
| experiments/dataset#4 | not | — |
| experiments/dataset#5 | not | — |
| experiments/experiment#1 | partly | `tests/test_experiment_arguments.py::test_instance_attributes_hold_effective_values`, `tests/test_motmaster_experiment.py::test_declared_and_searched_parameters_are_sent_typed` (declared arguments and a dotted extra at construction; `set_argument` extras not) |
| experiments/experiment#2 | covered | `tests/test_experiment_simulation.py::test_run_local_simulated_experiment`, `tests/test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices` |
| experiments/experiment#3 | covered | `tests/test_experiment_arguments.py::test_device_is_resolved_lazily_once_and_closed` |
| experiments/experiment#4 | partly | `tests/test_experiment_arguments.py::test_device_is_resolved_lazily_once_and_closed`, `tests/test_experiment_runner.py::test_devices_closed_even_on_failure` (no close that raises) |
| experiments/introspect#1 | covered | `tests/test_experiment_e2e.py::test_catalogue_lists_runnable_experiments` (leaves out `NotRunnable` and the imported `Experiment`) |
| experiments/introspect#2 | not | — |
| experiments/introspect#3 | not | — |
| experiments/motmaster#1 | partly | `tests/test_motmaster_experiment.py::test_master_is_untriggered_and_followers_start_first` (trigger mode only: the fake ignores script, `run_until_stopped`, iterations and save toggle, and removing `set_iterations` from `prepare` left every test passing in a scratch copy; the lone-master case is not checked) |
| experiments/motmaster#2 | covered | `tests/test_motmaster_experiment.py::test_declared_and_searched_parameters_are_sent_typed`, `tests/test_motmaster_experiment.py::test_master_is_untriggered_and_followers_start_first` |
| experiments/motmaster#3 | covered | `tests/test_motmaster_experiment.py::test_declared_and_searched_parameters_are_sent_typed`, `tests/test_motmaster_experiment.py::test_bad_script_parameter_fails_before_any_sequence_runs` |
| experiments/motmaster#4 | covered | `tests/test_motmaster_experiment.py::test_bad_script_parameter_fails_before_any_sequence_runs` |
| experiments/motmaster#5 | covered | `tests/test_motmaster_experiment.py::test_bad_script_parameter_fails_before_any_sequence_runs` |
| experiments/motmaster#6 | covered | `tests/test_motmaster_experiment.py::test_dotted_names_that_are_not_searchable_are_rejected_at_submit` |
| experiments/motmaster#7 | partly | `tests/test_motmaster_experiment.py::test_master_is_untriggered_and_followers_start_first` (start order only; `follower_arm_delay` is set but never measured) |
| experiments/motmaster#8 | not | — (`test_master_or_follower_failure_fails_the_task` fails one sequencer at a time) |
| experiments/motmaster#9 | partly | `tests/test_motmaster_experiment.py::test_follower_that_never_finishes_fails_after_its_timeout` (the new client on the next call is not checked) |
| experiments/motmaster#10 | covered | `tests/test_motmaster_experiment.py::test_declaration_rules_are_checked_at_class_definition` |
| experiments/queue#1 | covered | `tests/test_experiment_queue.py::test_rids_are_sequential_and_order_is_priority_then_rid` |
| experiments/queue#2 | covered | `tests/test_experiment_queue.py::test_only_waiting_tasks_can_be_held_deleted_or_dropped` |
| experiments/queue#3 | partly | `tests/test_experiment_queue.py::test_due_time_holds_a_task_back_and_naive_times_are_local` (the naive-local half depends on the machine's time zone: reading a naive time as UTC passed under `TZ=UTC` and failed only under `TZ=Europe/London`) |
| experiments/queue#4 | partly | `tests/test_experiment_queue.py::test_only_one_task_runs_at_a_time` (running only; paused not) |
| experiments/queue#5 | partly | `tests/test_experiment_queue.py::test_pause_resume_terminate_reach_the_worker_as_actions`, `tests/test_experiment_e2e.py::test_pause_resume_and_terminate_at_a_point_boundary` (the action is checked; that the task stays active until the worker reports is not) |
| experiments/queue#6 | covered | `tests/test_experiment_queue.py::test_only_waiting_tasks_can_be_held_deleted_or_dropped` |
| experiments/queue#7 | covered | `tests/test_experiment_queue.py::test_pause_resume_terminate_reach_the_worker_as_actions` |
| experiments/queue#8 | covered | `tests/test_experiment_queue.py::test_pause_resume_terminate_reach_the_worker_as_actions` |
| experiments/queue#9 | covered | `tests/test_experiment_queue.py::test_store_round_trip_and_unreadable_file`, `tests/test_experiment_manager.py::test_state_survives_a_restart_and_rids_never_repeat` |
| experiments/queue#10 | covered | `tests/test_experiment_queue.py::test_store_round_trip_and_unreadable_file` |
| experiments/recipes#1 | covered | `tests/test_experiment_recipes.py::test_replay_applies_overrides_and_keeps_the_rest`, `tests/test_experiment_manager.py::test_notebook_recipes_round_trip` |
| experiments/recipes#2 | covered | `tests/test_experiment_recipes.py::test_replay_applies_overrides_and_keeps_the_rest` |
| experiments/recipes#3 | covered | `tests/test_experiment_recipes.py::test_unknown_arguments_lists_names_the_experiment_lacks`, `tests/test_experiment_manager.py::test_a_recipe_that_no_longer_fits_its_experiment_is_refused` |
| experiments/recipes#4 | covered | `tests/test_experiment_recipes.py::test_a_name_is_taken_until_overwritten` |
| experiments/recipes#5 | covered | `tests/test_experiment_recipes.py::test_a_name_is_taken_until_overwritten` |
| experiments/recipes#6 | covered | `tests/test_experiment_recipes.py::test_saving_clears_the_start_time_and_stamps_the_save` |
| experiments/runner#1 | covered | `tests/test_experiment_runner.py::test_hook_order_and_scanned_values` |
| experiments/runner#2 | covered | `tests/test_experiment_runner.py::test_control_stops_at_a_point_boundary` |
| experiments/runner#3 | covered | `tests/test_experiment_runner.py::test_failure_runs_finish_and_keeps_completed_points` |
| experiments/runner#4 | not | — |
| experiments/runner#5 | not | — (`tests/test_results_browser.py` writes files with `run_local(path=...)` but never compares them with the returned `Measurement`) |
| experiments/runner#6 | partly | `tests/test_experiment_runner.py::test_run_local_stores_effective_arguments_and_results`, `tests/test_motmaster_experiment.py::test_declared_and_searched_parameters_are_sent_typed` (a given value and a dotted extra; defaults and the schema of scanned-only dotted names are not checked) |
| experiments/runner#7 | covered | `tests/test_experiment_runner.py::test_control_sees_each_points_scalars_and_times` |
| experiments/runner#8 | partly | `tests/test_experiment_runner.py::test_failure_runs_finish_and_keeps_completed_points` (a failing `finish` after a good run; keeping the first traceback is not tested) |
| experiments/runner#9 | not | — |
| experiments/scan#1 | partly | `tests/test_experiment_scan.py::test_nested_first_axis_outermost` (axis indices only; `Point.index` and the values are not checked) |
| experiments/scan#2 | covered | `tests/test_experiment_scan.py::test_repeat_point_vs_scan` |
| experiments/scan#3 | covered | `tests/test_experiment_scan.py::test_snake_changes_one_axis_by_one_step` |
| experiments/scan#4 | partly | `tests/test_experiment_scan.py::test_shuffle_gets_a_seed_and_is_reproducible_from_json` (one pass only; never with repetitions) |
| experiments/scan#5 | covered | `tests/test_experiment_scan.py::test_shuffle_gets_a_seed_and_is_reproducible_from_json` |
| experiments/scan#6 | partly | `tests/test_experiment_scan.py::test_linear_integer_axis_rounds_and_rejects_duplicates` (1 to 5 in 3 steps is already whole: replacing `round` with `int` truncation passed in a scratch copy) |
| experiments/scan#7 | covered | `tests/test_experiment_scan.py::test_linear_integer_axis_rounds_and_rejects_duplicates` |
| experiments/scan#8 | covered | `tests/test_experiment_scan.py::test_values_are_coerced_against_the_declared_argument`, `tests/test_experiment_scan.py::test_unknown_or_duplicate_axes_are_rejected`, `tests/test_experiment_manager.py::test_notebook_calls_check_arguments_before_sending` (below `min`) |
| experiments/scan#9 | covered | `tests/test_experiment_scan.py::test_no_axes_is_a_single_point_per_repetition` |
| experiments/simulation#1 | covered | `tests/test_experiment_simulation.py::test_simulated_backends_are_built_even_when_config_says_real` |
| experiments/simulation#2 | partly | `tests/test_experiment_simulation.py::test_simulated_backends_are_built_even_when_config_says_real` (repeated `get` only; no check that sub-devices of one composite share one build) |
| experiments/simulation#3 | partly | `tests/test_experiment_simulation.py::test_simulated_backends_are_built_even_when_config_says_real`, `tests/test_experiment_simulation.py::test_run_local_simulated_experiment` (teardown runs; a failing teardown is not tested) |
| experiments/tweezer_baseline#1 | not | — |
| experiments/tweezer_baseline#2 | not | — |
| experiments/tweezer_baseline#3 | not | — |
| experiments/tweezer_baseline#4 | not | — |
| experiments/tweezer_baseline#5 | not | — |
| experiments/summary#1 | covered | `tests/test_results_browser.py::test_aggregate_means_and_standard_errors` |
| experiments/summary#2 | covered | `tests/test_results_browser.py::test_aggregate_means_and_standard_errors` |
| experiments/summary#3 | partly | `tests/test_results_browser.py::test_aggregate_means_and_standard_errors`, `tests/test_results_browser.py::test_panel_shows_metadata_plots_and_resubmits` (series keys and a curve count; each curve's means are not checked against its own samples) |
| experiments/summary#4 | covered | `tests/test_results_browser.py::test_aggregate_means_and_standard_errors` |
| experiments/summary#5 | not | — (the metadata check uses `gain`, which has no unit) |
| experiments/worker#1 | covered | `tests/test_experiment_e2e.py::test_pause_resume_and_terminate_at_a_point_boundary` (integration only, through real processes) |
| experiments/worker#2 | not | — |
| experiments/worker#3 | covered | `tests/test_experiment_e2e.py::test_worker_survives_a_manager_restart` (integration only, using sleeps) |
| experiments/worker#4 | not | — (the manager's stale-token reply carries an explicit `terminate`, so the worker's default is never exercised) |
| experiments/worker#5 | not | — |

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_experiment_arguments.py::test_values_are_coerced_to_the_declared_type` | rewrite | guards four kinds in one; `== -1.0` cannot see a non-float come back; make it one parametrised accepted-value test that checks types |
| `tests/test_experiment_arguments.py::test_invalid_values_are_rejected` | keep | one behaviour, parametrised; out-of-range or wrong-kind values would otherwise reach hardware quietly (rank 1) |
| `tests/test_experiment_arguments.py::test_bad_default_is_rejected_at_declaration` | keep | the declared limits also hold for the default, which would otherwise run quietly (rank 1) |
| `tests/test_experiment_arguments.py::test_schema_is_read_from_the_class_without_instantiating` | rewrite | guards several behaviours (no instantiation, order, override, schema fields); whole-dict equality breaks when a schema field is added; split into arguments#7 and #8 |
| `tests/test_experiment_arguments.py::test_coerce_arguments_fills_defaults_and_rejects_unknown_names` | rewrite | name needs "and": two rank-1 behaviours (arguments#1, #2) that fail differently |
| `tests/test_experiment_arguments.py::test_instance_attributes_hold_effective_values` | keep | one behaviour: the instance and `argument_values()` hold the effective values, and the class default is untouched (rank 1/2) |
| `tests/test_experiment_arguments.py::test_device_is_resolved_lazily_once_and_closed` | rewrite | uses the right stand-in (`fake_devices`) but guards caching, `fresh`, closing and cache clearing at once; split into experiment#3 and #4, and add a close that raises |
| `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements` | keep | the one integration test for Experiment Manager → worker → h5 file and database, normal case |
| `tests/test_experiment_e2e.py::test_catalogue_lists_runnable_experiments` | keep | the one integration test of the catalogue's introspect subprocess; a hidden experiment would be quiet |
| `tests/test_experiment_e2e.py::test_pause_resume_and_terminate_at_a_point_boundary` | rewrite | a second integration test of the same path, taking 3 s with sleeps; queue and runner sides are unit-tested already; the worker's pause loop (worker#1) needs a `ManagerLink` unit test with a stand-in `ReqClient` |
| `tests/test_experiment_e2e.py::test_crashed_worker_is_recorded` | rewrite | an abnormal case over real processes (1 s); the manager's handling of a worker that exits belongs in a manager unit test like `test_restart_settles_a_dead_workers_task` |
| `tests/test_experiment_e2e.py::test_abort_kills_the_worker` | delete | a loud failure: an abort that does nothing is seen at once in the queue view |
| `tests/test_experiment_e2e.py::test_worker_survives_a_manager_restart` | rewrite | guards worker#3 and adoption together, with 3 s of sleeps; adoption already has a manager unit test, so the retry should be a `ManagerLink` unit test |
| `tests/test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices` | keep | the simulate flag crossing to the worker; if it broke, a simulating manager's worker would drive real hardware, quietly |
| `tests/test_experiment_e2e.py::test_guis_see_the_queue_and_each_point_live` | keep | the one integration test for the manager → GUI sync path, normal case |
| `tests/test_experiment_queue.py::test_rids_are_sequential_and_order_is_priority_then_rid` | keep | queue#1 (rank 1); the rid assertion only fixes the inputs for the tie-break |
| `tests/test_experiment_queue.py::test_only_one_task_runs_at_a_time` | keep | one rank-1 behaviour (queue#4), checked through the public API |
| `tests/test_experiment_queue.py::test_due_time_holds_a_task_back_and_naive_times_are_local` | rewrite | two behaviours; the naive-local half passes under `TZ=UTC` when naive times are read as UTC (confirmed in a scratch copy), so it needs a fixed non-UTC zone |
| `tests/test_experiment_queue.py::test_only_waiting_tasks_can_be_held_deleted_or_dropped` | rewrite | guards hold/release, delete, terminate of a waiting task, lookup errors and refusals on the running task in one; split out queue#2 and #6 |
| `tests/test_experiment_queue.py::test_pause_resume_terminate_reach_the_worker_as_actions` | rewrite | three rank-1 behaviours (queue#5, #7, #8) in one; queue#5 should also check that the task stays active |
| `tests/test_experiment_queue.py::test_finish_moves_to_history_newest_first` | rewrite | snapshot, ordering, `points_done`, paging and a non-final status in one; keep only the move to history with its final status and `points_done` |
| `tests/test_experiment_queue.py::test_history_is_capped` | delete | guards nothing a maintainer would miss: the cap only bounds the size of the state file |
| `tests/test_experiment_queue.py::test_store_round_trip_and_unreadable_file` | rewrite | name needs "and": queue#9 (rank 2) and queue#10 (rank 4) are separate behaviours |
| `tests/test_experiment_recipes.py::test_names_are_stripped_and_required` | delete | an empty name fails loudly; whitespace stripping is already exercised by `test_a_name_is_taken_until_overwritten` |
| `tests/test_experiment_recipes.py::test_saving_clears_the_start_time_and_stamps_the_save` | rewrite | recipes#6 mixed with the `saved_at` stamp and `get`; keep only `due_time` being cleared |
| `tests/test_experiment_recipes.py::test_a_name_is_taken_until_overwritten` | keep | one behaviour (recipes#4, #5): another user's recipe is not replaced quietly |
| `tests/test_experiment_recipes.py::test_find_filters_and_sorts` | rewrite | the per-experiment filter matters (the wrong experiment's recipe offered); the sort order is cosmetic; keep the filter only |
| `tests/test_experiment_recipes.py::test_missing_recipes_are_reported_by_name` | delete | loud failures (an error message's wording) |
| `tests/test_experiment_recipes.py::test_store_round_trip` | keep | saved recipes read back identical (rank 2) |
| `tests/test_experiment_recipes.py::test_replay_applies_overrides_and_keeps_the_rest` | rewrite | recipes#1 and #2 in one function; split out the refusal of an override of a scanned argument |
| `tests/test_experiment_recipes.py::test_unknown_arguments_lists_names_the_experiment_lacks` | keep | recipes#3 (rank 1), called the way the manager calls it; catches a fault (confirmed in a scratch copy) |
| `tests/test_experiment_runner.py::test_hook_order_and_scanned_values` | keep | runner#1 (rank 1); catches stale point values (confirmed in a scratch copy) |
| `tests/test_experiment_runner.py::test_run_local_stores_effective_arguments_and_results` | rewrite | arguments, results, constants and an unknown-name error in one; the arguments check never looks at defaults (runner#6) |
| `tests/test_experiment_runner.py::test_failure_runs_finish_and_keeps_completed_points` | keep | one behaviour parametrised (failure → failed, `finish` runs, completed points kept); catches a skipped `finish` (confirmed in a scratch copy) |
| `tests/test_experiment_runner.py::test_control_stops_at_a_point_boundary` | keep | runner#2 (rank 1), parametrised |
| `tests/test_experiment_runner.py::test_control_sees_each_points_scalars_and_times` | keep | runner#7 (rank 2), checked against what the file stores |
| `tests/test_experiment_runner.py::test_devices_closed_even_on_failure` | keep | one behaviour with the shared `fake_devices` stand-in; leaked connections would be quiet |
| `tests/test_experiment_scan.py::test_no_axes_is_a_single_point_per_repetition` | keep | scan#9 (rank 1) |
| `tests/test_experiment_scan.py::test_nested_first_axis_outermost` | rewrite | checks axis indices, not the values and `Point.index` that callers use (scan#1); the green trial rewrote it the same way |
| `tests/test_experiment_scan.py::test_snake_changes_one_axis_by_one_step` | keep | scan#3, checked against the adjacency property itself |
| `tests/test_experiment_scan.py::test_shuffle_gets_a_seed_and_is_reproducible_from_json` | rewrite | guards scan#5 and the permutation of scan#4 together; the permutation needs its own per-pass test with repetitions (as the green trial added) |
| `tests/test_experiment_scan.py::test_repeat_point_vs_scan` | keep | scan#2, one behaviour in its two modes |
| `tests/test_experiment_scan.py::test_values_are_coerced_against_the_declared_argument` | keep | scan#8 (rank 1) |
| `tests/test_experiment_scan.py::test_linear_integer_axis_rounds_and_rejects_duplicates` | rewrite | two behaviours, and the rounding case is already whole, so truncation passes (confirmed in a scratch copy) |
| `tests/test_experiment_scan.py::test_unknown_or_duplicate_axes_are_rejected` | rewrite | cases of one behaviour written in sequence; make them one parametrised test |
| `tests/test_experiment_simulation.py::test_simulated_backends_are_built_even_when_config_says_real` | rewrite | forced simulation, name matching, reuse, teardown and an unknown name in one; simulation#2 and #3 are only half checked |
| `tests/test_experiment_simulation.py::test_run_local_simulated_experiment` | rewrite | several behaviours; the last line runs `simulate=False` against unreachable real devices, relying on that failing; split out the `simulated` attribute in the file |
| `tests/test_experiment_simulation.py::test_manager_tells_guis_it_is_simulating` | keep | one behaviour; a GUI not told it is simulating would mislead quietly |
| `tests/test_experiment_simulation.py::test_data_root_resolution` | rewrite | several cases of one behaviour written in sequence; make it one parametrised test |
| `tests/test_motmaster_experiment.py::test_declared_and_searched_parameters_are_sent_typed` | rewrite | routing (motmaster#2), typing (#3) and recording in the file in one; split typing into a parametrised test |
| `tests/test_motmaster_experiment.py::test_bad_script_parameter_fails_before_any_sequence_runs` | keep | motmaster#3–#5, one behaviour, parametrised, checking what reaches the sequencer |
| `tests/test_motmaster_experiment.py::test_dotted_names_that_are_not_searchable_are_rejected_at_submit` | keep | motmaster#6 (rank 1), parametrised |
| `tests/test_motmaster_experiment.py::test_master_is_untriggered_and_followers_start_first` | rewrite | trigger mode, start order and sent parameters in one; the fake ignores the other `prepare` settings, so motmaster#1 needs a fake that records them, and #7 should measure the delay |
| `tests/test_motmaster_experiment.py::test_follower_client_is_opened_once_per_task` | delete | counts calls to a fake; guards no behaviour a caller sees |
| `tests/test_motmaster_experiment.py::test_failure_opening_a_follower_client_starts_no_follower` | keep | one behaviour: no follower is left armed for a trigger that never comes |
| `tests/test_motmaster_experiment.py::test_master_or_follower_failure_fails_the_task` | keep | a follower failure that was ignored would mark the task completed quietly; parametrised |
| `tests/test_motmaster_experiment.py::test_follower_that_never_finishes_fails_after_its_timeout` | keep | motmaster#9's timeout half |
| `tests/test_motmaster_experiment.py::test_declaration_rules_are_checked_at_class_definition` | rewrite | six cases of one behaviour in sequence, so the first failure hides the rest; parametrise |
| `tests/test_motmaster_experiment.py::test_schema_describes_the_sequencers_and_parameters` | delete | duplicates the GUI tests, which build their forms from a `MotMasterExperiment` schema and fail if it is wrong |
| `tests/test_motmaster_experiment.py::test_driver_raises_instead_of_printing` | keep | driver: a vendor error that was only printed would leave the previous script selected, quietly |
| `tests/test_motmaster_experiment.py::test_script_parameters_leaves_the_selected_script_unchanged` | rewrite | right behaviour (the next shot must run the selected script), but it asserts the vendor call sequence; check only the script left selected |
| `tests/test_motmaster_experiment.py::test_simulated_sequencer_serves_and_validates_parameters` | rewrite | four behaviours of the simulated twin in one; split, and add the check that the twin has the real class's public methods |
| `tests/test_motmaster_experiment.py::test_simulated_end_to_end_with_searched_parameters` | keep | searched parameters through the real simulated twin, which checks the types the script expects |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices` | `worker.run` passes the manager's `simulate` to `open_writer`, so a simulating manager's tasks get simulated devices | yes — `worker.py` is in this area, and if this broke a simulating manager's worker would drive real hardware quietly |
| `tests/test_experiment_simulation.py::test_run_local_simulated_experiment` | the file records `simulated` truthfully for `run_local` | yes — rank 2: simulated data that was not marked as simulated would pass for real data |
| `tests/test_experiment_runner.py::test_devices_closed_even_on_failure` | the runner closes device clients on every exit path | yes — experiment#4 covers `close_devices` itself but not that the runner calls it after a failure |
| `tests/test_experiment_recipes.py::test_store_round_trip` | saved recipes, with their scans, read back identical | yes — rank 2; queue#9 covers only `QueueStore` |
| `tests/test_motmaster_experiment.py::test_master_or_follower_failure_fails_the_task` | the failure of any one sequencer fails the task | yes — motmaster#8 covers only several failing; an ignored follower failure would mark the task completed |
| `tests/test_motmaster_experiment.py::test_failure_opening_a_follower_client_starts_no_follower` | no follower is armed when another follower cannot be reached | yes, minor — an armed follower blocks its device server until it times out |
| `tests/test_experiment_arguments.py::test_bad_default_is_rejected_at_declaration` | the declared limits apply to the default | yes, minor — the declaration-time half of arguments#3 |
| `tests/test_experiment_recipes.py::test_find_filters_and_sorts` | recipes are listed per experiment, and one name may be used by two experiments | yes, minor — otherwise another experiment's recipe could be offered or clash |
| `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements` | the manager → worker → h5 file and database path | no — `experiment_manager.py` is in the servers-config area |
| `tests/test_experiment_e2e.py::test_guis_see_the_queue_and_each_point_live` | the manager → GUI sync path | no — servers-config area |
| `tests/test_experiment_manager.py::test_state_survives_a_restart_and_rids_never_repeat` | rids are never reused after a restart, so files are not overwritten | no — servers-config area |
| `tests/test_experiment_simulation.py::test_data_root_resolution` | simulated data is kept apart from real data, and other PCs find the share | no — `storage.py` is in the data-recording area |
| `tests/test_motmaster_experiment.py::test_script_parameters_leaves_the_selected_script_unchanged` | reading another script's parameters leaves the selected script active | no — `drivers/motmaster.py` is in the drivers area |

## Verdict
**rework**: 29 of the 63 decided tests (46%) are kept as they are, which is more than a third. Almost every rewrite has the same cause: one function guards several behaviours or runs cases in sequence instead of parametrising them. The behaviours themselves are mostly the right ones. Four tests are unsound, not just untidy. Three were confirmed in a scratch copy run with `PYTHONPATH` set to the copy: the integer-axis rounding test passes when rounding is replaced by truncation, the naive due-time test passes under `TZ=UTC` when naive times are read as UTC, and the MOTMaster fake ignores the iterations, script and save settings, so removing `set_iterations` goes unnoticed. The fourth is a driver test that checks the vendor call sequence. The three slow end-to-end tests (pause, crash, restart) should become unit tests of `ManagerLink` and the manager. Coverage gaps are larger than test-quality problems: `dataset.py`, `tweezer_baseline.py`, most of `catalogue.py`/`introspect.py`, the `worker.py` orphan, disowned and failed-start paths, a failing `prepare`, `KeyboardInterrupt`, and the `run_local` file-versus-memory check have no tests at all. Six tests should go: four loud or trivial ones, one that counts calls to a fake, and one that duplicates the GUI tests.
