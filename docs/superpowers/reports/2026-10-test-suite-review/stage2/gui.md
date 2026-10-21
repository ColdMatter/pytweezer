# Stage 2: gui

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| gui/gui#1 | not | — |
| gui/gui#2 | not | — |
| gui/gui#3 | not | — |
| gui/gui#4 | not | — |
| gui/managed_panel#1 | partly | `tests/test_gui.py::test_remote_device_toggle_sends_requests_and_waits_for_the_feed` (start reads "starting" on "down", running on "up"; grace expiry untested) |
| gui/managed_panel#2 | not | — |
| gui/managed_panel#3 | partly | `tests/test_gui.py::test_remote_device_toggle_sends_requests_and_waits_for_the_feed` (stopping until the feed reads "down"; `_STOPPING_GRACE_S` fallback untested) |
| gui/managed_panel#4 | covered | `tests/test_gui.py::test_devices_panel_carries_out_requests_only_for_its_own_devices` |
| gui/managed_panel#5 | partly | `tests/test_gui.py::test_devices_panel_carries_out_requests_only_for_its_own_devices` (other host ignored; composite sub-devices untested) |
| gui/managed_panel#6 | covered | `tests/test_gui.py::test_remote_device_toggle_sends_requests_and_waits_for_the_feed` |
| gui/managed_panel#7 | covered | `tests/test_gui.py::test_stopping_an_unowned_device_hands_it_to_stop_orphan` |
| gui/managed_panel#8 | covered | `tests/test_gui.py::test_devices_panel_check_host_is_case_and_whitespace_insensitive` |
| gui/managed_panel#9 | partly | `tests/test_gui.py::test_client_control_panel_marks_portless_servers_unprobeable` (unprobeable only; probe → up/down untested) |
| gui/managed_panel#10 | partly | `tests/test_gui.py::test_devices_panel_breaks_composites_into_sub_device_rows` (state per row; `host:port` and age untested) |
| gui/analysismanager#1 | covered | `tests/test_analysis_gui.py::test_filter_state_distinguishes_a_died_process_from_a_stopped_one` |
| gui/analysismanager#2 | partly | `tests/test_analysis_gui.py::test_table_updates_in_place_and_keeps_the_selection` (same snapshot only; no add, removal or reorder, so the suspected bug passes) |
| gui/analysismanager#3 | not | — |
| gui/analysismanager#4 | not | — |
| gui/analysismanager#5 | not | — |
| gui/analysismanager#6 | not | — |
| gui/applet_launcher#1 | partly | `tests/test_applet_launcher.py::test_set_active_persists_locally_and_not_to_properties`, `tests/test_applet_launcher.py::test_shared_applet_list_carries_no_running_state` (through `_set_active`, not start/stop) |
| gui/applet_launcher#2 | covered | `tests/test_applet_launcher.py::test_shared_applet_list_carries_no_running_state` |
| gui/applet_launcher#3 | covered | `tests/test_applet_launcher.py::test_active_set_round_trips_through_local_settings` |
| gui/applet_launcher#4 | covered | `tests/test_applet_launcher.py::test_start_active_applets_starts_local_ones_and_prunes_deleted` |
| gui/applet_launcher#5 | covered | `tests/test_applet_launcher.py::test_applet_that_exits_on_its_own_is_forgotten` |
| gui/applet_launcher#6 | covered | `tests/test_applet_launcher.py::test_shutdown_terminates_children_but_keeps_the_active_set` |
| gui/applet_launcher#7 | not | — |
| gui/applet_launcher#8 | not | — |
| gui/applet#1 | not | — |
| gui/components#1 | not | — |
| gui/arg_editor#1 | covered | `tests/test_experiment_gui.py::test_form_shows_display_units_and_returns_si` |
| gui/arg_editor#2 | partly | `tests/test_experiment_gui.py::test_form_shows_display_units_and_returns_si`, `tests/test_experiment_gui.py::test_script_parameters_are_fetched_once_and_added_from_the_search` (defaults returned, but none smaller than the displayed decimals) |
| gui/arg_editor#3 | covered | `tests/test_experiment_gui.py::test_scan_axes_and_point_count` |
| gui/arg_editor#4 | partly | `tests/test_experiment_gui.py::test_parse_list_handles_units_and_kinds` (number scale and true/0 only) |
| gui/arg_editor#5 | partly | `tests/test_experiment_gui.py::test_parse_list_handles_units_and_kinds`, `tests/test_experiment_gui.py::test_an_invalid_form_is_reported_not_submitted_or_saved` (empty, bad choice, integer below min; SI-scaled min/max and unknown bool untested) |
| gui/arg_editor#6 | partly | `tests/test_experiment_gui.py::test_load_request_round_trips`, `tests/test_experiment_gui.py::test_load_request_restores_searched_rows_and_flags_missing_ones` (order and repeat not checked) |
| gui/arg_editor#7 | covered | `tests/test_experiment_gui.py::test_load_request_round_trips` |
| gui/arg_editor#8 | partly | `tests/test_experiment_gui.py::test_a_non_whole_value_for_an_integer_parameter_is_kept_and_reported`, `tests/test_experiment_gui.py::test_an_integer_parameter_scans_in_whole_steps_but_a_draft_keeps_any` (linear scan only; non-whole value and list value untested) |
| gui/arg_editor#9 | covered | `tests/test_experiment_gui.py::test_script_parameters_are_fetched_once_and_added_from_the_search`, `tests/test_experiment_gui.py::test_a_searched_parameter_can_be_scanned_removed_and_reset`, `tests/test_experiment_gui.py::test_an_unreachable_sequencer_shows_the_error_and_a_retry`, `tests/test_experiment_gui.py::test_load_request_restores_searched_rows_and_flags_missing_ones` |
| gui/arg_editor#10 | covered | `tests/test_experiment_gui.py::test_rows_restored_before_the_fetch_are_retyped_from_the_script` |
| gui/catalogue_view#1 | partly | `tests/test_experiment_gui.py::test_recipes_appear_under_their_experiment_and_filter` (listed under its class; "only there" unchecked) |
| gui/catalogue_view#2 | covered | `tests/test_experiment_gui.py::test_selecting_a_recipe_emits_it_and_a_refresh_keeps_it_quietly` |
| gui/catalogue_view#3 | partly | `tests/test_experiment_gui.py::test_recipes_appear_under_their_experiment_and_filter` (recipe match keeps class visible; module match untested) |
| gui/catalogue_view#4 | covered | `tests/test_experiment_gui.py::test_clicks_on_the_catalogue_only_reload_a_recipe_on_a_left_click` |
| gui/feed#1 | covered | `tests/test_experiment_gui.py::test_real_feed_signals_queue_changes_and_points` |
| gui/feed#2 | covered | `tests/test_experiment_gui.py::test_real_feed_signals_queue_changes_and_points` |
| gui/feed#3 | partly | `tests/test_experiment_gui.py::test_real_feed_signals_queue_changes_and_points` (alive-only drain; init after reconnect untested) |
| gui/feed#4 | not | — |
| gui/feed#5 | covered | `tests/test_experiment_gui.py::test_real_feed_signals_queue_changes_and_points` |
| gui/motmaster_params#1 | covered | `tests/test_experiment_gui.py::test_script_parameters_are_fetched_once_and_added_from_the_search` |
| gui/motmaster_params#2 | partly | `tests/test_experiment_gui.py::test_script_parameters_are_fetched_once_and_added_from_the_search`, `tests/test_experiment_gui.py::test_an_unreachable_sequencer_shows_the_error_and_a_retry` (concurrent duplicates untested) |
| gui/motmaster_params#3 | partly | `tests/test_experiment_gui.py::test_the_panel_reads_simulated_sequencers_when_the_manager_simulates` (simulated read; "no real client built" unchecked) |
| gui/panel#1 | partly | `tests/test_experiment_gui.py::test_panel_submits_and_forwards_actions` (one arg and submitter checked, not the whole request) |
| gui/panel#2 | covered | `tests/test_experiment_gui.py::test_submit_now_queues_the_recipe_or_shows_why_not` |
| gui/panel#3 | covered | `tests/test_experiment_gui.py::test_save_as_recipe_asks_for_a_name_and_confirms_a_replacement` |
| gui/panel#4 | partly | `tests/test_experiment_gui.py::test_panel_prefills_last_request_and_keeps_drafts` (draft restored via private calls; drop after submit untested) |
| gui/panel#5 | covered | `tests/test_experiment_gui.py::test_panel_prefills_last_request_and_keeps_drafts` |
| gui/panel#6 | covered | `tests/test_experiment_gui.py::test_panel_resubmit_selects_the_experiment` |
| gui/panel#7 | covered | `tests/test_experiment_gui.py::test_the_panel_reads_simulated_sequencers_when_the_manager_simulates` |
| gui/panel#8 | covered | `tests/test_experiment_gui.py::test_panel_submits_and_forwards_actions`, `tests/test_experiment_gui.py::test_panel_lists_recipes_and_refetches_when_they_change` |
| gui/queue_view#1 | partly | `tests/test_experiment_gui.py::test_queue_view_rows_and_buttons` (selection follows rid; buttons after the move unchecked) |
| gui/queue_view#2 | not | — |
| gui/queue_view#3 | partly | `tests/test_experiment_gui.py::test_queue_view_rows_and_buttons` (abort confirm and +1; −1 untested) |
| gui/queue_view#4 | partly | `tests/test_experiment_gui.py::test_queue_view_rows_and_buttons` (running and queued only, not parametrised) |
| gui/queue_view#5 | partly | `tests/test_experiment_gui.py::test_queue_view_rows_and_buttons` (row order; progress and "pause requested" text untested) |
| gui/results#1 | partly | `tests/test_results_browser.py::test_resubmission_reuses_arguments_and_scan` (no shuffle seed) |
| gui/results#2 | covered | `tests/test_results_browser.py::test_aggregate_means_and_standard_errors`, `tests/test_results_browser.py::test_panel_shows_metadata_plots_and_resubmits` |
| gui/results#3 | partly | `tests/test_results_browser.py::test_panel_shows_metadata_plots_and_resubmits` (curve count; labels unchecked) |
| gui/results#4 | partly | `tests/test_results_browser.py::test_live_running_measurement_plots_from_the_feed` (unshuffled; planned-index placement untested) |
| gui/results#5 | covered | `tests/test_results_browser.py::test_live_running_measurement_plots_from_the_feed` |
| gui/results#6 | covered | `tests/test_results_browser.py::test_panel_lists_days_and_hides_running_files_until_they_finish` |
| gui/results#7 | not | — |
| gui/results#8 | partly | `tests/test_results_browser.py::test_panel_shows_metadata_plots_and_resubmits` (one curve drawn; positions and labels unchecked) |
| gui/grafana#1 | covered | `tests/test_grafana.py::test_run_url_shows_the_run_window_with_margins` |
| gui/grafana#2 | covered | `tests/test_grafana.py::test_run_url_shows_the_run_window_with_margins` |
| gui/grafana#3 | covered | `tests/test_grafana.py::test_run_url_shows_the_run_window_with_margins` (the +01:00 case; dropping the offset makes it fail) |
| gui/property_editor#1 | not | — |
| gui/property_editor#2 | not | — |
| gui/property_editor#3 | not | — |
| gui/property_editor#4 | not | — |
| gui/property_editor#5 | not | — |
| gui/property_editor#6 | not | — |
| gui/property_editor#7 | not | — |
| gui/streammonitor#1 | not | — |
| gui/streammonitor#2 | partly | `tests/test_stream_monitor.py::test_log_monitor_filters_by_level_and_text` (unknown levels and case are untested) |
| gui/streammonitor#3 | partly | `tests/test_stream_monitor.py::test_stream_monitor_lists_newest_first_and_filters` (paused drop only; hidden-tab buffer untested) |
| gui/subscription_editor#1 | not | — |
| gui/subscription_editor#2 | not | — |
| gui/subscription_editor#3 | not | — |
| gui/theme#1 | not | — |
| gui/zmq_ROI#1 | not | — |
| gui/zmq_ROI#2 | not | — |
| gui/zmq_ROI#3 | not | — |
| gui/data_sidebar#1 | not | — |
| gui/image_monitor#1 | not | — |
| gui/image_monitor#2 | not | — |
| gui/image_monitor#3 | not | — |
| gui/image_plot_monitor#1 | not | — |
| gui/image_plot_monitor#2 | not | — |
| gui/live_plot#1 | not | — |
| gui/live_plot#2 | not | — |
| gui/scalar_history#1 | not | — |
| gui/scalar_history#2 | not | — |

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_analysis_gui.py::test_filter_state_distinguishes_a_died_process_from_a_stopped_one` | keep | GUI state logic; quiet failure; one behaviour |
| `tests/test_analysis_gui.py::test_table_updates_in_place_and_keeps_the_selection` | rewrite | asserts item identity (an internal); mixes in status text; never adds, removes or reorders rows, so it misses the suspected selection bug |
| `tests/test_applet_launcher.py::test_active_set_round_trips_through_local_settings` | keep | pytweezer's own logic on top of QSettings' list collapse; parametrised |
| `tests/test_applet_launcher.py::test_active_set_is_per_launcher_name` | delete | QSettings keying by name; nobody relies on two launchers on one PC |
| `tests/test_applet_launcher.py::test_set_active_persists_locally_and_not_to_properties` | rewrite | wanted behaviour, but drives the private `_set_active` rather than start/stop, so a start that wrote to Properties would pass |
| `tests/test_applet_launcher.py::test_start_active_applets_starts_local_ones_and_prunes_deleted` | rewrite | replaces the launcher's own `_start_applet` with a list: stand-in inside the unit rather than at the process edge (Popen) |
| `tests/test_applet_launcher.py::test_shutdown_terminates_children_but_keeps_the_active_set` | keep | fake process at the process edge; checks the outcome callers care about |
| `tests/test_applet_launcher.py::test_user_stop_clears_the_active_set` | keep | quiet failure (stopped viewer relaunched next session); one behaviour |
| `tests/test_applet_launcher.py::test_applet_that_exits_on_its_own_is_forgotten` | keep | quiet failure; one behaviour |
| `tests/test_applet_launcher.py::test_shared_applet_list_carries_no_running_state` | rewrite | three behaviours in one (legacy drop on load, defaults, `add_applet`) |
| `tests/test_experiment_gui.py::test_form_shows_display_units_and_returns_si` | keep | display units to SI: priority 1 (experiment gets the value asked) |
| `tests/test_experiment_gui.py::test_scan_axes_and_point_count` | keep | scan expansion from the form, in SI: priority 1 |
| `tests/test_experiment_gui.py::test_an_invalid_form_is_reported_not_submitted_or_saved` | rewrite | two behaviours: an invalid list is blocked, and non-MOTMaster integers accept non-whole linear steps |
| `tests/test_experiment_gui.py::test_parse_list_handles_units_and_kinds` | rewrite | valid and invalid cases mixed; should be two parametrised tests covering every kind and SI-scaled bounds |
| `tests/test_experiment_gui.py::test_load_request_round_trips` | keep | round trip of a request through the form: priority 1 |
| `tests/test_experiment_gui.py::test_the_recipe_name_is_cleared_by_anything_that_replaces_the_form` | delete | a display label; overwriting a recipe is already guarded by the confirm dialog |
| `tests/test_experiment_gui.py::test_queue_view_rows_and_buttons` | rewrite | about eight behaviours in one (order, tooltip, enable states, pause, abort confirm, Grafana, priority, selection) |
| `tests/test_experiment_gui.py::test_panel_submits_and_forwards_actions` | rewrite | three behaviours (submit, terminate forwarding, catalogue refetch); submit check looks at one arg, not the whole request |
| `tests/test_experiment_gui.py::test_panel_prefills_last_request_and_keeps_drafts` | rewrite | two behaviours; draft part drives `_save_draft` and sets `_current_key` instead of switching experiments as a user does |
| `tests/test_experiment_gui.py::test_panel_resubmit_selects_the_experiment` | keep | Resubmit loads into the right experiment; quiet failure; one behaviour |
| `tests/test_experiment_gui.py::test_real_feed_signals_queue_changes_and_points` | keep | the one integration test for the Experiment Manager → GUI path |
| `tests/test_experiment_gui.py::test_script_parameters_are_fetched_once_and_added_from_the_search` | rewrite | several behaviours (caching, which names are offered, keyboard completion, typing the row) |
| `tests/test_experiment_gui.py::test_a_searched_parameter_can_be_scanned_removed_and_reset` | rewrite | three behaviours (scan, remove drops its axis, reset drops rows) |
| `tests/test_experiment_gui.py::test_load_request_restores_searched_rows_and_flags_missing_ones` | keep | round trip of MOTMaster rows: priority 1 |
| `tests/test_experiment_gui.py::test_an_unreachable_sequencer_shows_the_error_and_a_retry` | rewrite | three behaviours (error shown, add refused without a value, retry not cached); the error text is loud |
| `tests/test_experiment_gui.py::test_a_threaded_fetch_is_delivered_on_the_gui_thread` | keep | pytweezer's threading on top of Qt; a cross-thread delivery fails intermittently, not loudly |
| `tests/test_experiment_gui.py::test_the_panel_reads_simulated_sequencers_when_the_manager_simulates` | keep | simulation must not reach hardware; quiet failure; one behaviour |
| `tests/test_experiment_gui.py::test_rows_restored_before_the_fetch_are_retyped_from_the_script` | keep | user's value and scan survive the late fetch: priority 1 |
| `tests/test_experiment_gui.py::test_a_non_whole_value_for_an_integer_parameter_is_kept_and_reported` | keep | non-whole value never reaches an Int32 parameter: priority 1 |
| `tests/test_experiment_gui.py::test_an_integer_parameter_scans_in_whole_steps_but_a_draft_keeps_any` | rewrite | several behaviours; draft half reads private `_drafts` and sets `_current_key` |
| `tests/test_experiment_gui.py::test_recipes_appear_under_their_experiment_and_filter` | rewrite | two behaviours (placement, filter); name has "and" |
| `tests/test_experiment_gui.py::test_selecting_a_recipe_emits_it_and_a_refresh_keeps_it_quietly` | keep | refresh keeps the selection without reloading the form; quiet failure |
| `tests/test_experiment_gui.py::test_the_recipe_menu_submits_or_deletes` | delete | a menu that emits nothing fails loudly; the downstream effects are tested at panel level |
| `tests/test_experiment_gui.py::test_panel_lists_recipes_and_refetches_when_they_change` | keep | refetch on `recipes_version`; quiet failure; one behaviour |
| `tests/test_experiment_gui.py::test_selecting_a_recipe_fills_the_form_and_a_list_refresh_keeps_edits` | rewrite | two behaviours (recipe fills the form; refresh keeps edits) |
| `tests/test_experiment_gui.py::test_save_as_recipe_asks_for_a_name_and_confirms_a_replacement` | rewrite | wanted (shared recipe overwrite), but five cases in sequence; should be one parametrised test |
| `tests/test_experiment_gui.py::test_submit_now_queues_the_recipe_or_shows_why_not` | keep | the right recipe is queued: priority 1 |
| `tests/test_experiment_gui.py::test_deleting_a_recipe_needs_confirmation` | keep | a recipe shared with every PC is not deleted by a mis-click; one behaviour |
| `tests/test_experiment_gui.py::test_clicks_on_the_catalogue_only_reload_a_recipe_on_a_left_click` | keep | real clicks, as a user makes them; quiet loss of edits |
| `tests/test_grafana.py::test_run_url_shows_the_run_window_with_margins` | keep | parametrised; expected epoch ms computed independently; fails when the UTC offset is dropped |
| `tests/test_grafana.py::test_run_url_accepts_datetimes` | delete | no caller passes datetimes |
| `tests/test_grafana.py::test_run_url_base_defaults_to_config` | delete | config plumbing; a wrong host fails loudly |
| `tests/test_grafana.py::test_dashboards_have_unique_uids_and_use_the_provisioned_datasource` | rewrite | two checks in one; a wrong datasource shows as an error panel (loud), a duplicate uid silently hides a dashboard (keep only that) |
| `tests/test_grafana.py::test_run_dashboard_takes_the_rid_the_gui_links_with` | keep | contract between `run_url` and the dashboard; a renamed variable gives an unfiltered dashboard quietly |
| `tests/test_grafana.py::test_alert_rules_query_the_provisioned_datasource_or_expressions` | keep | a misconfigured alert never fires, and nobody looks |
| `tests/test_gui.py::test_devices_panel_check_host_is_case_and_whitespace_insensitive` | keep | quiet failure (device spawned nowhere or everywhere); one behaviour |
| `tests/test_gui.py::test_this_pc_address_ignores_hostname_case` | keep | same quiet failure, at the config lookup |
| `tests/test_gui.py::test_bwidget_constructs_headless_without_property_hub` | delete | construction only: a loud failure, and asserts private attributes |
| `tests/test_gui.py::test_managed_row_toggle_reflects_controllability_and_state` | rewrite | three behaviours; only "degraded counts as running" is worth guarding |
| `tests/test_gui.py::test_client_control_panel_marks_portless_servers_unprobeable` | keep | client shows n/a rather than a false state; one behaviour |
| `tests/test_gui.py::test_devices_panel_breaks_composites_into_sub_device_rows` | rewrite | two behaviours (row building, which is visible; snapshot mapping, which is quiet) |
| `tests/test_gui.py::test_remote_device_toggle_sends_requests_and_waits_for_the_feed` | rewrite | start and stop paths in one test; each deserves its own |
| `tests/test_gui.py::test_devices_panel_carries_out_requests_only_for_its_own_devices` | rewrite | two behaviours (host filtering, ignoring a repeated request) in one |
| `tests/test_gui.py::test_stopping_an_unowned_device_hands_it_to_stop_orphan` | keep | quiet failure; stop and terminate contrasted as one behaviour |
| `tests/test_results_browser.py::test_aggregate_means_and_standard_errors` | keep | numerical: mean and SEM checked against hand-computed values |
| `tests/test_results_browser.py::test_resubmission_reuses_arguments_and_scan` | keep | Resubmit repeats the same measurement: priority 1 |
| `tests/test_results_browser.py::test_panel_lists_days_and_hides_running_files_until_they_finish` | keep | a finished file appears on refresh; quiet failure |
| `tests/test_results_browser.py::test_panel_shows_metadata_plots_and_resubmits` | rewrite | five behaviours (metadata, y choices, series plot, categorical axis, resubmit signal) |
| `tests/test_results_browser.py::test_open_in_grafana_only_for_queued_runs` | delete | a Grafana button that does nothing fails loudly |
| `tests/test_results_browser.py::test_panel_reports_an_unreadable_file` | delete | the message is loud; doesn't test the retry that matters (results#7) |
| `tests/test_results_browser.py::test_tree_updates_in_place_without_a_click` | rewrite | several behaviours, and asserts item identity (an internal) |
| `tests/test_results_browser.py::test_selected_running_measurement_reloads_and_keeps_choices` | keep | a selected running file updates on refresh; quiet staleness otherwise |
| `tests/test_results_browser.py::test_live_running_measurement_plots_from_the_feed` | rewrite | two behaviours (live plot, switch to file); spies on `load_measurement` calls |
| `tests/test_results_browser.py::test_queue_changes_drive_refreshes_and_polling_slows` | rewrite | counts calls to a wrapped `refresh`; should check that the tree updates when the feed is down |
| `tests/test_results_browser.py::test_a_missing_data_root_says_so` | delete | a hint message; loud |
| `tests/test_stream_monitor.py::test_stream_monitor_lists_newest_first_and_filters` | rewrite | three behaviours; misses the hidden-tab buffer (streammonitor#3) |
| `tests/test_stream_monitor.py::test_log_monitor_filters_by_level_and_text` | keep | errors not hidden by the filter; quiet failure |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_experiment_gui.py::test_a_threaded_fetch_is_delivered_on_the_gui_thread` | MOTMaster parameters fetched on a worker thread reach widgets on the GUI thread | yes — a cross-thread delivery gives intermittent crashes that are hard to trace back |
| `tests/test_experiment_gui.py::test_deleting_a_recipe_needs_confirmation` | a shared recipe is deleted only after confirmation | yes — the same quiet, every-PC loss as panel#3's overwrite |
| `tests/test_experiment_gui.py::test_a_searched_parameter_can_be_scanned_removed_and_reset` | removing a scanned MOTMaster row also removes its scan axis | yes — a removed row that stays scanned changes what the experiment runs (priority 1) |
| `tests/test_results_browser.py::test_selected_running_measurement_reloads_and_keeps_choices` | a selected running file's display updates on refresh with no feed | yes — results#5 covers only the feed path |
| `tests/test_gui.py::test_managed_row_toggle_reflects_controllability_and_state` | a degraded composite still counts as running (button reads Stop) | yes — theme#1 covers the colour, not that a degraded rig offers Start and can be started twice |
| `tests/test_gui.py::test_this_pc_address_ignores_hostname_case` | this PC's address is found whatever the hostname's case | no — `configuration/config.py` is in the servers-config area |
| `tests/test_grafana.py::test_run_dashboard_takes_the_rid_the_gui_links_with` | the run dashboard declares the `rid` variable `run_url` sets | no — the dashboard JSON is outside this area's file list, though the contract is worth keeping |
| `tests/test_grafana.py::test_alert_rules_query_the_provisioned_datasource_or_expressions` | alert rules query a provisioned datasource | no — deployment data, outside this area |
| `tests/test_simulation_mode.py::test_only_a_server_session_off_the_server_pc_simulates` | only a server session off the server PC simulates | no — config and launch code, in the servers-config area |

Merged into the stage-1 lists on 2026-10-21: every row above marked "yes" (and the "partly" schema-retry row) is now a stage-1 entry that names this test as its cover.

## Verdict
**rework**: 32 of the 67 decided tests (48 %) are kept as they are, well above a third, so the suite is a sound base. Its weaknesses follow a pattern: long scenario tests that check several behaviours at once (queue view, results panel, devices panel), and applet-launcher tests that build the object with `__new__` and replace its own methods. Most rewrites mean splitting these tests and moving stand-ins out to the process edge (Popen, the control channel). The bigger gap is coverage: 41 of the 103 stage-1 behaviours have no test at all. That includes every viewer, the property and subscription editors, `theme.resolve_state`, `build_gui`/`closeEvent` teardown, and the Analysis Manager commands. Several partly-covered entries miss exactly the case behind a suspected bug: the selection after a row is removed (analysismanager#2, queue_view#2), a default smaller than the display precision (arg_editor#2), and a shuffled live plot (results#4). Kept as is / total decided: 32 / 67.
