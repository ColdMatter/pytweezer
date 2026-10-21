# Stage 2: servers-config

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| servers-config/launch#1 | covered | `tests/test_simulation_mode.py::test_launcher_marks_the_session_before_the_gui_imports`, `tests/test_simulation_mode.py::test_launcher_imports_no_pytweezer_code` |
| servers-config/config#1 | covered | `tests/test_simulation_mode.py::test_only_a_server_session_off_the_server_pc_simulates` (lower-case `ph-beast` covers the case-folding; the flags are only checked at their defaults) |
| servers-config/config#2 | covered | `tests/test_gui.py::test_this_pc_address_ignores_hostname_case` |
| servers-config/config#3 | not | — |
| servers-config/logging_utils#1 | not | — |
| servers-config/logging_utils#2 | not | — |
| servers-config/logging_utils#3 | not | — |
| servers-config/logging_utils#4 | not | — |
| servers-config/parallel#1 | covered | `tests/test_parallel.py::test_calls_run_concurrently_and_return_in_call_order` (timing-based: fails when the calls run one after another, checked by mutation) |
| servers-config/parallel#2 | covered | `tests/test_parallel.py::test_calls_run_concurrently_and_return_in_call_order` |
| servers-config/parallel#3 | covered | `tests/test_parallel.py::test_after_staggers_start` |
| servers-config/parallel#4 | partly | `tests/test_parallel.py::test_single_exception_propagates`, `tests/test_parallel.py::test_multiple_exceptions_grouped` (nothing checks that the other calls finish before the raise) |
| servers-config/parallel#5 | covered | `tests/test_parallel.py::test_timeout_raises` |
| servers-config/analysis_manager#1 | not | — |
| servers-config/analysis_manager#2 | not | — |
| servers-config/analysis_manager#3 | not | — |
| servers-config/analysis_manager#4 | not | — |
| servers-config/clients#1 | not | — |
| servers-config/clients#2 | not | — |
| servers-config/clients#3 | not | — |
| servers-config/device_client#1 | partly | `tests/test_servers.py::test_get_device_endpoint` (checks the host, port and target handed to a faked `RPCClient`, with no real server answering as each sub-device) |
| servers-config/device_client#2 | covered | `tests/test_servers.py::test_get_device_endpoint` |
| servers-config/device_control#1 | not | — |
| servers-config/device_control#2 | not | — |
| servers-config/device_server#1 | partly | `tests/test_servers.py::test_build_spec_constructs_class_from_config_and_wires_teardown`, `tests/test_rearrangement.py::test_simulated_rig_builds_and_resolves_roles` (config keys reach the constructor; ignored keys and class defaults are not checked) |
| servers-config/device_server#2 | covered | `tests/test_servers.py::test_build_spec_simulation_uses_sim_class_else_a_generated_stand_in`, `tests/test_slm.py::test_build_real_slm_attempts_the_sdk` |
| servers-config/device_server#3 | partly | `tests/test_rearrangement.py::test_simulated_rig_builds_and_resolves_roles` (inheritance is caught, by mutation, only because the real camera DLL is missing; "unless they set their own" is not tested) |
| servers-config/device_server#4 | covered | `tests/test_servers.py::test_composite_serves_what_started` |
| servers-config/device_server#5 | partly | `tests/test_rearrangement.py::test_simulated_rig_builds_and_resolves_roles` (roles keyed by explicit `role`; the default to the device name is not tested) |
| servers-config/device_server#6 | partly | `tests/test_servers.py::test_get_device_endpoint` (sub-device and coordinator addressing; the `KeyError` for names that fold together is not tested) |
| servers-config/device_server#7 | covered | `tests/test_servers.py::test_a_failing_rpc_call_is_logged_on_the_device_server_and_still_raised` |
| servers-config/device_server#8 | not | — |
| servers-config/device_server#9 | partly | `tests/test_servers.py::test_composite_rejects_a_nested_composite` (one of the three errors, and not that it raises before any backend is built) |
| servers-config/device_status#1 | covered | `tests/test_servers.py::test_build_snapshot_composite` |
| servers-config/device_status#2 | covered | `tests/test_servers.py::test_build_snapshot_composite_reuses_targets_while_the_server_is_busy`, `tests/test_servers.py::test_build_snapshot_composite` |
| servers-config/device_status#3 | covered | `tests/test_servers.py::test_build_snapshot_plain_device`, `tests/test_servers.py::test_build_snapshot_composite` |
| servers-config/device_status#4 | not | — |
| servers-config/device_status#5 | not | — |
| servers-config/experiment_manager#1 | covered | `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements` |
| servers-config/experiment_manager#2 | partly | `tests/test_experiment_manager.py::test_worker_events_drive_the_task_and_reach_the_database` (a stale token on a heartbeat only, not a stale rid or other events) |
| servers-config/experiment_manager#3 | covered | `tests/test_experiment_manager.py::test_a_recipe_that_no_longer_fits_its_experiment_is_refused` |
| servers-config/experiment_manager#4 | covered | `tests/test_experiment_manager.py::test_recipes_are_filled_in_published_and_survive_a_restart`, `tests/test_experiment_manager.py::test_notebook_recipes_round_trip` |
| servers-config/experiment_manager#5 | covered | `tests/test_experiment_manager.py::test_state_survives_a_restart_and_rids_never_repeat` |
| servers-config/experiment_manager#6 | covered | `tests/test_experiment_manager.py::test_restart_settles_a_dead_workers_task`, `tests/test_experiment_e2e.py::test_crashed_worker_is_recorded`, `tests/test_experiment_e2e.py::test_abort_kills_the_worker` |
| servers-config/experiment_manager#7 | covered | `tests/test_experiment_manager.py::test_worker_events_drive_the_task_and_reach_the_database`, `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements`, `tests/test_experiment_e2e.py::test_guis_see_the_queue_and_each_point_live` |
| servers-config/experiment_manager#8 | partly | `tests/test_experiment_manager.py::test_worker_events_drive_the_task_and_reach_the_database` (arguments from the file; the git commit is not checked) |
| servers-config/experiment_manager#9 | covered | `tests/test_experiment_manager.py::test_restart_adopts_a_live_worker_but_not_a_reused_pid`, `tests/test_experiment_e2e.py::test_worker_survives_a_manager_restart` |
| servers-config/experiment_manager#10 | covered | `tests/test_experiment_e2e.py::test_tasks_run_in_order_and_write_measurements` |
| servers-config/messageclient#1 | not | — |
| servers-config/messageclient#2 | not | — |
| servers-config/process_cleanup#1 | covered | `tests/test_servers.py::test_find_device_processes_matches_the_device_name_exactly` |
| servers-config/process_cleanup#2 | not | — |
| servers-config/properties#1 | covered | `tests/test_properties.py::test_edits_read_back_at_once_and_reach_other_processes` |
| servers-config/properties#2 | not | — |
| servers-config/properties#3 | covered | `tests/test_properties.py::test_unreachable_server_falls_back_to_the_file`, `tests/test_properties.py::test_clients_resync_after_a_server_restart` |
| servers-config/properties#4 | partly | `tests/test_properties.py::test_edits_read_back_at_once_and_reach_other_processes`, `tests/test_properties.py::test_clients_resync_after_a_server_restart` (a changed key is reported; the clearing and the option-property path are not checked) |
| servers-config/properties#5 | not | — |
| servers-config/property_server#1 | partly | `tests/test_properties.py::test_clients_resync_after_a_server_restart` (a set survives a close and restart; a delete does not get tested) |
| servers-config/property_server#2 | covered | `tests/test_properties.py::test_edits_are_saved_to_the_file_and_non_json_refused` |
| servers-config/property_server#3 | covered | `tests/test_properties.py::test_edits_are_saved_to_the_file_and_non_json_refused` |
| servers-config/property_tree#1 | partly | `tests/test_property_tree.py::test_parse_key` (no absolute key with a trailing `/`, the case that fails today) |
| servers-config/property_tree#2 | covered | `tests/test_property_tree.py::test_option_property_only_accepts_its_options` |
| servers-config/property_tree#3 | partly | `tests/test_property_tree.py::test_option_property_only_accepts_its_options`, `tests/test_property_tree.py::test_lookup_returns_a_copy_and_raises_for_missing` (the copy is checked only on a leaf list, so a shallow copy would pass) |
| servers-config/property_tree#4 | covered | `tests/test_property_tree.py::test_set_creates_missing_levels_overwrites_leaves_and_copies_the_value`, `tests/test_property_tree.py::test_ops_drive_a_notifier_and_publish_each_change` |
| servers-config/property_tree#5 | covered | `tests/test_property_tree.py::test_delete_only_existing_entries` |
| servers-config/simulated_device#1 | partly | `tests/test_servers.py::test_build_spec_simulation_uses_sim_class_else_a_generated_stand_in` (the real class is not constructed or subclassed; arbitrary constructor arguments are not passed) |
| servers-config/simulated_device#2 | partly | `tests/test_simulate_decorator.py::test_simulate_stubs_missing_public_methods_and_keeps_hand_written_ones` (methods inherited from a simulated base are not tested) |
| servers-config/simulated_device#3 | covered | `tests/test_simulate_decorator.py::test_simulate_defaults_and_exclude` |
| servers-config/sipyco_patches#1 | partly | `tests/test_sipyco_patches.py::test_reassembles_a_long_line_with_large_recvs` (a fake socket with ASCII only; no real `Client` and no multibyte character across a chunk boundary) |
| servers-config/sipyco_patches#2 | covered | `tests/test_sipyco_patches.py::test_patch_is_installed_on_import_and_reapplying_keeps_it` |
| servers-config/sync#1 | covered | `tests/test_sync.py::test_late_mirror_gets_snapshot_then_every_change` |
| servers-config/sync#2 | covered | `tests/test_sync.py::test_mirror_reconnects_after_server_restart`, `tests/test_properties.py::test_clients_resync_after_a_server_restart` (the latter for local edits being replaced) |
| servers-config/sync#3 | partly | `tests/test_sync.py::test_mirror_reconnects_after_server_restart`, `tests/test_sync.py::test_late_mirror_gets_snapshot_then_every_change` (order seen; "each once" for `drain()` and "after it has been applied" for `on_mod` are not checked) |
| servers-config/xsub_xpub#1 | not | — |

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_experiment_manager.py::test_submit_and_query` | rewrite | guards several commands at once (submit, get, last_request, snapshot); one behaviour per test |
| `tests/test_experiment_manager.py::test_errors_come_back_as_replies` | rewrite | wanted (a bad request must not kill the manager), but three cases of one behaviour should be one parametrised test |
| `tests/test_experiment_manager.py::test_state_survives_a_restart_and_rids_never_repeat` | rewrite | priority-2 rid continuation, but the name needs "and": split the queue reload from the rid check |
| `tests/test_experiment_manager.py::test_worker_events_drive_the_task_and_reach_the_database` | rewrite | guards about six behaviours in one function (stale-token terminate, pause/resume, final status, DB points, run row from file); split by behaviour |
| `tests/test_experiment_manager.py::test_restart_settles_a_dead_workers_task` | keep | quiet mis-recording of run status, one parametrised behaviour, checked in queue, DB and file |
| `tests/test_experiment_manager.py::test_restart_adopts_a_live_worker_but_not_a_reused_pid` | keep | one parametrised behaviour, with psutil on the test's own process as the process edge |
| `tests/test_experiment_manager.py::test_recipes_are_filled_in_published_and_survive_a_restart` | rewrite | priority-1 saved arguments, mixed with version counters, persistence and deletion; one behaviour per test |
| `tests/test_experiment_manager.py::test_a_recipe_that_no_longer_fits_its_experiment_is_refused` | keep | priority 1 (arguments not silently ignored), one parametrised behaviour, nothing queued on refusal |
| `tests/test_experiment_manager.py::test_notebook_recipes_round_trip` | rewrite | wanted client→manager recipe path, but covers save, list, submit, overwrite, string form and delete in one test |
| `tests/test_experiment_manager.py::test_notebook_calls_check_arguments_before_sending` | rewrite | priority-1 argument checking, but five cases of one behaviour should be one parametrised test |
| `tests/test_parallel.py::test_calls_run_concurrently_and_return_in_call_order` | rewrite | both priority-1 behaviours are wanted but the name needs "and"; split, and test concurrency by two calls waiting on each other's event rather than wall-clock time |
| `tests/test_parallel.py::test_after_staggers_start` | keep | priority 1 (trigger timing), one behaviour, through the public API |
| `tests/test_parallel.py::test_single_exception_propagates` | keep | a swallowed exception would be quiet; one behaviour |
| `tests/test_parallel.py::test_multiple_exceptions_grouped` | keep | no failure is dropped; one behaviour |
| `tests/test_parallel.py::test_timeout_raises` | keep | stage-1 entry, one behaviour, quick; caught a sequential-run mutation |
| `tests/test_properties.py::test_client_starts_from_the_servers_tree_and_creates_its_entry` | rewrite | name needs "and"; the starting-tree half is wanted, while creating an entry is not worth a test of its own |
| `tests/test_properties.py::test_edits_read_back_at_once_and_reach_other_processes` | rewrite | the cross-process integration test the skill wants, but it also checks `changes()` and delete propagation; split |
| `tests/test_properties.py::test_option_properties_are_enforced` | delete | duplicates `test_option_property_only_accepts_its_options`, since the option logic is `property_tree`'s; the cross-process path is covered by the edits test |
| `tests/test_properties.py::test_edits_are_saved_to_the_file_and_non_json_refused` | rewrite | two quiet behaviours (save without `close()`, non-JSON refused) in one test; split |
| `tests/test_properties.py::test_clients_resync_after_a_server_restart` | keep | quiet stale-state failure, one behaviour across a real restart |
| `tests/test_properties.py::test_unreachable_server_falls_back_to_the_file` | keep | quiet failure, one behaviour (about 1.1 s, slightly over the time budget) |
| `tests/test_property_tree.py::test_parse_key` | keep | parametrised, one behaviour, results checked against independently known paths |
| `tests/test_property_tree.py::test_set_creates_missing_levels_overwrites_leaves_and_copies_the_value` | rewrite | three behaviours in one test; split, or parametrise them as cases |
| `tests/test_property_tree.py::test_option_property_only_accepts_its_options` | keep | quiet option corruption, one behaviour; caught a mutation |
| `tests/test_property_tree.py::test_lookup_returns_a_copy_and_raises_for_missing` | rewrite | copy-isolation is wanted but should use a nested dict to tell deep from shallow; the loud `KeyError` half adds nothing |
| `tests/test_property_tree.py::test_delete_only_existing_entries` | keep | one behaviour (delete removes only what exists) |
| `tests/test_property_tree.py::test_ops_drive_a_notifier_and_publish_each_change` | rewrite | asserts the exact mod sequence, which is internal; compare the Notifier's tree with the dict's for the same ops instead |
| `tests/test_servers.py::test_resolve_device_matches_leniently` | delete | a failed lookup is a loud `KeyError`, and three cases use up the budget the nine device_server entries need |
| `tests/test_servers.py::test_unknown_device_error_lists_available` | delete | checks an error message; loud |
| `tests/test_servers.py::test_get_device_config_finds_sub_device` | delete | loud on failure and duplicated by the sub-device cases of `test_get_device_endpoint` |
| `tests/test_servers.py::test_build_spec_missing_class_key` | delete | loud config error |
| `tests/test_servers.py::test_build_spec_constructs_class_from_config_and_wires_teardown` | rewrite | two behaviours; keep the config→constructor part and add ignored keys and class defaults (device_server#1) |
| `tests/test_servers.py::test_build_spec_simulation_uses_sim_class_else_a_generated_stand_in` | rewrite | wanted (device_server#2), but two cases inline and no not-simulating case: make it one parametrised test |
| `tests/test_servers.py::test_composite_serves_what_started` | keep | quiet half-built rigs, parametrised over the cases of one behaviour |
| `tests/test_servers.py::test_composite_rejects_a_nested_composite` | rewrite | the raise is loud; the quiet part (no backend opened first) is unchecked, and the other two config errors belong in the same parametrised test |
| `tests/test_servers.py::test_get_device_endpoint` | keep | guards addressing and overrides, with RPC stubbed at the process edge; parametrised |
| `tests/test_servers.py::test_get_device_unaddressable` | delete | loud errors |
| `tests/test_servers.py::test_get_device_async_uses_config_host_and_port` | delete | checks a fake client's call arguments; the address resolution is shared with `get_device`, which is already tested |
| `tests/test_servers.py::test_build_snapshot_plain_device` | keep | quiet status mix-ups, parametrised, probes stubbed at the network edge (the fixture's `__new__` and private `_last_targets` could become plain construction with `get_config` patched) |
| `tests/test_servers.py::test_build_snapshot_composite` | keep | quiet "dead camera shows as up", parametrised |
| `tests/test_servers.py::test_build_snapshot_composite_reuses_targets_while_the_server_is_busy` | keep | quiet flapping, one behaviour; caught a mutation |
| `tests/test_servers.py::test_is_reachable` | delete | `is_reachable` only wraps `socket.create_connection`, so the test checks the library's guarantee |
| `tests/test_servers.py::test_a_failing_rpc_call_is_logged_on_the_device_server_and_still_raised` | keep | a real sipyco server and client, one behaviour, quiet missing log |
| `tests/test_servers.py::test_find_device_processes_matches_the_device_name_exactly` | keep | quiet kill of the wrong device, process list stubbed at the OS edge; caught a prefix-match mutation |
| `tests/test_simulate_decorator.py::test_public_methods_enumerates_the_public_surface_without_instantiating` | delete | tests a helper; its effect is already guarded through `simulate` in the next test |
| `tests/test_simulate_decorator.py::test_simulate_stubs_missing_public_methods_and_keeps_hand_written_ones` | keep | one behaviour (stubs fill gaps without replacing hand-written methods); quiet if broken |
| `tests/test_simulate_decorator.py::test_simulate_defaults_and_exclude` | rewrite | two behaviours; keep the callable-default freshness, and drop or separate `exclude` |
| `tests/test_simulate_decorator.py::test_simulated_backend_covers_real_interface` | keep | exactly the twin-has-public-methods check the skill asks for each driver |
| `tests/test_simulate_decorator.py::test_simulated_backend_methods_are_pyon_safe` | delete | an encode failure is loud at the first RPC, and the loop silently skips every method that needs arguments |
| `tests/test_simulation_mode.py::test_only_a_server_session_off_the_server_pc_simulates` | keep | quiet real-hardware risk, parametrised; a subprocess is needed because CONFIG is set at import |
| `tests/test_simulation_mode.py::test_launcher_marks_the_session_before_the_gui_imports` | keep | quiet real-hardware risk, one behaviour |
| `tests/test_simulation_mode.py::test_launcher_imports_no_pytweezer_code` | keep | the other half of launch#1, one behaviour |
| `tests/test_sipyco_patches.py::test_patch_is_installed_on_import_and_reapplying_keeps_it` | keep | quiet slowdown if the patch is lost; one behaviour |
| `tests/test_sipyco_patches.py::test_reassembles_a_long_line_with_large_recvs` | rewrite | correctness is wanted, but the recv-count assertion is internal; round-trip a multi-MB reply with a multibyte character across a chunk boundary through a real `Client` |
| `tests/test_sipyco_patches.py::test_raises_eof_when_the_peer_closes_mid_line` | delete | loud (an exception or a hang on disconnect), and stage 1 gives it no entry |
| `tests/test_sipyco_patches.py::test_read_time_scales_linearly_not_quadratically` | keep | guards the defect the patch exists for; fast (under 0.1 s), with a loose ratio |
| `tests/test_sync.py::test_late_mirror_gets_snapshot_then_every_change` | keep | the cross-process sync path, normal case, against a real server |
| `tests/test_sync.py::test_mirror_never_shares_objects_with_later_mods` | keep | quiet drift through shared references; one behaviour |
| `tests/test_sync.py::test_mutations_from_many_threads_stay_consistent` | keep | quiet drift under concurrent writers; one behaviour |
| `tests/test_sync.py::test_mirror_reconnects_after_server_restart` | keep | quiet stale GUI state, one behaviour |
| `tests/test_sync.py::test_mirror_waits_for_a_server_that_starts_later` | keep | quiet failure (a GUI started first never fills); one behaviour |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_sync.py::test_mirror_never_shares_objects_with_later_mods` | a mirror's state never aliases objects in later mods, so a list is not appended to twice | yes — quiet drift between the manager and GUIs, the same harm sync#1 names |
| `tests/test_sync.py::test_mutations_from_many_threads_stay_consistent` | `SyncServer.call` from several threads gives the mirror exactly the notifier's state | yes — the manager mutates from worker and request threads; a race would drift quietly |
| `tests/test_sync.py::test_mirror_waits_for_a_server_that_starts_later` | a mirror started before its server connects once the server comes up | yes — a GUI opened before the manager would otherwise stay empty with no error |
| `tests/test_experiment_e2e.py::test_pause_resume_and_terminate_at_a_point_boundary` | the manager relays pause, resume and terminate to the worker, which acts at point boundaries | yes — priority 1 (experiments do what they are asked); stage 1 has no entry for control relay |
| `tests/test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices` | a simulating manager's workers get simulated devices and mark the file simulated | yes — quiet real-hardware risk on the manager side; config#1 covers only the CONFIG flag |
| `tests/test_experiment_e2e.py::test_abort_kills_the_worker` | abort ends the worker process and records `aborted` | no — a failed abort is visible at once in the queue |
| `tests/test_simulate_decorator.py::test_simulated_backend_covers_real_interface` | each driver's simulated twin has the real class's public methods | no — it belongs to the drivers area, where the skill asks for it |
| `tests/test_experiment_manager.py::test_notebook_calls_check_arguments_before_sending` | client-side argument checks before anything is sent | no — it guards `pytweezer/experiment/client.py`, which is outside this area |
| `tests/test_gui.py::test_devices_panel_carries_out_requests_only_for_its_own_devices` | a Devices panel acts only on requests for devices on its own PC | no — GUI area, though it is the receiving half of device_control#1 |

## Verdict
**rework**: 31 of 61 decided tests are kept as they are (19 rewrite, 11 delete), well above the one-third threshold. The kept tests cover the behaviours that matter most here: restart reconciliation, recipe refusal, composite partial builds, device-status states, the launch and simulation guard, and sync and Properties resync over real sockets. Mutations of `run_parallel`, rid continuation, busy-target reuse, the non-JSON guard, option checks and exact device-name matching were each caught. The rewrites are mostly of form, not substance. `test_experiment_manager.py` and `test_properties.py` bundle many behaviours per function and should be split or parametrised. A few tests assert internals: recv counts, exact Notifier mod sequences and private fixture attributes. The deletions are loud-failure checks and library wrappers. The larger gap is not in these files: 23 of 74 stage-1 entries have no test at all, including the whole of `logging_utils`, `analysis_manager`, `clients` (with its Fortran-order bug), `device_control` and `messageclient`, composite teardown order, and the real-CONFIG port and name checks. Another 17 are only partly covered.
