# Existing tests by area

Each test file has one **deciding** area, whose stage-2 agent decides keep, rewrite or delete for every test in it. The **also read by** areas use the file only to judge coverage of their own stage-1 entries. Areas come from each file's `pytweezer`/`bin` imports and patch targets. The deciding area was chosen by hand from the file's subject. `conftest.py` and `helpers.py` are shared fixtures, decided in stage 3: keep whatever the kept tests use.

| test file | deciding area | also read by |
|---|---|---|
| `tests/conftest.py` | shared (stage 3) | all |
| `tests/helpers.py` | shared (stage 3) | all |
| `tests/test_analysis_gui.py` | gui | — |
| `tests/test_applet_launcher.py` | gui | — |
| `tests/test_db_analysis.py` | data-recording | experiments |
| `tests/test_db_integration.py` | data-recording | — |
| `tests/test_db_writer.py` | data-recording | — |
| `tests/test_experiment_arguments.py` | experiments | — |
| `tests/test_experiment_e2e.py` | experiments | servers-config |
| `tests/test_experiment_gui.py` | gui | experiments, servers-config |
| `tests/test_experiment_manager.py` | servers-config | experiments |
| `tests/test_experiment_queue.py` | experiments | — |
| `tests/test_experiment_recipes.py` | experiments | — |
| `tests/test_experiment_runner.py` | experiments | — |
| `tests/test_experiment_scan.py` | experiments | — |
| `tests/test_experiment_simulation.py` | experiments | data-recording, servers-config |
| `tests/test_experiment_storage.py` | data-recording | experiments |
| `tests/test_grafana.py` | gui | — |
| `tests/test_gui.py` | gui | servers-config |
| `tests/test_hamamatsu.py` | drivers | — |
| `tests/test_loggers.py` | data-recording | — |
| `tests/test_motmaster_experiment.py` | experiments | drivers |
| `tests/test_parallel.py` | servers-config | — |
| `tests/test_properties.py` | servers-config | — |
| `tests/test_property_tree.py` | servers-config | — |
| `tests/test_rearrangement.py` | numerical | analysis, drivers, servers-config |
| `tests/test_results_browser.py` | gui | experiments |
| `tests/test_servers.py` | servers-config | drivers |
| `tests/test_simulate_decorator.py` | servers-config | drivers |
| `tests/test_simulation_mode.py` | servers-config | gui |
| `tests/test_sipyco_patches.py` | servers-config | — |
| `tests/test_slm.py` | drivers | servers-config |
| `tests/test_stream_monitor.py` | gui | — |
| `tests/test_sync.py` | servers-config | — |
