# Stage 2: drivers

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| drivers/camera_base#1 | not | — |
| drivers/camera_base#2 | not | — |
| drivers/camera_base#3 | not | — |
| drivers/camera_base#4 | not | — |
| drivers/camera_base#5 | not | — |
| drivers/camera_base#6 | not | — |
| drivers/hamamatsu#1 | covered | `tests/test_hamamatsu.py::test_falls_back_to_model_without_serial`, `tests/test_hamamatsu.py::test_ambiguous_model_raises` (the ORCA + ImagEM fixture is the stage-1 example; the "none match" branch shares the `len != 1` check that `test_missing_serial_raises` also pins) |
| drivers/hamamatsu#2 | partly | `tests/test_hamamatsu.py::test_selects_by_serial_regardless_of_model` parametrises only the configured side; the camera side is normalised in `list_dcam_cameras`, which the autouse fixture replaces with already-clean serials |
| drivers/imagemX2#1 | not | — |
| drivers/imagemX2#2 | not | — |
| drivers/imagemX2#3 | not | — |
| drivers/imagemX2#4 | covered | `tests/test_simulate_decorator.py::test_simulated_backend_covers_real_interface` (checks `SimulatedImagEMX2Camera`, the parent of the configured `SimulatedTweezerArrayCamera`, which inherits every method) |
| drivers/motmaster#1 | partly | `tests/test_motmaster_experiment.py::test_script_parameters_leaves_the_selected_script_unchanged` — the fake's `GetParameters` never raises, so replacing the `try/finally` with straight-line code passes (confirmed by mutation) |
| drivers/motmaster#2 | not | — |
| drivers/motmaster#3 | not | — (`test_declared_and_searched_parameters_are_sent_typed` types values in the experiment layer against a fake sequencer; `python_to_cs_dict` is never called) |
| drivers/motmaster#4 | not | — |
| drivers/motmaster#5 | not | — |
| drivers/motmaster#6 | not | — |
| drivers/motmaster#7 | not | — |
| drivers/motmaster#8 | covered | `tests/test_motmaster_experiment.py::test_simulated_sequencer_serves_and_validates_parameters` |
| drivers/motmaster#9 | covered | `tests/test_motmaster_experiment.py::test_simulated_sequencer_serves_and_validates_parameters` |
| drivers/motmaster#10 | partly | `tests/test_motmaster_experiment.py::test_simulated_end_to_end_with_searched_parameters` fails if `@simulate` is removed (confirmed by mutation), but only for the methods `MotMasterExperiment` calls; `SimulatedMotMasterInterface` is not in `test_simulated_backend_covers_real_interface` |
| drivers/orca#1 | not | — |
| drivers/orca#2 | not | — |
| drivers/orca#3 | covered | `tests/test_simulate_decorator.py::test_simulated_backend_covers_real_interface` |
| drivers/slm#1 | not | — (stripping the contiguity/dtype cast in `_as_c_uint8` fails nothing; confirmed by mutation) |
| drivers/slm#2 | not | — |
| drivers/slm#3 | not | — (removing the real `preload_image` bound fails nothing; confirmed by mutation) |
| drivers/slm#4 | not | — (`test_simulated_slm_records_what_was_written` counts the simulated twin's frames, not the real driver's order) |
| drivers/slm#5 | partly | `tests/test_slm.py::test_simulated_slm_records_what_was_written` checks the values, but its input is already `uint8` and `assert_array_equal` ignores dtype, so dropping the cast passes (confirmed by mutation) |
| drivers/slm#6 | covered | `tests/test_simulate_decorator.py::test_simulated_backend_covers_real_interface` |
| drivers/thorcam#1 | not | — |
| drivers/thorcam#2 | not | — |
| drivers/thorcam#3 | not | — |
| drivers/thorcam#4 | not | — |
| drivers/tweezermonitorcam#1 | not | — |
| drivers/tweezermonitorcam#2 | not | — |
| drivers/tweezermonitorcam#3 | not | — |
| drivers/tweezermonitorcam#4 | not | — |

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_hamamatsu.py::test_selects_by_serial_regardless_of_model` | keep | Quiet failure (wrong camera opened if the model filter overrode the serial); one behaviour, parametrised over the prefix; fails under mutation |
| `tests/test_hamamatsu.py::test_falls_back_to_model_without_serial` | keep | Quiet failure (wrong camera by model); pytweezer's own selection logic over a fake enumeration at the hardware edge |
| `tests/test_hamamatsu.py::test_missing_serial_raises` | keep | Guards that an absent serial never falls back to a model match, which would open the wrong camera quietly; one behaviour |
| `tests/test_hamamatsu.py::test_ambiguous_model_raises` | keep | Quiet failure (first of two matching cameras picked silently); fails under mutation |
| `tests/test_hamamatsu.py::test_open_returns_the_requested_camera` | keep | Normal case of the open path; the counterpart that stops the wrong-serial check passing by always raising, and the only check that the opened side's `S/N:` prefix is normalised |
| `tests/test_hamamatsu.py::test_open_rejects_a_camera_with_the_wrong_serial` | keep | Quiet failure (index shift opens another process's camera); fake vendor class only supplies data; fails under mutation |
| `tests/test_slm.py::test_simulated_slm_records_what_was_written` | rewrite | Guards five things at once (dimensions, temperature type, return value, `last_mask`, frame count), all on the twin; keep only slm#5, passing a non-`uint8` or non-contiguous input and checking dtype |
| `tests/test_slm.py::test_build_simulated_slm_and_teardown_closes_it` | delete | Duplicates `tests/test_servers.py::test_build_spec_constructs_class_from_config_and_wires_teardown` and `test_build_spec_simulation_uses_sim_class_else_a_generated_stand_in`; device-server wiring, not driver behaviour |
| `tests/test_slm.py::test_build_real_slm_attempts_the_sdk` | rewrite | Wanted behaviour (`simulate: False` must never quietly simulate), but it depends on the Blink DLL being absent (on the SLM PC it would open the real board) and `pytest.raises(Exception)` passes on any error; belongs in `tests/test_servers.py` with a recording fake class |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_hamamatsu.py::test_open_rejects_a_camera_with_the_wrong_serial` | `open_dcam_camera` closes and refuses a camera whose serial differs from the configured one (DCAM indices shift when another process holds a camera) | yes — without it the wrong camera is opened quietly, and stage 1 has no `open_dcam_camera` entry at all |
| `tests/test_motmaster_experiment.py::test_driver_raises_instead_of_printing` | A failed `SetScriptPath` raises and leaves no script selected; a failed `Go` raises | yes — if the driver swallowed these, the next shot would run the previous script or report a run that never happened, both quiet; the test also guards two behaviours at once and should be split |
| `tests/test_hamamatsu.py::test_open_returns_the_requested_camera` | Opened-side `S/N:` prefix is normalised before comparison | no — close to hamamatsu#2 in spirit, and a failure here is loud (the camera refuses to open) |
| `tests/test_slm.py::test_build_real_slm_attempts_the_sdk` | `simulate: False` builds the real class even when a `sim_class` is configured | no — device-server behaviour, owned by the servers-config area, not a driver promise |
| `tests/test_motmaster_experiment.py::test_simulated_sequencer_serves_and_validates_parameters` | The simulated twin serves typed canned defaults (`int` for Int32, `float` for Double) | no — simulation-only, and the typing it feeds is the experiment layer's concern; #8 and #9 already take the validation half |

## Verdict
**rework**: 6 of 9 decided tests are kept as they are. `tests/test_hamamatsu.py` is sound: every test guards pytweezer's own selection logic over a fake enumeration, and mutations to the ambiguity check, the serial-over-model precedence and the post-open serial check are each caught. `tests/test_slm.py` is where the work is. None of its tests reaches the real `SLM` class, so the driver's only data transformation (`_as_c_uint8`, slm#1/#2) and its frame-slot bound (slm#3) can be broken without failing anything, as mutation confirmed. It should be rebuilt around a fake Blink SDK that records the bytes it is handed, as the green trial does (its `test_update_mask_sends_the_pixels_row_major_as_uint8`, `test_run_sequence_displays_every_frame_in_order` and `test_preload_sequence_uploads_frames_in_order` map directly onto slm#1, #4 and #2), with the twin test cut down to slm#5. Beyond these two files, 29 of the 39 stage-1 entries have no coverage. The largest gaps are in `camera_base.py` (frame indexing, TIFF round-trip and no-overwrite), the four pylablib camera drivers (frame range and ROI, where stage 1 suspects an off-by-one) and the `motmaster.py` scan and `_auto_mot` logic. These are new tests to add in each area's existing file. None of them calls for discarding what is there.
