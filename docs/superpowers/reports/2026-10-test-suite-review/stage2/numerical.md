# Stage 2: numerical

## Coverage
| stage-1 id | status | covering test(s) |
|---|---|---|
| numerical/rearrangement#1 | not | — |
| numerical/rearrangement#2 | not | — |
| numerical/rearrangement#3 | not | — |
| numerical/rearrangement#4 | not | — |
| numerical/rearrangement#5 | not | — |
| numerical/rearrangement#6 | not | — |
| numerical/rearrangement#7 | not | — |
| numerical/rearrangement#8 | not | — |
| numerical/rearrangement#9 | not | — |
| numerical/transport#1 | not | — |
| numerical/transport#2 | not | — |
| numerical/transport#3 | not | — |
| numerical/optimisation#1 | not | — |
| numerical/optimisation#2 | not | — |
| numerical/optimisation#3 | not | — |
| numerical/optimisation#4 | not | — |
| numerical/optimisation#5 | not | — |
| numerical/optimisation#6 | not | — |
| numerical/phasemask#1 | not | — |
| numerical/phasemask#2 | not | — |
| numerical/phasemask#3 | not | — |
| numerical/phasemask#4 | not | — |
| numerical/phasemask#5 | not | — |
| numerical/phasemask#6 | not | — |
| numerical/phasemask#7 | not | — |
| numerical/phasemask#8 | not | — |
| numerical/phasemask#9 | not | — |
| numerical/phasemask#10 | not | — |

## Existing tests
| test | decision | skill rule that decides it |
|---|---|---|
| `tests/test_rearrangement.py::test_missing_slm_role_names_available` | delete | A missing role raises `KeyError` at construction: a loud failure, which ranks last. |
| `tests/test_rearrangement.py::test_phasemask_kwargs_merge_config_over_defaults` | delete | `Rearrangement` no longer has `phasemask_kwargs`, so the test fails on an attribute that does not exist and guards no current behaviour. |
| `tests/test_rearrangement.py::test_status_without_gpu` | delete | It pins the exact shape of the `status()` dict. A wrong readiness report shows up as soon as `initialise` is called, so this is loud, low-value and breaks under any reorganisation. |
| `tests/test_rearrangement.py::test_gpu_operations_without_cupy_raise_clearly` | delete | A raised error on a machine without a GPU is loud. The test also checks two calls, so it guards two behaviours. |
| `tests/test_rearrangement.py::test_arm_before_initialise_raises` | delete | Calling before `initialise` raises: a loud failure. |
| `tests/test_rearrangement.py::test_arm_streams_sequence_to_slm` | rewrite | It guards something wanted: the measured occupancy mask, `d0` and `profile` reach the generator, and the frames reach the SLM. But it checks six things at once, including `to_host`, the image shapes and the timings key. It stubs pytweezer's own analysis functions, which are not a process or hardware edge, and it patches the wrong module (`pytweezer.analysis.analysis` rather than the `rearr.an` the code uses), so it fails today with `AttributeError`. It counts frames written (`== 6`) instead of checking that each frame arrives once and in order. Split it into one test for the forwarding and one for frame order, both with a synthetic image rather than stubs. |
| `tests/test_rearrangement.py::test_arm_reraises_slm_upload_error` | rewrite | It guards something wanted (rearrangement#6: an upload error reaches the caller). But it goes through the same broken occupancy patch, so it errors before it reaches the SLM. It also uses 4 frames, fewer than `UPLOAD_QUEUE_DEPTH = 5`, so the case where a full queue blocks the producer is never exercised. Drive `_play_sequence_pipelined` directly with more than 5 frames. |
| `tests/test_rearrangement.py::test_shutdown_stops_the_camera_only_if_initialised` | delete | It covers two behaviours (no stop when uninitialised, and a stop plus state cleared when initialised) for teardown, which fails loudly. It also misses the one quiet case, rearrangement#9: `stop_acquisition` raising. |
| `tests/test_rearrangement.py::test_simulated_rig_builds_and_resolves_roles` | keep for now (ruling) | **Ruling:** keep until `tests/test_servers.py` has its own test for servers-config `device_server#3` (sub-devices inherit the composite's `simulate` flag), then move or delete it. This is its only cover. Original reasoning: This is generic composite and role wiring in `device_server`, which belongs in `tests/test_servers.py`. That file already covers which targets a composite serves. Wrong wiring would raise as soon as the coordinator called a method (loud), and the `roles` assertion duplicates `test_status_without_gpu`. |

## Missed by the review
| test | what it guards | was stage 1 wrong to omit it? |
|---|---|---|
| `tests/test_rearrangement.py::test_arm_streams_sequence_to_slm` | `arm_rearrangement` passes the occupancy mask it measured, and `d0`/`profile` from `initialise`, to `iter_rearrangement_sequence` | yes. A wrong `d0` or a stale occupancy mask makes the atoms move at the wrong speed, or to and from the wrong sites, with no error. That is priority 4 and quiet. Stage 1 listed exactly this for `Transport` (transport#2) but not for `Rearrangement`, and rearrangement#1 stops at the mask without checking that it reaches the generator. |

Merged into the stage-1 lists on 2026-10-21: every row above marked "yes" (and the "partly" schema-retry row) is now a stage-1 entry that names this test as its cover.

## Verdict
**start again**: none of the 9 decided tests can be kept as they are (0/9 kept, 2 rewrite, 7 delete). Seven guard only loud failures, stale API or generic framework wiring. The two worth having both fail today, before they reach the behaviour they target. Their occupancy stub patches `pytweezer.analysis.analysis`, while the coordinator calls the empty `pytweezer.analysis` package. That confirms stage 1's first suspected bug: with the patch pointed at `rearr.an`, both pass. Today no stage-1 entry is covered: nothing touches `phasemask.py`, `optimisation.py` or `transport.py`, nor the rearrangement plumbing (pairing record, preload slots, panel pacing, trigger arming). The file should be rebuilt around the stage-1 list plus the missed forwarding entry above. Its stand-ins should sit only at the camera, SLM and pulser edges, not at pytweezer's own analysis functions. The `phasemask.py` entries also need the import-time cupy dependency resolved before they can run on CI.
