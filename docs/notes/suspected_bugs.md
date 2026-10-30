# Suspected bugs not yet fixed

The test-suite review of October 2026 found 84 suspected bugs, mostly by
reading the code (commit `c95e9f5`). The 22 urgent ones were fixed in PRs #10
and #13–#17, each with a regression test. These are the other 62, with
the smallest input that shows each one, copied from the review's
`stage1/<area>.md` lists (branch `test-suite-review`, PR #11). The full review
is in `docs/superpowers/reports/2026-10-test-suite-review.md` on that branch.

Most have not been run. Before fixing one, reproduce it with a failing test
(see the `write-tests` skill); if it doesn't reproduce, drop it from this list.
Some may be deliberate, so check with the people who use that code first.

## Noticed while fixing the urgent bugs

- `pytweezer/experiment/client.py`: after `submit` raises `ManagerUnavailable`,
  the task may still have been queued by a manager that was only slow. The
  client can't tell the two apart without a request id the manager
  deduplicates on.
- `pytweezer/analysis/analysis.py`: `detect_trap_sites(_grid)` no longer hangs,
  but when the requested site count is never matched it returns the wrong
  number of sites (or `None` for `_grid`) without raising.
- `pytweezer/analysis/analysis.py`: `fit_gaussian` reports `sx` as the major
  axis, so `extract_cloud_temperature`'s "Tx"/"Ty" are major- and minor-axis
  temperatures, not x and y.
- `pytweezer/experiment/queue.py`: `ModelStore.load` may not catch a
  `UnicodeDecodeError` from a corrupt queue or recipes file (it is a
  `ValueError`, not an `OSError`), so such a file would stop the manager
  starting. Read from the code, not reproduced.
- `pytweezer/drivers/motmaster.py`: the public `move_yag_spot` paths swallow
  every exception with a `print`, so a failing picomotor move is easy to miss.

## Data recording (0)

None left.

## Experiments (3)

- pytweezer/experiment/arguments.py: `Bool.coerce` refuses numpy bools, although its comment says they are accepted (`np.bool_` is not a `numbers.Integral`). A notebook passing `flag=arr.any()` or `ListAxis(values=list(mask))` gets an error. Smallest input: `Bool(False).coerce(np.True_)` raises `ValueError`.
- pytweezer/experiment/catalogue.py: introspection output is the child's whole stdout parsed as one JSON document. So an experiment module, or anything it imports, that prints at import time vanishes from the catalogue, and the misleading error is "introspection exited with code 0". Smallest input: an experiment module whose first line is `print("hi")`; its entry has `classes == []`.
- pytweezer/experiment/motmaster.py (low confidence): dotted overrides are always declared as `_MotMasterOverride`, a `Number`, so a script parameter whose default is a bool cannot be set by name, although `_typed` has a branch for bool defaults. Smallest input: a script with `{"flag": False}`; `submit(Exp, **{"rb.flag": True})` is refused ("is not a number"), and `1.0` is passed to .NET unchanged as a float.

## Servers and config (11)

- pytweezer/servers/analysis_manager.py: in `serve_forever`, a request that is received but fails to decode is followed by `continue` without a reply. The REP socket is then stuck in its send state: every later `recv_json` raises `EFSM`, which is caught and logged, and the loop spins at full CPU and never answers again. Checked against pyzmq — smallest input: a REQ socket sends `b"x"` to the Analysis Manager port.
- pytweezer/servers/experiment_manager.py: `handle` calls `request.get("command")` outside its `try`, and `serve_forever` calls `recv_json` without one. Valid JSON that is not a dict, or a body that is not JSON, raises out of `serve_forever` and stops the manager. Its worker survives and is adopted on restart — smallest input: `send_json([])` to the manager's REQ port.
- pytweezer/logging_utils.py: `DailyJsonFileHandler._ensure_path` creates `self._log_dir` but takes the file path from `get_daily_log_path(today)`, which uses `$PYTWEEZER_LOG_DIR` or `<repo>/logs`. An explicit `log_dir` is ignored — smallest input: `DailyJsonFileHandler(log_dir=Path(tmp)).emit(record)` with `PYTWEEZER_LOG_DIR` unset creates an empty `tmp` and appends to `<repo>/logs/pytweezer-<date>.jsonl`.
- pytweezer/servers/property_tree.py: `parse_key` strips a trailing `/` only from relative keys. `Properties.get` strips it first, but `set` and `delete` do not, so the two address different entries — smallest input: `parse_key("x", "/a/b/")` returns `["a", "b", ""]`, while `parse_key("x", "a/b/")` returns `["x", "a", "b"]`. So `Properties("x").set("/a/b/", 1)` writes the key `""` under `/a/b`.
- pytweezer/servers/imagelogger.py: `updated` is reset at the top of every loop iteration, so `active` is published only when a message arrives at least 1 s after the last publish. The last messages before a stream goes quiet are never published — smallest input: one message 0.5 s after start, then silence; `active` is never set.
- pytweezer/servers/datalogger.py: the same pattern (`self.updated = False` on every iteration), so a stream that sends a short burst and stops is never published. `active` changes only when the entry ages out after `maxStreamInactive[s]` (1000 s) — smallest input: one message 0.5 s after start, then silence.
- pytweezer/servers/datalogger.py, pytweezer/servers/imagelogger.py: `__main__` hardcodes the names `"Servers/DataStream"` and `"Servers/ImageStream"` and ignores the CLI argument. This breaks the `python <script> <name>` launch protocol — smallest input: `python datalogger.py Datalogger` still uses `Properties("Servers/DataStream")`.
- pytweezer/servers/datamgr.py (unused): `lastupdate` starts at 0, so the `timeout_without_endtime` branch in `_processCompleteData` fires at once. The first run is moved to `completeData` as soon as its `Experiment.start` arrives, and so is any run that starts more than 60 s after the last completion. Their data is then discarded as "arrived after timeout" — smallest input: on a fresh `DataSummary`, `processIncomingData("Experiment.start", {"_task": 1, "_starttime": time.time()}, None)` leaves `completeData == [start entry]`, with `_streams == ["Experiment.start"]`.
- pytweezer/servers/datamgr.py (unused): when column names collide, `get_unique_flist` returns bare names without dtypes, so `np.dtype(...)` raises. String widths are also taken from the last run only, which truncates longer earlier strings — smallest input: `_determineFormatstring([{"a": [1, 2], "a1": "x"}])` raises `TypeError`, and `[{"s": "long"}, {"s": "x"}]` gives width `S1`.
- pytweezer/servers/property_server.py: `_terminate_stale_instances` SIGTERMs and then SIGKILLs any process on the PC whose command line contains `property_server.py`, including editors, pagers or a pytest run naming the file. `xsub_xpub._terminate_stale_instances` has the same issue with `xsub_xpub.py` plus a substring match on the hub name — smallest input: `less pytweezer/servers/property_server.py` is killed when the Properties server starts.
- pytweezer/configuration/config.py: possibly the skill that is stale rather than the code. The `pytweezer-architecture` skill says the Experiment Manager is appended after the `CONFIG` literal so its ports come last. Here it sits inside `CONFIG["Servers"]`, before `"Devices"`, so its two ports shift every device port — smallest input: `CONFIG["Devices"]["Rb MotMaster"]["port"] == 7292`, where an appended entry would leave it at 7290.

## Drivers (11)

- pytweezer/drivers/thorcam.py, pytweezer/drivers/tweezermonitorcam.py: `setup_acquisition(self, nframes)` narrows the base signature `(acq_mode, nframes)`. The documented call fails on the real camera, while the simulated twin (from `SimulatedCamera`) needs `acq_mode`, so `optimisation.py`'s `setup_acquisition(nframes=...)` fails in simulation instead. Smallest input: `ThorCam.setup_acquisition(cam, "sequence", 10)` gives TypeError; `SimulatedThorLabsCamera().setup_acquisition(nframes=10)` gives TypeError.
- pytweezer/drivers/orca.py, imagemX2.py, thorcam.py, tweezermonitorcam.py: `set_roi` passes `x0 + width - 1` and `y0 + height - 1` as the end coordinates. pylablib's `set_roi(hstart, hend, vstart, vend)` treats the ends as exclusive (by pylablib's documented convention; I could not check this here because pylablib is not in the blind copy), so the real ROI comes out one pixel narrower and shorter than requested, while `SimulatedCamera` gives the full size. Smallest input: `set_roi(0, 512, 0, 512)` gives a 511x511 frame on hardware and 512x512 in simulation.
- pytweezer/drivers/motmaster.py: in `scan_picomotor_steps`, `results` is first assigned inside the `try`, so an early exception becomes UnboundLocalError at `return results`. Smallest input: with `self.motmaster = None`, `scan_picomotor_steps("X", "m1", 10, 100)` raises UnboundLocalError.
- pytweezer/drivers/motmaster.py: `python_to_cs_dict` sends bools as `Int32`, because `bool` is `numbers.Integral`. Smallest input: `python_to_cs_dict({"flag": True})` sends `Int32(1)`.
- pytweezer/drivers/motmaster.py: `SimulatedMotMasterInterface` rejects numpy integers that the real driver accepts (it checks `type(value) is not int`, while the real driver converts any `numbers.Integral` to `Int32`), so scans built from `np.arange` fail only in simulation. Smallest input: after `set_motmaster_experiment("RbTweezerBasic")`, `start_motmaster_experiment({"tDelay1": np.int64(5)})` raises TypeError.
- pytweezer/drivers/slm.py: `update_mask`, `preload_image` and `preload_sequence` never check the mask shape against `(height, width)`, and `preload_sequence` never checks `n <= 752`, so the DLL reads past the end of the buffer. Smallest input: on a 1024x1024 board, `update_mask(np.zeros((512, 512), np.uint8))`.
- pytweezer/drivers/slm.py: `_connect` enables `SetWaitForTrigger` before its blank `update_mask`, which then (according to `set_wait_for_trigger`'s own docstring) blocks until a trigger arrives or `timeout_ms` runs out, and then raises. Smallest input: `SLM(wait_for_trigger=True)` with no trigger within 5 s.
- pytweezer/drivers/camera_base.py: `SimulatedCamera._read_frames` returns one frame when asked for none. Smallest input: `acquire_n_frames(0)` returns shape `(1, 256, 256)`.
- pytweezer/drivers/orca.py, imagemX2.py: `trigger()` lacks `@requires_camera`, so after `relinquish_camera()` it fails with an AttributeError on `None` rather than the clear error. Smallest input: `cam.relinquish_camera(); cam.trigger()`.
- pytweezer/drivers/imagemX2.py, thorcam.py, tweezermonitorcam.py: in the legacy `main()`, `name` is a required positional, so the `--host/--port/--simulate` branch can never run, and `server_conf["image_dir"]` raises KeyError for an entry without it. Smallest input: `python thorcam.py --host 127.0.0.1 --port 1234` fails with "the following arguments are required: name".
- pytweezer/drivers/tweezermonitorcam.py: it copies `thorcam.py` with only the serial changed, including `get_logger("thorcam")`, so both cameras' JSONL logs share one logger name and cannot be told apart. Smallest input: any log line from "Tweezer Monitor ThorCam".

## Coordinators, phase masks and optimisation (16)

The live version of this code is on `rearrangement-optimisation-RnD`, so these were left alone on `exp-handler`. Check each one against that branch before fixing it.

- pytweezer/coordinators/rearrangement.py, also transport.py: `from pytweezer import analysis as an` binds the empty package `pytweezer/analysis/__init__.py`, which shadows `pytweezer/analysis.py`. So the default (no-detector) occupancy path raises `AttributeError: module 'pytweezer.analysis' has no attribute 'morphological_tophat_high_pass'`. Smallest input: `_extract_occupancy(np.zeros((10, 10)), 0)` with `_state={"detector": None, "grid_positions": {0: (5, 5)}, "window_size": 3, "feature_size": 2}`.
- pytweezer/coordinators/rearrangement.py, also transport.py: `initialise` uses `self.pm_gen`, taken from role `"phasemask_generator"`. The configured "Rb Rearrangement Rig" has no such sub-device and nothing else builds one, though the module docstring says `initialise` builds it. So `initialise` fails with `'NoneType' object has no attribute 'generate_phasemask'`. Smallest input: `Rearrangement({"slm": SimulatedSLM(), "camera": cam}, {}).initialise(...)` on a GPU host.
- pytweezer/coordinators/rearrangement.py, also transport.py: the panel pacing waits *after* each `update_mask`, so the first two writes are back to back. Frame 0 is shown only for one write time, which breaks the documented minimum interval. The `_play_sequence_pipelined_timings` path has the same bug. Smallest input: `panel_period_s=0.1` and `_play_sequence_pipelined([np.zeros((2, 2), np.uint8)] * 3)` with an SLM stub that records `perf_counter` per call. The gaps come out ≈ `[0.0, 0.1]`, not `[0.1, 0.1]`.
- pytweezer/coordinators/rearrangement.py, also transport.py: `shutdown` calls `print(..., exc_info=True)`, which raises `TypeError`. Smallest input: an initialised coordinator whose `camera.stop_acquisition` raises.
- pytweezer/coordinators/rearrangement.py, also transport.py: `arm_rearrangement_timings` runs the `_pairing_record`/`ttff_ms`/`med` block twice. The extra device syncs are harmless to the numbers, but it is a copy-paste duplicate. Smallest input: any `arm_rearrangement_timings()` call.
- pytweezer/coordinators/transport.py: it imports `linear_path, lift_path` from `pytweezer.phasemask`, and neither is defined there. The module therefore fails to import on every machine, which contradicts its docstring ("this module imports and status works on any machine"). Smallest input: `import pytweezer.coordinators.transport`.
- pytweezer/coordinators/transport.py: `arm_transport_pattern` cannot run against either generator in `phasemask.py`, for three reasons:
  - It passes `init_refine_iters`/`init_refine_damping`, which `iter_rearrangement_sequence` does not accept.
  - It passes 5-element `(w, phi, x, y, z)` terms where the 2D class unpacks 4.
  - It calls `PM.iter_transport_sequence`, which neither class defines.

  Smallest input: any `arm_transport_pattern()` call after `initialise`.
- pytweezer/coordinators/transport.py: `arm_rearrangement_timings` reads `s["num_images"]`, which `Transport.initialise` never stores. This raises `KeyError` only after the whole sequence has played; it is raised inside the `try` and again in the `except`. `arm_rearrangement_timings` and `arm_rearrangement_preload_trigger` would also hit the 4-vs-5 term unpack. `d0_z` and `fps` are stored but never used. Smallest input: `arm_rearrangement_timings()` after `initialise`.
- pytweezer/phasemask.py: `plan_rearrangement` builds `curr_w/phi/x/y` with `cp.asarray(..., dtype=float32)` and no `.copy()`. The fix and its comment are in `iter_rearrangement_sequence`. So, for float32 cupy `terms1`:
  - `curr_w[off_mask] = 0` zeroes the caller's weights at plan time.
  - `iter_rearrangement_sequence_timings` walks the caller's positions to the targets in place.

  The second `arm_rearrangement_timings` then starts from the wrong array. Smallest input: float32 cupy `terms1` with one empty site; after `plan_rearrangement(terms1, terms2, occ)` and consuming the timings generator, `terms1[2]` equals the target x and `terms1[0]` has a zero.
- pytweezer/phasemask.py: `superimpose` returns phase in [0, 2π), but `transform_phase_8bit` assumes [−π, π). It adds π, so half the range lands at 255–382 before `astype(uint8)`. Out-of-range float→uint8 is undefined: if the cast wraps, the result is a harmless global π offset; if it saturates, which CUDA's `cvt` does by default, half the phase range is clipped to 255. Smallest input: `transform_phase_8bit(superimpose([cp.full((1, 1), 1.5 * cp.pi)]))` gives 318.75 before the cast.
- pytweezer/phasemask.py: when every atom already sits on its target (`max_dist == 0`), `n_steps` is 0. `dw` is then divided by zero and no frames are yielded, and `arm_rearrangement_preload_trigger` goes on to call `start_auto_increment(0)` and `pulser.send_pulses(-1)`. An all-empty `occ_mask` instead raises on `.max()` of an empty array, and also on `1.0 / len(moving_idx)`. Smallest input: `terms1 == terms2` with `occ_mask` all True.
- pytweezer/phasemask.py: at import, the module sets `os.environ['CUDA_PATH']` to a hard-coded Windows path on every machine. It also imports `cupy`/`lap` and compiles `cp.ElementwiseKernel`. Without a GPU stack, even the pure-numpy helpers (`get_zernike_polynomial`, `rotate_*`, `ZernikeCalibrator`) therefore cannot be imported, so none of the entries above can run on CI. Smallest input: `import pytweezer.phasemask` without cupy.
- pytweezer/phasemask.py: `OptimisationBasedPhasemaskGeneratorGPU3D.generate_rearrangement_sequence` has two faults:
  - `n_steps` comes from the in-plane distance only, so a pure-z move gives 0 frames.
  - `z2[final_idx]` indexes the numpy `Zn` returned by `generate_weighted_array` with a cupy index, which raises `TypeError`.

  Its terms also carry `array_shape`, which the 2D-style unpack expects. Smallest input: terms from `generate_weighted_array` for both arrays, with a full `occ_mask`.
- pytweezer/phasemask.py: `ZernikeCalibrator.get_zernike_coeff` ignores `polynomial_degree` and always evaluates the 10-term cubic. A degree-2 file (6 coefficients) raises `IndexError`; a degree-4 file is silently truncated. Smallest input: a JSON with `{"polynomial_degree": 2, "coefficients": [0, 0, 0, 0, 0, 0]}`.
- pytweezer/optimisation.py: when `an.fit_gaussian` raises, `DetectMOTPosition` prints "Could not detect MOT" and then fails at `return` with `UnboundLocalError` on `centre_x`. Every caller dies on the same path. Smallest input: blank frames from the camera, so the fit fails.
- pytweezer/optimisation.py: `OptimiseMOTShimParameters` and `OptimiseMolassesShimParameters` divide by `n_calls - 1` in the xi schedule. Smallest input: `n_calls=1` gives `ZeroDivisionError`.

## Analysis (10)

- pytweezer/analysis/analysis.py: `get_array_loading_statistics_grid` passes `grid_shape` positionally into `sum_pixel_values(image, grid_positions, grid_shape, window_size=...)` and into `SiteScorer.from_images(images, grid_positions, grid_shape, window_size=...)`, so both paths raise `TypeError: got multiple values for argument 'window_size'`. Its later `photon_array[:, n, m]` indexing also assumes a 3-D array that neither scorer returns. Smallest input: `get_array_loading_statistics_grid([np.zeros((10, 10))], {(0, 0): (5, 5)}, (1, 1))`.
- pytweezer/analysis/analysis.py: `extract_temperature` drops the NaNs from `survival_prob_list` first, then builds the mask for `dropt` from the already-filtered list, so the lengths and the pairing no longer match. Smallest input: `extract_temperature(np.array([1., 2., 3.]), [0.9, np.nan, 0.5], plot=False)` raises `IndexError`. Its bare `except:` also returns a fabricated 100 µK when the fit fails, and with `plot=True` it then raises `NameError` on `params`.
- pytweezer/analysis/analysis.py: `index_grid_positions` converts pixels to µm and back, then uses `.astype("int64")`, which truncates float round-off and moves about 4 % of sites one pixel down. Smallest input: x pixel 1 with `x_centre = 10.5`, since `int((1 - 10.5) * -hamcam_um_per_px * -hamcam_px_per_um + 10.5)` gives `0` (value 0.9999999999999982).
- pytweezer/analysis/analysis_base.py: `run()` unpacks `msgstr, head, data = msg`, but `GenericClient.recv` returns a 2-tuple for header-only messages, so the first such message raises `ValueError` and kills the process. `gaussian_fit.py`'s `if data is None` branch can therefore never run. Smallest input: a client whose `recv()` returns `("x", {"a": 1})`.
- pytweezer/analysis/data_average.py: `_single_experiment` defaults to the string `"True"` and is tested for truthiness, so any non-empty value, including `"False"`, selects single-experiment mode and the continuous branch is unreachable. Smallest input: property `_single_experiment = "False"`. Also:
  - After a mid-shot reset (filter change), the partial sum is still divided by `n_average`. With `n_average=2` and the filter value changing at `_imgindex 1`, it publishes `data1 / 2`.
  - For integer-dtype traces, `average /= n_average` raises `UFuncTypeError` outside the `try`.
  - `n_average` is read once, so live retuning is ignored.
  - Every instance writes to the hard-coded `/Analysis/Data/axcol_bg_substract/background`.
- pytweezer/analysis/experiment_timer.py: an `Experiment.end` with no preceding start (`starttime == 0`) satisfies `0 < endtime` and publishes the absolute end time as the duration. Smallest input: one end message with `_endtime = 100` and no start gives `duration = 100`.
- pytweezer/analysis/floating_point_arithmetics.py: `round_floating_prec` rounds to `dig − 3` = 12 decimal places, not significant figures, so small values are zeroed. Smallest input: `round_floating_prec(1e-13)` gives `0.0`. It is currently unused.
- pytweezer/analysis/random_number.py: `self.dataq.send(data, "_n")` passes `"_n"` as the array argument `A`, so the number is published on `<name>`, not `<name>_n`. Smallest input: any `Experiment.start` message.
- pytweezer/analysis/substract.py: `np.float` was removed in NumPy 1.24 (2.4.4 is installed), so the first loop iteration raises `AttributeError` and the process dies. The default background `"[]"` is a string, which `np.array(..., dtype=float)` also rejects. Smallest input: `np.array("[]", dtype=np.float)`.
- pytweezer/coordinators/transport.py (caller of this area, for the coordinators reviewer): `from pytweezer import analysis as an` resolves to the empty package `pytweezer/analysis/__init__.py`, not to the stale duplicate module `pytweezer/analysis.py`, because a package takes precedence over a module of the same name. The no-detector occupancy path therefore raises `AttributeError` on `an.morphological_tophat_high_pass` / `an.sum_pixel_values`. Smallest input: `_extract_occupancy(image, threshold)` with no `detector` in `_state`.

## GUI (11)

- pytweezer/GUI/applet_launcher.py: the catalogue is read from Properties once, at construction, and every add or delete writes the whole cached dict back. A launcher on another PC therefore silently undoes additions and deletions made since it started. Smallest input: two `AppletLauncher`s on one Properties; `L1.add_applet("X", s)`; `L2.add_applet("Y", s)`; the shared `"Applets"` no longer contains `"X"`.
- pytweezer/GUI/streammonitor.py: `LogMonitor._load_daily_logs` appends backfilled records oldest first (`prepend=False`) while live records are prepended, so the table is not "newest first". Once 200 rows are full, each live record trims the bottom row, which is the newest backfilled record. Smallest input: a daily log holding messages "a" then "b"; the table shows "a" in row 0 and "b" in row 1.
- pytweezer/GUI/viewers/image_monitor.py: the image-index filter reads `head["index"]`, but publishers use `_imgindex` (`imageselector.py`, `image_stats.py`). Any index ≥ 0 therefore raises KeyError in `poll` on every frame. Smallest input: `image_index = 1` and a frame with header `{"_imgindex": 1, ...}`.
- pytweezer/GUI/viewers/image_monitor.py: `init_gui` unpacks `imgdata, head = self.imgstream.recv()`, but `recv()` returns `(channel, head, array)`, which raises ValueError if a frame is already queued at start-up. Also, `poll` and `update_mask` (and `image_plot_monitor._update_image`) unpack three values unconditionally, and `update_mask` indexes `_imgresolution` without a fallback. Smallest input: one image message queued before `init_gui`.
- pytweezer/GUI/viewers/image_monitor.py and image_plot_monitor.py: pyqtgraph's default `imageAxisOrder` is `col-major` (it is not changed anywhere, and `apply_theme` doesn't set it), so array axis 0 (rows) is drawn along x. Meanwhile `setRect` uses `shape[1]` for the x extent, and `roi_slice.py` maps x to columns. Non-square frames are therefore drawn transposed and stretched, and an ROI drawn on screen crops a different part of the array. Smallest input: `np.zeros((2, 3))` with pixel `[0, 2]` set, resolution [1, 1] and offset [0, 0]; it is drawn at x∈[0, 1.5], y∈[4/3, 2], not at x∈[2, 3], y∈[0, 1].
- pytweezer/GUI/property_editor.py: `save()` and `load()` don't handle a cancelled dialog. The empty filename becomes ".json", so cancelling Save writes the properties to a stray `./.json` in the working directory. Smallest input: click "Save to file…" and press Cancel.
- pytweezer/GUI/property_editor.py: `TreeEdit._update` rebuilds a remotely changed entry from `props.get(...)`, which unwraps an option property to its bare value, so after any outside change the dropdown turns into a plain field. Smallest input: subtree "/" with `{"A": {"value": "x", "options": ["x", "y"]}}`; another client sets `/A` to "y"; `update()` then shows A as a str parameter.
- pytweezer/GUI/subscription_editor.py: `add()` recovers the name with `split("[")[0]`, so a stream whose name contains "[" is saved truncated. Smallest input: an active stream "cam[1]" is stored as "cam".
- pytweezer/GUI/fitfunctions.py: `tof_temperature` uses `m_Li` (lithium-6) for a Rb/CaF experiment, which would give T too low by m_Li/m_Rb ≈ 0.07. Also, the `saturation_parameter` and `lorentzian` `function_text` strings don't match the code (a √(1+s0) that isn't there; misplaced brackets). The module is unused. Smallest input: `tof_temperature([0, 1e-6, 0], 1e-3)` returns √(k_B·1 µK/m_Li)·1 ms.
- pytweezer/GUI/models.py: `DictSyncModel.__setitem__` *swaps* the old and new rows in `row_to_key` instead of moving the key. A priority change that moves a row by more than one place therefore reorders an unrelated row, and a move to the end raises IndexError. The module is unused. Smallest input: keys a, b, c, d at priority 0, then set c's priority to 1; `row_to_key` becomes [c, b, a, d], not [c, a, b, d].
- pytweezer/GUI/pytweezerQt.py: `SearchLineEdit.focusOutEvent` calls `super().focusInEvent`, and `lastText` is unset until the first focus-out, so Escape before that raises AttributeError. The class is unused. Smallest input: press Escape in a fresh `SearchComboBox`.
