# Stage 1: Drivers

## pytweezer/drivers/anapico.py

No entries: nothing in CONFIG or the package imports `AnapicoSYN420`, and its property-only API (`enabled`, `cw_frequency`) cannot be reached over sipyco RPC anyway, so there is no caller to protect.

## pytweezer/drivers/camera_base.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/camera_base#1 | `acquire_n_frames(n, start_frame=s, broadcast=True)` sends each frame to the image stream unmodified and in order, tagged `index = s + i`. | 2 | unit | quiet: shots are misnumbered or reordered on the stream, so analysis pairs the wrong images | — |
| drivers/camera_base#2 | `save_tiff` never overwrites: saving twice into the same directory produces `<prefix>0000_1.tif` and `<prefix>0000_2.tif`. | 2 | unit | quiet: earlier autosaved frames are lost | — |
| drivers/camera_base#3 | A frame written by `save_tiff` reads back with identical pixel values and dtype (`uint16`). | 2 | unit | quiet: saved images are truncated or recast | — |
| drivers/camera_base#4 | `acquire_n_frames(n, autosave=True)` writes exactly one TIFF per frame, numbered in frame order. | 2 | unit | quiet: frames missing from disk or saved out of order | — |
| drivers/camera_base#5 | After `close()`, `is_connected()` is False and `@requires_camera` methods raise, even when the vendor `_disconnect` itself raises. | 4 | unit | quiet: the GUI shows a relinquished camera as connected, and the next call fails on a `None` handle | — |
| drivers/camera_base#6 | After `set_roi(x0, width, y0, height)`, `SimulatedCamera` frames have shape `(height, width)`. | 4 | unit | quiet: in simulation, analysis runs on frames of the wrong size | — |

## pytweezer/drivers/hamamatsu.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/hamamatsu#1 | With `serial_number=None`, `find_dcam_index` returns the one camera whose model starts with the prefix, and raises when none or several match (e.g. an ORCA and an ImagEM on one host). | 4 | unit | quiet: the wrong Hamamatsu is opened, and nothing checks the serial afterwards | — |
| drivers/hamamatsu#2 | `find_dcam_index` matches a configured serial whether or not either side carries the `"S/N:"` prefix (parametrised). | 4 | unit | loud: the camera fails to open with "found 0" | — |
| drivers/hamamatsu#3 | `open_dcam_camera` closes and refuses a camera whose serial differs from the configured one (DCAM indices shift when another process holds a camera). *Added from stage 2; covered by `tests/test_hamamatsu.py::test_open_rejects_a_camera_with_the_wrong_serial`.* | 2 | unit | quiet: the wrong camera is opened and its frames recorded as the configured one's | — |

## pytweezer/drivers/imagemX2.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/imagemX2#1 | `_read_frames(n, s)` asks the vendor for exactly the frame range `(s, s + n)`. | 2 | unit | quiet: a shot is recorded with another shot's frames | — |
| drivers/imagemX2#2 | `set_roi(x0, width, y0, height)` reaches the vendor so that frames come out `width` x `height`, starting at `(x0, y0)`. | 4 | unit | quiet: the ROI is a pixel short or shifted, which moves tweezer site positions | — |
| drivers/imagemX2#3 | After each `start_acquisition`, `SimulatedTweezerArrayCamera` frames alternate atom image, background image, starting with atoms. | 4 | unit | quiet: simulated two-image shots subtract the wrong frame | — |
| drivers/imagemX2#4 | `SimulatedTweezerArrayCamera` (the configured `sim_class`) has every public method of `ImagEMX2Camera`. | 4 | unit | loud: the RPC call fails in simulation | — |

## pytweezer/drivers/motmaster.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/motmaster#1 | `script_parameters(other)` leaves MOTMaster's script path on the previously selected script (the last `SetScriptPath` names its `.cs`), including when `GetParameters` raises. | 1 | unit | quiet: the next shot runs the probed script instead of the selected one | — |
| drivers/motmaster#2 | `scan_motmaster_parameter` runs one `Go` per value, in the given order, each with `{parameter: value}`, and returns the callback results in that order. | 1 | unit | quiet: the scan runs points out of order or with the wrong value | — |
| drivers/motmaster#3 | `python_to_cs_dict` passes Python and numpy integers as `Int32` and floats unchanged. | 1 | unit | loud: MOTMaster raises a cast error on `Go` (quiet if it coerces instead) | — |
| drivers/motmaster#4 | `_auto_mot` leaves each laser on the set point with the greatest background-subtracted counts: the last `setSlaveFrequency` for the laser is that value. | 1 | unit | quiet: the MOT is "optimised" onto the wrong frequency | — |
| drivers/motmaster#5 | `_auto_mot` reads camera frames in consecutive windows `start_frame = 0, n, 2n, ...` across the background, scan and parking shots. | 2 | unit | quiet: each point's counts come from another shot's frames | — |
| drivers/motmaster#6 | The `numbers` and `errors` from `_auto_mot` are the mean and standard error of the per-frame background-subtracted counts. | 3 | unit | quiet: wrong optimum, or misleading error bars | frames that are background plus a constant `k` give exactly `k*H*W` counts; two frames with counts `a` and `b` give mean `(a+b)/2` and a known spread |
| drivers/motmaster#7 | `_move_picomotor_with_default_settings` moves the configured motor by the default steps (with the default speed and acceleration) and reverses direction once `max_steps` is reached. | 1 | unit | quiet: `move_yag_spot=True` scans never move the YAG spot (currently the case, see bugs) | — |
| drivers/motmaster#8 | `SimulatedMotMasterInterface.start_motmaster_experiment` rejects a parameter the selected script does not have. | 1 | unit | quiet: typos in experiment arguments pass in simulation and reach the lab | — |
| drivers/motmaster#9 | `SimulatedMotMasterInterface.start_motmaster_experiment` rejects a non-integer value for an Int32 script parameter. | 1 | unit | quiet: type errors in experiments go unseen until real MOTMaster runs | — |
| drivers/motmaster#10 | `SimulatedMotMasterInterface` has every public method of `MotMasterInterface`. | 4 | unit | loud: the RPC call fails in simulation | — |
| drivers/motmaster#11 | A failed `SetScriptPath` raises and leaves no script selected. *Added from stage 2; covered by `tests/test_motmaster_experiment.py::test_driver_raises_instead_of_printing`.* | 1 | unit | quiet: the next shot runs the previous script | — |
| drivers/motmaster#12 | A failed `Go` raises rather than printing. *Added from stage 2; covered by `tests/test_motmaster_experiment.py::test_driver_raises_instead_of_printing`.* | 1 | unit | quiet: a run that never happened is reported as done | — |

## pytweezer/drivers/orca.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/orca#1 | `_read_frames(n, s)` asks the vendor for exactly the frame range `(s, s + n)`. | 2 | unit | quiet: a shot is recorded with another shot's frames | — |
| drivers/orca#2 | `set_roi(x0, width, y0, height)` reaches the vendor so that frames come out `width` x `height`, starting at `(x0, y0)`. | 4 | unit | quiet: the ROI is a pixel short or shifted | — |
| drivers/orca#3 | `SimulatedORCACamera` has every public method of `ORCACamera` (binning, readout speed, gain, external exposure, trigger). | 4 | unit | loud: the RPC call fails in simulation | — |

## pytweezer/drivers/slm.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/slm#1 | `update_mask` hands `Write_image` a C-contiguous `uint8` buffer holding the mask's pixels in row-major order, including when given a non-contiguous view (a transpose or strided slice). | 4 | unit | quiet: a scrambled hologram on the SLM, so the tweezers land in the wrong places | — |
| drivers/slm#2 | `preload_sequence` hands `PreLoad_sequence` a contiguous `uint8` `(n, H, W)` buffer with `list_length = n`. | 4 | unit | quiet: the wrong or partial frames are clocked out on trigger | — |
| drivers/slm#3 | `preload_image` refuses a frame slot outside 0–751 before it calls the DLL. | 4 | unit | quiet: a write to an invalid on-board slot | — |
| drivers/slm#4 | `run_sequence` writes the frames one per `update_mask`, in sequence order. | 4 | unit | quiet: the frames are displayed out of order | — |
| drivers/slm#5 | `SimulatedSLM.update_mask` keeps `last_mask` as the `uint8` array the real driver would send. | 4 | unit | quiet: simulated rearrangement checks inspect the wrong mask | — |
| drivers/slm#6 | `SimulatedSLM` has every public method of `SLM`. | 4 | unit | loud: the RPC or coordinator call fails in simulation | — |

## pytweezer/drivers/thorcam.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/thorcam#1 | `_read_frames(n, s)` asks the vendor for exactly the frame range `(s, s + n)`. | 2 | unit | quiet: a shot is recorded with another shot's frames | — |
| drivers/thorcam#2 | `set_roi(x0, width, y0, height)` reaches the vendor so that frames come out `width` x `height`, starting at `(x0, y0)`. | 4 | unit | quiet: the ROI is a pixel short or shifted | — |
| drivers/thorcam#3 | `ThorCam`'s hooks accept the `Camera` base signatures, including `setup_acquisition(acq_mode, nframes)`. | 4 | unit | loud: TypeError on the real camera only, because the simulated twin accepts the base signature | — |
| drivers/thorcam#4 | `SimulatedThorLabsCamera` has every public method of `ThorCam`. | 4 | unit | loud: the RPC call fails in simulation | — |

## pytweezer/drivers/tweezermonitorcam.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| drivers/tweezermonitorcam#1 | `_read_frames(n, s)` asks the vendor for exactly the frame range `(s, s + n)`. | 2 | unit | quiet: a shot is recorded with another shot's frames | — |
| drivers/tweezermonitorcam#2 | `set_roi(x0, width, y0, height)` reaches the vendor so that frames come out `width` x `height`, starting at `(x0, y0)`. | 4 | unit | quiet: the ROI is a pixel short or shifted | — |
| drivers/tweezermonitorcam#3 | `ThorCam`'s hooks accept the `Camera` base signatures, including `setup_acquisition(acq_mode, nframes)`. | 4 | unit | loud: TypeError on the real camera only | — |
| drivers/tweezermonitorcam#4 | `SimulatedThorLabsCamera` has every public method of `ThorCam`. | 4 | unit | loud: the RPC call fails in simulation | — |

## Suspected bugs

- pytweezer/drivers/thorcam.py, pytweezer/drivers/tweezermonitorcam.py: `setup_acquisition(self, nframes)` narrows the base signature `(acq_mode, nframes)`. The documented call fails on the real camera, while the simulated twin (from `SimulatedCamera`) needs `acq_mode`, so `optimisation.py`'s `setup_acquisition(nframes=...)` fails in simulation instead. Smallest input: `ThorCam.setup_acquisition(cam, "sequence", 10)` gives TypeError; `SimulatedThorLabsCamera().setup_acquisition(nframes=10)` gives TypeError.
- pytweezer/drivers/orca.py, imagemX2.py, thorcam.py, tweezermonitorcam.py: `set_roi` passes `x0 + width - 1` and `y0 + height - 1` as the end coordinates. pylablib's `set_roi(hstart, hend, vstart, vend)` treats the ends as exclusive (by pylablib's documented convention; I could not check this here because pylablib is not in the blind copy), so the real ROI comes out one pixel narrower and shorter than requested, while `SimulatedCamera` gives the full size. Smallest input: `set_roi(0, 512, 0, 512)` gives a 511x511 frame on hardware and 512x512 in simulation.
- pytweezer/drivers/motmaster.py: `picomotor_default_axis` is set to None in `connect()` and never assigned (the config's `"motor"` goes into `picomotor_default_motor`), so `_move_picomotor_with_default_settings` never moves the stage. Smallest input: picomotor config `{"connect": true, "defaults": {"motor": "m1", "steps": 10, "max_steps": 100}}`, then `_move_picomotor_with_default_settings()` makes no `move_by` call.
- pytweezer/drivers/motmaster.py: the defaults check tests for the misspelt key `"aceeleration"`, so the acceleration is never read and `setup_velocity` is never called. Smallest input: defaults `{"speed": 100, "acceleration": 1000}` leaves `picomotor_default_acceleration` as None.
- pytweezer/drivers/motmaster.py: once `picomotor_steps_moved >= max_steps`, the counter is never reset, so the direction flips on every later call instead of travelling back. The steps also become a float (`*= -1.0`). Smallest input: `steps=10, max_steps=20` moves +10, +10, -10.0, +10.0, -10.0, ...
- pytweezer/drivers/motmaster.py: in `scan_picomotor_steps`, `results` is first assigned inside the `try`, so an early exception becomes UnboundLocalError at `return results`. Smallest input: with `self.motmaster = None`, `scan_picomotor_steps("X", "m1", 10, 100)` raises UnboundLocalError.
- pytweezer/drivers/motmaster.py: `python_to_cs_dict` sends bools as `Int32`, because `bool` is `numbers.Integral`. Smallest input: `python_to_cs_dict({"flag": True})` sends `Int32(1)`.
- pytweezer/drivers/motmaster.py: `SimulatedMotMasterInterface` rejects numpy integers that the real driver accepts (it checks `type(value) is not int`, while the real driver converts any `numbers.Integral` to `Int32`), so scans built from `np.arange` fail only in simulation. Smallest input: after `set_motmaster_experiment("RbTweezerBasic")`, `start_motmaster_experiment({"tDelay1": np.int64(5)})` raises TypeError.
- pytweezer/drivers/motmaster.py: `_auto_mot`'s "standard error" uses `np.std` with `ddof=0`, so it is 0 for `shots_per_point=1` and too small for small n. This is a convention question rather than a certain bug. Smallest input: per-frame counts `[0, 2]` give 0.707 rather than 1.0.
- pytweezer/drivers/slm.py: `update_mask`, `preload_image` and `preload_sequence` never check the mask shape against `(height, width)`, and `preload_sequence` never checks `n <= 752`, so the DLL reads past the end of the buffer. Smallest input: on a 1024x1024 board, `update_mask(np.zeros((512, 512), np.uint8))`.
- pytweezer/drivers/slm.py: `_connect` enables `SetWaitForTrigger` before its blank `update_mask`, which then (according to `set_wait_for_trigger`'s own docstring) blocks until a trigger arrives or `timeout_ms` runs out, and then raises. Smallest input: `SLM(wait_for_trigger=True)` with no trigger within 5 s.
- pytweezer/drivers/camera_base.py: `SimulatedCamera._read_frames` returns one frame when asked for none. Smallest input: `acquire_n_frames(0)` returns shape `(1, 256, 256)`.
- pytweezer/drivers/orca.py, imagemX2.py: `trigger()` lacks `@requires_camera`, so after `relinquish_camera()` it fails with an AttributeError on `None` rather than the clear error. Smallest input: `cam.relinquish_camera(); cam.trigger()`.
- pytweezer/drivers/imagemX2.py, thorcam.py, tweezermonitorcam.py: in the legacy `main()`, `name` is a required positional, so the `--host/--port/--simulate` branch can never run, and `server_conf["image_dir"]` raises KeyError for an entry without it. Smallest input: `python thorcam.py --host 127.0.0.1 --port 1234` fails with "the following arguments are required: name".
- pytweezer/drivers/tweezermonitorcam.py: it copies `thorcam.py` with only the serial changed, including `get_logger("thorcam")`, so both cameras' JSONL logs share one logger name and cannot be told apart. Smallest input: any log line from "Tweezer Monitor ThorCam".
