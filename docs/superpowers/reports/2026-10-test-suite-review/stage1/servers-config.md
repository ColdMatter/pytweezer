# Stage 1: Servers and config

## bin/kill_stale.py

No entries: a thin argparse wrapper that prints what `process_cleanup.kill_stale_processes` returns, so any fault in it is obvious at the terminal.

## bin/launch.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/launch#1 | Importing `bin.launch` imports nothing from `pytweezer`, and `server_main` sets `PYTWEEZER_ROLE=server` before `bin.gui`, and so the config, is imported. | 4 | integration | quiet: `pytweezer-server` on a lab PC stops simulating and spawns real device servers that drive real hardware | — |

## pytweezer/configuration/config.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/config#1 | A process with the server role on any host other than `SERVER_PC` (hostname case ignored) gets `SIMULATING=True` and a localhost `SERVER_HOST`; on `SERVER_PC`, or without the role, both follow the `SIMULATING`/`LOCAL` flags (parametrised over role × hostname). | 4 | unit | quiet: a dev or lab-PC server session drives real hardware and tries the lab addresses, or the real server PC runs simulated | — |
| servers-config/config#2 | `this_pc_address()` returns this PC's `HOSTS` address whatever case the OS reports the hostname in, and `None` for a PC not listed. | 4 | unit | quiet: a GUI treats its own devices as remote and spawns none of them | — |
| servers-config/config#3 | In the real `CONFIG`, no two port-owning entries on the same host share a port, and device names are unique (case- and whitespace-insensitively) across composites. | 4 | unit | loud: a server fails to bind, or `device_index()` raises at device start | — |

## pytweezer/configuration/paths.py

No entries: module-level path constants and a two-line JSON loader whose failures show up at once as missing files or missing icons.

## pytweezer/logging_utils.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/logging_utils#1 | Each record is appended as one JSON line (timestamp, level, host, logger, module, message, and the traceback when `exc_info` is set) to `pytweezer-<date>.jsonl` in `$PYTWEEZER_LOG_DIR`. | 4 | unit | quiet: the log of a crashed run is missing or unparseable when someone goes looking for it | — |
| servers-config/logging_utils#2 | A `DailyJsonFileHandler` given an explicit `log_dir` writes its file in that directory. | 4 | unit | quiet: logs land in `<repo>/logs` instead (fails today, see bugs) | — |
| servers-config/logging_utils#3 | Calling `get_logger` many times attaches the message-hub and file handlers only once, so each record is written once. | 4 | unit | quiet: duplicated log lines that multiply with each import | — |
| servers-config/logging_utils#4 | `PYTWEEZER_LOG_LEVEL` sets the root level, and an unknown level name falls back to INFO. | 4 | unit | quiet: debug output missing, or a flood of it | — |

## pytweezer/parallel.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/parallel#1 | The calls really run at the same time: two calls that each wait for an event the other sets (an armed MotMaster waiting for the trigger the other fires) both finish. | 1 | unit | quiet: sequences run back to back, the armed MotMaster misses its trigger and the shot is lost | — |
| servers-config/parallel#2 | Results come back in the order the calls were given, whatever order they finish in. | 1 | unit | quiet: a camera frame is unpacked as a MotMaster return value and the other way round | — |
| servers-config/parallel#3 | `after(delay, call)` holds back only its own call by `delay`, and the other calls start at once. | 1 | unit | quiet: the trigger fires before the camera or follower is armed | — |
| servers-config/parallel#4 | When one call raises, `run_parallel` waits for the others to finish and then raises that call's own exception; when several raise, it raises an `ExceptionGroup` of all of them. | 4 | unit | mostly loud; quiet part: control returns while a sequence is still running | — |
| servers-config/parallel#5 | With `timeout=`, a call still running at the deadline makes `run_parallel` raise `TimeoutError` instead of blocking. | 4 | unit | loud: the experiment hangs | — |

## pytweezer/servers/analysis_manager.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/analysis_manager#1 | `start` launches the filter's script from `analysisdir` with the token `Analysis/<category>/<name>` and marks it active in Properties; `stop` terminates it and marks it inactive. | 4 | unit | quiet: the analysis reads the wrong property namespace and runs on defaults, or the GUI shows the wrong state | — |
| servers-config/analysis_manager#2 | Starting a filter that is already running replaces its process rather than running a second one. | 4 | unit | quiet: two copies publish the same derived stream, so every result appears twice | — |
| servers-config/analysis_manager#3 | `snapshot` lists every configured Image/Data filter with its script and streams, and reports as running only the processes that are still alive. | 4 | unit | quiet: the Analysis Manager tab shows a dead filter as running | — |
| servers-config/analysis_manager#4 | After a bad request (an unknown filter, a missing script, an unknown command, or a body that is not JSON), the service replies `ok: False` with the reason and goes on answering (parametrised). | 4 | integration | quiet: the service stops answering for good and the GUI's controls do nothing (fails today for non-JSON, see bugs) | — |

## pytweezer/servers/clients.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/clients#1 | An array sent by `DataClient`/`ImageClient` through a running hub arrives with the same dtype, shape and values, including a transposed (Fortran-ordered) image (parametrised over dtype and memory order). | 4 | integration | quiet: analyses and viewers get a scrambled image (fails today for Fortran-ordered arrays, see bugs) | — |
| servers-config/clients#2 | A subscriber receives messages under the topic `name + channel` (`CommandClient`: `name + " " + cmd`; with `prefix`: `prefix + "." + channel`) and nothing outside the prefixes it subscribed to. | 4 | integration | quiet: an analysis or applet waits for a stream that never arrives | — |
| servers-config/clients#3 | A message without an array arrives as a `(topic, dict)` pair, and `recv` returns `None` once the receive timeout passes with nothing sent. | 4 | integration | quiet: callers that branch on the tuple length (device control, datamgr) mis-parse or drop messages | — |

## pytweezer/servers/datalogger.py

No entries: nothing in the code reads the `active` property it publishes (StreamMonitor subscribes to the hub directly), so a maintainer would not miss a test; its update bug is listed below.

## pytweezer/servers/datamgr.py

No entries: nothing in CONFIG or the codebase imports or launches it, so it has no callers to protect; its bugs are listed below in case it is revived.

## pytweezer/servers/device_client.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/device_client#1 | `get_device(name)` reaches the right backend: each sub-device of a composite served on localhost by `device_server` answers as itself, and the composite's own name reaches its coordinator. | 4 | integration | quiet: a call meant for one device in a rig drives another | — |
| servers-config/device_client#2 | An explicit `host`, `port` or `target_name` given to `get_device` overrides the config's value. | 4 | unit | quiet: a notebook pointed at a test server talks to the lab device instead | — |

## pytweezer/servers/device_control.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/device_control#1 | A `request(action, device)` from one channel is emitted by every channel's `requested` signal, through a running Commandhub, as `(action, device, requester_host)`. | 4 | integration | quiet: starting or stopping a device from another PC does nothing | — |
| servers-config/device_control#2 | Messages on the channel with an unknown action or a device that is not a string are ignored rather than emitted. | 4 | unit | quiet: a stray command message starts or stops a device | — |

## pytweezer/servers/device_server.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/device_server#1 | `build_spec` passes the backend exactly the config keys that name its constructor parameters: other keys are ignored, and absent ones keep the class default. | 4 | unit | quiet: a serial number, timeout or stream name in config is not applied, and the driver runs on defaults or opens the wrong camera | — |
| servers-config/device_server#2 | The class built is `class` when not simulating, `sim_class` when simulating and it is given, and otherwise a generated stand-in, with the real class never constructed (parametrised). | 4 | unit | quiet: a simulated session opens real hardware, or a lab session runs a fake | — |
| servers-config/device_server#3 | Composite sub-devices inherit the composite's `simulate` flag unless they set their own. | 4 | unit | quiet: one device in a simulated rig is driven for real | — |
| servers-config/device_server#4 | A sub-device that fails to build is listed in `failed`, the other sub-devices are still served under their folded target names, and no coordinator is built. | 4 | unit | quiet: a half-built rig serves a coordinator that fails mid-experiment, or one dead SLM takes its camera down with it | — |
| servers-config/device_server#5 | The coordinator receives the backends keyed by each sub-config's `role`, which defaults to the device name. | 4 | unit | quiet: the coordinator drives the wrong backend in a rig whose devices are named differently | — |
| servers-config/device_server#6 | `device_index` addresses a sub-device at its composite's host and port with its folded target name, maps the composite's own name to the coordinator target, and raises `KeyError` for two names differing only in case or whitespace. | 4 | unit | quiet: a name resolves to the wrong device | — |
| servers-config/device_server#7 | A call that raises on the server is logged with the device, target, method and shortened arguments, and the caller still receives the exception. | 4 | integration | quiet: an error seen in a notebook leaves no trace in the device's log | — |
| servers-config/device_server#8 | A composite's teardown shuts the coordinator down before the sub-devices' teardowns, and one teardown raising does not stop the rest. | 4 | unit | quiet: hardware handles left open, so the next start fails to connect | — |
| servers-config/device_server#9 | Config errors (a nested composite, two sub-device names folding to one target, a coordinator name colliding with a sub-device) raise before any backend is constructed. | 4 | unit | loud: the server refuses to start, though hardware may already have been opened | — |

## pytweezer/servers/device_status.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/device_status#1 | For a reachable composite, each sub-device is `up` if its target is served and `failed` if not, and a composite whose coordinator target is missing is `degraded` (parametrised). | 4 | unit | quiet: the Devices tab shows a dead camera as up | — |
| servers-config/device_status#2 | When a composite that once answered fails a handshake (busy in a long call), its sub-devices keep their last known states; one that has never answered reports `unknown`. | 4 | unit | quiet: every sub-device of a rig flips to failed during each frame grab | — |
| servers-config/device_status#3 | An inactive device and its sub-devices are reported `disabled`, and an unreachable one `down`. | 4 | unit | quiet: a disabled device shows as broken, or the other way round | — |
| servers-config/device_status#4 | `server_targets` returns the set of target names a real sipyco server on localhost serves, and `None` (not an empty set) when nothing is listening. | 4 | integration | quiet: an unreachable rig reads as "all parts failed" | — |
| servers-config/device_status#5 | A `DeviceStatusClient` connected to a running `DeviceStatusServer` emits the `devices` dict of the newest snapshot. | 4 | integration | quiet: every PC's Devices tab freezes on stale states | — |

## pytweezer/servers/experiment_manager.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/experiment_manager#1 | Only one task runs at a time: a due task is not started while the previous worker process is still alive. | 1 | unit | quiet: two experiments drive the same devices at once | — |
| servers-config/experiment_manager#2 | A worker whose rid or token does not match the current worker record is told to terminate on every event. | 1 | unit | quiet: a stale worker from before a restart keeps driving devices alongside the new task | — |
| servers-config/experiment_manager#3 | `submit_recipe` refuses overrides that name arguments the experiment does not declare, and recipes that use arguments it no longer declares, instead of queuing the task (parametrised). | 1 | unit | quiet: the task runs with those arguments silently ignored | — |
| servers-config/experiment_manager#4 | `save_recipe` stores every unscanned declared argument, with defaults filled in and the user's values winning, and leaves scanned arguments out. | 1 | unit | quiet: replaying a recipe after a default changes runs with different values from those saved | — |
| servers-config/experiment_manager#5 | After a restart with a lost or stale `queue_state.json`, new rids continue above the highest rid already on disk. | 2 | unit | quiet: a reused rid collides with an existing file and DB run | — |
| servers-config/experiment_manager#6 | When a worker ends without reporting, its task takes the status and error in its h5 file if the file says it finished; otherwise the fallback status (crashed, aborted, interrupted) is written into the file and the queue (parametrised). | 2 | unit | quiet: a completed run is recorded as crashed, or a crashed file still says running | — |
| servers-config/experiment_manager#7 | Each worker point event with an index yields exactly one DB `points` row with the worker's index, times, scanned values and scalars, and the same row in the published points for that rid. | 2 | unit | quiet: points missing or misattributed in the database and the live plot | — |
| servers-config/experiment_manager#8 | A finished run's DB `runs` row takes its arguments (defaults included) and git commit from the h5 file, not from the submitted request or the manager's checkout. | 2 | unit | quiet: the database records arguments or code that the run did not use | — |
| servers-config/experiment_manager#9 | On restart, a still-running worker with the recorded pid and create time is adopted; a pid since reused by another process is not, and its task is settled as interrupted. | 4 | unit | quiet: the queue waits for ever on an unrelated process, or orphans a live worker | — |
| servers-config/experiment_manager#10 | A submit sent over the manager's REQ socket, as `ExperimentManagerClient` sends it, is queued and answered with its rid. | 4 | integration | loud: submit from notebooks and the GUI fails | — |

## pytweezer/servers/imagelogger.py

No entries: nothing in the code reads the `active` property it publishes, so a maintainer would not miss a test; its update bug is listed below.

## pytweezer/servers/__init__.py

No entries: imports, a shared ZMQ context and the sipyco patch call, which `sipyco_patches#2` already covers.

## pytweezer/servers/logger_server.py

No entries: name lookup and a registry dispatch whose every failure is a `KeyError` at logger start.

## pytweezer/servers/messageclient.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/messageclient#1 | A record logged with `get_logger` reaches a `MessageClient` subscribed to `"Logs"` through a running Messagehub as `(topic, payload)`, with its level, logger name and message. | 4 | integration | quiet: errors from device and server processes never reach the GUI's log view | — |
| servers-config/messageclient#2 | `send_error`/`send_warning`/`send_info`/`send_debug` log at the matching level, and an unknown level name logs an error naming it. | 4 | unit | quiet: a warning is shown as info, or a message vanishes | — |

## pytweezer/servers/process_cleanup.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/process_cleanup#1 | `find_device_processes(name)` matches the device argument exactly, so `"Cam"` never matches `"Cam 2"`, on both POSIX and quoted Windows command lines (parametrised). | 4 | unit | quiet: stopping one device also kills another device's server | — |
| servers-config/process_cleanup#2 | `find_stale_processes` finds processes started the way the GUI starts them (servers by their script, devices by `DEVICE_SERVER_SCRIPT`) and labels them with category and name, flags script-only matches, and skips unrelated processes and itself. | 4 | unit | quiet: kill-stale kills an unrelated process, or leaves a port-holder behind | — |

## pytweezer/servers/properties.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/properties#1 | A value set through `Properties` reaches the Properties server and appears in another connection's mirror. | 4 | integration | quiet: applet settings and stream subscriptions differ between PCs | — |
| servers-config/properties#2 | `get` of a missing key creates it with the default and returns the default, with an option-property default unwrapped to its value. | 4 | unit | quiet: the tree stops building itself, and viewers lose their settings | — |
| servers-config/properties#3 | With the server unreachable, `Properties` starts from the saved file and writes apply locally; when the server comes up, its tree replaces the local edits. | 4 | integration | quiet: a stale offline edit overwrites newer shared state, or a GUI starts empty | — |
| servers-config/properties#4 | `changes()` returns each changed key with its parents once and then clears, and a remote change to an option property's value is reported under the property's path, not `.../value`. | 4 | unit | quiet: viewers miss setting changes made elsewhere | — |
| servers-config/properties#5 | A `PropertyAttribute` reads and writes through its owner's `Properties`, under its name and with its default. | 4 | unit | quiet: a viewer setting is not saved or not shared | — |

## pytweezer/servers/property_server.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/property_server#1 | Edits survive a restart: after `set`/`delete` through the store and `close()`, a new `PropertyServer` on the same file serves the edited tree. | 4 | integration | quiet: shared settings revert at every restart | — |
| servers-config/property_server#2 | An edit reaches the file within the save window without `close()` being called (the GUI hard-kills the process on Windows). | 4 | integration | quiet: every edit since startup is lost when the server is stopped | — |
| servers-config/property_server#3 | The store refuses a value that is not JSON-serialisable and leaves the tree unchanged. | 4 | unit | quiet: every later save fails, and no further edit reaches the file | — |

## pytweezer/servers/property_tree.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/property_tree#1 | `parse_key` places relative keys under the owner's name and absolute keys from the root, passes lists through, and drops a trailing `/` (parametrised). | 4 | unit | quiet: settings are written to and read from different places (trailing `/` on absolute keys fails today, see bugs) | — |
| servers-config/property_tree#2 | Setting an option property to a plain value outside its options plans nothing; one inside them changes only `"value"`; a whole option dict replaces it (parametrised). | 4 | unit | quiet: a dropdown property loses its options or takes an invalid value | — |
| servers-config/property_tree#3 | `lookup` unwraps an option property to its value and returns a deep copy, so changing the result leaves the tree alone. | 4 | unit | quiet: a caller's local edit silently changes the shared mirror | — |
| servers-config/property_tree#4 | `plan_set` creates every missing intermediate level, and `apply_ops` gives the same tree whether it runs on a dict or a `Notifier`. | 4 | unit | quiet: the client's copy and the server's tree drift apart | — |
| servers-config/property_tree#5 | `plan_delete` of a path that does not exist plans nothing. | 4 | unit | loud: a delete raises on the server | — |

## pytweezer/servers/reachability.py

No entries: a direct wrapper of `socket.create_connection`, with no pytweezer logic of its own.

## pytweezer/servers/simulated_device.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/simulated_device#1 | `default_simulated(cls)` accepts any constructor arguments and never runs the real class's code, because it does not subclass it. | 4 | unit | quiet: a simulated session opens real hardware | — |
| servers-config/simulated_device#2 | `@simulate(real)` gives the decorated class every public method of the real class, leaving methods written on it or inherited from a simulated base untouched. | 4 | unit | quiet: a simulated device lacks a method the experiment calls, or a hand-written fake is replaced by a no-op | — |
| servers-config/simulated_device#3 | A `defaults` entry that is callable produces a fresh return value on each call, and a plain one is returned as given. | 4 | unit | quiet: simulated calls share one mutable return value | — |

## pytweezer/servers/sipyco_patches.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/sipyco_patches#1 | A reply of several megabytes, with a non-ASCII character across a 64 kB chunk boundary, reads back through a sipyco `Client` identical to what the server returned. | 2 | integration | loud: frame grabs over RPC fail to decode | — |
| servers-config/sipyco_patches#2 | Importing `pytweezer.servers` installs the linear-time reader into `sipyco.pc_rpc`, and applying it twice changes nothing. | 4 | unit | quiet: a 100-frame grab takes minutes rather than under a second | — |

## pytweezer/servers/sync.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/sync#1 | A mirror gets the full state on connect and then each mutation in order, and after nested mutations (append a dict, edit inside it, delete a key) it equals the notifier's `raw_view`. | 4 | integration | quiet: GUIs show a queue or points that drift from the manager's | — |
| servers-config/sync#2 | After the server restarts on the same port, the mirror reconnects, reports `connected` again, and its data, local edits included, is replaced by the new snapshot. | 4 | integration | quiet: a GUI keeps showing pre-restart state | — |
| servers-config/sync#3 | With `keep_mods`, `drain()` returns each mod once and in order, and `on_mod` sees each mod after it has been applied. | 4 | integration | quiet: the experiment feed misses or repeats points | — |

## pytweezer/servers/xsub_xpub.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| servers-config/xsub_xpub#1 | Stale-instance cleanup before binding terminates only `xsub_xpub.py` processes for the same hub name, not other hubs. | 4 | unit | quiet: starting one hub kills another, and its streams stop | — |

## Suspected bugs

- pytweezer/servers/clients.py: `GenericClient.send` and `ImageClient.send` send the array's buffer in memory order, but `recv` rebuilds it with `frombuffer(...).reshape(shape)` in C order. A Fortran-ordered array (any `frame.T`, or a driver returning F-order) therefore arrives scrambled with no error. A non-contiguous slice raises instead. Checked against pyzmq — smallest input: `ImageClient("cam").send({}, np.arange(6, dtype=np.uint16).reshape(2, 3).T)` is received as `[[0, 1], [2, 3], [4, 5]]` instead of `[[0, 3], [1, 4], [2, 5]]`.
- pytweezer/servers/analysis_manager.py: in `serve_forever`, a request that is received but fails to decode is followed by `continue` without a reply. The REP socket is then stuck in its send state: every later `recv_json` raises `EFSM`, which is caught and logged, and the loop spins at full CPU and never answers again. Checked against pyzmq — smallest input: a REQ socket sends `b"x"` to the Analysis Manager port.
- pytweezer/servers/experiment_manager.py: `handle` calls `request.get("command")` outside its `try`, and `serve_forever` calls `recv_json` without one. Valid JSON that is not a dict, or a body that is not JSON, raises out of `serve_forever` and stops the manager. Its worker survives and is adopted on restart — smallest input: `send_json([])` to the manager's REQ port.
- pytweezer/logging_utils.py: `DailyJsonFileHandler._ensure_path` creates `self._log_dir` but takes the file path from `get_daily_log_path(today)`, which uses `$PYTWEEZER_LOG_DIR` or `<repo>/logs`. An explicit `log_dir` is ignored — smallest input: `DailyJsonFileHandler(log_dir=Path(tmp)).emit(record)` with `PYTWEEZER_LOG_DIR` unset creates an empty `tmp` and appends to `<repo>/logs/pytweezer-<date>.jsonl`.
- pytweezer/servers/property_tree.py: `parse_key` strips a trailing `/` only from relative keys. `Properties.get` strips it first, but `set` and `delete` do not, so the two address different entries — smallest input: `parse_key("x", "/a/b/")` returns `["a", "b", ""]`, while `parse_key("x", "a/b/")` returns `["x", "a", "b"]`. So `Properties("x").set("/a/b/", 1)` writes the key `""` under `/a/b`.
- pytweezer/servers/device_server.py: in `_make_composite`, two sub-devices with the same `"role"` silently overwrite each other in `roles`, so the coordinator gets only the last one. The other config errors fail loudly — smallest input: a composite with `"devices": {"A": {..., "role": "camera"}, "B": {..., "role": "camera"}}` and a coordinator; it receives `{"camera": <B>}` without any error.
- pytweezer/servers/imagelogger.py: `updated` is reset at the top of every loop iteration, so `active` is published only when a message arrives at least 1 s after the last publish. The last messages before a stream goes quiet are never published — smallest input: one message 0.5 s after start, then silence; `active` is never set.
- pytweezer/servers/datalogger.py: the same pattern (`self.updated = False` on every iteration), so a stream that sends a short burst and stops is never published. `active` changes only when the entry ages out after `maxStreamInactive[s]` (1000 s) — smallest input: one message 0.5 s after start, then silence.
- pytweezer/servers/datalogger.py, pytweezer/servers/imagelogger.py: `__main__` hardcodes the names `"Servers/DataStream"` and `"Servers/ImageStream"` and ignores the CLI argument. This breaks the `python <script> <name>` launch protocol — smallest input: `python datalogger.py Datalogger` still uses `Properties("Servers/DataStream")`.
- pytweezer/servers/datamgr.py (unused): `lastupdate` starts at 0, so the `timeout_without_endtime` branch in `_processCompleteData` fires at once. The first run is moved to `completeData` as soon as its `Experiment.start` arrives, and so is any run that starts more than 60 s after the last completion. Their data is then discarded as "arrived after timeout" — smallest input: on a fresh `DataSummary`, `processIncomingData("Experiment.start", {"_task": 1, "_starttime": time.time()}, None)` leaves `completeData == [start entry]`, with `_streams == ["Experiment.start"]`.
- pytweezer/servers/datamgr.py (unused): when column names collide, `get_unique_flist` returns bare names without dtypes, so `np.dtype(...)` raises. String widths are also taken from the last run only, which truncates longer earlier strings — smallest input: `_determineFormatstring([{"a": [1, 2], "a1": "x"}])` raises `TypeError`, and `[{"s": "long"}, {"s": "x"}]` gives width `S1`.
- pytweezer/servers/property_server.py: `_terminate_stale_instances` SIGTERMs and then SIGKILLs any process on the PC whose command line contains `property_server.py`, including editors, pagers or a pytest run naming the file. `xsub_xpub._terminate_stale_instances` has the same issue with `xsub_xpub.py` plus a substring match on the hub name — smallest input: `less pytweezer/servers/property_server.py` is killed when the Properties server starts.
- pytweezer/configuration/config.py: possibly the skill that is stale rather than the code. The `pytweezer-architecture` skill says the Experiment Manager is appended after the `CONFIG` literal so its ports come last. Here it sits inside `CONFIG["Servers"]`, before `"Devices"`, so its two ports shift every device port — smallest input: `CONFIG["Devices"]["Rb MotMaster"]["port"] == 7292`, where an appended entry would leave it at 7290.
