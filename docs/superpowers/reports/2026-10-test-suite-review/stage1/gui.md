# Stage 1: GUI

## bin/gui.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/gui#1 | `build_gui(server)` builds all nine tabs offscreen, in order, each a dock with the stable objectName `dock:<label>` (parametrised over server/client). | 4 | gui | Loud: the window won't open or a tab is missing; with an unstable objectName the saved layout quietly stops restoring. | — |
| gui/gui#2 | `_safe_panel` returns a placeholder label when a panel's factory raises, so the other tabs still build. | 4 | unit | Loud: one unreachable service (Analysis Manager, Properties) takes the whole control window down. | — |
| gui/gui#3 | `TabbedGUI.closeEvent` calls `close()` on every embedded panel, so the panels' own teardown (terminating child processes) runs. | 4 | gui | Quiet: device servers and loggers are left running as orphans after the GUI closes and hold their ports until someone runs kill-stale. | — |
| gui/gui#4 | Results' `resubmit_requested` reaches `ExperimentsPanel.load_request`, so "Resubmit" fills the Experiments form. | 4 | gui | Loud: the Resubmit button does nothing. | — |

## bin/managed_panel.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/managed_panel#1 | After a start, a feed-driven row reads "starting" through "down"/"stopped" feed readings within `_STARTING_GRACE_S`, switches to running on "up", and accepts the feed once the grace period has run out. | 4 | unit | Quiet: a device that is coming up reads Stopped and gets started a second time, or a failed start reads Starting for ever. | — |
| gui/managed_panel#2 | While starting, a row whose own subprocess has already exited reads "crashed" immediately instead of waiting out the grace period. | 4 | unit | Quiet: a driver that dies on init shows "Starting" for 30 s and the operator waits on a dead process. | — |
| gui/managed_panel#3 | A remote stop shows "stopping" until the feed stops reporting a running state, then shows the feed's state (or the feed's state once `_STOPPING_GRACE_S` has passed). | 4 | unit | Quiet: a device on another PC still reads Running after a stop, or reads Stopped while it is still serving. | — |
| gui/managed_panel#4 | `DevicesPanel._handle_request` ignores a request that matches the row's current state (start of a running device, stop of a stopped one). | 4 | unit | Quiet: a repeated click from another PC restarts a camera or sequencer mid-run. | — |
| gui/managed_panel#5 | `DevicesPanel._handle_request` acts only on rows this PC controls; requests for other hosts' devices and for composite sub-devices are ignored. | 4 | unit | Quiet: two PCs both spawn the same device server, and the second fails on its port or drives the hardware twice. | — |
| gui/managed_panel#6 | Toggling a row whose device runs on another PC calls `remote_request(action)` and never spawns a local process. | 4 | unit | Quiet: a device server is started on the wrong PC, with no hardware attached to it. | — |
| gui/managed_panel#7 | `stopProcess` on a row that is running according to the feed but did not spawn the process hands the name to `stop_orphan`; `terminateProcess` (panel close) never does. | 4 | unit | Quiet: the Stop button does nothing to a device left behind by a crashed GUI, or closing a GUI kills devices it doesn't own. | — |
| gui/managed_panel#8 | `DevicesPanel.check_host` matches this PC's address while ignoring case and surrounding whitespace, and treats a missing host as not local. | 4 | unit | Quiet: a device is spawned nowhere, or on every PC. | — |
| gui/managed_panel#9 | `ControlPanel` in view-only mode maps probe results to up/down, and shows "unprobeable" (n/a) for entries with no port. | 4 | unit | Quiet: a client shows a server as Stopped (or Running) when it cannot actually see it. | — |
| gui/managed_panel#10 | `DevicesPanel._apply_snapshot` applies each snapshot entry, sub-devices included, to the row of the same name, with the address shown as `host:port` and last-seen shown as an age. | 4 | unit | Quiet: a sub-device that failed to open reads as up, because its state lands on the wrong row or none. | — |
| gui/managed_panel#11 | A degraded composite counts as running: its button reads Stop, so it can't be started twice. *Added from stage 2; covered by `tests/test_gui.py::test_managed_row_toggle_reflects_controllability_and_state`.* | 4 | unit | quiet: a second copy is started on top of the first | — |

## pytweezer/GUI/analysismanager.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/analysismanager#1 | `filter_state` gives running when the filter is running, crashed when it is active but not running, and stopped otherwise (parametrised). | 4 | unit | Quiet: a filter whose process died reads as Stopped, so nobody notices the analysis stream has stopped. | — |
| gui/analysismanager#2 | `FilterTable.update_filters` keeps the same filter key selected when rows are added, removed or reordered by a new snapshot. | 4 | gui | Quiet: Start/Stop or Configure acts on a different filter from the one the user picked. | — |
| gui/analysismanager#3 | `toggle_filter` sends `set_active` with the negation of the selected filter's running state, for that filter's category and name. | 4 | unit | Quiet: Stop starts the filter, or the command goes to another filter. | — |
| gui/analysismanager#4 | `add_filter` refuses an existing `category/name` without calling the manager; otherwise it sends `add_filter` with the chosen script, and `["nostream"]` when no stream is chosen. | 4 | unit | Quiet: an existing filter and its tuned properties are replaced. | — |
| gui/analysismanager#5 | `AnalysisManagerClient.request` returns `{"ok": False, ...}` when nothing answers within the timeout, and the next request succeeds once a manager is listening (the REQ socket is reset). | 4 | integration | Quiet: after one timeout the tab reads "not responding" for ever, even with the manager up. | — |
| gui/analysismanager#6 | `AddFilterSheet._scan_scripts` puts a script under Image or Data from the stream keys it mentions, and leaves out scripts that mention both or neither. | 4 | unit | Quiet: a script appears under the wrong type and is fed the wrong kind of stream. | — |

## pytweezer/GUI/applet_launcher.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/applet_launcher#1 | Starting or stopping an applet changes only the local QSettings active set; the shared Properties `"Applets"` catalogue never gains an `"active"` flag. | 4 | unit | Quiet: starting a viewer on one PC starts it on every lab PC next session. | — |
| gui/applet_launcher#2 | A legacy `"active"` key in the Properties catalogue is dropped on load, so the next save doesn't republish it. | 4 | unit | Quiet: an old shared running flag survives the migration and keeps spreading. | — |
| gui/applet_launcher#3 | The local active set survives a QSettings round trip for zero, one and several names (parametrised; QSettings turns a one-item list into a str and an empty one into None). | 4 | unit | Quiet: the applets running last session are not restored, or one is restored as a set of single characters. | — |
| gui/applet_launcher#4 | On construction the launcher starts the applets in the active set that are still in the catalogue, and drops names no longer in the catalogue from the saved set. | 4 | unit | Quiet: viewers are not restored, or a deleted applet name stays in the set for ever. | — |
| gui/applet_launcher#5 | `refresh_status` marks an applet whose process has exited as stopped and removes it from the active set. | 4 | unit | Quiet: a closed viewer shows Running, and is relaunched next session. | — |
| gui/applet_launcher#6 | `closeEvent` terminates every applet process but keeps the active set, so the next session restores them. | 4 | unit | Quiet: closing the GUI either orphans viewers or forgets which were open. | — |
| gui/applet_launcher#7 | `_start_applet` with a script path that does not exist shows "Missing", spawns nothing, and removes the name from the active set. | 4 | unit | Loud: a Python traceback in a child process; no row change. | — |
| gui/applet_launcher#8 | `unique_name` returns the base name if it is free, otherwise the lowest free `"<base> N"` starting at 2. | 4 | unit | Quiet: two applets share a name and so share one Properties namespace and window geometry. | — |

## pytweezer/GUI/applet.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/applet#1 | `stream_color(stream, index)` returns the per-stream `<stream>/color` property when one is set, and the `index`-th palette colour (cycling) otherwise. | 4 | unit | Quiet: every curve gets the same colour, or a colour the user chose is ignored, so streams are confused on a shared plot. | — |

## pytweezer/GUI/components.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/components#1 | `CompactDoubleSpinBox.textFromValue` strips only trailing fractional zeros: 1.5 gives "1.5", 100.0 gives "100", 0.0 gives "0", and decimals=0 leaves "10" alone (parametrised). | 4 | unit | Quiet: an argument field reads "1" for 100, and the operator submits a value they misread. | — |

## pytweezer/GUI/experiments/arg_editor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/arg_editor#1 | A number argument shown in display units is returned in SI by `request()`: 5 in a MHz field (scale 1e6) gives 5e6 in `args` (parametrised over scales). | 1 | unit | Quiet: the experiment runs at a value a factor `scale` off. | — |
| gui/arg_editor#2 | An untouched form's `request()` gives every argument exactly its schema default, including SI values smaller than the displayed decimals (e.g. `Number(2e-7)`). | 1 | unit | Quiet: a submitted pulse time or small field is silently rounded to 0. | — |
| gui/arg_editor#3 | A scanned argument is left out of `args` and appears as a scan axis whose start/stop (linear) or values (list) are in SI. | 1 | unit | Quiet: the experiment runs at a fixed value, or the scan covers the wrong range. | — |
| gui/arg_editor#4 | `parse_list` turns comma-separated display-unit text into SI values of the argument's kind (number×scale, integer, the bool spellings true/false/1/0, choice). | 1 | unit | Quiet: a list scan runs at values off by the scale factor, or with bools read the wrong way. | — |
| gui/arg_editor#5 | `parse_list` raises ValueError for an empty list, a value outside min/max (compared in SI), a choice that is not an option, or an unknown bool spelling (parametrised). | 1 | unit | Quiet: an out-of-range scan point reaches the hardware. | — |
| gui/arg_editor#6 | `load_request(r)` followed by `request()` gives back `r`'s args, scan axes, repetitions, order, repeat, priority and label, MOTMaster search rows included. | 1 | unit | Quiet: "Last submitted", a draft or Resubmit puts different settings in the form from the ones loaded, and those get submitted. | — |
| gui/arg_editor#7 | `load_request` ignores arguments and axes the experiment no longer declares, and fills in the rest. | 1 | unit | Loud: loading an old request fails, or quietly drops the arguments it does know. | — |
| gui/arg_editor#8 | `request()` raises ValueError when a searched Int32 script parameter has a non-whole value, list value or linear-scan point; `validate=False` skips the check. | 1 | unit | Quiet: a non-whole value is sent to an Int32 MOTMaster parameter. | — |
| gui/arg_editor#9 | `add_motmaster_row` types the row from the script default (int gives integer, float gives number), uses the given value when the script lacks the parameter, and raises when neither is known. | 1 | unit | Quiet: an Int32 parameter is sent as a float, or a row is created with no value behind it. | — |
| gui/arg_editor#10 | A MOTMaster row added before the script's parameters arrived keeps its typed value and scan once the parameters arrive and the row is rebuilt (`_parameters_fetched`). | 1 | unit | Quiet: the user's edit is replaced by the script default just before they press Submit. | — |
| gui/arg_editor#11 | Removing a scanned MOTMaster row also removes its scan axis. *Added from stage 2; covered by `tests/test_experiment_gui.py::test_a_searched_parameter_can_be_scanned_removed_and_reset`.* | 1 | unit | quiet: the run still scans a parameter the user removed | — |

## pytweezer/GUI/experiments/catalogue_view.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/catalogue_view#1 | Each recipe is listed under the experiment whose (module, class_name) it names, and only there. | 4 | gui | Quiet: a recipe is loaded into the form of a different experiment. | — |
| gui/catalogue_view#2 | Rebuilding after new modules or recipes arrive keeps the selected experiment or recipe selected, and emits no selection signal. | 4 | gui | Quiet: a catalogue refresh reloads the form under the user and discards their edits. | — |
| gui/catalogue_view#3 | The filter keeps a class and its module visible when only one of its recipes matches, and shows everything under a matching module. | 4 | gui | Quiet: a saved recipe looks as if it is gone. | — |
| gui/catalogue_view#4 | Clicking the recipe that is already current emits `recipe_selected` again, so it reloads over edits; moving to a different item emits once. | 4 | gui | Quiet: clicking the recipe does not reset the form, and the user submits their edits thinking they are the recipe. | — |

## pytweezer/GUI/experiments/feed.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/feed#1 | Points that a manager-side notifier publishes for the running task arrive through a real `ExperimentFeed` as `point_received` with that task's rid, index, values and scalars (Experiment Manager → GUI, normal case). | 4 | integration | Quiet: the Results tab's live plot of a running measurement stays empty or stale. | — |
| gui/feed#2 | Rows appended after the manager replaces `points` for a new task carry the new rid, in the order the mods were drained. | 4 | unit | Quiet: one run's live points are plotted onto another run. | — |
| gui/feed#3 | A drain that holds only `alive` heartbeats emits no `queue_changed`; any other change, and every init after a (re)connect, emits it once. | 4 | unit | Quiet: the queue view misses changes, or redraws every 2 s and loses the selection. | — |
| gui/feed#4 | `connected` turns False when no mod has arrived for `STALE_AFTER_S`, even though the socket is still connected. | 4 | unit | Quiet: a hung manager still reads "Connected". | — |
| gui/feed#5 | `points(rid)` returns the held rows only for the rid they belong to, and `[]` for any other rid. | 4 | unit | Quiet: an old measurement is plotted with the running one's points. | — |

## pytweezer/GUI/experiments/__init__.py

No entries: the file holds only a docstring.

## pytweezer/GUI/experiments/motmaster_params.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/motmaster_params#1 | `MotMasterBox.choose` emits `parameter_chosen` only for a name in the numeric (non-bool) script defaults that the experiment does not already declare, and parses the "name · type · default" completer label. | 1 | unit | Quiet: a declared parameter is added a second time with a conflicting value, or a bool parameter is sent as a number. | — |
| gui/motmaster_params#2 | `ParameterFetcher` calls the source once per (device, script): repeats are served from the cache and concurrent duplicates are not refetched, while a failure is not cached and a retry calls the source again. | 4 | unit | Quiet: the device is hammered with RPCs, or a transient failure leaves the box broken until the GUI restarts. | — |
| gui/motmaster_params#3 | `DeviceParameterSource` reads parameters from the simulated device when `simulated()` is True, and never builds a real device client. | 4 | unit | Quiet: a simulation session on a laptop opens RPC connections to lab hardware, or hangs on timeouts. | — |
| gui/motmaster_params#4 | Parameters fetched on a worker thread reach the widgets on the GUI thread. *Added from stage 2; covered by `tests/test_experiment_gui.py::test_a_threaded_fetch_is_delivered_on_the_gui_thread`.* | 4 | unit | quiet: intermittent crashes that are hard to trace back | — |

## pytweezer/GUI/experiments/panel.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/panel#1 | Pressing Submit hands `client.submit` the editor's request unchanged, apart from `submitter` being set to the current user (fake client). | 1 | integration | Quiet: the queued task differs from what the form shows. | — |
| gui/panel#2 | The recipe "Submit now" action calls `submit_recipe` with that recipe's experiment, class_name and name. | 1 | unit | Quiet: a different recipe or experiment is queued. | — |
| gui/panel#3 | Saving a recipe under a name that is already taken asks first; declining sends nothing, and accepting sends `overwrite=True` with the current user as submitter. | 4 | unit | Quiet: a recipe shared with every PC is overwritten without warning. | — |
| gui/panel#4 | Switching to another experiment and back restores the unsubmitted draft; after a successful submit the draft is dropped, and the next selection loads the last submitted request. | 4 | gui | Quiet: the form shows stale or unexpected values, which then get submitted. | — |
| gui/panel#5 | Selecting an experiment that has no draft loads the manager's last request for that same (module, class_name). | 4 | unit | Quiet: the form starts from defaults, or from another class's settings. | — |
| gui/panel#6 | `load_request` for an experiment not in the catalogue shows an error and leaves the form alone; otherwise it selects that experiment and fills the form with the request. | 4 | unit | Quiet: Resubmit loads into whichever experiment happened to be open. | — |
| gui/panel#7 | A queue snapshot with `simulated` set shows the banner and switches MOTMaster parameter lookups to the simulated devices. | 4 | unit | Quiet: in simulation the form reads parameters from (or times out on) real hardware. | — |
| gui/panel#8 | A queue snapshot whose `catalogue_version` or `recipes_version` differs from the last one seen refetches the catalogue or the recipes. | 4 | unit | Quiet: a new or edited experiment class never appears until the GUI is restarted. | — |
| gui/panel#9 | Deleting a shared recipe asks for confirmation first. *Added from stage 2; covered by `tests/test_experiment_gui.py::test_deleting_a_recipe_needs_confirmation`.* | 4 | unit | quiet: one mis-click deletes a recipe on every PC | — |

## pytweezer/GUI/experiments/queue_view.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/queue_view#1 | After a new snapshot moves the selected task to another row, the selection follows its rid, and the buttons act on that task. | 1 | gui | Quiet: Terminate, Abort or Delete hits a different task. | — |
| gui/queue_view#2 | When the selected task is no longer in the snapshot, no task is selected and the action buttons are disabled. | 1 | gui | Quiet: a second click on Delete or Abort acts on whichever task moved into that row. | — |
| gui/queue_view#3 | Abort asks for confirmation and emits nothing if declined; Priority +/− emits `set_priority` with the task's priority ±1. | 1 | gui | Quiet: a running measurement is killed on a mis-click, or a task jumps the queue by the wrong amount. | — |
| gui/queue_view#4 | Each control is enabled only for the statuses it applies to (pause for running, delete for queued/held, release for held, ...) (parametrised over statuses). | 4 | gui | Quiet: a command the manager can't honour is sent, or a needed one is unavailable. | — |
| gui/queue_view#5 | Rows run running task first, then queue, then history, and show progress as `done/total` and a pending request as "running (pause requested)". | 4 | gui | Quiet: the operator misreads what is running or how far it has got. | — |

## pytweezer/GUI/experiments/results.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/results#1 | `resubmission(measurement)` for a file written by `run_local` gives the original request's experiment, class, non-scanned args, scan (shuffle seed included) and label. | 1 | integration | Quiet: "Resubmit" repeats a different measurement from the one shown. | — |
| gui/results#2 | The quick plot of a finished file shows y against the scanned argument in display units (SI / scale), with the mean and standard error over repetitions at each x. | 3 | gui | Quiet: the plot's x axis is off by the scale factor, or the error bars are wrong, and that misleads a decision mid-run. | Synthetic file, 2 repetitions with y = 3·x_SI ± δ: plotted x = x_SI/scale, mean = 3·x_SI, SEM = δ (sample std δ√2 over √2). |
| gui/results#3 | One curve per value of the chosen series argument, each labelled `<series> = <display value>`. | 4 | gui | Quiet: two series are merged into one, or labelled with the wrong values. | — |
| gui/results#4 | While a measurement is running and the feed is connected, each received point is plotted at its own planned index, with points not yet measured left as NaN (shuffled scan). | 4 | gui | Quiet: during a shuffled scan the live plot puts values at the wrong x. | — |
| gui/results#5 | When the feed reports the selected running task finished, the file is reloaded and the plot switches from the feed's points to the file's results. | 4 | gui | Quiet: the plot stays on partial live data after the run ends. | — |
| gui/results#6 | Running files are left out of the tree unless "Show running" is ticked, and a file that has finished appears on the next refresh. | 4 | gui | Quiet: a finished measurement never shows up in the tree. | — |
| gui/results#7 | A selected file that can't be read yet is retried on the poll until it loads. | 4 | gui | Quiet: a just-started measurement stays "unreadable" until it is clicked again. | — |
| gui/results#8 | A scan over string values is plotted at integer positions labelled with those values, in the order they first appear. | 4 | gui | Loud: a crash, or an empty plot for categorical scans. | — |
| gui/results#9 | A selected running measurement reloads on refresh without a live feed, keeping the user's choices. *Added from stage 2; covered by `tests/test_results_browser.py::test_selected_running_measurement_reloads_and_keeps_choices`.* | 4 | unit | quiet: the display stays on stale data | — |

## pytweezer/GUI/fitfunctions.py

No entries: nothing in the repository imports it, so no caller relies on it (but see the TOF mass under Suspected bugs before anyone revives it).

## pytweezer/GUI/grafana.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/grafana#1 | `run_url` gives a window from start − margin to end + margin in epoch ms, with margin = max(10 % of the duration, 30 s) (parametrised over a short and a long run). | 4 | unit | Quiet: Grafana shows readings from outside the run next to its points, or cuts some off. | Hand-computed epoch ms for fixed UTC datetimes (e.g. a 10 min run gives a 60 s margin, a 1 min run gives 30 s). |
| gui/grafana#2 | A missing or empty `t_end` ends the window at `now`. | 4 | unit | Loud: a crash, or an empty window for a running task. | — |
| gui/grafana#3 | ISO strings with different UTC offsets that name the same instant give the same `from`/`to`. | 4 | unit | Quiet: the window is shifted by the offset in hours. | The same instant written in +00:00 and +01:00. |

## pytweezer/GUI/__init__.py

No entries: it only re-exports `tweezerpath`.

## pytweezer/GUI/models.py

No entries: nothing imports these ARTIQ-era table models (but see Suspected bugs).

## pytweezer/GUI/property_editor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/property_editor#1 | Editing a str- or list-valued entry evaluates the text and writes the resulting Python value back, so a list stays a list and a quoted string stays a str (parametrised). | 4 | gui | Quiet: an applet's stream list becomes the string "['A', 'B']" and the applet subscribes to nothing useful. | — |
| gui/property_editor#2 | An int, float or bool edit in a `PropEdit` opened on a subtree (e.g. `/Analysis/Image/foo/`) is written to that subtree's absolute key. | 4 | gui | Quiet: a filter threshold edit lands in a different key, and the analysis keeps its old value. | — |
| gui/property_editor#3 | An option property (a dict with `options`) is shown as a choice list, and picking an option writes it as the property's value, keeping its options. | 4 | gui | Quiet: the options are lost and the property turns into a bare value. | — |
| gui/property_editor#4 | An int colour property ending in `color` is shown as the colour that QColor reads from that ARGB int, and editing it writes back an int for the colour picked. | 4 | gui | Quiet: an ROI or curve colour changes channel order (red shown as blue). | — |
| gui/property_editor#5 | Delete removes the selected entry's full absolute key under the subtree, and nothing else. | 4 | gui | Quiet: a sibling entry or a whole parent is deleted for every PC. | — |
| gui/property_editor#6 | Loading a JSON file sets each top-level key in it under the editor's subtree. | 4 | gui | Quiet: a saved configuration is restored into the wrong place. | — |
| gui/property_editor#7 | A change made elsewhere to a key under the subtree is shown in the tree on the next `update()`. | 4 | gui | Quiet: the editor shows a stale value, and the user's next edit is based on it. | — |

## pytweezer/GUI/pytweezerQt.py

No entries: `BWidget` only restores and saves geometry, which is checked by eye, and `BFrame`, `BMainWindow` and `SearchComboBox` have no callers.

## pytweezer/GUI/qled.py

No entries: nothing imports `LedIndicator`.

## pytweezer/GUI/streammonitor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/streammonitor#1 | `LogMonitor` starts by showing the newest `max_rows` records of today's log, newest first like live records, and skips lines that are blank, not JSON, or not objects. | 4 | gui | Quiet: the latest error before a restart is buried at the bottom, and live records push the newest backfilled ones out. | — |
| gui/streammonitor#2 | The level filter "Warnings and errors" hides INFO and DEBUG, and counts unknown level names as INFO; the text filter is case-insensitive (parametrised). | 4 | gui | Quiet: errors are hidden, or the filter appears to do nothing. | — |
| gui/streammonitor#3 | `StreamMonitor` keeps the last 40 messages that arrive while the tab is hidden and shows them newest first when it is shown; messages that arrive while paused are dropped. | 4 | gui | Quiet: the tab shows stale traffic, or none, when opened. | — |

## pytweezer/GUI/subscription_editor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/subscription_editor#1 | Adding a stream from the combo writes the bare stream name, without the "[N s]" age suffix, to the stream-list property, and adds nothing if the name is already there. | 4 | gui | Quiet: the applet subscribes to "Cam[3 s]" and never shows anything. | — |
| gui/subscription_editor#2 | Deleting the selected row removes exactly that stream from the property. | 4 | gui | Quiet: a different stream is unsubscribed. | — |
| gui/subscription_editor#3 | The property key defaults to `<category>.lower() + "streams"`, and `streamkey` (with `propprefix`) replaces it. | 4 | unit | Quiet: the dialog edits a key the applet never reads, e.g. the mask list instead of the image list. | — |

## pytweezer/GUI/table_parameter.py

No entries: nothing creates a `"table"` parameter, and `makeWidget` uses Qt4-era API that no longer exists under PyQt6.

## pytweezer/GUI/theme.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/theme#1 | `resolve_state` maps every status the panels and the device-status feed produce (running, up, starting, stopping, degraded, stopped, down, crashed, failed, disabled, unknown, unprobeable, True, False) to itself or up/down, and anything else to "unknown" (parametrised). | 4 | unit | Quiet: a failed sub-device or a degraded rig reads "Unknown" in grey. | — |

## pytweezer/GUI/viewers/archive/zmq_ROI.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/zmq_ROI#1 | After the ROI is moved or resized, the next tick writes the new pos and size to `<name>/pos` and `<name>/size` in plot coordinates. | 4 | gui | Quiet: `roi_slice.py` keeps cropping the old region while the viewer shows the new one. | — |
| gui/zmq_ROI#2 | A change to `<name>/pos` or `size` made elsewhere moves the ROI on the next tick, and the ROI does not then write its old geometry back. | 4 | gui | Quiet: two viewers of the same ROI fight, and the region analysed flips between them. | — |
| gui/zmq_ROI#3 | The ROI registers `"/<name>"` in `/Servers/roilist` exactly once. | 4 | unit | Quiet: the ROI is missing from the editors' ROI choice list, or listed several times. | — |

## pytweezer/GUI/viewers/data_sidebar.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/data_sidebar#1 | `update()` drains the queue and shows only the newest header's fields, leaving out `dtype*` keys, with floats to 6 significant figures and lists joined. | 4 | gui | Quiet: the sidebar shows a stale shot's statistics next to the current image. | — |

## pytweezer/GUI/viewers/image_monitor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/image_monitor#1 | A frame with `_offset` and `_imgresolution` is drawn with array element [row, col] at physical ((col + off_x)·res_x, (row + off_y)·res_y), the same convention `roi_slice.py` uses to crop. | 4 | gui | Quiet: an ROI drawn over the atoms crops a different region of the array, so the counts are from the wrong place. | Non-square frame with one bright pixel at a known [row, col]; map its bounds through the ImageItem transform by hand. |
| gui/image_monitor#2 | With an image index selected, only frames whose header carries that index are drawn; the others are skipped without raising. | 4 | gui | Loud: the poll raises on every frame and the image freezes. | — |
| gui/image_monitor#3 | `poll` drains every queued frame and shows the newest, and skips header-only or `None` messages without raising. | 4 | gui | Loud: a traceback from the timer, and the viewer stops updating. | — |

## pytweezer/GUI/viewers/image_plot_monitor.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/image_plot_monitor#1 | A `[coords, values]` array on a top stream is drawn as values against coords; on a left stream it is drawn as coords up the y axis against values; a 1-D array uses its index as coords. | 4 | gui | Quiet: a profile is drawn on the wrong axis, or misaligned with the image. | — |
| gui/image_plot_monitor#2 | `update_subscriptions` removes the old curves before adding one per new stream, skipping "--None--". | 4 | gui | Quiet: ghost curves of removed streams stay on the plot. | — |

## pytweezer/GUI/viewers/live_plot.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/live_plot#1 | Each stream's curve shows the newest array: a 2×N array as row 0 against row 1, a 1-D array against its index, swapped when `rotated` is set; header-only messages are skipped. | 4 | gui | Quiet: the plot shows an old trace, or the axes swapped. | — |
| gui/live_plot#2 | `update_subscriptions` clears the old curves and makes one per subscribed stream, skipping "--None--". | 4 | gui | Quiet: ghost curves of streams the user removed. | — |

## pytweezer/GUI/viewers/scalar_history.py

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| gui/scalar_history#1 | Each subscribed stream's curve holds that stream's last `history` values of the header `field`, one per message in arrival order, all queued messages drained. | 4 | gui | Quiet: shots are dropped, or the trace drifts behind the live data. | — |
| gui/scalar_history#2 | Messages that lack the field, carry a non-numeric or bool value, or come from an unsubscribed channel add no point. | 4 | gui | Quiet: fake zeros or ones appear in the atom-number trace. | — |

## Suspected bugs

- pytweezer/GUI/experiments/arg_editor.py: `ValueField` rounds number values to `ndecimals` (6 when unset) in *display* units, so an argument whose SI value is smaller than 1e-6×scale is submitted as 0 (and its label reads "modified" even though nobody touched it). MOTMaster search rows have no scale either, so a float parameter like 3e-7 is affected too. Confirmed: a `QDoubleSpinBox` with 6 decimals returns 0.0 after `setValue(2e-7)`. Smallest input: `ValueField({"kind": "number", "default": 2e-7, "scale": 1.0, "ndecimals": None}).value()` gives `0.0`.
- pytweezer/GUI/experiments/queue_view.py: `set_snapshot` replaces items in place, and Qt keeps the old row index selected. So when the selected task disappears, the task that moves into its row becomes selected, with its buttons enabled (confirmed: QTableWidget keeps row 0 selected after `setRowCount` and `setItem`). The same happens in `analysismanager.FilterTable.update_filters` after the selected filter is deleted, where Start/Stop then acts on the neighbouring filter without asking. Smallest input: `set_snapshot({"queue": [t1, t2]})`, select row 0, `set_snapshot({"queue": [t2]})`; `selected_task()["rid"]` is then t2's rid.
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
