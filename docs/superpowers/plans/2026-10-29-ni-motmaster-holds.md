# NI Cards MOTMaster Holds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While MOTMaster runs a sequence, or a queued task holds that MOTMaster, its pattern boards are greyed out in the NI Cards tab and refused by the driver. After each run, the tab shows each analogue pattern channel's end-of-sequence value.

**Architecture:** The hardware-file parser learns MOTMaster's pattern boards. The `NICards` driver gains two reservation sources:
- MOTMaster's own `IsRunning`, polled over .NET remoting on the same PC;
- the Experiment Manager's `"experiment"` notifier, mirrored with `SyncMirror`.

It also gains an end-of-run sync that reads one new read-only C# method, `Controller.GetLastRunOutputs()`. The tab labels MOTMaster end values and shows notices.

**Tech Stack:** Python, pythonnet / .NET remoting, sipyco `SyncMirror`, PyQt6. C# (.NET Framework, EDMSuite MOTMaster).

**Spec:** `docs/superpowers/specs/2026-10-29-ni-motmaster-holds-design.md`

## Global Constraints

- pytweezer work happens in the worktree `/home/twalker/RbCaF/pytweezer-ni-motmaster`, on branch `ni-motmaster-holds` (based on `main`, already pushed). Commit and push after each task. Open a PR against `main` after Task 1 is pushed.
- EDMSuite work happens in a new worktree, `/home/twalker/RbCaF/EDMSuite-last-run`, on a new branch `feature/motmaster-last-run-outputs`. Create it from `feature/linked-motmaster` at `c1bebf23`. Never touch the `../EDMSuite` checkout.
- **EDMSuite safety:** never run a MOTMaster script, never call `Run`/`Go`, never launch MOTMaster.
- **MOTMaster's behaviour must not change for anyone who doesn't use it over remoting.** The C# change is one read-only method plus bookkeeping that can never throw into, or delay, a run.
- `get_logger(name)`, never `logging.getLogger`. Use Google-style docstrings in UK English that describe current behaviour only, and comment only what is surprising.
- RPC return values are PYON-safe (dict, list, str, float, bool, None).
- At most 10 test functions per test module (`write-tests` skill). `tests/test_ni_cards_gui.py` is already over that, so add no new function there; extend existing parametrisations instead.
- `QT_QPA_PLATFORM=offscreen` for GUI tests.
- Before each commit, run `uv run pytest tests/ -q` and `uv run ruff check <touched> && uv run ruff format --check <touched>`.
- **The C# remoting contract.** `Controller.GetLastRunOutputs()` returns `null` before the first run that has finished since MOTMaster started. After that it returns a `Dictionary<String, Object>` with these keys:
  - `"RunNumber"`: `int`, starting at 1 and increasing with every finished run;
  - `"Script"`: `string`, the script's file name without `.cs`;
  - `"Finished"`: `double`, Unix seconds (UTC);
  - `"OutputsKnown"`: `bool`;
  - `"Values"`: `Dictionary<String, Object>`, MOTMaster channel name to a `double` (analogue) or `bool` (digital). It is empty when `OutputsKnown` is false.
- **Holder and notice text** (copy exactly; `<mm>` is the MOTMaster device name, e.g. `CaF MotMaster`):
  - holders:
    - `"<mm> (running a sequence)"`
    - `"<mm> (queued run <rid>)"`
    - `"<mm> (not answering)"`
  - notices:
    - `"<mm> is not reachable: its runs are not greyed out"`
    - `"<mm> does not report end-of-run values: it needs a newer MOTMaster build"`
    - `"The Experiment Manager is not reachable: queued runs are not greyed out"`
    - `"The hardware file names no MOTMaster pattern boards: MOTMaster runs are not greyed out"`
- **Last-set label text** for a MOTMaster value: `"end of <script>: <value> at HH:MM"`, e.g. `"end of LoadTweezer: 1.25 V at 14:02"`.

## Review Focus

1. **MOTMaster is restarted mid-session.** Its run number drops back, or `GetLastRunOutputs()` returns `null`. Expected: the shown values stay as they are until a run newer than the one already applied (compared by finish time) arrives. Test in Task 4.
2. **The device server starts after a panel write that came after MOTMaster's last run.** This happens when another GUI writes and then the NI device server restarts. Expected: a "last set" entry newer than the run's `Finished` survives, and older entries on MOTMaster's boards are replaced. Since `last_set` is in memory, this reduces to the ordering rule, which Task 4 tests.
3. **MOTMaster is frozen, so a remoting call never returns.** Expected:
   - `state()` returns within one status timeout per remoting call (2 s each), with the boards held by `"<mm> (not answering)"`;
   - a write to those boards is refused;
   - the tab never freezes.

   Test in Task 3 (timeout) and Task 4 (hold).
4. **The lab PC runs an older MOTMaster build without `GetLastRunOutputs`.** Expected: the `IsRunning` holds still work, and the notice says a newer build is needed. Test in Task 3.
5. **A linked queued run.** Expected: the task's `devices` lists both MOTMasters, so both PCs' NI devices hold their boards, while an NI device whose MOTMaster is not listed holds nothing. Test in Task 4.

---

### Task 1: Pattern boards from the hardware file

**Files:**
- Modify: `pytweezer/drivers/ni_hardware_file.py`
- Test: `tests/test_ni_hardware_file.py`

**Interfaces:**
- Produces: `HardwareFile.pattern_boards: list[str]`, the board addresses (e.g. `"/Dev1"`) in this order:
  1. `PatternGeneratorBoard`;
  2. then `AdditionalPatternGeneratorBoards`' values;
  3. then `AnalogBoards`' values.

  Duplicates are dropped. It defaults to `[]`, so existing `HardwareFile(boards, aliases)` calls keep working.

- [ ] **Step 1: Write the failing test** `test_pattern_boards_are_read_from_info`. It uses an inline snippet modelled on `MoleculeMOTHardware.cs` lines 425-460:
  - `string dp = "/Dev1"; string dp2 = "/PXI1Slot6"; string ap = "/PXI1Slot2";`
  - `Info.Add("PatternGeneratorBoard", dp);`
  - `Dictionary<string, string> additionalPatternBoards = new Dictionary<string, string>(); additionalPatternBoards.Add(dp2, dp2); Info.Add("AdditionalPatternGeneratorBoards", additionalPatternBoards);`
  - `Dictionary<string, string> analogBoards = new Dictionary<string, string>(); analogBoards.Add("AO", ap); Info.Add("AnalogBoards", analogBoards);`

  Expected: `pattern_boards == ["/Dev1", "/PXI1Slot6", "/PXI1Slot2"]`. A second snippet with no such `Info.Add` calls gives `[]`. That's one test function, parametrised over the two snippets. The module must stay at or under 10 test functions.
- [ ] **Step 2: Run** `uv run pytest tests/test_ni_hardware_file.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Extend `_STATEMENT` to match `Info.Add(` and `<var>.Add(` for any other identifier (not `Boards`). Record each dictionary variable's `.Add` values (evaluated addresses), and resolve the three `Info` keys at the end of the pass. Unresolvable values are skipped with `logger.debug`.
- [ ] **Step 4: Run** the module's tests. Expected: PASS. Then run a smoke check against the real files, without committing a test: `uv run python -c "from pytweezer.drivers.ni_hardware_file import read_hardware_file as r; print(r('../EDMSuite/DAQ/MoleculeMOTHardware.cs').pattern_boards, r('../EDMSuite/DAQ/RbTweezerHardware.cs').pattern_boards)"`.
  - Expected for CaF: four addresses, starting with `/Dev1`, including `/PXI1Slot2`.
  - Expected for Rb: `['/PXI1Slot5', '/PXI1Slot4']`.
  - The alias counts must not change: 131 for CaF, 32 for Rb.
- [ ] **Step 5: Commit and push.** Open the PR ("NI Cards: MOTMaster holds and end-of-run values", body linking the spec and noting the EDMSuite branch).

---

### Task 2: Reservation sources that hold whole cards

**Files:**
- Modify: `pytweezer/drivers/ni_cards.py`
- Test: `tests/test_ni_cards.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Holds:
      channels: dict[str, str] = field(default_factory=dict)  # physical → holder
      cards: dict[str, str] = field(default_factory=dict)     # address → holder

  class Reservations:
      def __init__(self, sources: list[Callable[[], Holds]])
      def current(self) -> Holds   # merged; per key, the earliest source wins
  ```
  `Reservations.holds()` and `NICards._card_holds` go away. `state()["holds"]` and `state()["card_holds"]` come from one `current()` call. A source that holds a card lists every physical on it in `channels` too. `NICards._card_channels(address) -> list[str]` returns the physicals of a card, aliased and unaliased.
  - Each `write_do`, `write_ao` and `read` call evaluates `current()` exactly once. Task 4's sources make remoting calls, so `read()` must not evaluate it once per physical.

- [ ] **Step 1: Write the failing test** `test_earlier_source_wins_for_channels_and_cards`. It builds `Reservations` with two lambdas returning overlapping `Holds`, and asserts that both `current().channels` and `current().cards` keep the first source's holder.
- [ ] **Step 2: Run** `uv run pytest tests/test_ni_cards.py -q`. Expected: the new test fails, and the existing ones still pass.
- [ ] **Step 3: Implement** `Holds`, the new `Reservations` and the static source as a `Holds`, and route `_usable` through a single `current()` per call.
- [ ] **Step 4: Run** the full suite. Expected: PASS, including the GUI tests, since the `state()` shape is unchanged.
- [ ] **Step 5: Commit and push.**

---

### Task 3: MOTMaster and Experiment Manager clients

**Files:**
- Create: `pytweezer/drivers/ni_cards_sources.py`
- Modify: `pytweezer/drivers/motmaster.py` (add `motmaster_controller`)
- Test: `tests/test_ni_cards_sources.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class LastRun:
      number: int
      script: str
      finished: float                    # Unix seconds
      outputs_known: bool
      values: dict[str, float | bool]    # by MOTMaster channel name

  def last_run_from_report(report: Mapping[str, Any] | None) -> LastRun | None

  class MotMasterNotAnswering(Exception): ...
  class MotMasterTooOld(Exception): ...   # the build has no GetLastRunOutputs

  class RemotingMotMasterStatus:
      def __init__(self, device_name: str, timeout_s: float = 2.0)
      def is_running(self) -> bool | None    # None: MOTMaster not reachable
      def last_run(self) -> LastRun | None   # None: unreachable, or no run yet
      # both raise MotMasterNotAnswering after timeout_s; last_run raises MotMasterTooOld

  class SimulatedMotMasterStatus:          # never running, no last run
      def is_running(self) -> bool | None
      def last_run(self) -> LastRun | None

  class ManagerQueue:
      def __init__(self)                   # SyncMirror(*sync_address(), NOTIFIER_NAME)
      def running_devices(self) -> dict[str, int] | None   # device → rid; None: manager unreachable
      def close(self) -> None
  ```
  In `motmaster.py`: `motmaster_controller(config_file: str)` returns the remoting proxy for the `"motmaster"` entry of the JSON config named by `config_file`, resolved as `MotMasterInterface` does against `CONFIG_DIR`. It adds the references (`dll_paths` and the `exe_path`) and returns `Activator.GetObject(MOTMaster.Controller, remote_path)`. It does not start MOTMaster.
  - `RemotingMotMasterStatus` reads `get_config()["Devices"][device_name]["config_file"]`.
  - It builds the proxy lazily, on the first call.
  - Every call runs on a worker thread and waits at most `timeout_s`.
  - A .NET connection failure (`SocketException` or `RemotingException`, as pythonnet surfaces them) drops the proxy and means "unreachable". The next call reconnects.
  - A missing `GetLastRunOutputs` (pythonnet `AttributeError`, or `MissingMethodException`) raises `MotMasterTooOld`.
  - `running_devices` counts every task in `snapshot["running"]` (paused tasks are in that list) and maps each name in its `devices` to its `rid`.

- [ ] **Step 1: Write the failing tests:**
  - `test_last_run_from_report_converts_the_contract`: parametrised over:
    - `None` gives `None`;
    - a full dict gives the matching `LastRun`, with `values` converted to `float`/`bool`;
    - `OutputsKnown=False` gives `outputs_known is False` and `values == {}`.
  - `test_status_call_that_never_returns_raises_not_answering`: monkeypatch the proxy factory to return an object whose `IsRunning` sleeps for 5 s, construct the status with `timeout_s=0.1`, and assert that `is_running()` raises `MotMasterNotAnswering` within 1 s. *(Review Focus 3)*
  - `test_build_without_last_run_method_raises_too_old`: the proxy has `IsRunning` but no `GetLastRunOutputs`. `last_run()` raises `MotMasterTooOld`, and `is_running()` still works. *(Review Focus 4)*
  - `test_unreachable_motmaster_reads_as_none`: the proxy factory raises `ConnectionRefusedError`. `is_running()` and `last_run()` return `None`.
  - `test_running_devices_maps_reserved_devices_to_rid`: replace the mirror with a stub whose `snapshot()` holds two running tasks, `{"rid": 42, "devices": ["CaF MotMaster", "Rb MotMaster"]}` and `{"rid": 43, "devices": ["CaF ThorCam"]}`. The result maps both MOTMasters to 42 and `CaF ThorCam` to 43. A stub with `connected` false gives `None`.

  Fake the proxy by monkeypatching `ni_cards_sources.motmaster_controller`. No pythonnet is needed.
- [ ] **Step 2: Run** `uv run pytest tests/test_ni_cards_sources.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the module and `motmaster_controller`. Map exceptions by class name for the .NET ones, so that the module imports without pythonnet.
- [ ] **Step 4: Run** the module's tests, then the full suite. Expected: PASS.
- [ ] **Step 5: Commit and push.**

---

### Task 4: MOTMaster holds and end values in `NICards`

**Files:**
- Modify: `pytweezer/drivers/ni_cards.py`, `pytweezer/configuration/config.py`
- Test: `tests/test_ni_cards_motmaster.py` (new)

**Interfaces:**
- Consumes: `HardwareFile.pattern_boards` (Task 1), `Holds`/`Reservations` (Task 2), and the status and queue classes (Task 3).
- Produces:
  - `NICards(hardware_file, reserved, protected, motmaster: str | None = None, *, io=None, motmaster_status=None, queue=None)`. With `motmaster` set and no status or queue given, it uses `RemotingMotMasterStatus(motmaster)` and `ManagerQueue()`. With `motmaster=None` it has no MOTMaster sources and no notices.
  - `SimulatedNICards(hardware_file, reserved, protected, motmaster=None)` uses `SimulatedMotMasterStatus()` and, when `motmaster` is set, `ManagerQueue()`.
  - `NICards.close()` closes the queue mirror. Calling it again does nothing.
  - `state()` gains `"notices": list[str]`.
  - Every `last_set` entry gains `"source": "panel" | "motmaster"`. MOTMaster entries also carry `"script"`.
  - Module constant `SHOW_DIGITAL_END_VALUES = False`. Its comment carries the spec's TODO: check on the rig whether the PCI-6534 and PXI-6535 digital lines hold the pattern's last sample, and if they do, set it to `True`.
- Behaviour:
  - The two new sources come after the static one. Both hold every card in `pattern_boards` that is in the layout, with the holder texts from Global Constraints:
    - **running source:** `is_running()` is `True`. A `MotMasterNotAnswering` holds with `"(not answering)"`. `None` holds nothing and adds the not-reachable notice.
    - **queue source:** `running_devices()` contains `motmaster`. `None` adds the manager notice.
  - **End-of-run sync** runs in `state()` and once in `__init__`. `MotMasterTooOld` adds that notice, and `None` changes nothing. A `LastRun` whose `(number, finished)` differs from the last one applied, and whose `finished` is later than the last applied run's, is applied as follows. A restarted MOTMaster's older or `None` record is ignored. *(Review Focus 1)*
    1. Every `last_set` entry on a pattern board whose `time < finished` is removed. Newer panel writes survive. *(Review Focus 2)*
    2. If `outputs_known`, then for each value whose MOTMaster name is an alias of an AO channel on a pattern board, set `{"value": float, "time": finished, "source": "motmaster", "script": script}`. Do the same for DO only if `SHOW_DIGITAL_END_VALUES`.
  - An empty `pattern_boards` with `motmaster` set adds the no-pattern-boards notice and no MOTMaster holds.
- Config: add `"motmaster": "CaF MotMaster"` and `"teardown": "close"` to `"CaF NI Cards"`, and `"motmaster": "Rb MotMaster"` and `"teardown": "close"` to `"Rb NI Cards"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ni_cards_motmaster.py`. Move `FakeIO` and `HARDWARE` from `tests/test_ni_cards.py` into `tests/helpers.py`, which other tests already import as `from tests.helpers import ...`, and import them in both modules. Add these `Info.Add` lines to a local copy of the snippet: pattern boards `/PXI1Slot4` and `/PXI1Slot2`. Use a `FakeStatus` with settable `running` and `last` (or an exception to raise), and a `FakeQueue` with settable `devices`.
  - `test_running_motmaster_holds_its_pattern_boards_and_refuses_writes`:
    - `card_holds` holds both boards with `"CaF MotMaster (running a sequence)"`, and so does an unaliased line on `/PXI1Slot4`;
    - `write_ao` raises `ChannelHeld`, and `FakeIO` records nothing;
    - `/PXI1Slot8` is not held.
  - `test_write_checks_motmaster_fresh_not_from_the_last_poll`: call `state()` while not running, then set `running = True`. `write_do` is refused.
  - `test_not_answering_motmaster_holds_and_refuses`: the holder is `"CaF MotMaster (not answering)"`. *(Review Focus 3)*
  - `test_unreachable_motmaster_holds_nothing_and_says_so`: the notices contain the not-reachable text, and writes go through.
  - `test_queued_run_holds_while_task_reserves_this_motmaster`: parametrised over `{"CaF MotMaster": 42, "Rb MotMaster": 42}` (held, `"CaF MotMaster (queued run 42)"`) and `{"Rb MotMaster": 7}` (not held). *(Review Focus 5)*
  - `test_end_of_run_sets_analogue_values_and_resets_the_rest`: with a prior panel write on a DO line and on an AO channel, both older than `finished`, a known-outputs run with `values={"v21Lock": 0.4, "MOT1Shutter": True}` gives:
    - `v21Lock`'s physical has `{"value": 0.4, "source": "motmaster", "script": "LoadTweezer"}`;
    - the DO line has no entry, because `SHOW_DIGITAL_END_VALUES` is false;
    - `/PXI1Slot8` entries are untouched.
  - `test_panel_write_newer_than_the_run_survives`: the write's time is greater than `finished`, so it stays. *(Review Focus 2)*
  - `test_unknown_outputs_reset_pattern_channels`: `outputs_known=False`, so every entry on the pattern boards is removed.
  - `test_restarted_motmaster_does_not_roll_values_back`: apply run 5, then report run 1 with an older `finished`, then `None`. The run-5 values remain. *(Review Focus 1)*
  - `test_too_old_build_says_so_but_still_holds_while_running`: `last` raises `MotMasterTooOld`. The notice is present, and the running hold still applies. *(Review Focus 4)*
- [ ] **Step 2: Run** `uv run pytest tests/test_ni_cards_motmaster.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, including the config keys. Then add `close` to the simulation parity check in `tests/test_simulate_decorator.py`, which needs no new case.
- [ ] **Step 4: Run** the full suite, then build both config entries in simulation, as the previous plan did:
  `uv run python -c "from pytweezer.configuration.config import get_config; from pytweezer.drivers.ni_cards import SimulatedNICards; from pytweezer.servers.device_server import _config_kwargs as k; [print(n, SimulatedNICards(**k(SimulatedNICards, get_config()['Devices'][n])).state()['notices']) for n in ['CaF NI Cards', 'Rb NI Cards']]"`.
  - Expected: both build.
  - The notices list only the Experiment Manager one, because no manager is running here.
- [ ] **Step 5: Commit and push.**

---

### Task 5: The tab shows MOTMaster values and notices

**Files:**
- Modify: `pytweezer/GUI/ni_cards/cells.py`, `pytweezer/GUI/ni_cards/panel.py`
- Test: `tests/test_ni_cards_gui.py`

**Interfaces:**
- Consumes: the `last_set` entry shape and `state()["notices"]` (Task 4).
- Produces:
  - `last_set_text(entry, kind)` returns `"end of <script>: <value> at HH:MM"` when `entry.get("source") == "motmaster"`. Entries without a `source` (an older server) read as panel writes.
  - `DeviceView.notices`: a `QLabel` with `role="regionHint"`, hidden when there are no notices. It shows the notices one per line, under `message`. An older server's `state()` without `"notices"` shows none.

- [ ] **Step 1: Extend the existing tests; no new functions:**
  - add `({"value": 1.25, "time": 1793300000.0, "source": "motmaster", "script": "LoadTweezer"}, "ao", "end of LoadTweezer: 1.25 V at {}")` to `test_last_set_text`'s parametrisation;
  - in `test_state_marks_held_cells_and_card_banner`, emit a state with `notices=["CaF MotMaster is not reachable: its runs are not greyed out"]` (extend the `state()` helper with a `notices` argument), and assert that `view.notices` is visible with that text.
- [ ] **Step 2: Run** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_ni_cards_gui.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite.
  - Then take a screenshot: write a scratchpad factory along the lines of the previous session's `nifakes.py`. It builds `SimulatedNICards` for `"CaF NI Cards"`, wrapped so that `state()` shows a held `analogPattern` card and a MOTMaster end value.
  - Shoot it with `.claude/skills/pytweezer-gui-design/scripts/shoot.py ... --size 1600x1000 --wait 2000`. Note that `--wait` is in milliseconds.
  - Check: the banner reads "Held by CaF MotMaster (running a sequence)", an AO cell reads "end of LoadTweezer: …", and the notice line sits under the header without crowding it.
- [ ] **Step 5: Commit and push.**

---

### Task 6: `GetLastRunOutputs` in MOTMaster (EDMSuite)

**Files** (all in `/home/twalker/RbCaF/EDMSuite-last-run`):
- Create: `MOTMaster/LastRunOutputs.cs`
- Modify: `MOTMaster/Controller.cs`, `MOTMaster/MOTMaster.csproj` (add `<Compile Include="LastRunOutputs.cs" />` next to `Controller.cs`)
- Test: `MOTMasterTests/LastRunOutputsTests.cs`

**Interfaces:**
- Produces:
  - `public static class LastRunOutputs` with:
    - `public static Dictionary<String, Object> EndValues(MOTMasterSequence sequence, Hashtable digitalOutputChannels)`. It returns each analogue channel's value at its latest event time, over every board in `sequence.AnalogPattern.Boards`. For each digital channel registered on a board in `sequence.DigitalPattern.Boards`, it returns its bit in the board's last pattern sample.
    - `public static Dictionary<String, Object> Record(int runNumber, string script, DateTime finishedUtc, bool outputsKnown, Dictionary<String, Object> values)`. This builds the contract dictionary from Global Constraints. `values` is replaced by an empty dictionary when `outputsKnown` is false.
  - `Controller.GetLastRunOutputs()`: `public Dictionary<String, Object> GetLastRunOutputs()` returns the latest record, or `null`.
- Bookkeeping in `Controller`, all inside `try`/`catch` that logs with `SharedCode.AppLog.Error`. A failure leaves the outputs unknown and never throws or delays the run.
  - At the start of each run (in `goClaimed`, and at the start of the secondary's run in `linkedSecondaryLoop`), set `lastShotCompleted = false` and `pendingValues = null`.
  - At the start of each shot, set `lastShotCompleted = false`.
  - When a shot's pattern finishes (`runPattern` when `completed`, and the matching point in `linkedSecondaryLoop`), set `lastShotCompleted = true` and `pendingValues = LastRunOutputs.EndValues(sequence, Environs.Hardware.DigitalOutputChannels)`. This happens before any `clearDigitalPattern`.
  - In `Go`'s `finally` and in `linkedSecondaryLoop`'s `finally`, after `setSafeOutputs()`, publish `Record(++runNumber, script, DateTime.UtcNow, lastShotCompleted && !config.Debug, pendingValues)`. Assign the new dictionary to a `volatile` field; never mutate a published one.

- [ ] **Step 1: Create the worktree:** `git -C /home/twalker/RbCaF/EDMSuite worktree add -b feature/motmaster-last-run-outputs /home/twalker/RbCaF/EDMSuite-last-run c1bebf23`.
- [ ] **Step 2: Write the tests** in `MOTMasterTests/LastRunOutputsTests.cs`, as `public static void Test*()` methods using `MOTMasterTests.Assert`:
  - `TestAnalogueEndValueIsTheLatestEvent`: a channel with events at times 0, 100 and 50 (in insertion order) gives the value at 100.
  - `TestDigitalEndValueIsTheLastSample`: a two-channel board whose last sample has bit 3 set and bit 4 clear gives `true` and `false`. Use `TestHardware.cs`'s registered channels.
  - `TestRecordDropsValuesWhenOutputsUnknown`: `Record(..., outputsKnown: false, values with one entry)` has an empty `"Values"`, and every contract key with the right type.
- [ ] **Step 3: Implement** `LastRunOutputs.cs`, the `Controller` bookkeeping and the csproj entry.
- [ ] **Step 4: Verify.**
  - This machine (Linux) has no C# toolchain. Re-read the diff against the bookkeeping list above, then check that nothing outside the `try`/`catch` blocks changed in the run path: `git diff c1bebf23 -- MOTMaster/Controller.cs`.
  - The build (`msbuild`) and `MOTMasterTests\run-tests.ps1` must be run on a Windows PC with VS2019. They are listed for the user in the EDMSuite PR description, together with the skill's note on building to a scratch `OutDir` when MOTMaster is running. Do not report them as passed.
- [ ] **Step 5: Commit and push** the EDMSuite branch, and open a PR against `feature/linked-motmaster`. Its description gives the remoting contract, the Windows verification steps, and the fact that it has not yet been compiled.
  - Per the EDMSuite `motmaster-controller` skill, ask the user whether that skill should mention `GetLastRunOutputs`. Don't edit it unasked.
  - Link the EDMSuite PR from the pytweezer PR.
