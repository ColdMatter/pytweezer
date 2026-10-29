# NI Test Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An "NI Cards" GUI tab that drives and reads individual NI channels on the lab PCs by their MOTMaster alias, with reserved and protected channels greyed out.

**Architecture:** A pure parser turns a MOTMaster hardware `.cs` file into boards and aliases. A device driver (`NICards`, one `CONFIG["Devices"]` entry per PC) combines that with NI-DAQmx's own channel list, enforces reservations and protection, does single static writes and on-demand reads, and remembers the last value written. The GUI tab talks to each driver through `get_device()` on a worker thread and lays each card's channels out in a reflowing grid of cells.

**Tech Stack:** Python, `nidaqmx` 1.x (already a dependency), sipyco RPC via the generic device server, PyQt6.

**Spec:** `docs/superpowers/specs/2026-10-29-ni-test-panel-design.md`

## Global Constraints

- Work in the worktree `/home/twalker/RbCaF/pytweezer-ni-test-panel`, branch `ni-test-panel`, based on `main`. Another agent is using the main checkout: do not touch it. Do not depend on `experiment-pipelines`.
- Commit and push after each task. Stage only your own files. Open a PR for the branch after Task 1 is pushed (every feature branch has one).
- `get_logger(name)` from `pytweezer/logging_utils.py`, never `logging.getLogger`.
- Google-style docstrings, UK English, describe current behaviour only. Comment only what is surprising.
- Styling only through `DARK_STYLESHEET` rules in `pytweezer/GUI/theme.py`, keyed by `objectName`/properties. No per-widget `setStyleSheet`.
- RPC arguments and return values are PYON-safe (dicts, lists, str, float, bool, None).
- Physical channel names are NI-DAQmx names without a leading slash: `PXI1Slot4/port3/line5`, `PXI1Slot2/ao17`, `PXI1Slot8/ai4`.
- At most 10 test cases per module (`write-tests` skill). Tests need no hardware, no NI-DAQmx runtime and no EDMSuite checkout: hardware-file tests use inline C# snippets.
- `QT_QPA_PLATFORM=offscreen` for any GUI test or screenshot.
- Run `uv run pytest tests/ -q` and `uv run ruff check . && uv run ruff format --check .` before each commit.

## Review Focus

1. **Hardware file missing or unreadable on the PC** (wrong path, EDMSuite not checked out). Expected: the device server still starts, `layout()` returns an `"error"` naming the path, and the tab shows that message, not a crash. Test in Task 2.
2. **A card in the hardware file that NI-DAQmx doesn't see** (e.g. an unplugged USB-6008). Expected: the card block is shown with "not found on this PC", its cells are disabled, and the driver rejects writes to it. Test in Task 2.
3. **A typo in `reserved` or `protected` config.** Expected: the driver fails at construction, naming the unknown alias or board, rather than silently leaving a coil channel unprotected. Test in Task 2.
4. **Device server unreachable, or an RPC timing out while watching.** Expected: the tab says it is not reachable, watch polling stops, and the UI never freezes. Test in Task 4.
5. **Another GUI writes a channel.** Expected: this GUI's cell shows the new "last set" within one state poll. Test in Task 4.

---

### Task 1: Hardware-file parser

**Files:**
- Create: `pytweezer/drivers/ni_hardware_file.py`
- Test: `tests/test_ni_hardware_file.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Board:
      name: str        # "digitalPattern2"
      address: str     # "/PXI1Slot4"
      @property
      def device(self) -> str: ...   # "PXI1Slot4"

  @dataclass(frozen=True)
  class Alias:
      name: str                                # "MOT1Shutter"
      kind: Literal["ao", "ai", "do", "di"]
      physical: str                            # "PXI1Slot4/port3/line5"
      board: str                               # board name, "digitalPattern2"
      ao_range: tuple[float, float] | None     # AO only, when declared
      terminal: str | None                     # AI only: "Rse", "Nrse", "Differential", ...

  @dataclass(frozen=True)
  class HardwareFile:
      boards: dict[str, Board]      # insertion order = Boards.Add order
      aliases: list[Alias]          # declaration order

  def parse_hardware_file(source: str) -> HardwareFile
  def read_hardware_file(path: str) -> HardwareFile   # raises OSError if unreadable
  ```

- [ ] **Step 1: Write the failing tests** in `tests/test_ni_hardware_file.py`, each on a short inline C# snippet:
  - `test_board_address_resolves_through_string_variable`: `string a = "/PXI1Slot2"; Boards.Add("analogPattern", a); AddAnalogOutputChannel("xShim", a + "/ao17");` gives `Alias("xShim", "ao", "PXI1Slot2/ao17", "analogPattern", None, None)`.
  - `test_board_address_resolves_through_boards_lookup`: `Boards.Add("pg", "/PXI1Slot5"); string pgBoard = (string)Boards["pg"]; AddDigitalOutputChannel("tCoolSwitch", pgBoard, 0, 0);` gives physical `"PXI1Slot5/port0/line0"`, board `"pg"`. Also `(string)Boards["pg"] + "/ai2"` used inline as an AI address resolves.
  - `test_commented_out_declarations_are_ignored`, parametrised over a `//` line and a `/* ... */` block spanning several lines (including one that starts mid-line, `/*AddDigitalOutputChannel(...`): no alias produced.
  - `test_alias_on_commented_out_board_is_skipped`: `//string daqBoard = (string)Boards["daq"];` then `AddAnalogInputChannel("iodine", daqBoard + "/ai2", AITerminalConfiguration.Nrse);` gives no alias.
  - `test_ao_range_and_ai_terminal_are_read`: `AddAnalogOutputChannel("v21Lock", usb + "/ao0", 0.0, 1.0)` gives `ao_range == (0.0, 1.0)`; `AddAnalogOutputChannel("bXLock", b + "/ao2")` gives `None`; `AddAnalogInputChannel("v00PD", b + "/ai0", AITerminalConfiguration.Rse)` gives `terminal == "Rse"`.
  - `test_digital_input_is_parsed_as_di`: `AddDigitalInputChannel("bXLockBlockFlag", b, 0, 1)` gives `kind == "di"`, physical `".../port0/line1"`.
  - `test_unrelated_calls_are_ignored`: `Info.Add(...)`, `AddCounterChannel(...)`, `AddAnalogInputThermocoupleChannel(...)`, `wmlConfig.AddSlaveLaser(...)` produce no aliases.
  - `test_two_aliases_on_one_line_are_both_kept`: `MOT3Shutter` and `rbOPShutter` on the same board/port/line give two `Alias` objects with equal `physical`.

- [ ] **Step 2: Run** `uv run pytest tests/test_ni_hardware_file.py -q`. Expected: FAIL, module not found.

- [ ] **Step 3: Implement `parse_hardware_file`** in `pytweezer/drivers/ni_hardware_file.py`. Approach: strip `/* */` blocks and `//` to end of line (ignoring the contents of string literals). Then make one pass in source order: collect `string X = "<literal>";` and `string X = (string)Boards["key"];` assignments and `Boards.Add(<name>, <address>)`, and resolve each `Add{Analog,Digital}{Input,Output}Channel(` call's arguments. An address expression is a literal, a variable, or `(string)Boards["key"]`, optionally followed by `+ "/aoN"`. A call whose board can't be resolved is skipped, with a `logger.debug`. The `AITerminalConfiguration.X` suffix becomes `terminal`. `read_hardware_file` reads the file as UTF-8 with `errors="replace"`.

- [ ] **Step 4: Run** `uv run pytest tests/test_ni_hardware_file.py -q`. Expected: PASS. Also run a smoke check against the real files, without committing a test for it: `uv run python -c "from pytweezer.drivers.ni_hardware_file import read_hardware_file as r; h=r('../EDMSuite/DAQ/MoleculeMOTHardware.cs'); print(len(h.boards), len(h.aliases))"`. Expected: 9 boards, and a nonzero alias count with no `valve` and no `iodine` among Rb's aliases (check the Rb file the same way).

- [ ] **Step 5: Commit and push**; open the PR against `main` ("NI test panel", body linking the spec).

---

### Task 2: `NICards` driver logic: layout, reservations, protection, last-set

**Files:**
- Create: `pytweezer/drivers/ni_cards.py`
- Test: `tests/test_ni_cards.py`

**Interfaces:**
- Consumes: `read_hardware_file`, `HardwareFile`, `Alias`, `Board` (Task 1).
- Produces:
  ```python
  @dataclass
  class CardInfo:              # what NI-DAQmx reports for one device
      model: str               # "PXI-6535"
      ao: list[str]            # physical names
      ai: list[str]
      lines: list[str]         # digital lines, "PXI1Slot4/port0/line0", ...
      ao_range: tuple[float, float]   # widest AO voltage range, (0, 0) if no AO

  class CardIO(Protocol):      # Task 3 provides DaqmxIO and SimulatedIO
      def card(self, device: str) -> CardInfo | None    # None: not on this PC
      def write_do(self, physical: str, high: bool) -> None
      def write_ao(self, physical: str, volts: float, low: float, high: float) -> None
      def read_ai(self, physicals: list[str], terminals: list[str | None]) -> list[float]
      def read_di(self, physicals: list[str]) -> list[bool]

  class ChannelHeld(Exception): ...        # message names the holder

  PROTECTED = "Protected"

  class NICards:
      def __init__(self, hardware_file: str, reserved: dict | None = None,
                   protected: list[str] | None = None, *, io: CardIO | None = None)
      def layout(self) -> dict
      def state(self) -> dict
      def write_do(self, physical: str, high: bool) -> None
      def write_ao(self, physical: str, volts: float) -> None
      def read(self, physicals: list[str]) -> dict[str, float | bool]
  ```
  `reserved` is `{holder: {"channels": [alias, ...], "cards": [board_name, ...]}}`, with either key optional. `io=None` means `DaqmxIO()` (Task 3); until Task 3 lands, tests always pass an `io`.

  `layout()` returns:
  ```python
  {"error": str | None,               # e.g. "Cannot read C:\...\MoleculeMOTHardware.cs: ..."
   "cards": [{"board": "digitalPattern2", "address": "/PXI1Slot4",
              "model": "PXI-6535" | None,
              "error": None | "not found on this PC",
              "channels":  [CHANNEL, ...],     # aliased, one per physical
              "unaliased": [CHANNEL, ...]}]}
  # CHANNEL = {"physical": str, "kind": "ao"|"ai"|"do"|"di",
  #            "aliases": [str, ...], "range": [lo, hi] | None, "terminal": str | None}
  ```
  Cards come in `Boards.Add` order. Within a card, channels are ordered by kind (ai, ao, di, do) and then by natural physical order (`ao2` before `ao10`; port, then line). Aliases sharing a physical merge into one CHANNEL. `range` is the declared AO range, else the card's `ao_range`, and `None` for non-AO. Unaliased digital lines are `"do"`. A card NI-DAQmx doesn't see still lists its aliased channels, with empty `unaliased`.

  `state()` returns `{"holds": {physical: holder}, "card_holds": {address: holder}, "last_set": {physical: {"value": float | bool, "time": float}}}`. `holds` covers every held physical, including every channel of a held card and protected aliases (holder `PROTECTED`). Protection wins over a reservation.

  Reservations are assembled by a `Reservations` class in the same module: `Reservations(sources: list[Callable[[], dict[str, str]]])` with `holds() -> dict[str, str]`. Only the static source exists now. The MOTMaster source plugs in here later (spec, **Reservations**).

- [ ] **Step 1: Write the failing tests** in `tests/test_ni_cards.py`. Use a `FakeIO` implementing `CardIO` that records writes and returns fixed readings, and a `tmp_path` hardware file written from an inline snippet with two boards (a DO board and an AO+AI board) plus a third board the `FakeIO` reports as `None`.
  - `test_layout_merges_shared_aliases_and_lists_unaliased_lines`: two aliases on one line become one CHANNEL with both names. `FakeIO`'s extra lines appear in `unaliased` as `"do"`, in natural order.
  - `test_missing_hardware_file_reports_error_instead_of_raising`: a nonexistent path constructs, and `layout()["error"]` contains the path. *(Review Focus 1)*
  - `test_card_not_on_this_pc_is_listed_and_rejects_writes`: the `None` card has `error == "not found on this PC"`, and `write_do` to its alias's physical raises `ChannelHeld` or `ValueError` with "not found" in the message. *(Review Focus 2)*
  - `test_unknown_reserved_or_protected_name_fails_construction`, parametrised over a typo in `reserved[...]["channels"]`, in `reserved[...]["cards"]` and in `protected`: raises `ValueError` naming it. *(Review Focus 3)*
  - `test_reserved_card_holds_every_channel_including_unaliased`: `{"Hardware Control": {"cards": ["tclBoard2"]}}` puts that card's address in `card_holds`, and every physical on it, aliased or not, in `holds` with that holder.
  - `test_held_and_protected_channels_refuse_writes_and_reads`, parametrised over a reserved alias and a protected alias, and over `write_ao` and `read`: raises `ChannelHeld` whose message names the holder (or `"Protected"`), and `FakeIO` records nothing.
  - `test_ao_write_outside_declared_range_is_rejected`: `ao_range (0.0, 1.0)`, `write_ao(p, 1.5)` raises `ValueError`, nothing reaches `FakeIO`, and `write_ao(p, 0.5)` passes `(p, 0.5, 0.0, 1.0)` to `FakeIO`.
  - `test_last_set_records_value_and_time_of_each_write`: after `write_do(p, True)`, `state()["last_set"][p]["value"] is True` and `"time"` is within a second of `time.time()`.
  - `test_read_returns_values_keyed_by_physical`: a mixed AI+DI request comes back as `{physical: float}` / `{physical: bool}`, with AI terminals passed through from the aliases (`None` for unaliased).

- [ ] **Step 2: Run** `uv run pytest tests/test_ni_cards.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement** `Reservations`, `ChannelHeld`, `PROTECTED` and `NICards` (with `io` required for now) in `pytweezer/drivers/ni_cards.py`. Validate `reserved`/`protected` names against the parsed file in `__init__`. Every write and read checks holds first, then card presence, then range. Keep `last_set` in memory only.

- [ ] **Step 4: Run** `uv run pytest tests/test_ni_cards.py -q`. Expected: PASS.

- [ ] **Step 5: Commit and push.**

---

### Task 3: NI-DAQmx and simulated IO, config entries, simulated twin

**Files:**
- Modify: `pytweezer/drivers/ni_cards.py` (add `DaqmxIO`, `SimulatedIO`, `SimulatedNICards`; `io=None` → `DaqmxIO()`)
- Modify: `pytweezer/configuration/config.py` (append two entries at the **end** of the `"Devices"` block, after `"CaF ThorCam"`, so no existing port moves)
- Modify: `tests/test_simulate_decorator.py` (add `(NICards, SimulatedNICards)` to `test_simulated_backend_covers_real_interface`)

**Interfaces:**
- Consumes: `CardIO`, `CardInfo`, `NICards` (Task 2).
- Produces: `DaqmxIO()`, `SimulatedIO()`, and `SimulatedNICards(hardware_file, reserved=None, protected=None)`, decorated `@simulate(NICards)` and subclassing `NICards` with `io=SimulatedIO()`.

- [ ] **Step 1: Add the failing parametrisation** to `test_simulated_backend_covers_real_interface`. Run `uv run pytest tests/test_simulate_decorator.py -q`. Expected: FAIL (import error).

- [ ] **Step 2: Implement `DaqmxIO`.** Import `nidaqmx` lazily inside methods, as `ni_adc_logger.py` does.
  - `card`: `nidaqmx.system.System.local().devices`. Return `None` if the device name is absent. Otherwise read `product_type`, `ao_physical_chans.channel_names`, `ai_physical_chans.channel_names`, `do_lines.channel_names` (falling back to `di_lines` when there are no DO lines), and the widest pair from `ao_voltage_rngs`.
  - Each write and read opens a `nidaqmx.Task`, adds the channels (AI: one task per device per call, terminal mapped `Rse→RSE`, `Nrse→NRSE`, `Differential→DIFF`, `None→DEFAULT`), does one `write`/`read` and closes it in a `with` block. No task outlives a call (spec anti-goal).
- [ ] **Step 3: Implement `SimulatedIO`:**
  - every board is present, model `"Simulated"`;
  - it reports `ao0…ao7`, `ai0…ai7` and `port0…port3 × line0…line7` for each device, plus any physical the aliases name;
  - `ao_range (-10, 10)`;
  - writes are remembered, DI reads return the last DO write to that line (else `False`), and AI reads return `math.sin(2π·0.1·t + index) + gauss(0, 0.01)`.

  `SimulatedNICards.__init__`: if `hardware_file` doesn't exist, try `<tweezerpath>/../EDMSuite/DAQ/<basename>` (the dev-machine sibling checkout) before falling back to the missing-file error.
- [ ] **Step 4: Add the config entries** with `"active": True`, `"class": "pytweezer.drivers.ni_cards:NICards"`, `"sim_class": "pytweezer.drivers.ni_cards:SimulatedNICards"`, `"port": get_next_port()` and `"simulate": SIMULATING`:
  - `"CaF NI Cards"`: `"host": HOSTS["ph-bonesaw"]`, `"hardware_file": r"C:\ControlPrograms\EDMSuite\DAQ\MoleculeMOTHardware.cs"`, `"reserved": {"Wavemeter Lock": {"channels": ["v00Lock", "v10Lock", "v21Lock", "v32Lock", "bXLock", "bXBeastLock"]}, "Hardware Control": {"cards": ["tclBoard2"]}}`, `"protected": ["MOTCoilsCurrent", "transferCoils", "TweezerCoils"]`.
  - `"Rb NI Cards"`: `"host": HOSTS["IC-CZC221CCHS"]`, `"hardware_file": r"C:\ControlPrograms\EDMSuite\DAQ\RbTweezerHardware.cs"`, `"reserved": {"Wavemeter Lock": {"channels": ["tRbD1Lock"]}}`, `"protected": []`. The Rb file declares no safe-state channels.
- [ ] **Step 5: Verify.**
  - `uv run pytest tests/ -q`. Expected: PASS.
  - `uv run python .claude/skills/add-device-driver/scripts/check_device.py "CaF NI Cards"` and the same for `"Rb NI Cards"`. Expected: every config key reaches the constructor, the device builds in simulation, and the RPC surface lists `layout`, `state`, `write_do`, `write_ao` and `read`.
  - Real hardware is untested at this point; say so in the PR.
- [ ] **Step 6: Commit and push.**

---

### Task 4: The "NI Cards" tab

**Files:**
- Create: `pytweezer/GUI/ni_cards/__init__.py`, `client.py`, `flow_grid.py`, `cells.py`, `panel.py`
- Modify: `bin/gui.py` (add `("NI Cards", _safe_panel("NI Cards", NICardsPanel))` after Results)
- Modify: `pytweezer/GUI/theme.py` (rules below)
- Modify: `.claude/skills/pytweezer-gui-internals/SKILL.md` (add "NI Cards" to the tab list)
- Test: `tests/test_ni_cards_gui.py`

**Interfaces:**
- Consumes: the `NICards` RPC surface and the `layout()`/`state()` shapes (Task 2).
- Produces:
  - `ni_card_devices(devices_config: dict) -> list[str]` in `panel.py`: the names of `CONFIG["Devices"]` entries whose `"class"` is `"pytweezer.drivers.ni_cards:NICards"`, in config order.
  - `CardsClient(QObject)` in `client.py`: `CardsClient(name: str, connect: Callable[[], Any])`. One daemon worker thread drains a queue of calls against the object `connect()` returns, reconnecting after a failure.
    - Methods (non-blocking): `fetch_layout()`, `fetch_state()`, `write_do(physical, high)`, `write_ao(physical, volts)`, `read(physicals)`, `close()`.
    - Signals: `layout_received(dict)`, `state_received(dict)`, `values_received(dict)`, `failed(str)` (a write or read the driver refused, e.g. `ChannelHeld`), `connection_changed(bool)`.
    - Default `connect`: `get_device(name, timeout=5)`, or `SimulatedDevices().get(name)` when `config.SIMULATING`, following `motmaster_params.DeviceParameterSource`.
  - `FlowGrid(QWidget)` in `flow_grid.py`: `add(widget)`, `clear()`. Column count is `max(1, width // CELL_WIDTH)` with `CELL_WIDTH = 170`, recomputed in `resizeEvent`, filled row by row.
  - `last_set_text(entry: dict | None, kind: str) -> str` in `cells.py`: `None` → `"unknown"`; AO → `"last set 1.250 V at 14:02"`; DO → `"last set high at 14:02"`.
  - `ChannelCell(QFrame)` in `cells.py`, `objectName("ChannelCell")`, built from a CHANNEL dict.
    - Shows the aliases joined by `" / "`, or the physical name when unaliased, plus a muted address line: `port3/line5`, `ao17 · 0…1 V`, `ai4 · Rse`.
    - Per kind:
      - DO: click toggles and emits `write_requested(physical, bool)`.
      - AO: `CompactDoubleSpinBox` limited to `range`, committed on Enter, emits `write_requested(physical, float)`.
      - AI and DI: click toggles watch and emits `watch_toggled(physical, bool)`. While watching it shows the value (`"2.413 V"` / `"high"`) and, for AI, a `Sparkline` (custom-painted polyline of the last 100 readings).
    - `set_hold(holder: str | None)` disables the cell and shows `"held by <holder>"`, or `"protected"`. `set_last(entry)` and `set_reading(value)`.
    - Dynamic property `state` ∈ `"idle" | "high" | "watching" | "held"` via `set_state`.
  - `NICardsPanel(QWidget)` in `panel.py`: `NICardsPanel(devices: list[str] | None = None, make_client: Callable[[str], CardsClient] | None = None)`. One sub-tab per device, labelled `f"{host name} — {device}"`, using the `HOSTS` name for the entry's host address.
    - Each card is a `Region("well", f"{board}", f"{address} · {model}")` with AI / AO / DI / DO `FlowGrid` sections. The unaliased channels go in a collapsed `QToolButton`-toggled "Unaliased channels" section at the bottom.
    - A `QLabel#CardBanner` reads `"Held by <holder>"` when the card's address is in `card_holds`. A card with `error` shows it as the hint, with every cell disabled.
    - Timers run only while visible: `fetch_state()` every 1 s, and `read(watched)` every 200 ms while anything is watched.
    - On `state_received`, apply holds and last-set to the cells, and drop any watched physical that is now held.
    - On `connection_changed(False)`, stop the read timer, disable the cells and show `"<device> is not reachable. Start it in the Devices tab."` On reconnect, re-fetch the layout.
    - `layout()["error"]` is shown in place of the cards.
    - `closeEvent` closes the clients.
- Theme rules in `theme.py`:
  - `QFrame#ChannelCell` uses the well-on-sheet field colour `#30313b` with a `#44454f` edge.
  - `[state="high"]` gets a `#2ecc71` edge and tinted background.
  - `[state="watching"]` gets the accent `#5b8def` edge.
  - `[state="held"]` uses muted text `#8d8e99` and a transparent background.
  - `QLabel#CardBanner` uses a warning `#f5a623` edge and tint.

- [ ] **Step 1: Write the failing tests** in `tests/test_ni_cards_gui.py`, using a `FakeClient` QObject with the same signals as `CardsClient`, which records calls:
  - `test_ni_card_devices_picks_entries_by_class`: given a config dict with an NI entry, a camera entry and a second NI entry, returns the two NI names in order.
  - `test_last_set_text`, parametrised over `None`, an AO entry and a DO entry, against the exact strings above, with the expected time written as `time.strftime("%H:%M", time.localtime(ts))` for the entry's `ts` so the test is timezone-independent.
  - `test_flow_grid_column_count_follows_width`: widths 170, 510 and 1000 give 1, 3 and 5 columns.
  - `test_panel_builds_offscreen_with_a_layout`: emitting a two-card layout creates one cell per channel, and the unaliased section starts collapsed.
  - `test_state_marks_held_cells_and_card_banner`: a `state` with one held physical and one `card_holds` entry disables that cell with `"held by Wavemeter Lock"` and shows the banner on that card.
  - `test_channel_becoming_held_stops_being_watched`: watch an AI cell, then emit a `state` holding it; the next read timer tick does not include it. *(spec: mid-use release)*
  - `test_unreachable_device_stops_reads_and_says_so`: watch a cell, emit `connection_changed(False)`, and check the read timer is stopped and the "not reachable" text is visible. *(Review Focus 4)*
  - `test_other_gui_write_shows_as_last_set`: emit a `state` whose `last_set` has an entry for a DO cell; the cell's text is `last_set_text(...)` of it. *(Review Focus 5)*
  - `test_do_click_requests_write_of_opposite_state`: a DO cell last set high, when clicked, calls `client.write_do(physical, False)`.

- [ ] **Step 2: Run** `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_ni_cards_gui.py -q`. Expected: FAIL.

- [ ] **Step 3: Implement** `client.py`, `flow_grid.py`, `cells.py`, `panel.py` and the theme rules, and register the tab in `bin/gui.py`.

- [ ] **Step 4: Run** `uv run pytest tests/ -q`. Expected: PASS.

- [ ] **Step 5: Look at it.**
  - Write a scratch factory that builds `NICardsPanel(devices=["CaF NI Cards"])` against `SimulatedNICards` on `../EDMSuite/DAQ/MoleculeMOTHardware.cs`, with the config's reservations.
  - Shoot it: `QT_QPA_PLATFORM=offscreen uv run python .claude/skills/pytweezer-gui-design/scripts/shoot.py fakes:panel <scratchpad>/ni.png --path <scratchpad> --size 1600x1000 --wait 2`. Do this at 1600 and at 800 px wide.
  - Check against the agreed mockup (`layout-v2`): cells reflow, tclBoard2 has a "Held by Hardware Control" banner, wavemeter-lock cells say "held by Wavemeter Lock", the three coil channels say "protected", and the unaliased sections are collapsed.
  - Fix what floats, truncates or crams (`pytweezer-gui-design` loop), then screenshot the whole window via the `run-pytweezer` skill.
- [ ] **Step 6: Commit and push.** Put the screenshots in the PR description.
