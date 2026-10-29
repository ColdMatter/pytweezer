# NI test panel — design

## Intent

A GUI tab for driving and reading individual channels on the lab PCs' NI
cards by hand, like the test panels in NI MAX, but with channels named by
their MOTMaster aliases (`MOT1Shutter`, `xShimCoilCurrent`) rather than by
physical address (`/PXI1Slot4/port3/line5`). It is for poking hardware between
runs: open a shutter, put a voltage on a shim coil, watch a photodiode.

The aliases already exist in the MOTMaster hardware files
(`EDMSuite/DAQ/MoleculeMOTHardware.cs` for CaF on ph-bonesaw,
`RbTweezerHardware.cs` for Rb on IC-CZC221CCHS). The panel reads them from
there, so a new `AddDigitalOutputChannel(...)` line shows up without touching
pytweezer. `CONFIG` says which PCs exist, which hardware file describes each,
and which channels are reserved.

## Goals

- One place, in the GUI, to set any DO line or AO channel and read any AI
  channel on any configured PC, by alias.
- Never fight another program for a channel: channels or cards that are
  reserved are shown but greyed out and cannot be driven or read.
- Read inputs only while the user is looking at them. Nothing polls in the
  background.

## Non-goals

- No live detection of whether MOTMaster, the wavemeter lock or the hardware
  controller is running. Reservations are static (config) for now; see
  **Reservations**.
- MOTMaster greying is not built yet: it waits for `experiment-pipelines`
  (see **Reservations → later**). The design leaves room for it.
- No readback of output state. On most of these cards an AO's present value
  cannot be read back, and DO readback is not reliable on every card. A cell
  shows the last value set from the panel, or "unknown".
- No timed/buffered output or acquisition, no counters, no waveform
  generation. Single static writes and on-demand reads only.
- No editing of the hardware files from pytweezer.

## Anti-goals

- Writing to a channel another program owns (wavemeter-lock outputs, the
  hardware controller's card, a card MOTMaster is mid-sequence on), or to a
  protected channel.
- The panel holding a DAQmx task open on a card when it is not actively
  writing or watching, so that it blocks MOTMaster or another program.
- A stale or invented value presented as the real output state.

## Where it lives

A new tab in the GUI (`bin/gui.py`, alongside Experiments and Results), in
both the server and client GUIs. Client GUIs are not view-only: they drive
outputs too.

## Layout (agreed with mockups)

- One sub-tab per PC (e.g. "ph-bonesaw (CaF)", "IC-CZC221CCHS (Rb)").
- Within it, one block per card, headed by the board's name from the
  hardware file (`digitalPattern2`), its address (`/PXI1Slot4`) and its model
  (`PXI-6535`, from NI-DAQmx on that PC).
- Inside each block, channels are cells in a **grid** that reflows to the
  window width (as many columns as fit), not a list. On a card with several
  kinds of channel, cells are grouped into AI, AO and DO sub-sections.
- Cells:
  - **DO**: alias, `port/line`, a high/low state; click toggles.
  - **AO**: alias, `aoN`, declared range if any, an inline value entry.
  - **AI**: alias, `aiN`, terminal configuration (e.g. RSE); click toggles
    "watch", which shows a live value and a short trace. Unwatched cells show
    no value.
- Two aliases on one physical line (e.g. `MOT3Shutter` and `rbOPShutter` on
  Slot4 port3/line7) share one cell showing both names.
- **DI** (`AddDigitalInputChannel`, e.g. `bXLockBlockFlag`): watched like an
  AI cell, showing high/low. *(Builder-check call.)*
- Physical lines/channels on a card with **no alias** go in their own
  section(s) at the bottom of that card's block, collapsed by default, named
  by physical address. They are fully usable: unaliased digital lines are
  driven as DO cells *(builder-check call: as outputs, not inputs)*, unaliased
  AOs are settable within the card's range, unaliased AIs are watchable.
- Reserved cells are greyed out and say who holds them ("held by Wavemeter
  Lock"). A card reserved as a whole (Hardware Control's tclBoard2; later, a
  card MOTMaster is running on) is greyed as a whole with a banner naming the
  holder ("Held by Hardware Control", "CaF MOTMaster is running a sequence on
  this card"), including its unaliased sections.
- Protected cells are greyed out and say "protected".

Mockups are in `.superpowers/brainstorm/` (local, not committed);
`layout-v2.html` is the agreed one.

## Reading the hardware files

- Channels come from the four declaration calls:
  `AddAnalogOutputChannel(name, board + "/aoN"[, min, max])`,
  `AddAnalogInputChannel(name, board + "/aiN", AITerminalConfiguration.X)`,
  `AddDigitalOutputChannel(name, board, port, line)`,
  `AddDigitalInputChannel(name, board, port, line)`.
- Boards come from `Boards.Add(name, address)`, with the address usually
  bound to a local string variable first (`string tclBoard1Address =
  "/PXI1Slot3";`), sometimes inline (`Boards.Add("pg", "/PXI1Slot5")`) and
  referenced via variables or `(string)Boards["x"]`. Resolve those.
- Ignore commented-out code: `//` lines and `/* ... */` blocks.
- Aliases on boards whose `Boards.Add` is commented out (e.g. Rb's `daqBoard`)
  are skipped.
- An AO declared with `min, max` is clamped to that range in the panel.
  Without one, use the card's own AO range.
- Only these declarations matter. Everything else in the file (Info entries,
  instruments, calibrations, WML/TCL config) is ignored.

## Config

Each PC with cards gets an entry in `CONFIG["Devices"]` (e.g. "CaF NI Cards"
on `HOSTS["ph-bonesaw"]`, "Rb NI Cards" on `HOSTS["IC-CZC221CCHS"]`), naming:

- the path to the hardware file on that PC (its EDMSuite checkout, the source
  of truth for what is compiled there);
- the statically reserved channels and cards, grouped by holder. A holder can
  reserve individual aliases or whole cards (by board name), e.g.
  `{"Wavemeter Lock": {"channels": ["v00Lock", "v10Lock", ...]},
  "Hardware Control": {"cards": ["tclBoard2"]}}`;
- the protected channels, by alias (see **Protected channels**);
- *(later, with pipelines)* which MOTMaster device entry (e.g.
  `"CaF MotMaster"`) drives which of its cards. Leave the schema open to
  adding this key; not needed now.

## Reservations

Reservations come from **sources**. Build the panel and driver around a
single "what is reserved, and by whom" view assembled from those sources, so
that a dynamic source can be added later without reshaping the cells or the
driver. Only the static source exists now.

- **Static (now)**: config lists, per holder, reserved aliases and reserved
  whole cards. Reserved things are always greyed out with the holder's name;
  they cannot be written or watched.
  - Wavemeter Lock: its outputs `v00Lock`, `v10Lock`, `v21Lock`, `v32Lock`,
    `bXLock`, `bXBeastLock`.
  - Hardware Control: the **whole of tclBoard2** (`/PXI1Slot8`), every
    channel aliased or not, because it holds AI tasks there and the card runs
    only one AI task at a time. Narrowing this is not in scope.
- **MOTMaster (later, with `experiment-pipelines`)**: that branch (design in
  `docs/superpowers/specs/2026-10-29-experiment-pipelines-design.md` on the
  `experiment-pipelines` branch, not yet on `main`) gives every `Task` the set
  of CONFIG device names it reserves; a linked run reserves both MOTMasters.
  When it lands, a second source follows the Experiment Manager's snapshot and
  reserves a MOTMaster's cards (the config key above) while a running task
  reserves that MOTMaster. With it come: stop watching / release tasks on a
  channel that becomes reserved mid-use, and reset "last set" values on that
  MOTMaster's cards to "unknown" when the reservation ends. Build the
  mid-use release generally now (it applies to any source); the rest is for
  then. Runs started outside the queue (MOTMaster's own window, a notebook
  `Rig`) will not grey anything even then; this is accepted. The pipelines
  design releases a reservation as soon as a task ends, including on abort
  while a sequence may still be finishing; the panel will inherit that.

Until then, a MOTMaster run does not grey or reset anything, and a panel write
during a run can collide with it. The "last set at HH:MM" label is what keeps
displayed values honest. *(Builder-check call.)*

## Protected channels

A config list of aliases that the panel cannot touch at all: greyed out,
marked "protected", never written or read, in every GUI. The driver enforces
it, not just the GUI. Seeded with MOTMaster's safe-state channels
(`MOTCoilsCurrent`, `transferCoils`, `TweezerCoils`, from
`DAQMxAnalogSafeState` in `MoleculeMOTHardware.cs`) *(builder-check call:
the seed list)*. Add Rb's equivalents if its hardware file declares any.

## Hardware access

- A device driver run by the standard device server on each PC
  (`get_device()`), using `nidaqmx` locally. It parses that PC's hardware
  file, enumerates each card's physical channels and model through NI-DAQmx
  (for the unaliased sections and the card headers), and performs writes and
  reads.
- Writes are single static writes; open a task, write, close it. Do not keep
  output tasks open.
- An AI watch holds a task only while at least one GUI is watching a channel
  on that card.
- The last value written to each output is held by the device server, so
  every GUI on every PC shows the same "last set" value, labelled as last set
  from the panel with its time, never as the present output. It resets to
  "unknown" when the device server restarts (and, later, when a MOTMaster
  reservation on that card ends).
- Reservations and protection are enforced in the driver as well as the GUI,
  so a notebook calling the driver directly cannot bypass them.
  *(Builder-check call.)*
- It must simulate (no NI hardware, no hardware file needed beyond a copy) so
  the tab works in the simulating server GUI, like the other devices.

## Left to the builder

- Module layout, class names and the config keys' exact names.
- The parser's implementation (regex over stripped source is expected to be
  enough; it need not be a C# parser).
- AI watch rate and trace length; how AO values are committed (Enter / focus
  out); the DO cell's exact look within `pytweezer-gui-design`'s rules.
- Base: `main`, branch `ni-test-panel`. Do not depend on `experiment-pipelines`.
