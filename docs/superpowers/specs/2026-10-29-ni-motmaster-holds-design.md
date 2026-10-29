# NI Cards: MOTMaster holds and end-of-run values — design

Follows `2026-10-29-ni-test-panel-design.md` (the NI Cards tab, merged in #33),
whose **Reservations → later** section this replaces and extends.

## Intent

The NI Cards tab drives the same pattern boards that MOTMaster drives. Today
nothing stops a panel write landing during a sequence, and after a sequence the
tab's "last set" values are stale: the outputs are wherever the pattern left
them. This work makes the tab and its driver aware of MOTMaster:

- while MOTMaster is running a sequence, **or** a queued task holds that
  MOTMaster, its pattern boards are greyed out and the driver refuses writes
  and reads on them;
- after each sequence, every channel the pattern drove shows the value the
  pattern ended on, labelled as such.

MOTMaster itself reports both (whether it is running, and what its last run
ended on), so this covers runs from MOTMaster's own window and from notebooks,
not only runs through the queue.

## Goals

- No panel write or read reaches a MOTMaster pattern board while MOTMaster is
  running a sequence, whoever started it.
- No panel write reaches those boards between the shots of a queued run: the
  whole queued task holds them, from start to finish (including scans). *(User's
  decision.)*
- After a run, the tab shows each pattern channel's end-of-sequence value, from
  MOTMaster's own record, so it does not rely on the run having gone through
  pytweezer.
- Enforced in the driver as well as the tab, as the static reservations are.

## Non-goals

- No push notifications from MOTMaster. pytweezer polls. *(User's decision.)*
- No per-sequence board detection: a hold covers all of MOTMaster's pattern
  boards, not just the ones the current script uses (see **What is held**).
- No change to how MOTMaster runs, builds, saves or stops a sequence.
- No readback of the hardware: end-of-run values are what the pattern says the
  last sample was, not a measurement.

## Anti-goals

- **Any change in MOTMaster's behaviour for people who do not use it from
  Python over .NET remoting.** The C# additions are read-only methods plus
  passive bookkeeping at the end of a run. Nothing in the GUI, the run path,
  saving, or timing changes, and MOTMaster never learns whether anyone is
  polling. Bookkeeping that fails must never fail or delay a run. *(User's
  constraint.)*
- A stale or invented value presented as the output state. An aborted or failed
  run, or a run whose end values are unknown, resets the affected channels to
  "unknown"; it does not leave the previous value up.
- The NI device server failing to start, or the tab breaking, because MOTMaster
  or the Experiment Manager is not running.

## Constraints

- **EDMSuite safety**: automated work never runs a MOTMaster script, never calls
  `Run`/`Go`, never launches MOTMaster to click through it (EDMSuite's
  `motmaster-controller` skill). C# changes are verified by compiling and by
  `MOTMasterTests`, not by running the experiment.
- **EDMSuite branching**: the C# change goes on its own branch off
  `feature/linked-motmaster`, which the lab PCs run and which has the linked
  mode and `MOTMasterTests` this relies on (`master` has neither). Work in a
  separate worktree. *(User's decision.)*
- The MOTMaster device for a PC is on the same PC as that PC's NI cards
  (`CaF MotMaster` and `CaF NI Cards` on ph-bonesaw; `Rb MotMaster` and
  `Rb NI Cards` on IC-CZC221CCHS).

## How it works

### MOTMaster (C#, additive)

`Controller.IsRunning` already exists and is public. It is true from the
moment a run claims the hardware until it has fully finished, and it covers
a linked secondary's shots as well (`PrepareLinkedRun` sets the same flag).
No change is needed for running state.

Add one read-only method to `Controller`, callable over remoting, returning
the **last run outputs**: a record of the most recent run that has finished:

- a run number that increases with every finished run;
- the script name;
- the time it finished;
- whether its **outputs are known**, and if so each channel's value at the
  pattern's last sample: volts for analogue channels, high/low for digital,
  keyed by MOTMaster channel name.

"Outputs known" means the last shot's pattern played to its end, whatever
happened to the run as a whole. *(Builder-check call.)* So:

- A Stop between shots of an iteration run: known, because the last shot
  finished.
- A failure after the final shot, such as saving throwing or the data check
  returning early: known.
- A Stop that cuts a shot short, which is how a run-until-stopped run usually
  ends: unknown. The outputs really are mid-pattern.
- An exception or a lost link mid-shot: unknown.

The values are taken when a shot's pattern finishes, before
`clearDigitalPattern` empties it. Each linked side records its own shots. Only
channels present in the run's patterns are listed. A Debug-mode run drives no
hardware, so its outputs are unknown. If the bookkeeping throws, the exception
is caught and logged, and the outputs are unknown.

MOTMaster's pattern boards are not asked for over remoting. They come from the
hardware file (see **What is held**).

Return types must cross .NET remoting into pythonnet the way
`GetExperimentReport()`'s `Dictionary<String, Object>` already does.

### What is held

A MOTMaster hold covers **every channel on MOTMaster's pattern boards**,
aliased or not, as a whole-card hold (banner, every cell disabled).

The pattern boards are read from the hardware file the driver already parses,
from its `PatternGeneratorBoard`, `AdditionalPatternGeneratorBoards` and
`AnalogBoards` info. For CaF these are digitalPattern (`/Dev1`),
digitalPattern2, analogPattern (`/PXI1Slot2`) and analogPattern2. For Rb they
are `pg` and `aoBoard`. Reading them from the file means the list is known in
simulation, and while MOTMaster is unreachable, so a queued-run hold works in
both cases. *(Builder-check call.)* A hardware file that names no pattern
boards gets no MOTMaster holds, and the tab shows a notice saying so. Holding
whole boards rather than the channels a script uses keeps it simple and safe.
Digital pattern boards are driven a whole port at a time, so lines a script
never names are driven too.

Two sources, added to the driver's `Reservations` alongside the static one:

- **MOTMaster running**: `IsRunning` is true. Holder text: "CaF MotMaster
  (running a sequence)".
- **Queued run**: a running task in the Experiment Manager's snapshot lists that
  MOTMaster in its `devices`. A linked task lists both MOTMasters, so both PCs'
  boards are held. Holder text: "CaF MotMaster (queued run 42)".

Precedence is unchanged: protected first, then the static reservations, then
these. A channel held for any reason refuses writes and reads in the driver.
The tab already stops watching a channel when it becomes held.

`Reservations` sources today return `{physical: holder}` only. They also need
to return whole-card holds (`{address: holder}`), so that the dynamic sources
can raise the card banner. `state()["card_holds"]` becomes the merge of every
source's card holds.

### Polling

- **Display**: the driver asks MOTMaster for `IsRunning` and for the last-run
  record whenever the tab requests `state()` (about once a second while the tab
  is visible). With no tab open, nothing polls.
- **Enforcement**: every write and read asks `IsRunning` fresh, immediately
  before touching the hardware. A refusal does not depend on a poll having
  noticed the run.
- **Missed runs**: the run number means a run that starts and finishes between
  polls still updates the end values. Only the latest run matters.
- **Queue**: the driver keeps a `SyncMirror` (`pytweezer/servers/sync.py`) of
  the manager's `"experiment"` notifier. It is thread-safe and reconnects on its
  own.
- **Known gap**: a run can start between the `IsRunning` check and the write.
  The write then either lands just before the pattern starts, and the pattern
  overwrites it, or it hits lines the pattern task already holds, and DAQmx is
  expected to refuse it, which the tab reports as an error. This is accepted.

### End-of-run values

When the last-run record's run number changes:

- **Outputs known**: each listed analogue channel (and digital, once the
  hardware check allows it; see **Assumptions to verify on hardware**) whose
  MOTMaster name is an alias in
  the hardware file gets a "last set" entry with the value, the run's finish
  time, `source: "motmaster"` and the script name. Every other channel on
  MOTMaster's boards resets to unknown, since the pattern may have driven it.
- **Outputs unknown**: every channel on MOTMaster's boards resets to
  unknown.

Safe-state channels are protected and show no value either way. MOTMaster sets
them to their safe values after every run.

Whichever is newer wins between a panel write and a MOTMaster end value. That
falls out naturally: a panel write on those boards can only happen after the
run.

When the NI device server starts, it reads the last-run record once, so a
restart shows the previous run's end values rather than "unknown" everywhere.

### Config

Each NI Cards entry gains `"motmaster": "<MOTMaster device name>"`
(`"CaF MotMaster"`, `"Rb MotMaster"`). The driver gets MOTMaster's remoting
address from that device's own config, in the same way the MotMaster driver
connects (`Activator.GetObject` on its `remote_path`). The key is optional: an
entry without it has no MOTMaster holds.

### Failure behaviour (fail-open, said in the tab)

- **MOTMaster not reachable** (closed, or remoting not up): no "running" hold,
  and no end-of-run values. `state()` carries a notice, "CaF MotMaster is not
  reachable: its runs are not greyed out", which the tab shows as a hint line.
  A closed MOTMaster cannot run a sequence, so failing open there is safe.
  Writes still check `IsRunning` and go ahead only when the answer is "not
  running" or MOTMaster can't be reached.
- **Experiment Manager not reachable**: no queued-run holds, and a similar
  notice. MOTMaster's own `IsRunning` still greys the boards during each shot.
- The NI device server always starts. Neither MOTMaster nor the manager being
  down is an error at construction.

### The tab

- A last-set value from MOTMaster reads "end of `LoadTweezer`: 1.25 V at
  14:02" (DO: "end of `LoadTweezer`: high at 14:02"). A panel write still reads
  "last set 1.25 V at 14:02".
- MOTMaster holds use the existing card banner and held-cell styling, with the
  holder text above.
- Notices from `state()` are shown as a hint line under the sub-tab's header.

### Simulation

`SimulatedNICards` gets a simulated MOTMaster source. MOTMaster never runs in
simulation, so it is always "not running" with no last run, and no notice is
shown. The queue source works in simulation as normal: a simulated queued task
that reserves a MOTMaster greys its boards.

## Assumptions to verify on hardware

- **Digital lines are assumed to go off after a run.** Nobody yet knows whether
  the PCI-6534 (`/Dev1`) and the PXI-6535-class board keep the pattern's last
  sample once the pattern task stops, or tristate or reset. Build on the
  assumption that they do not hold it. *(User's decision.)* MOTMaster still
  reports digital end values, but pytweezer does not show them: after a run,
  every digital channel on MOTMaster's boards shows "unknown". Analogue
  channels show their end values as designed, since MOTMaster's own comment
  says a stopped task "leaves its outputs at whatever value they had". Leave a
  `TODO` at the switch: check on the rig whether the digital lines hold, and if
  they do, turn digital end values on.
- A static write to a pattern board while its pattern task runs is refused by
  DAQmx (this only matters for the known gap above).

## Left to the builder

- The name and exact signature of the C# method.
- How the driver caches `IsRunning` for `state()` (writes always ask fresh).
- Where the `SyncMirror` lives and how the driver closes it on teardown.
- Splitting the remoting connection out of the MotMaster driver for reuse,
  versus a small dedicated client in `ni_cards.py`.
- C# tests in `MOTMasterTests` for the end-of-run record (each case under
  "outputs known", and Debug), alongside the pytweezer tests.
