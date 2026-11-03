# Handoff: cut MOTMaster's per-`Go()` overhead

For whoever picks this up on a MOTMaster PC (the Rb PC, IC-CZC221CCHS, is where
it was measured). Written 2026-11-03 on PH-BEAST, which has no current EDMSuite
checkout and cannot build MOTMaster.

## The problem

A pytweezer experiment with N repetitions runs N separate `Go()` calls, one
point per shot. It is much slower than one `Go()` with N MOTMaster iterations.

## What was measured (Rb, `LoadTweezers`, RbTweezerBasic2026_2)

| | per shot |
|---|---|
| Sequence: `PatternLength` 60000 at the default 100 kHz clock | 0.60 s |
| pytweezer point time, `t_end - t_start` (rids 128–153, stable to ±10 ms) | 0.99 s |
| **Overhead** | **≈ 0.39 s** |

How it splits:

- **pytweezer between points: ~1 ms.** Next `t_start` minus previous `t_end`.
  The queue, manager, HDF5 and database cost nothing worth chasing.
- **pytweezer inside a point: ~20 ms.** Camera start/stop, tophat and site
  sums, and frame transfer, each timed separately.
- **Driver sleep: 0.10 s, already removed.** `MotMasterInterface.start_motmaster_experiment`
  slept `self.interval` after `Go()`. It is gone in pytweezer `3237c63` (on
  main). `Go()` is synchronous, holding `runLock` until the run and any zip are
  done. **Not live until the Rb PC pulls pytweezer and restarts the
  `Rb MotMaster` device server.**
- **Script compile on every `Go()`: 0.163 s.** Timed through
  `get_device("Rb MotMaster").get_params()`, which runs the same
  `prepareScript` compile without firing anything. Repeatable to ±5 ms, against
  3 ms for a bare RPC (`is_linked()`).
- **~0.1 s not yet measured.** It is presumably the rest of the work done once
  per `Go()`: `GetSequence`, `buildPattern`, `setSafeOutputs`,
  `beginRunOutputs`, `clearDigitalPattern` and the two
  `controllerWindow.RefreshRunControls()` calls.

## Where the time goes in MOTMaster

References are to `MOTMaster/Controller.cs` at EDMSuite master `b0243324`.

- `Go(dict)` → `goClaimed(dict)` → `prepareScript(scriptPath, dict)`, or
  `prepareScriptOrThrow(linkScriptPath, dict)` when linked. Both compile with
  `CodeDom` `CompileAssemblyFromFile` (`compileFromFile` /
  `compileFromFileOrThrow`), load the assembly (`loadScriptFromDLL` /
  `loadScriptOrThrow` → `createScriptInstance`, "last class wins"), and only
  then call `script.EditDictionary(dict)`. **The compile does not depend on
  the parameters**, so caching it helps scans as well as repetitions.
- Every compile loads another assembly that is never unloaded. MOTMaster's
  memory therefore grows with every shot; the cache fixes that too.
- `GetParameters`, `GetSwitchConfiguration` and the pattern preview also go
  through `prepareScript`. pytweezer calls `GetParameters` once per task in
  `prepare()`.
- When linked, the primary compiles a temp copy at a **new path every run**
  (`LinkedScriptBuilder.PrimaryScriptPathFor(linkRunId, ...)`). The secondary
  compiles too (`LinkedScriptBuilder.BuildSecondary(..., compileAndLoadOrThrow)`,
  about line 1325).
- In iterations mode the loop calls `runPattern(sequence)` repeatedly on one
  built sequence. Only `initializeHardware` / `run` / `releaseHardware` are paid
  per iteration, so running a sequence does not consume it. After the loop,
  `clearDigitalPattern(sequence)` clears the digital boards.

## The work

Read EDMSuite's own `CLAUDE.md` and its `motmaster-controller` and
`motmaster-linked-mode` skills first. **Work on a new EDMSuite branch** (e.g.
`motmaster-script-compile-cache`) and open a PR. `Controller.cs` is shared by
every MOTMaster build (Rb and CaF), so CaF gets the change too.

### 1. Compile cache (do this)

- Key on a **hash of the script text**, not path + mtime: the linked temp copy
  has a fresh path every run. Read the file once, hash it, and compile from
  that same text on a miss (or compile the file, if the text is known to match).
- Cache the compiled `Assembly`/`CompilerResults`. Keep making a **fresh script
  instance per run**, as now. Don't cache failed compiles. Keep the
  MessageBox-vs-throw split between the normal and linked paths.
- Route every `prepareScript*` caller through it, including the linked
  secondary's `compileAndLoadOrThrow`.
- `Go` arrives on remoting threads and from the Run button, so guard the cache
  with a lock.
- **Watch for static fields in scripts.** Every compile used to be a new
  assembly with fresh statics. With a cache, a static field in a script class
  now keeps its value from one run to the next. Grep the script folders (`RbTweezerMasterScripts`,
  `MoleculeMOTMasterScripts`, ...) for `static` fields that are written, and
  report any you find before relying on the cache.
- Only the script text is in the key, so a change to a DLL the scripts
  reference (DAQ, SharedCode) is not seen. That needs a MOTMaster restart
  anyway, since MOTMaster loads them, so say so in a comment and move on.

Expected gain: about 0.16 s per shot, on every shot, scans included.

### 2. Measure before a sequence cache

On the Rb PC, with pytweezer's Rb MotMaster device server running, the script
set and saving off: time `start_motmaster_experiment()` with `set_iterations(1)`
and with `set_iterations(5)`, a few times each, both before and after the
compile cache. Ask the user before firing; it is the normal sequence, about 10
shots. The difference divided by 4 is the cost per iteration. What remains of
the iterations = 1 time, less 0.60 s, is the cost paid once per `Go()`.

### 3. Sequence cache (only if step 2 shows it is worth it)

The idea is to reuse the built `MOTMasterSequence` when the script-text hash
and the full `script.Parameters`, after `EditDictionary`, match a cached
entry. This skips `GetSequence` and `buildPattern`.

- It only helps repeated parameter sets: repetitions, or a shuffled or
  repeated scan. Every new value in a scan misses. A small LRU (tens of
  entries, a few MB each) catches shuffled scans; keeping only the last
  sequence does not.
- `clearDigitalPattern` must not clear a cached sequence. Check what else
  mutates a sequence after `buildPattern`.
- It assumes `GetSequence` is deterministic for the same parameters. Grep the
  scripts for `Random`, `DateTime`, file reads and static state.
- Leave linked runs out of it at first (`BuildPrimary` and the secondary's
  handshake).

## Verifying

- In pytweezer: submit `LoadTweezers` with `Scan(repetitions=20)` and compare
  the mean of `points/t_end - points/t_start` in its h5 file with the 0.99 s
  above. Expect about 0.73 s after the sleep removal and the compile cache.
- MOTMaster's output must not change. With saving on, compare the pattern files
  in a zip from a cached run with one from an uncached run (first `Go()` after
  a restart). Repeat with a scanned parameter, to show that cached runs still
  pick up new values.
- Run a linked task (`experiments/linked_scope_test.py`) once to show the
  linked path still works.

## Out of scope here, for later

- `MotMasterExperiment.follower_arm_delay` (0.5 s) is added to every shot of a
  two-MOTMaster experiment. It could probably be much shorter. `LoadTweezers`
  doesn't use it.
- To match iterations mode exactly, k identical repetitions could run as one
  `Go()` with iterations = k, with the HamCam buffering 2k frames. That is a
  design change in pytweezer: pause and terminate would only act between batches.
