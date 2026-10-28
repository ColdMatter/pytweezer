# pytweezer ↔ linked MOTMaster: plan and first-cut code

Status: code written and unit-tested off the rig; **not yet run against real
MOTMasters**. The live steps below are for a person at the machines. pytweezer
never fires a sequence on its own initiative.

## What linked mode is (MOTMaster side, already built in EDMSuite)

One script on the **primary** (PH-BONESAW, CaF MOTMaster) drives the pattern
generators of both PH-BONESAW and the **secondary** (RbTweezer, `10.59.3.2:1187`).
The primary claims the secondary over .NET remoting (`Controller.SetLinked(true)`),
sends it the script text and full parameter set on each `Go`, and fires it with
a TTL on `rbTrigger`. Every failure of a linked run, on either machine, is
thrown from the primary's `Go`. A plain unlinked run can fail without raising:
it shows a MessageBox and `Go` returns normally. While claimed, the secondary
refuses its own `Go` / `SetScriptPath` / `SetTriggered`. Full details: EDMSuite
`.claude/skills/motmaster-linked-mode` and
`docs/superpowers/specs/2026-10-22-linked-motmaster-design.md`.

The remoting surface pytweezer needs: `SetLinked(bool)` and the `IsLinked`
property on the primary's `Controller`. `SetScriptPath` / `GetParameters` / `Go`
stay as they are.

## How pytweezer uses it now

Before this change, pytweezer ran two machines with **two** `MotMaster` devices.
The follower went into trigger mode, its `Go` was started in a thread, and the
master was started `follower_arm_delay` later (`MotMasterExperiment.run_sequences`).
Linked mode replaces that pattern for CaF+Rb: **one** device, one script, one
`Go`, and timing set by `SecondaryTriggerDelay` rather than a software sleep.

## Design

```python
class LinkedLoad(MotMasterExperiment):
    mm = MotMaster("CaF MotMaster", script="LinkedTweezerLoad", linked=True)
    delay = MotMasterInteger(0, parameter="SecondaryTriggerDelay")
    tof = MotMasterInteger(100, parameter="tDelay1")      # an Rb-side parameter
```

- `MotMaster(..., linked=True)` must be the only `MotMaster` in the class. This
  is checked when the class is defined. All parameters for both machines belong
  to the one script.
- **The link lasts one task.** `prepare()` calls `set_linked(...)` on every
  declared sequencer, unlinking ones first. `close_devices()` always runs at task
  end, even on failure or when a subclass overrides `finish()`, and it releases
  the link there. So a finished linked task never leaves RbTweezer locked
  against Rb-only tasks. An unlinked task on CaF also unticks a Linked box left
  on by hand.
- **Driver** (`MotMasterInterface`):
  - `set_linked(value)`: turning it on when MOTMaster already reports linked
    keeps the existing claim rather than re-claiming. Turning it off on a
    MOTMaster build that predates linked mode does nothing.
  - `is_linked()`: reads MOTMaster's own state.
  - `start_motmaster_experiment` refuses to run if pytweezer asked for a link but
    MOTMaster has dropped it. MOTMaster drops the link by itself when RbTweezer's
    user presses *Take back control*. Without this check, the next `Go` would run
    a linked script on CaF alone.
- **Simulation:** `SimulatedMotMasterInterface` has a hand-written
  `set_linked` / `is_linked`, and canned parameters for `LinkedScopeTest`.
- **Example / first live test:** `pytweezer/experiments/linked_scope_test.py`
  runs EDMSuite's `MoleculeMOTMasterScripts/LinkedScopeTest.cs`, one 10 ms pulse
  on each machine. Scan `secondary_trigger_delay` with both pulses on a scope.
- **GUI:** the Experiments-tab MOTMaster box of a linked sequencer is titled
  "…(script, linked)", with a tooltip saying the secondary's parameters are set
  there too.
- **Config fix:** `rb_mm_config.json` pointed at `MOTMaster\bin\x86\RbTweezer\`
  and `DAQ\bin\x86\RbTweezer\`. Those are dead folders from the old solution
  mapping, and a MOTMaster started from there is stale and 32-bit (the primary
  refuses a bitness mismatch). They now point at `bin\RbTweezer\`, which is the
  output of `Configuration=RbTweezer`.

## Live test plan (done by a person at the rig)

0. Both PCs are on the same EDMSuite commit. RbTweezer is built as
   `Configuration=RbTweezer` into `MOTMaster\bin\RbTweezer\`. Pull this branch on
   both PCs, then restart the "Rb MotMaster" and "CaF MotMaster" device servers
   (the Rb one picks up the new exe path).
1. **Claim/release only, no pattern.** From a notebook:
   `mm = get_device("CaF MotMaster"); mm.set_linked(True); mm.is_linked()`.
   Expect `True` and the gold banner on RbTweezer. Then `mm.set_linked(False)`:
   the banner goes.
2. **Scope check.** Queue `LinkedScopeTest` with probes as listed in
   `LinkedScopeTest.cs`. Expect both pulses each shot, and the Linked box ticked
   during the task and unticked after it. Scan `secondary_trigger_delay` to
   cancel the measured lag.
3. **Failure paths.**
   - Press *Take back control* on RbTweezer between points of a long scan. The
     task should fail with the link-lost message, and the next linked task should
     re-claim.
   - Put RbTweezer in Debug. A linked task should fail with MOTMaster's
     "Debug mode" message.
   - Check the messages read sensibly in the Experiments tab. .NET exceptions
     cross the device RPC as text.
4. **No regressions.** Run an Rb-only task (e.g. `TweezerBaseline`) right after
   a linked task. It should run normally, because the link was released.

## Decisions to confirm

- **Task-scoped link** (above), rather than leaving the link on between tasks.
  It costs one claim round-trip per task.
- **An unlinked CaF task turns linked mode off,** even if someone ticked it by
  hand in the MOTMaster GUI. This follows the existing rule that `prepare()`
  configures every sequencer completely on each task.
- **An Rb-only task while CaF holds the link fails** with MOTMaster's "linked
  to PH-BONESAW; SetScriptPath is refused" message. It does not release the
  link through the CaF device.

## Follow-ups (not done)

- **Record the `LinkRunId`.** Both MOTMasters save it in `_link.txt`, but
  `Controller` doesn't expose it after `Go`. A small EDMSuite change (e.g. a
  `LastLinkRunId` property) would let pytweezer record it per point and tie each
  h5 point to both zip files.
- **Real linked scripts:** port the Rb+CaF sequences pytweezer actually runs
  (e.g. `RbTweezerBasic2026_2` plus the CaF load) into one linked script in
  `MoleculeMOTMasterScripts`, each channel name on exactly one machine.
  Experiments like `TweezerBaseline` can then move to `linked=True`.
- `parallel.py` / `optimisation.py` still use the two-device trigger pattern.
  Leave them until the linked path is proven on the rig.
- Fill in EDMSuite's `motmaster-python-integration` stub skill with what the
  live tests show (ask first, as that skill asks).
