# CaF laser-tuning optimiser

## Intent

Today the CaF MOT's wavemeter-lock lasers are tuned from a notebook with
pycaf's `MOTAutoTunerWML` (`auto_mot()`), ported unchanged and untested into
`Rig.auto_mot` (`pytweezer/notebook/rig.py`). Each enabled laser is scanned over
a range around its current set point, parked on the point with the most
background-subtracted fluorescence, and the next laser is scanned from there.
The scan ranges are rebuilt by hand in every notebook, which lasers run is
toggled by hand, and nothing is recorded in the queue or the saved parameters.

Make this an optimiser on pytweezer's optimiser framework: a `StagedScan` with
one stage per laser, run by the Experiment Manager, whose proposed set points
are accepted in the Optimisers tab and recorded as saved parameters. A
notebook's `auto_mot(...)` stays the way to call it. The reference for current
behaviour is pycaf `mot_auto_tuner_wml.py` and the 27 Oct 2026 lab-book
notebook; v0, v1 and TCool are enabled there, BX, v2 and v3 are disabled in
that notebook only.

## Goals

- One stage per laser: v0, BX, v1, v2, v3, TCool (the `lasers` of
  `caf_mm_config.json`). Every one can run.
- **Choose which stages run**, per run, from the form and from `auto_mot`.
  A skipped stage is not scanned, proposes nothing, and leaves its laser alone.
- Each stage scans a span around a *centre* that defaults to the laser's saved
  set point and can be set to another value, per run.
- Each laser's saved set point is a saved parameter: "the last good scan
  result", not a mirror of the lock. It is not kept in step with hand changes
  to the lock, and is only written by accepting an optimiser run.
- A run changes nothing lasting: it leaves every laser at the set point it had
  when the run started, even if the run fails or is terminated.
- **Accepting** a run saves the ticked set points *and* ramps the wavemeter lock
  to them.
- `Rig.auto_mot` submits the optimiser, waits and shows the same per-laser
  scans (counts against detuning, chosen point marked) that pycaf showed.

## Non-goals

- The BX recapture-rate scan (`BX_BMOT_OP_magtrap`, three-image recapture
  rate). It is a later task, and it will be a second optimiser reusing the
  laser-setting machinery here, so keep that machinery separable from the
  fluorescence objective.
- A joint (Bayesian) search over the lasers. Staged only.
- Replacing the argmax choice with a fit. `StagedScan.best` is the default.
- Supporting pycaf's `MOTAutoTunerWML` after this lands. `Rig.auto_mot`'s
  current implementation is untested and is replaced, not kept alongside.

## Anti-goals

- A rejected or failed run that leaves lasers moved.
- Absolute-THz scan ranges stored in the optimiser's defaults: set points
  drift, so a range fixed in the class goes stale.
- Hidden hardware effects from accepting beyond the saved set points and the
  lock ramp, or an accept that half-applies without saying so.

## Constraints

- No CaF or Rb hardware is available while this is built. Everything is tested
  against the simulated MOTMaster (`SimulatedMotMasterInterface`, which already
  keeps set points) and a simulated camera, per `write-tests`. Hardware
  behaviour that cannot be checked this way is listed in the handoff, not
  guessed.
- Follow the add-optimiser and add-experiment skills: the optimiser is a module
  under `pytweezer/optimisers/`, the measurement a module under
  `pytweezer/experiments/`, nothing touches hardware at import.
- Production code is not bent to suit tests (CLAUDE.md).
- Spelling is UK; docstrings Google style and describe current behaviour only.

## What was decided

By the user:

- Staged scan, all six lasers, the stages to run chosen per run.
- Set points are saved parameters (option "saved parameters"), not read only
  from the lock; accepting applies them to the lock.
- Saved values record the last good scan, not the current lock value; a scan is
  around a chosen value, defaulting to the saved one.
- Fluorescence tuning only; the recapture scan is later.

Design, as the author's calls (marked *call*; the user has not seen these):

### Framework additions (`pytweezer/experiment/optimisers.py`)

1. *call* **Stage enable.** Every `Stage` generates a Boolean argument
   `<setting>_run` (default per stage, `Stage(run=True)`). `StagedScan`'s
   schedule, `number_of_points`, `stages_done/_total` and the `proposed.*`
   records cover enabled stages only; later stages receive the best values of
   the enabled earlier ones. A run with no stage enabled is refused at
   prepare. A stage that was skipped is absent from the Optimisers tab's
   proposals.
2. *call* **Span stage.** `Stage(..., span=..., save_as=...)` as an alternative
   to `start/stop/points`, parallel to `Dimension(span=...)`: the sweep runs
   from `centre - span` to `centre + span` in `<setting>_points` points. The
   centre is the argument `<setting>_centre`, empty by default, in which case
   it is the saved parameter `save_as` read when the run starts (`KeyError` if
   that does not exist, as for `Dimension`). The resolved centre is recorded
   as `<setting>_centre_used`. Class-definition checks mirror `Dimension`'s
   (span and bounds are exclusive, span positive).
3. *call* **Apply on accept.** An optimiser may define `apply(proposals)`,
   where `proposals` is `{saved key: value}` of the ticked values. The
   Optimisers tab calls it after the saved parameters are written and reports
   a failure beside the accepted run without undoing the save. The default does
   nothing, so existing optimisers are unchanged. How it reaches a device from
   the tab (the tab is a GUI process) is the builder's to settle, using
   `get_device()`.

### Measurement (`pytweezer/experiments/`)

*call* A CaF fluorescence measurement, a `MotMasterExperiment` on
`CaF MotMaster` with the `CaF ThorCam`. For `measure(**set_points)`:
`None`/absent keeps a laser where it is; otherwise it is ramped to the value
(`ramp_laser_set_point_wml`, as `Rig.scan_laser` does) and held `settle_s`
before the first shot after a change. It runs `iterations` shots of the script
at the MOT field on, subtracts the background, sums each frame and returns
`counts` (mean over frames) and `counts_error` (standard error; NaN for one
frame).

- The background is one shot with `MOTCoilsCurrentValue` at 0, taken in
  `prepare`, averaged over frames, as pycaf and `Rig.auto_mot` do. Argument
  names and defaults for the field parameter and its on/off values follow
  `Rig.auto_mot`.
- Defaults: script `"0. DDSFourSidebands5W_noDDS"` (the notebook's), 3
  iterations (the notebook's). The camera is read whole-frame, as pycaf did;
  a crop is a later argument if wanted.
- `prepare` records every laser's set point at the start of the run;
  `finish` ramps every laser back to that and must run on failure and
  termination.
- Camera frame retrieval follows the CaF ThorCam driver and the camera-frame
  protocol the `notebook-rig` skill describes; read it before choosing how a
  shot's frames come back.

### Optimiser (`pytweezer/optimisers/`)

*call* `OptimiseMOTLasers(StagedScan, <measurement>)`, `objective = "counts"`,
`goal = "max"`. Stages in the order v0, BX, v1, v2, v3, TCool, each a span
stage with `save_as` the laser's set-point saved parameter. Defaults come from
the 27 Oct notebook and are the only values this spec takes from it: spans
(THz) v0 10e-6, BX 30e-6, v1 40e-6, v2 150e-6, v3 150e-6, TCool 40e-6; points
30, 30, 20, 30, 30, 30; enabled by default v0, v1 and TCool, as in the
notebook. `proposal_units = THz`, `proposal_group = "CaF lasers"`. `apply`
ramps each accepted laser on `CaF MotMaster` to its accepted value.

*call* The saved parameters are named for the lasers (the builder picks the
names, e.g. `caf_laser_v0`). They must exist before the first run
(`save_as` rule), so the builder adds a one-off way to create the missing ones
from the lock's current set points, and tells the user how to run it.

### Notebook (`pytweezer/notebook/rig.py`)

*call* `Rig.auto_mot(...)` keeps its name. Its arguments name the stages to run
(and optionally spans, points and centres), it submits the optimiser, waits,
and returns per-laser results of the same shape as today (`scan_range`,
`numbers`, `errors`, `set_point`, `start_set_point`) read back from the run's
file, for `plot_auto_mot_results`. It does not accept the proposals; that is
done in the Optimisers tab. Whether it should also offer `accept=True` is
left open, below.

## Left to the builder

- Exact argument names, group names and tooltips; spans and centres are in THz
  to match MOTMaster, so how the form shows them (MHz offsets read better) is a
  presentation choice.
- How `apply` reaches the device from the tab, and how failures are reported.
- Whether an argument that is "empty = saved value" is a new argument type or
  an optional `Number`.
- Settle time default (the notebook slept 10 s only for the BX start).
- Tests: stage enable, span/centre resolution, restoration on failure, the
  accept-then-apply path, and an end-to-end run against the simulated
  MOTMaster; whatever of `write-tests` applies.
- Updating the `add-optimiser` and `notebook-rig` skills for the framework
  additions. The `add-optimiser` changes are significant under CLAUDE.md, so
  they are proposed to the user, not made unasked.

## Open

- Should `Rig.auto_mot` offer `accept=True` (save and apply from the notebook)
  in addition to accepting in the tab?
