# Test suite review: report

Written 21 October 2026. Reviews `exp-handler` at commit `c95e9f5`, following
`docs/superpowers/specs/2026-10-21-test-suite-review-design.md`. The raw
outputs are in `2026-10-test-suite-review/`. `stage1/` holds the blind
behaviour lists, `stage2/` the per-test decisions, and `test-map.md` records
which area decided each test file.

Nothing in `tests/` or in the product code has been changed.

## The short answer

- **Rework the suite; don't start from scratch.** 108 of today's 239 tests are
  worth keeping as they are, and 94 guard the right thing but need rewriting.
  Only 37 should go. Five of the seven areas come out as "rework". The two
  "start again" areas are numerical/coordinators, where none of its 9 tests
  survives as it is, and analysis, which has no tests at all.
- **Is everything from the review covered?** No. The blind review proposed
  418 behaviours. 133 are covered, 90 partly covered and 195 not covered at
  all. The gaps matter more than the quality of the existing tests.
- **Did the review miss anything important?** Yes. 25 existing tests guard
  something stage 1 had no entry for (26 rows, because one test was flagged by
  two areas). The most serious is that a simulating Experiment Manager's
  workers must also simulate; otherwise they drive real hardware.
- **The suite doesn't pass today.** 3 tests in `tests/test_rearrangement.py`
  fail on `exp-handler`. Of pytest's 300 items (239 test functions, some
  parametrised), 294 pass and 3 are skipped. Two of the failures come from a
  real bug (see "Bugs"). The third tests a `phasemask_kwargs` attribute that no
  longer exists.
- **The review found 84 suspected bugs** as a by-product. That's 83 distinct
  ones, as one was found by two areas, and a few entries bundle several related
  faults. Several quietly damage data or run the wrong experiment. They are
  worth fixing before any test work.

## By area

| area | proposed | covered | partly | not | keep | rewrite | delete | verdict |
|---|---|---|---|---|---|---|---|---|
| Experiments | 91 | 46 | 21 | 24 | 29 | 28 | 6 | rework |
| Data recording | 42 | 14 | 19 | 9 | 10 | 17 | 3 | rework (exactly ⅓ kept) |
| Servers and config | 74 | 34 | 17 | 23 | 31 | 19 | 11 | rework |
| Drivers | 39 | 6 | 4 | 29 | 6 | 2 | 1 | rework |
| Numerical and coordinators | 28 | 0 | 0 | 28 | 0 | 2 | 7 | start again |
| Analysis | 41 | 0 | 0 | 41 | — | — | — | start again (no tests) |
| GUI | 103 | 33 | 29 | 41 | 32 | 26 | 9 | rework |
| **all** | **418** | **133** | **90** | **195** | **108** | **94** | **37** | |

Why each area got its verdict:

- **Experiments.** The behaviours tested are mostly the right ones. The
  rewrites split tests that guard several behaviours at once. The slow
  end-to-end pause, crash and restart tests should become unit tests. There are
  no tests at all for `dataset.py`, `tweezer_baseline.py`, most of the
  catalogue, or the worker's failure paths.
- **Data recording.** This area is on the line, with exactly a third kept, and
  it is a top-priority area. The `points_with_readings` tests can't catch a
  reading attached to the wrong point: every point shares one reading from
  before the run. Nothing tests `close()` flushing its queue, or NaN and numpy
  values in JSON columns.
- **Servers and config.** The deletes are checks of loud failures and library
  wrappers. There are no tests for `logging_utils`, `analysis_manager`,
  `clients`, `device_control` or `messageclient`.
- **Drivers.** All six Hamamatsu camera-selection tests are kept, confirmed by
  mutation. No test reaches the real `SLM` class: removing its uint8 conversion
  or its frame-slot bound fails nothing.
- **Numerical and coordinators.** The 9 tests guard loud failures or the
  shape of a status dict. Three fail today, and nothing touches `phasemask.py`,
  `optimisation.py` or `transport.py`.
- **Analysis.** It has no tests. The 41 proposals are mostly P3, each with an
  analytic check.
- **GUI.** It keeps 48% of its tests. The rewrites are long scenario tests.
  There are no tests for the viewers, the property and subscription editors,
  or `build_gui` teardown.

The 37 deletes are mostly:
- checks that a failure is loud (an error message, a missing role)
- checks on what a library already guarantees
- counts of calls to a fake
- duplicates of tests elsewhere

**Conflicts between areas.** Each test file had one deciding area, but two
deletes remove the only cover for another area's behaviour:
- `test_rearrangement.py::test_simulated_rig_builds_and_resolves_roles`
  (deleted by numerical) is the only cover for servers-config `device_server#3`.
  That entry checks that a composite's sub-devices inherit its `simulate` flag;
  otherwise one device in a simulated rig is driven for real. It also partly
  covers #1 and #5.
- `test_experiment_e2e.py::test_abort_kills_the_worker` (deleted by
  experiments) is cited by servers-config as covering `experiment_manager#6`.

Settled: **keep both** until `test_servers.py` and `test_experiment_manager.py`
have their own tests for those behaviours, then move or delete them. The counts
above still show them as deletes.

"Covered" means the test would fail if the behaviour broke. In four areas the
comparison agents checked this by deliberately breaking code in a scratch copy.
Those checks found tests that pass anyway: rounding replaced by truncation in
integer scans; MOTMaster's `set_iterations` removed; per-point database readings
joined to the wrong point; naive due times read as UTC.

Most rewrites are mechanical. They split tests that check several behaviours
at once, turn copied cases into a parametrised test, and replace checks on
internals or on calls to a fake with checks on results.

## How big the suite would be

| | P1 | P2 | P3 | P4 | total |
|---|---|---|---|---|---|
| already covered | 44 | 12 | 6 | 71 | 133 |
| partly covered (strengthen an existing test) | 23 | 19 | 2 | 46 | 90 |
| not covered (new test) | 15 | 20 | 41 | 119 | 195 |

P1 is experiments doing what they're asked, P2 is data recorded faithfully, P3
is numerical correctness, and P4 is anything else that would fail quietly.

Keeping and rewriting the 202 existing tests worth having, and adding all 195
new ones, gives **about 400 tests**. Adding only the new P1–P3 tests (76)
gives **about 280**. That's close to today's size, but each test has been
judged against the skill. The 119 new P4 tests are the place to trim:

| area | new P4 tests |
|---|---|
| GUI | 40 |
| servers | 23 |
| analysis | 16 |
| drivers | 15 |
| experiments | 11 |
| numerical | 11 |
| data recording | 3 |

The tier split covers stage-1 behaviours. The 202 existing tests weren't
tiered on their own, and some of them are P4 too.

The blind reviewers mostly didn't pad. 30 of the 121 modules got no entries,
and only 8 hit the limit of 10 (`stage1/SUMMARY.md`). The P4 entries are where
judgement was loosest. Many claim a quiet failure for something a physicist
would notice in the GUI within minutes.

## What the review missed

The comparison agents found 25 existing tests guarding something important that
stage 1 had no entry for. The full list is under "Missed by the review" in each
`stage2/<area>.md`. The ones that matter most:

- **A simulating manager's workers also simulate.** Otherwise they drive real
  hardware. The experiments and servers agents both flagged
  `test_experiment_e2e.py::test_simulating_manager_gives_workers_simulated_devices`.
- **Pause, resume and terminate requests reach the worker**, end to end.
- **Removing a scanned MOTMaster row in the argument editor drops its scan
  axis too.** Otherwise the run still scans it.
- **The runner closes its devices when a run fails.**
- **Simulated runs are written to their own data folder** (`data_root`), and
  **rids never repeat** (`highest_rid`). Either one breaking would quietly
  overwrite real data.
- **`run_local` runs record the `simulated` attribute in their file.**
- **Recipes survive a save-and-load round trip**, and a single follower's
  failure fails the whole task.
- **The manager-to-GUI mirror** (`servers/sync.py`) holds no shared
  references, takes writes from many threads, and copes with being started
  before its server.
- **The MOTMaster driver raises errors instead of printing them**, so the next
  shot can't quietly run the wrong script.
- **The Hamamatsu camera opener refuses a camera with the wrong serial
  number.**
- **`arm_rearrangement` passes the measured occupancy mask** to the sequence
  generator.

These should be added to the stage-1 lists before the rework starts. Some of
them, such as the simulate flag and rid reuse, belong in P1–P2.

## Did the blind review find the known bugs?

The trial runs found three bugs. Two are recorded in the storage note
(`docs/notes/storage_text_encoding.md`, on the `storage-text-encoding`
branch); the third came from the SLM trial. The blind reviewers weren't told
about any of them.

| known bug | found blind? |
|---|---|
| a text result and a dict result under one name make the file unloadable | **yes**, independently, with the same smallest input |
| complex-valued results can't be saved | not as a bug, but storage#1 includes complex in its round-trip test, which would catch it |
| SLM mask values outside 0–255 wrap round silently | **not in the drivers review**, which listed other SLM issues such as the missing shape check. The numerical review found a related problem: `transform_phase_8bit` produces values up to 382 before the uint8 cast. |

One found outright and one that a proposed test would catch, out of three. That
is reasonable evidence that the blind review works. It isn't proof.

## Bugs

There are 84 suspected bugs in all, each with the smallest input that shows it.
They are listed per area at the end of each `stage1/<area>.md`. None has been
fixed. These are the ones that quietly damage data or run the wrong
experiment, and so belong first in line:

**Experiments run the wrong thing**
- **Argument editor rounds to 6 decimals in display units.** A small SI
  default such as 2e-7 is submitted as 0. Confirmed. (`GUI/experiments/arg_editor.py`)
- **The queue view keeps the selected row index when that task disappears.**
  A second Delete or Abort then hits a different task. Confirmed.
  (`queue_view.py`; the same in `analysismanager.FilterTable`)
- **`ExperimentManagerClient` retries a slow reply**, so a manager that takes
  more than 2 s queues the same `submit` twice. (`experiment/client.py`)
- **Rearrangement can't run without a detector.** `rearrangement.py` and
  `transport.py` do `from pytweezer import analysis as an`, which gives the
  empty `analysis/` package, so the occupancy fallback raises. The fix is to
  import `pytweezer.analysis.analysis`, as `optimisation.py` already does. This
  is the cause of 2 of the 3 failing tests. `transport.py` can't be imported at
  all: it imports `linear_path` and `lift_path`, which `phasemask.py` lacks.
- **`plan_rearrangement` modifies the caller's trap arrays in place**, so the
  next shot starts from the wrong positions. (`phasemask.py`)
- **MOTMaster driver:** a typo (`"aceeleration"`) means the acceleration is
  never set. The picomotor axis is never assigned. The reversal flips direction
  on every later call.

**Data recorded wrongly or lost**
- **One NaN in a JSON column blocks every later database write** from that
  writer: the batch fails, is put back at the head of the queue, and fails
  again on every retry. This was worked out from the code, not run against
  Postgres. `np.bool_` values are silently dropped, which was confirmed.
  (`database/writer.py`)
- **Arrays sent over a stream in column-first order** (any `frame.T`)
  **arrive scrambled with no error**. Confirmed. (`servers/clients.py`)
- **A transient read error on the queue or recipes file discards it.** The
  file is moved aside and an empty state is loaded. (`experiment/queue.py`)
- **Two sub-devices in a composite with the same `role` silently overwrite
  each other.** (`servers/device_server.py`)
- **Backfill lets simulated runs overwrite real ones** in the database when
  their rids collide. (`database/backfill.py`)
- **Two ADC channels with the same short name**, such as `Dev1/ai0` and
  `Dev2/ai0`, collapse into one field. (`loggers/ni_adc_logger.py`)
- **Mixed text and dict results**: the known bug above.

**Numbers come out wrong**
- **"Fidelity" is actually the total error** (about 0 for clean data).
- **`detect_trap_sites` hangs for ever on `uint16` frames** when it can't
  match the requested site count.
- **Centres of mass are truncated, not rounded.**
- **`rotate_image` swaps the x and y offsets.**
- **`fit_gaussian` swaps the widths but not their errors.**

(All in `analysis/`.)

**Other:** `DBWriter.dsn_host` leaves the password in place for key=value DSNs,
so it can reach the logs.

## Dead code

**From the design, confirmed:**
- **`pytweezer/analysis.py`:** a stale duplicate of
  `pytweezer/analysis/analysis.py` that the package shadows. Delete it once the
  coordinators' import is fixed, as above.
- **`data_manager.py`, `arduino.py` and `analysis/analysis_old.py`:** nothing
  imports them.

**Also unused, per the reviewers:**
- `servers/datamgr.py`
- `GUI/models.py`
- `GUI/fitfunctions.py`, which uses the lithium mass for Rb temperatures
- `GUI/qled.py`
- `pytweezerQt.SearchLineEdit`
- `drivers/anapico.py`
- `analysis/floating_point_arithmetics.py`

## Limitations

- **The same model wrote the tests, reviewed them blind, and made the keep
  decisions**, so shared blind spots are likely. The "missed by the review"
  check helps, but doesn't remove the risk. The best extra check is a maintainer
  reading the P1–P2 lists for Experiments and Data recording.
- **The blinding was strong but not perfect.** The blind copy had no `tests/`,
  no git history and no old plans. A few docs and skills still name existing
  test files or describe what they check: `docs/notes/hamcam_rpc_slowness_investigation.md`
  on `test_sipyco_patches.py`, and the architecture, add-experiment and
  run-pytweezer skills. The final reviewer audited all seven stage-1 agents'
  transcripts. None opened `tests/`, the real repo, the trial runs or git
  history, and the `sipyco_patches` entries don't mirror that note.
- **Stage 1 read the code, so most bugs were found by reading.** Those marked
  "confirmed" were run.
- **The comparison agents broke code in scratch copies.** Three of them first
  imported the real repo by mistake (the editable install wins unless
  `PYTHONPATH` points at the copy). They threw those results away and redid the
  checks, and every result quoted here comes from a corrected run.
- **The integration tests in `test_db_integration.py` skip** unless
  `PYTWEEZER_TEST_DSN` is set. They pass against the local test database.

## Recommended order

1. **Fix the P1–P2 bugs above**, each with the test that exposes it. The
   argument editor, the queue selection, duplicate submits, the database writer
   and the storage note's encoding change come first.
2. **Experiments and data recording.** Rework the tests, add the missed
   behaviours, then the uncovered P1–P2 entries.
3. **Servers and config**, which includes the Experiment Manager and device
   server. Drivers has a larger share of P1–P2 entries (41%), but most of its
   P1 entries are MOTMaster, which depends on the manager working. So this
   order departs from a strict ranking by P1–P2 share.
4. **Numerical and coordinators.** Fix the coordinators' `analysis` import and
   the `transport.py` imports first. Then start again, using the 28 stage-1
   entries.
5. **Drivers.** Add the SLM data-path tests. The green trial in `trials/` maps
   onto slm#1, #2 and #4.
6. **Analysis.** Start from the 41 stage-1 entries. They are mostly P3, checked
   against analytic answers.
7. **GUI.** Do the P1 entries (argument editor, queue view) and decide whether
   the P4 entries are worth having at all.

Each step is one branch and one PR, following the `write-tests` skill.

No data-recording entry depends on the current pair of text encodings. All of
them are about values coming back unchanged, so they still hold after the
`storage-text-encoding` change.
