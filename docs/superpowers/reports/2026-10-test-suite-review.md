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
- **Did the review miss anything important?** Yes, 26 behaviours guarded by
  existing tests, listed below. The most serious is that a simulating Experiment
  Manager's workers must also simulate; otherwise they drive real hardware.
- **The suite doesn't pass today.** 3 tests in `tests/test_rearrangement.py`
  fail on `exp-handler` (294 pass, 3 skipped). They fail because of a real bug
  (see "Bugs"), not because of flaky tests.
- **The review found 84 suspected bugs** as a by-product. Several of them
  quietly damage data or run the wrong experiment. They are worth fixing before
  any test work.

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

"Covered" means the test would fail if the behaviour broke. In five areas the
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
gives **about 280**, close to today's size, but with every test justified. The
119 new P4 tests are the place to trim: most are GUI (85 of that area's 103
proposals are P4) and servers (62 of 74).

The blind reviewers mostly didn't pad. 30 of the 121 modules got no entries,
and only 8 hit the limit of 10 (`stage1/SUMMARY.md`). The P4 entries are where
judgement was loosest. Many claim a quiet failure for something a physicist
would notice in the GUI within minutes.

## What the review missed

The comparison agents found 26 existing tests guarding something important that
stage 1 had no entry for. The ones that matter most:

- **A simulating manager's workers also simulate.** Otherwise they drive real
  hardware. Experiments and servers both found this independently, in
  `test_experiment_e2e.py` and `test_experiment_simulation.py`.
- **Simulated runs are written to their own data folder** (`data_root`), and
  **rids never repeat** (`highest_rid`). Either one breaking would quietly
  overwrite real data.
- **The `simulated` attribute is recorded in every file.**
- **Recipes survive a save-and-load round trip**, and a single follower's
  failure fails the whole task.
- **The property sync mirror** holds no shared references, takes writes from
  many threads, and copes with being started before its server.
- **The MOTMaster driver raises errors instead of printing them**, so the next
  shot can't quietly run the wrong script.
- **The Hamamatsu camera opener refuses a camera with the wrong serial
  number.**
- **`arm_rearrangement` passes the measured occupancy mask** to the sequence
  generator.

These should be added to the stage-1 lists before the rework starts. Some of
them, such as the simulate flag and rid reuse, belong in P1–P2.

## Did the blind review find the known bugs?

The storage note (`docs/notes/storage_text_encoding.md`, on the
`storage-text-encoding` branch) records three bugs found during the trial runs.
The blind reviewers weren't told about them.

| known bug | found blind? |
|---|---|
| a text result and a dict result under one name make the file unloadable | **yes**, independently, with the same smallest input |
| complex-valued results can't be saved | not as a bug, but storage#1 includes complex in its round-trip test, which would catch it |
| SLM mask values outside 0–255 wrap round silently | **no**; the review listed other SLM issues (no shape check) |

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
- **Rearrangement can't run without a detector.** `from pytweezer import
  analysis` loads the empty `analysis/` package instead of
  `pytweezer/analysis.py`, which the package shadows, so the occupancy fallback
  raises. This is the cause of 2 of the 3 failing tests. `transport.py` can't be
  imported at all: it imports `linear_path` and `lift_path`, which
  `phasemask.py` lacks.
- **`plan_rearrangement` modifies the caller's trap arrays in place**, so the
  next shot starts from the wrong positions. (`phasemask.py`)
- **MOTMaster driver:** a typo (`"aceeleration"`) means the acceleration is
  never set. The picomotor axis is never assigned. The reversal flips direction
  on every later call.

**Data recorded wrongly or lost**
- **One NaN in a JSON column blocks every later database write** from that
  writer: the batch fails, is put back at the head of the queue, and fails
  again on every retry. `np.bool_` values are silently dropped.
  (`database/writer.py`)
- **Arrays stored column-first** (any `frame.T`) **arrive scrambled with no
  error**. Confirmed. (`servers/clients.py`)
- **Backfill lets simulated runs overwrite real ones** in the database when
  their rids collide. (`database/backfill.py`)
- **Two ADC channels with the same short name**, such as `Dev1/ai0` and
  `Dev2/ai0`, collapse into one field. (`loggers/ni_adc_logger.py`)
- **Mixed text and dict results**: the known bug above.

**Numbers come out wrong**
- **"Fidelity" is actually the total error** (about 0 for clean data).
- **`detect_trap_sites` hangs for ever on `uint16` frames.**
- **Centres of mass are truncated, not rounded.**
- **`rotate_image` swaps the x and y offsets.**
- **`fit_gaussian` swaps the widths but not their errors.**

(All in `analysis/`.)

**Other:** `DBWriter.dsn_host` leaves the password in place for key=value DSNs,
so it can reach the logs.

## Dead code, corrected

The design listed `pytweezer/analysis.py` as dead. It isn't. The coordinators
need it, and it is unreachable only because the `analysis/` package has the same
name. That makes it a bug, listed above. The rest of the dead-code list stands:
- **`data_manager.py`, `arduino.py` and `analysis/analysis_old.py`:** nothing
  imports them.
- **Also unused, per the reviewers:**
  - `servers/datamgr.py`
  - `GUI/models.py`, `GUI/fitfunctions.py` (which uses the lithium mass for Rb
    temperatures) and `pytweezerQt.SearchLineEdit`
  - `drivers/anapico.py`
  - `analysis/floating_point_arithmetics.py`

The `rearrangement-optimisation-RnD` branch may hold working versions of the
coordinator and phase-mask code. That's a guess I haven't checked, and worth
looking at before rewriting tests for that area.

## Limitations

- **The same model wrote the tests and reviewed them**, so shared blind spots
  are likely. The "missed by the review" check helps, but doesn't remove the
  risk. The best extra check is a maintainer reading the P1–P2 lists for
  Experiments and Data recording.
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
   server.
4. **Numerical and coordinators.** Fix the `analysis` shadowing and
   `transport.py` imports first. Then start again, using the 28 stage-1
   entries.
5. **Drivers.** Add the SLM data-path tests. The green trial in `trials/` maps
   onto slm#1, #2 and #4.
6. **Analysis.** Start from the 41 stage-1 entries. They are mostly P3, checked
   against analytic answers.
7. **GUI.** Do the P1 entries (argument editor, queue view) and decide whether
   the P4 entries are worth having at all.

Each step is one branch and one PR, following the `write-tests` skill.
