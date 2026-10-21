---
name: write-tests
description: The standard for what in pytweezer deserves a test and what a test should look like. Use whenever writing, adding, extending, reviewing, trimming or deleting pytest tests under tests/, when asked to "add tests for" or "cover" a module, when a change touches code the suite covers, or when judging whether an existing test is worth keeping — for drivers, experiments, storage, the database, analysis, coordinators or the GUI alike.
---

# Writing tests for pytweezer

The suite is maintained by physicists, not software engineers, and every test
is something one of them must understand the day it fails. A test earns its
place only if a maintainer would miss it were it deleted: it guards behaviour
someone relies on, and it fails when that behaviour breaks.

## What matters, in order

1. **Experiments do what they are asked.** Arguments, scan expansion, the order
   points run in, which devices get called with what.
2. **Data is recorded faithfully.** What goes into h5 files or the database
   reads back identical — nothing silently dropped, truncated, recast or
   overwritten.
3. **Numerical code gives the right answer** (phase masks, rearrangement, fits,
   analysis). Check against an answer known independently of the code — an
   analytic case, a conservation law, a symmetry — never against the code's own
   earlier output.
4. **Anything else that would fail quietly.**

Failures that are loud — a crash on start, a window that won't open, a driver
that raises — rank last: someone notices them straight away.

## The recipe

1. Read the module, then its existing test file in `tests/`. New cases go into
   that file, and may replace cases already there; one test file per area.
2. List the behaviours the module promises its callers and rank them by the
   list above.
3. Write tests for the top of the list, up to **10 test cases per module**,
   each parametrised case counting as one.
4. Reply with each test written (one line on what it guards) followed by the
   ranked behaviours left untested, so the user decides whether any are worth
   adding.

## What a test looks like

- It calls the code the way its callers do and checks what those callers care
  about, so it passes unchanged after the internals are reorganised.
- Stand-ins appear only at process and hardware edges. Where one exists, use
  it: `fake_devices` and `recording_db` in `tests/conftest.py`, helpers in
  `tests/helpers.py`, and each driver's `simulate()` twin
  (`pytweezer/servers/simulated_device.py`). A driver test that must see what
  reaches the hardware gets a fake vendor library that only records the data
  handed to it.
- Each test guards one behaviour; a test whose name needs "and" is two tests.
- It runs in well under a second, gives the same result every run, and needs no
  hardware, network or real database.
- Its name states the behaviour: `test_stopped_run_loads_only_completed_points`.
- Several cases of one behaviour are one parametrised test.

## By kind of code

| Code | Its tests are |
|---|---|
| Hardware driver (`drivers/`) | Any transformation of data before it reaches the hardware (dtype, memory layout, units, bounds), plus a check that the simulated twin has the real class's public methods. Which calls go to the vendor library is left to the real hardware. |
| pytweezer code built on a library (pydantic, h5py, Qt, numpy, sipyco) | pytweezer's own logic on top; the library's guarantees are the library's to test. |
| GUI | The logic behind the widgets (models, parsing, state) and one check that each tab builds offscreen. Layout and styling are checked by eye. |
| Cross-process paths (device server ↔ client, Experiment Manager → h5 file, writer → Postgres) | One integration test per path, covering the normal case. |

## When a test exposes a bug

Report it to the user with the smallest input that shows it, and leave the fix
for them to approve. The test that exposed it lands together with the fix.
