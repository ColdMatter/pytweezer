# Test suite review — design

## Intent

Claude wrote every test in `tests/`: 34 files, about 240 tests and 5.5k lines,
against about 30k lines of source. That is more than the maintainers, who are
physicists rather than software engineers, can realistically understand and keep
up. We want a suite where every test is one a maintainer would miss if it were
deleted. To get there without anchoring on what already exists, we first decide
from the source alone what deserves a test, and only then compare that with the
current tests.

The maintainers' priorities, which the `write-tests` skill encodes:

1. Experiments do what they are asked.
2. Data is recorded faithfully.
3. Numerical code gives the right answer.
4. Anything else that would fail quietly.

The code is still in development and has not been used much in the lab, so no
list of past incidents exists to check the review against.

## Goals

- For each area of the code, a ranked list of the behaviours worth testing, made
  without reading the existing tests.
- For each area, a verdict on whether to **rework** the existing tests or
  **start again**, backed by per-test decisions: keep, rewrite or delete.
- Answers to three questions: is everything from the review already covered;
  did the review miss anything the existing tests guard; and which existing
  tests should go.
- A short report a maintainer can read in one sitting and act on.

## Non-goals

- Changing anything in `tests/` or in product code. The review only proposes;
  carrying out the decisions is a separate plan, made after the maintainers
  approve the report.
- Fixing bugs the review finds. They are reported, like the storage bugs in
  `docs/notes/storage_text_encoding.md`.
- Reaching a coverage percentage.

## Anti-goals (failure even if it "works")

- A proposed test that a maintainer wouldn't miss if it were deleted. There is
  no target for the total size: a suite about as large as today's is fine if
  every test in it earns its place. The skill's limit of 10 per module is the
  only cap.
- A report too long to read, or one that says "rework everything" without
  per-area reasons.
- A blind review that saw the tests, through `tests/`, `conftest.py`, or git
  history and diffs touching `tests/`.

## Constraints

- **Snapshot.** Review `exp-handler` at commit `c95e9f5`, so the source doesn't
  move under the review. The work happens on the `test-suite-review` branch,
  which is based on `write-tests-skill` so that the skill is present.
- **Standard.** Both stages judge against `.claude/skills/write-tests/SKILL.md`:
  its priorities, its limit of 10 cases per module, and its table of what tests
  look like for each kind of code.
- **Blinding is enforced, not requested.** Stage-1 agents work in a git worktree
  of the snapshot with every copy of test code deleted, and their instructions
  forbid `git log`/`git show`/`git diff` on test paths. *(Builder's call:)* the
  deletions are:
  - `tests/`
  - `docs/superpowers/plans/`, which contains the original test bodies
  - `logs/`
  - the starter test files under `.claude/skills/*/assets/`

  Skill prose that mentions a test file by name stays. Names alone don't anchor
  the review, and editing the skills would leave agents reading guidance that
  isn't the real thing. The snapshot predates the `write-tests` skill, so it is
  copied into the worktree. In blind mode, stage-1 agents skip the skill's "read
  the existing test file" step.
- **Same model on both sides.** The reviewer is the same model that wrote the
  tests, so their blind spots are likely shared. Stage 2's "existing tests the
  review didn't ask for" check is the counterweight, and the report should say
  that this limitation remains.
- **Trial runs to reuse.** Before this design, three agents wrote tests for
  `experiment/scan.py`, `experiment/storage.py` and `drivers/slm.py`, once
  without the skill and once with it. Those tests were *not* blind: the
  with-skill agents read the existing test files. Their output is in the session
  scratchpad, and the builder copies it into the reports folder as
  `trials/`. It is a worked example of the skill's output: ranked behaviours,
  a list of what was left untested, and bugs reported rather than fixed. Stage 2
  may use it as a second opinion for those three modules. Stage 1 must never see
  it. "The SLM trial" means the with-skill `slm.py` run. That agent confirmed its
  tests catch real faults by breaking the driver's data conversion, after which
  6 of its 10 tests failed. None of those three modules has been reviewed by
  this method, so all three stay in scope.
- **Known bugs.** The bugs in `docs/notes/storage_text_encoding.md` (on the
  `storage-text-encoding` branch) are *not* given to stage 1, so they can't
  steer it. Stage 3 checks whether the Data recording review found them
  independently, and reports the result as evidence of how well the review
  works. Storage is reviewed as it stands at the snapshot. The planned JSON
  change keeps every stored value round-tripping unchanged, so behaviours stated
  at that level remain valid. Entries tied to the current text encoding are
  flagged so they can be revisited.
- **Cross-process paths** belong to the area on the receiving side, e.g.
  Experiment Manager → h5 file goes to Data recording *(builder's call)*.
- **UK English** throughout. The report and any committed files follow the
  repo's CLAUDE.md conventions.

## Scope

In scope: every kind of test (unit, integration, GUI), at the levels the skill's
table sets.

| Area | Source |
|---|---|
| Experiments | `pytweezer/experiment/` (except `storage.py`), `pytweezer/experiments/` |
| Data recording | `pytweezer/experiment/storage.py`, `pytweezer/database/`, `pytweezer/loggers/` |
| Servers and config | `pytweezer/servers/` (except `deprecated/`), `pytweezer/configuration/`, `pytweezer/parallel.py`, `pytweezer/logging_utils.py`, `bin/launch.py`, `bin/kill_stale.py` |
| Drivers | `pytweezer/drivers/` (except the vendored `Blink Plus/`) |
| Numerical and coordinators | `pytweezer/coordinators/`, `pytweezer/phasemask.py`, `pytweezer/optimisation.py` |
| Analysis | `pytweezer/analysis/` (except `analysis_old.py`) |
| GUI | `pytweezer/GUI/`, `bin/gui.py`, `bin/managed_panel.py` |

Out of scope: the notebooks, `servers/deprecated/`, the
`rearrangement-optimisation-RnD` branch, and the dead-code candidates below.

**Dead-code candidates.** These are excluded from the review, and the report
lists them so the maintainers can decide whether to delete them:

- `pytweezer/analysis.py`: the `pytweezer/analysis/` package shadows it, so it
  cannot be imported.
- `pytweezer/data_manager.py` and `pytweezer/arduino.py`: nothing imports them.
- `pytweezer/analysis/analysis_old.py`: nothing imports it.

## Stage 1: the blind review

One agent per area (seven in all). Each reads its area's source, CLAUDE.md, the
skills relevant to it and the `write-tests` skill, in the worktree that has no
`tests/`. It writes no test code.

*(Builder's call:)* One entry is one test case. Ten entries is a ceiling, not a
target. A module with nothing worth testing gets zero entries, and the prompt
says to expect that for many modules, e.g. widgets, thin wrappers and
re-export `__init__` files.

Output, per module, at most 10 entries ranked by the skill's priorities:

- **id** (`area/module#n`), used by stage 2 to cite the entry
- **behaviour**: one sentence about what a caller relies on, e.g. "a stopped
  run loads only its completed points"
- **priority**: 1–4, from the skill's list
- **level**: unit, integration or GUI
- **how it fails if broken**: quiet or loud, and what would go wrong in the lab
- for numerical behaviours, **the independent check**: an analytic case, a
  conservation law or a symmetry

Each area's output also lists any suspected bugs, each with the smallest input
that shows it.

## Stage 2: comparison

One agent per area. Its inputs are that area's stage-1 list and the existing
tests that exercise the area: every test file that imports a module from it,
which may mean one file counts for several areas.

Output:

1. **Coverage.** Each stage-1 entry marked covered, partly covered or not
   covered, citing the test(s) that cover it. "Covered" means the test would
   fail if the behaviour broke, not merely that it calls the code.
2. **Each existing test** marked keep, rewrite or delete, with the skill rule
   that decides it. Rewrite means it guards a wanted behaviour badly, for
   example by checking calls to a fake or by guarding several behaviours.
3. **Missed by the review.** Existing tests that guard something important that
   stage 1 left out, with a judgement on whether stage 1 was wrong to leave it
   out.
4. **Verdict for the area:** rework or start again, with one paragraph of
   reasons. As a rule of thumb, start again when fewer than about a third of the
   existing tests are kept as they are.

## Stage 3: the report

One synthesis, written for the maintainers:

- a table of areas showing behaviours proposed and covered, existing tests to
  keep, rewrite or delete, and the verdict
- the total size of the proposed suite against today's 239, broken down by
  priority tier, so the maintainers can see where the tests would go before
  deciding whether to trim
- a short section on what the review missed
- the bugs and dead-code candidates
- a recommended order for carrying it out, highest priority areas first
  (experiments and data recording first, as the maintainers' priorities say)

It is committed as `docs/superpowers/reports/2026-10-test-suite-review.md` on
this branch, with the raw stage-1 and stage-2 outputs alongside it in the same
folder.

## Left to the builder

- Prompt wording for each stage, and whether the agents run as parallel subagents.
- How the worktree is created and cleaned up.
- How test files are assigned to areas, as long as every file under `tests/` is
  reviewed by at least one comparison agent.
- Whether to use cheap objective signals in stage 2. Examples are a coverage run,
  or deliberately breaking a module, as the SLM trial did, to confirm that "keep"
  tests fail. Adding a dev dependency such as `pytest-cov` needs the maintainers'
  approval.
