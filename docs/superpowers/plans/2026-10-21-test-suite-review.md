# Test Suite Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a maintainer-readable report deciding, area by area, which
existing tests to keep, rewrite or delete, built from a blind review of the
source.

**Architecture:** This review is run by agents, so it writes no product code.
A shell step exports a test-free copy of the snapshot. Seven blind stage-1
agents write ranked behaviour lists into that copy, and they are copied out.
Seven stage-2 agents compare those lists with the real tests. A final synthesis
writes the report. Small throwaway scripts in the scratchpad do the mechanical
checks for completeness. Every output lands in one reports folder on the
`test-suite-review` branch.

**Tech Stack:** git, bash, Python 3 (stdlib only, for checks), Agent tool
(general-purpose subagents).

**Spec:** `docs/superpowers/specs/2026-10-21-test-suite-review-design.md`

## Global Constraints

- Snapshot commit: `c95e9f5`. Work branch: `test-suite-review`, whose `tests/`
  is identical to the snapshot's.
- Nothing under `tests/` or in product code changes. Bugs are reported, not
  fixed.
- Standard: `.claude/skills/write-tests/SKILL.md` (priorities 1–4, at most 10
  cases per module, the by-kind table).
- One stage-1 entry is one test case. Ten is a ceiling, not a target. Zero
  entries for a module is normal.
- There is no target for the total suite size. The report shows the total
  against today's 239.
- "Covered" means the test would fail if the behaviour broke, not merely that it
  calls the code.
- Rule of thumb for an area: start again when fewer than about a third of its
  existing tests are kept as they are.
- UK English in every committed file.
- Reports folder: `docs/superpowers/reports/2026-10-test-suite-review/`. The
  report itself: `docs/superpowers/reports/2026-10-test-suite-review.md`.
- Scratch location (`$SCRATCH`):
  `/tmp/claude-1000/-home-twalker-RbCaF-pytweezer/16e9fa87-2685-466e-af17-6fc080a9068d/scratchpad`.
  The blind copy is `$SCRATCH/blind`.

## Review Focus

1. **Blinding leak.** A stage-1 output that quotes a test name, a fixture name
   (`fake_devices`, `recording_db`) or a path under `tests/`. Expected: none.
   Task 3 greps for these.
2. **A module silently skipped.** A stage-1 agent that leaves a module out
   instead of giving it zero entries. Expected: every module in the area's file
   list has a heading. Task 3 diffs the two.
3. **A test silently skipped.** A stage-2 agent that never decides on an
   existing test function. Expected: every `def test_` in `tests/` gets a
   decision in at least one stage-2 file. Task 5 checks this.
4. **Conflicting decisions.** A test file shared by two areas, kept by one agent
   and deleted by the other. Expected: the report reconciles each conflict, and
   a test is kept if any area keeps it. Task 6 lists them.
5. **Padding.** Stage-1 lists that sit at the 10-per-module ceiling for most
   modules. Expected: a spread that includes many zeros. Task 3 prints the
   distribution, and Task 6 reports it.

---

### Task 1: Blind snapshot and area file lists

**Files:**
- Create: `$SCRATCH/blind/` (export of `c95e9f5`, not a git checkout)
- Create: `docs/superpowers/reports/2026-10-test-suite-review/areas/<slug>.txt` (7 files)

**Interfaces:**
- Produces: `$SCRATCH/blind/`, a test-free tree with no `.git`. It contains
  `$SCRATCH/blind/.claude/skills/write-tests/SKILL.md` and an empty
  `$SCRATCH/blind/review-output/`.
- Produces: `areas/<slug>.txt`, with one repo-relative `.py` path per line.
  The slugs are `experiments`, `data-recording`, `servers-config`, `drivers`,
  `numerical`, `analysis` and `gui`.

The spec said a git worktree. This plan uses `git archive` instead: it gives the
same tree with no history at all, so the ban on `git log` becomes unnecessary
rather than merely requested.

- [ ] **Step 1: Export and strip**

```bash
SCRATCH=/tmp/claude-1000/-home-twalker-RbCaF-pytweezer/16e9fa87-2685-466e-af17-6fc080a9068d/scratchpad
mkdir -p "$SCRATCH/blind" && cd /home/twalker/RbCaF/pytweezer
git archive c95e9f5 | tar -x -C "$SCRATCH/blind"
cd "$SCRATCH/blind"
rm -rf tests docs/superpowers logs .claude/skills/*/assets/test_*.py
mkdir -p .claude/skills/write-tests review-output
cp /home/twalker/RbCaF/pytweezer/.claude/skills/write-tests/SKILL.md .claude/skills/write-tests/
```

- [ ] **Step 2: Verify no test code remains**

Run: `cd "$SCRATCH/blind" && grep -rlE "^\s*def test_|import pytest|recording_db|fake_devices" --include=*.py --include=*.md . ; ls -a | grep -c '^\.git$'`
Expected: no file paths printed, then `0`.

- [ ] **Step 3: Write the area file lists**

Each area takes every `*.py` under its paths (from the spec's Scope table),
excluding `__pycache__/`, `pytweezer/servers/deprecated/`,
`pytweezer/drivers/Blink Plus/`, `pytweezer/analysis/analysis_old.py`,
`pytweezer/analysis.py`, `pytweezer/data_manager.py` and
`pytweezer/arduino.py`. `pytweezer/experiment/storage.py` belongs to
`data-recording`, not `experiments`. Generate the lists with `find` from the
repo root, one `areas/<slug>.txt` per area.

- [ ] **Step 4: Verify the lists partition the scope**

Run: `cat docs/superpowers/reports/2026-10-test-suite-review/areas/*.txt | sort | uniq -d; wc -l docs/superpowers/reports/2026-10-test-suite-review/areas/*.txt | tail -1`
Expected: no duplicates. The total is about 105, give or take what `__init__.py`
files add.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-10-test-suite-review/areas
git commit -m "Test review: area file lists"
```

### Task 2: Copy the trial runs

**Files:**
- Create: `docs/superpowers/reports/2026-10-test-suite-review/trials/{baseline,green}/{scan,storage,slm}/test_*.py`
- Create: `docs/superpowers/reports/2026-10-test-suite-review/trials/README.md`

- [ ] **Step 1:** Copy `$SCRATCH/baseline/` and `$SCRATCH/green/` into `trials/`,
  keeping their layout.
- [ ] **Step 2:** Write `trials/README.md` (under 15 lines). It says these are
  pre-design trials for `scan.py`, `storage.py` and `slm.py`: `baseline/` was
  written without the skill, `green/` with it, and neither was blind. They are
  for stage 2 only, and the `green/` storage run found two bugs and the SLM run
  checked its tests by deliberately breaking the driver.
- [ ] **Step 3:** Verify: `ls trials/*/*/` shows six `test_*.py` files.
- [ ] **Step 4:** Commit: `git commit -m "Test review: keep trial runs for stage 2"`.

### Task 3: Stage 1, the blind review

**Files:**
- Create: `docs/superpowers/reports/2026-10-test-suite-review/stage1/<slug>.md` (7 files)

**Interfaces:**
- Consumes: `$SCRATCH/blind/` and `areas/<slug>.txt` (Task 1).
- Produces: `stage1/<slug>.md` in this exact shape, parsed by Tasks 5 and 6:

```markdown
# Stage 1: <Area name>

## <repo/relative/module.py>

| id | behaviour | priority | level | if broken | independent check |
|---|---|---|---|---|---|
| <slug>/<module stem>#1 | ... | 1 | unit | quiet: ... | ... or — |

(or the single line `No entries: <one-sentence reason>.`)

## Suspected bugs

- <module>: <what is wrong> — smallest input: `...`
```

- [ ] **Step 1: Dispatch seven agents in parallel**, one per slug, with this
  prompt (fill in `<AREA>`, `<SLUG>` and the file list):

```
You are reviewing part of the pytweezer codebase to decide what deserves a
test. Work only inside <$SCRATCH>/blind. It is a copy of the repo with all
tests removed. Don't look for the tests anywhere else, in this copy or outside
it, and don't open /home/twalker/RbCaF/pytweezer.

Read CLAUDE.md, .claude/skills/write-tests/SKILL.md and any other skill under
.claude/skills/ relevant to your area. Then read every file below.

The skill is your standard. Skip its "read the existing test file" step:
there are no tests here, on purpose.

Area: <AREA>
Files:
<contents of areas/<SLUG>.txt>

For each file, list the behaviours its callers rely on that deserve a test,
ranked by the skill's priorities, at most 10. One entry is one test case. Ten
is a ceiling, not a target. Most files deserve fewer, and many deserve none:
say "No entries" with a one-sentence reason. Write no test code. A cross-process path
(e.g. Experiment Manager → h5 file) belongs to the area on its receiving side;
list it under that side's module.

Write your answer to <$SCRATCH>/blind/review-output/<SLUG>.md in exactly this
format: <the shape above>. Under "Suspected bugs", list anything that looks
wrong, with the smallest input that shows it. Don't fix anything.

Reply with: the number of entries by priority, the number of files with no
entries, and the bugs found.
```

- [ ] **Step 2: Copy out** `$SCRATCH/blind/review-output/*.md` to `stage1/`.

- [ ] **Step 3: Verify that every module has a heading** (Review Focus 2). For
  each slug, the `## ` headings (excluding "Suspected bugs") must match
  `areas/<slug>.txt` exactly.

Run: `for s in experiments data-recording servers-config drivers numerical analysis gui; do diff <(grep '^## ' stage1/$s.md | grep -v 'Suspected bugs' | sed 's/^## //' | sort) <(sort areas/$s.txt) >/dev/null || echo "MISMATCH $s"; done`
Expected: no output. For a mismatch, send the agent back to add the missing
headings.

- [ ] **Step 4: Verify no leak** (Review Focus 1).

Run: `grep -nE "tests/|test_[a-z_]+\.py|conftest|recording_db|fake_devices" stage1/*.md`
Expected: no output. If anything matches, read the line, and rerun that area if
it shows knowledge of the tests.

- [ ] **Step 5: Print the spread** (Review Focus 5). Count the table rows per
  module, then record a histogram of entries per module (0, 1–3, 4–6, 7–9, 10)
  for each area in `stage1/SUMMARY.md`, along with each area's entries by
  priority.

- [ ] **Step 6: Commit**: `git commit -m "Test review: stage 1 blind behaviour lists"`.

### Task 4: Map existing tests to areas

**Files:**
- Create: `docs/superpowers/reports/2026-10-test-suite-review/test-map.md`

**Interfaces:**
- Consumes: `areas/<slug>.txt`.
- Produces: `test-map.md`, a table with the columns `test file | areas`. Every
  file under `tests/` gets a row. `conftest.py` and `helpers.py` have the area
  `shared`.

- [ ] **Step 1:** Write `$SCRATCH/map_tests.py` (it stays out of the repo). For
  each `tests/*.py`, collect the `pytweezer.*` and `bin.*` modules it imports
  (`import x` / `from x import`). Map each module to the area whose file list
  contains it; a package import maps to the area of its `__init__.py`. Write
  the table.
- [ ] **Step 2: Verify**: the row count equals `ls tests/*.py | wc -l` (34),
  and every row has at least one area. For a file whose imports match no area,
  assign it by hand from its name, and note "assigned by hand".
- [ ] **Step 3: Commit**: `git commit -m "Test review: map tests to areas"`.

### Task 5: Stage 2, comparison

**Files:**
- Create: `docs/superpowers/reports/2026-10-test-suite-review/stage2/<slug>.md` (7 files)

**Interfaces:**
- Consumes: `stage1/<slug>.md`, `test-map.md` and `trials/`.
- Produces: `stage2/<slug>.md` with these four sections, parsed by Task 6:

```markdown
## Coverage
| stage-1 id | status (covered / partly / not) | covering test(s) `file::name` |

## Existing tests
| `file::name` | decision (keep / rewrite / delete) | skill rule that decides it |

## Missed by the review
| `file::name` | what it guards | was stage 1 wrong to omit it? (yes / no + why) |

## Verdict
**rework** or **start again**: one paragraph. Report kept-as-is / total.
```

- [ ] **Step 1: Dispatch seven agents in parallel** with this prompt:

```
You are comparing a blind review of part of the pytweezer codebase with its
existing tests. Repo: /home/twalker/RbCaF/pytweezer, branch
test-suite-review. Don't modify anything in the repo.

Read .claude/skills/write-tests/SKILL.md, which is the standard you judge by.
Then read:
- the blind review: docs/superpowers/reports/2026-10-test-suite-review/stage1/<SLUG>.md
- every test file listed for area <SLUG> in
  docs/superpowers/reports/2026-10-test-suite-review/test-map.md
- the source those tests exercise, as needed
- <only for experiments, data-recording, drivers:> the trial rewrites in
  docs/superpowers/reports/2026-10-test-suite-review/trials/green/, as a
  second opinion

"Covered" means the test would fail if the behaviour broke, not merely that
it calls the code. Decide on every test function in your files: keep,
rewrite (it guards a wanted behaviour, but badly — e.g. it checks calls to a
fake, or guards several behaviours at once) or delete. For the verdict:
start again when fewer than about a third of the tests are kept as they are;
otherwise rework.

Optionally, to confirm that a "keep" test really catches faults: export your
own copy with
`git archive HEAD | tar -x -C <$SCRATCH>/mut-<SLUG>`, break the code there,
and run `poetry run pytest` in that copy. Never edit the real repo.

Write to docs/superpowers/reports/2026-10-test-suite-review/stage2/<SLUG>.md
in exactly this format: <the four sections above>.
Reply with the counts of keep / rewrite / delete, the coverage counts, and
the verdict.
```

- [ ] **Step 2: Verify that every test got a decision** (Review Focus 3). Write
  `$SCRATCH/check_decisions.py`. It collects every `def test_` name per
  `tests/*.py` (as `file::name`, ignoring any class prefix), collects every
  `file::name` from the "Existing tests" tables in `stage2/*.md`, and prints
  the set difference.

Run: `python3 $SCRATCH/check_decisions.py`
Expected: an empty difference. For each missing name, send the owning agent
back to decide on it.

- [ ] **Step 3: Verify that every stage-1 id has a coverage row**, by the same
  method applied to the ids from `stage1/<slug>.md` against the "Coverage"
  tables. Expected: an empty difference.

- [ ] **Step 4: Commit**: `git commit -m "Test review: stage 2 comparisons"`.

### Task 6: Stage 3, the report

**Files:**
- Create: `docs/superpowers/reports/2026-10-test-suite-review.md`

**Interfaces:**
- Consumes: `stage1/`, `stage2/`, `test-map.md`, `stage1/SUMMARY.md`, and
  `docs/notes/storage_text_encoding.md` from the `storage-text-encoding` branch
  (`git show storage-text-encoding:docs/notes/storage_text_encoding.md`).

- [ ] **Step 1: List conflicts** (Review Focus 4). Using `check_decisions.py`
  extended with a `--conflicts` flag, print every `file::name` that has
  different decisions in different stage-2 files.

- [ ] **Step 2: Write the report**, for the maintainers, in the order the spec
  sets out. The whole report should be under about 300 lines; the details
  live in `stage1/` and `stage2/`.
  1. A table of areas showing behaviours proposed, covered, partly covered and
     not covered; existing tests kept, rewritten and deleted; and the verdict.
  2. The proposed suite size against today's 239: (kept + rewritten + new
     behaviours not covered), split by priority tier 1–4.
  3. What the review missed: the stage-2 "yes" rows.
  4. Conflicts and how each was settled (keep wins unless the stage-2 reasons
     say otherwise).
  5. Bugs: every stage-1 suspected bug, deduplicated. Also say whether stage 1
     independently found each of the three known bugs (mixed dict and str text
     columns; complex results; out-of-range SLM mask values), as evidence of
     how well the review works.
  6. The dead-code candidates from the spec, plus any data-recording entries
     that depend on the current two text encodings, flagged for revisiting once
     the `storage-text-encoding` change lands.
  7. The padding spread from `stage1/SUMMARY.md`.
  8. The limitation: the same model wrote and reviewed the tests.
  9. The recommended order for carrying it out: Experiments and Data recording
     first, then by each area's share of priority 1–2 behaviours.

- [ ] **Step 3: Verify the numbers.** For each area, the keep + rewrite +
  delete counts in the table equal the number of decided tests in its stage-2
  file, and the totals across all areas equal 239 plus the duplicates counted
  in Step 1.

- [ ] **Step 4: Commit and push**:

```bash
git add docs/superpowers/reports
git commit -m "Test review: report"
git push
```

- [ ] **Step 5:** Link the report on PR #11, and tell the user it is ready.
  Nothing in `tests/` changes until they approve.
