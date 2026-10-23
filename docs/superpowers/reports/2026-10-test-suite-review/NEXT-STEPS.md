# Next steps: fix the urgent bugs

Handoff for a new agent session. Read this file, then `CLAUDE.md` and the
`write-tests` skill, before touching code.

## Where things stand

- **A blind review of the test suite is finished.** The report is
  `docs/superpowers/reports/2026-10-test-suite-review.md` and its raw outputs
  are in this folder. The verdict is to rework the suite, not rewrite it, but
  **the bugs the review found come first**.
- **Every suspected bug, with the smallest input that shows it**, is in the
  `## Suspected bugs` section at the end of each `stage1/<area>.md`. Most were
  found by reading the code, not by running it.
- **Branches and PRs:**
  - `write-tests-skill` (PR #9): the testing standard.
  - `storage-text-encoding` (draft PR #10): a design note for the storage fix.
  - `test-suite-review` (draft PR #11): the review, this file included, based
    on #9.
- **The review files** live on `test-suite-review`. If PR #11 isn't merged when
  you start, read them with `git show origin/test-suite-review:<path>`.

## Before you start

1. **Check that PR #9 is merged** into `exp-handler`
   (`gh pr view 9 --json state`). If it isn't, stop and ask the user. They
   chose to merge it first, so that bug-fix tests follow the skill.
2. **Re-check each bug against the current `exp-handler`** before fixing it.
   The review read commit `c95e9f5`, and the code may have moved since.

## Decisions the user has made

| Question | Decision |
|---|---|
| Which bugs first? | The urgent list below, about 20. The other ~60 wait for the per-area test rework. |
| Behaviour that may be deliberate | **Ask per item.** Collect them, each with a one-line question, and leave the code alone until the user answers. |
| A text result at one point and a dict at another (storage) | **Refuse with a `RecordError`.** Update the "Still to decide" section of `docs/notes/storage_text_encoding.md` to say so. |
| How to group fixes into PRs | **One PR per area.** Each bug gets its own test and its own commit. |
| Coordinator and phase-mask code | **The `rearrangement-optimisation-RnD` branch is the live version.** Don't fix `coordinators/` or `phasemask.py` on `exp-handler`. |
| The 3 failing tests in `tests/test_rearrangement.py` | **Delete them**: `test_phasemask_kwargs_merge_config_over_defaults`, `test_arm_streams_sequence_to_slm` and `test_arm_reraises_slm_upload_error`. |
| Dead code | **Ask per file.** Don't delete anything yet. |
| Lowest-priority (P4) test proposals | Decided per area during the later test rework. Not part of this work. |

## How to work each bug

1. **Reproduce it first**, with a failing test written to the `write-tests`
   skill, using the smallest input from `stage1/<area>.md`. Put the test in the
   existing test file for that module.
2. **If it doesn't reproduce**, drop it and record it as "not reproduced" in the
   PR description. Don't fix it anyway.
3. **If it could be deliberate**, don't change it; add it to the questions for
   the user instead. Three are already known:
   - SLM mask values outside 0–255 wrap round (`drivers/slm.py`)
   - `subtract_bg` takes the absolute value of the residual
   - MOTMaster `_auto_mot` error bars use `ddof=0`, giving 0 for one shot per
     point
4. **Make the smallest fix** that turns the test green, then run
   `poetry run pytest tests/ -q`. The only failures allowed are the 3
   `test_rearrangement.py` tests, until that PR lands.
5. **Commit each bug on its own**, and push as you go.
6. **The PR description lists, for each bug:** the test that exposes it, a
   one-line description of the fix, or "not reproduced" or "asked the user".

If you check a test by deliberately breaking code in a scratch copy: run
pytest there with `PYTHONPATH` set to the copy, the copy as the working
directory, and the venv's `python`. Without that, the editable install imports
the real repo, and the check proves nothing.

## The work, in order

Each item is one branch and one PR into `exp-handler`.

### 0. Delete the 3 failing rearrangement tests
This one is tiny and goes first, so that every later PR can expect a green
suite.

### 1. Data recording
Build this on the `storage-text-encoding` branch, turning PR #10 into this
area's PR and renaming it.
- **`experiment/storage.py`:** store all non-numeric results as JSON, and refuse
  a mix of text and dict under one name. The design is in
  `docs/notes/storage_text_encoding.md`, including the `SCHEMA_VERSION` bump to
  2 and reading version-1 files.
- **`experiment/storage.py`:** complex results can't be saved, because the
  dataset's fill value is `np.nan`. Use `complex(np.nan, np.nan)`.
- **`database/writer.py`:** one NaN or infinity in a JSON column fails the
  batch on every retry and blocks all later writes. This was inferred from the
  code; reproduce it against the local test database. Set `PYTWEEZER_TEST_DSN`,
  as `tests/test_db_integration.py` does; the local dev stack is described in
  this project's Claude memory.
- **`database/writer.py`:** `np.bool_` values are silently dropped.
- **`database/writer.py`:** `dsn_host` leaves the password in place for
  key=value DSNs.
- **`database/backfill.py`:** `rglob` picks up `simulated/` files, which
  overwrite real `runs` rows that share their rid.
- **`loggers/ni_adc_logger.py`:** channels with the same short name
  (`Dev1/ai0`, `Dev2/ai0`) collapse into one field.

### 2. Experiments
- **`experiment/client.py`:** `ReqClient` resends after its timeout, so a
  manager that takes more than 2 s to reply queues the same `submit` twice.
- **`experiment/queue.py`:** `ModelStore.load` treats any `OSError` as
  corruption, moving the queue or recipes file aside on a transient read error.

### 3. GUI (only the experiment-facing items)
- **`GUI/experiments/arg_editor.py`:** number fields round to 6 decimals in
  display units, so a default of 2e-7 is submitted as 0.
- **`GUI/experiments/queue_view.py`, and `GUI/analysismanager.py`'s
  `FilterTable`:** when the selected row's task disappears, the row index stays
  selected, so the next Delete or Abort hits a different task.

### 4. Servers and config
- **`servers/clients.py`:** Fortran-order arrays (any `frame.T`) are sent in
  memory order but reshaped C-order, so they arrive scrambled.
- **`servers/device_server.py`:** two sub-devices in a composite with the same
  `role` silently overwrite each other. Make it raise, as the other config
  errors in that function do.

### 5. Drivers (`drivers/motmaster.py`)
- The `"aceeleration"` typo means the acceleration is never read and
  `setup_velocity` is never called.
- `picomotor_default_axis` is never assigned, so `move_yag_spot=True` never
  moves the stage.
- The picomotor reversal counter is never reset, so the direction flips on
  every later call.

These can't be checked against real hardware here. Test them through the
driver's own logic with a stand-in at the hardware edge, as the skill
describes, and say so in the PR.

### 6. Analysis (`analysis/analysis.py`, `analysis/rotate_image.py`)
**First** check whether `rearrangement-optimisation-RnD` changes these files
(`git diff exp-handler origin/rearrangement-optimisation-RnD -- pytweezer/analysis/`).
If it does, ask the user before fixing them here.
- The fixed-threshold "fidelity" is really the total error, about 0 for clean
  data.
- `detect_trap_sites` hangs on `uint16` frames when it can't match the site
  count, because the threshold wraps around.
- `detect_bright_points` truncates each centre of mass instead of rounding it.
- `fit_gaussian` swaps `sx`/`sy` but not their errors or `theta`.
- `rotate_image` swaps the x and y offsets: it is wrong at 0° and the identity
  at 90°.

## Questions to bring back to the user

Collect these in one message rather than asking as you go:
- each possibly-deliberate behaviour (see "How to work each bug")
- **the dead-code files, one by one**, each with what it contains and a check
  that nothing imports it:
  - `pytweezer/analysis.py`, a stale duplicate of
    `pytweezer/analysis/analysis.py`
  - `data_manager.py`
  - `arduino.py`
  - `analysis/analysis_old.py`
  - `servers/datamgr.py`
  - `GUI/models.py`
  - `GUI/fitfunctions.py`
  - `GUI/qled.py`
  - `drivers/anapico.py`
  - `analysis/floating_point_arithmetics.py`

## Not in this work

- **Bugs in `coordinators/`, `phasemask.py` and `optimisation.py`**, since
  the RnD branch is live.
- **The remaining suspected bugs** outside the urgent list.
- **The test rework itself.** When it starts, three things need doing first:
  - **Apply the conflict rulings** in the report's "Conflicts between areas"
    section. The `stage2/` tables still say "delete" for both tests.
  - **Merge the 25 "missed by the review" behaviours** into the `stage1/`
    lists.
  - **Re-check against the current code**, since `c95e9f5` will be out of
    date.
