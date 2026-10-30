# Saved parameters — design

Date: 2026-10-30. Status: design, agreed in conversation; not yet planned.
This is the first of two specs; the second, Optimisers, depends on it. The
decisions behind both are in `2026-10-30-optimisers-brainstorm.md`.

## Intent

Saved parameters are the lab's current calibrated values: resonances, MOT and
molasses settings, shim currents, camera conversion factors, the MOT centre.
Today they live in `pytweezer/experiment/parameters.json`, a file tracked in
git, read and written through `ExpParameterManager` by whichever checkout the
code runs in. That has three problems:

- Writing a value (as `optimisation.py` and `analysis.py` do) leaves the
  server PC's checkout with local changes, so `git pull` there fails whenever
  someone commits the file.
- Every other PC reads its own, stale copy.
- There is no record of who changed a value, when, or why.

This spec moves saved parameters into the Experiment Manager, which every PC
and every task already talks to, keeps a log of every change, and gives the
lab a tab to view and edit them. Reading saved values in experiments is the
norm for new experiments (as `TweezerBaseline` does today), so it is built
properly now.

## Goals

1. One set of saved values for the whole lab, owned by the Experiment Manager
   and the same from every PC, notebook and task.
2. Every change is logged with who, when, the old and new value, and why.
3. A task reads the values saved when it reads them; a value changed while a
   task is queued reaches that task when it runs.
4. A GUI tab to view saved values, edit one, and see its history.
5. The current values and every current reader move over without losing a
   value.

## Non-goals

- **No link between saved parameters and experiment arguments or MOTMaster
  parameters.** Saved parameters are a plain store. Code reads a value and
  does with it what it likes: pass it as an argument's default, send it to a
  MOTMaster parameter, use it in analysis.
- **No per-system namespacing.** Keys are flat and unique across the lab.
- **No list or array values** until something needs them.
- **No creating or deleting keys in the GUI.** Keys are created in code or by
  the import.
- **No offline or override mode** for `run_local()`. Tests replace the client
  functions through a fixture (see Testing).
- **Accepting optimiser results in the GUI** is the Optimisers spec; this spec
  provides the `accept` command it uses.

## 1. Data

In a new `pytweezer/experiment/saved_parameters.py`, following the pattern of
`recipes.py`.

```python
class SavedParameter(BaseModel):
    key: str
    value: float | int | bool | str
    group: str = ""
    unit: str = ""
    updated_at: datetime
    updated_by: str          # user@host, as submitter_name() gives it


class SavedParameterChange(BaseModel):
    time: datetime
    who: str
    key: str
    old: float | int | bool | str | None   # None when the key was created
    new: float | int | bool | str
    reason: str
    rid: int | None = None   # the run whose results were accepted, or that set it
```

- **Keys** are flat strings, unique across the lab, stripped of surrounding
  whitespace and non-empty. Today's keys are kept as they are. New CaF keys
  take their own names (e.g. `caf.tImgCoolVCO`, by convention only).
- **Groups** are for display only (`MOTMaster settings`, `Resonances`,
  `EMCCD`, `Image analysis` from the old categories). Lookups ignore them.
- **Values** are JSON scalars. A key keeps its type: an int stays an int, a
  float accepts an int (stored as float), a bool is not an int, a str is only
  a str. A value of the wrong type is refused.

`SavedParameterBook` holds the rules as plain data, like `RecipeBook`:

- `get(key)` raises `SavedParameterError` naming the key, with close matches
  (`difflib`) if any.
- `find(group=None)`, sorted by group then key.
- `set(values, reason, who, *, create=False, group="", unit="", rid=None)`
  applies every value or none. Without `create` an unknown key is refused; with
  it, new keys get `group` and `unit`. Returns the changes made. Setting a key
  to the value it already has is not a change and is not logged.
- `accept(rid, values, who)` is `set` with reason `accepted from run <rid>` and
  `rid` recorded.

`SavedParameterError` is a `ValueError`, so the manager returns it as a
refusal.

`SavedParameterState` (`schema_version`, `parameters: list[SavedParameter]`)
is persisted by `SavedParameterStore` (a `ModelStore`) to
`saved_parameters.json` in the manager's data directory, next to
`recipes.json`, with the same atomic replace and unreadable-file handling. The
change log is appended to `saved_parameters_log.jsonl` in the same directory,
one `SavedParameterChange` per line, never rewritten.

## 2. Manager commands and client functions

`ExperimentManager` gains a `SavedParameterBook`, saves it after every change,
appends to the log, and increments a `saved_parameters_version` that is
included in the published snapshot (as `recipes_version` is).

| command | fields | reply |
| --- | --- | --- |
| `saved_parameters` | optional `group` | `parameters`, `version` |
| `set_saved_parameters` | `values`, `reason`, `who`, optional `create`, `group`, `unit`, `rid` | `changes` |
| `accept_saved_parameters` | `rid`, `values`, `who` | `changes` |
| `saved_parameter_history` | optional `key`, `rid`, `limit` | `entries`, newest first |

`accept_saved_parameters` refuses a run ID the manager does not know.

In `pytweezer/experiment/client.py`, next to the recipe functions:

```python
get_saved_parameter(key) -> value              # KeyError naming the key if missing
get_saved_parameters(group=None) -> dict       # {key: value}
set_saved_parameters(values, reason, *, create=False, group="", unit="", rid=None)
accept_saved_parameters(rid, values)
saved_parameter_history(key=None, *, rid=None, limit=None) -> list[SavedParameterChange]
```

`who` is filled in with `submitter_name()`. Every call goes to the manager;
nothing is cached. With the manager unreachable they raise
`ManagerUnavailable`, as the other client functions do.

## 3. Using saved parameters in code

There is no special behaviour. Code reads values where it needs them:

```python
class MOTCounts(MotMasterExperiment):
    centre_x = Number(get_saved_parameter("x_centre"))    # as an argument's default


class TweezerBaseline(MotMasterExperiment):
    def prepare(self):
        self._saved = {
            parameter: get_saved_parameter(key)
            for parameter, key in SAVED_MOTMASTER_PARAMETERS.items()
        }                                                  # sent to MOTMaster parameters
        ...
```

- The Experiment Manager's worker imports the experiment module fresh for
  every task, so a value read at class level or in `prepare()` is the value
  saved when the task starts.
- A module that reads a value at class level contacts the manager when it is
  imported (by the worker, the catalogue, the tests). The Experiments form
  shows such a default as of when the catalogue was last loaded, and a recipe
  stores it like any other argument value.
- What a run records in its measurement file is up to the experiment, as
  `TweezerBaseline` already records `motmaster_parameters`.
- Ordinary experiments are expected only to read saved values. Writing is not
  prohibited: an experiment may call `set_saved_parameters(..., rid=...)`.
  Optimisers do not write; their results are accepted by hand.

## 4. The Saved parameters tab

A new tab, `Saved parameters`, after `Results` in `bin/gui.py`, in
`pytweezer/GUI/saved_parameters.py`. Mockup: the "Saved parameters tab"
artboard on https://claude.ai/artifact/2E9MRytEvLUcr7AVahEcPJ.

- **Well, left: "Saved parameters"**, hint "N values, read by experiments when
  they run; pick one to edit", with a filter box. A table grouped by group:
  key, value, unit, last changed (when, who), why. Values changed today are in
  bold.
- **Sheet, top right: the selected key.** Value field with its unit, a "Why"
  field (saved as the reason), Apply as the primary button with the note
  "tasks that start afterwards use the new value", and Revert. Apply calls
  `set_saved_parameters` with the reason prefixed `set by hand: `.
- **Well, bottom right: "History"** for the selected key, newest first: when,
  old, new, who and why.

The tab follows the manager's published snapshot through `ExperimentFeed`
and refetches when `saved_parameters_version` changes, updating in place:
selection, scroll position and an unsent edit survive an update. It polls
nothing while hidden. If the manager is unreachable the tab says so
("Saved parameters: the Experiment Manager is not responding") rather than
showing stale values as current.

## 5. Migration

**The values.** The real values are in the server PC's checkout of
`pytweezer/experiment/parameters.json`, possibly with local changes. A new
console script, `pytweezer-saved-parameters import <file>`, loads a
`parameters.json` into the store through `set_saved_parameters(create=True)`:

- each category becomes a group (`motmaster_parameters` → `MOTMaster
  settings`, `misc_parameters` → `Resonances`, `emccd_parameters` → `EMCCD`,
  `image_analysis` → `Image analysis`);
- units are filled in where obvious: `V` for keys containing `VCO` or `VVA`
  (including the two resonances), `px` for `x_centre` and `y_centre`, `mm/px`
  for `*_mm_per_px`; the rest are left blank;
- every value is logged with reason `imported from parameters.json`;
- it refuses if the store already holds values, unless given `--replace`.

On the server PC, the steps (in the PR description):

1. Copy `pytweezer/experiment/parameters.json` somewhere outside the checkout.
2. Discard its local changes and pull.
3. Start the Experiment Manager and run the import on the copy.
4. Check the values in the Saved parameters tab.

**The code**, in the same PR:

- `TweezerBaseline` reads its `SAVED_MOTMASTER_PARAMETERS` map with
  `get_saved_parameter` in `prepare()`; its behaviour is otherwise unchanged.
  The `parameters_file` argument is removed.
- `pytweezer/analysis/analysis.py`: the module-level read of
  `cool_vco_resonance` moves into the function that uses it, so importing
  analysis never contacts the manager. The `x_centre`/`y_centre` read and
  write use `get_saved_parameter` / `set_saved_parameters`.
- `pytweezer/helper_scripts/optimisation.py` keeps its behaviour of writing
  results directly: `ExpParams.get_parameter` becomes `get_saved_parameter`,
  and `set_parameter` plus `save_parameters` becomes `set_saved_parameters`
  with reason `optimisation.py`.
- `notebooks/tweezer_baseline.ipynb` writes `conversion_factor` with
  `set_saved_parameters`.
- `pytweezer/experiment/parameters.json` and
  `pytweezer/experiment/experiment_parameter_manager.py` are deleted.

Left alone: `notebooks/optimisation-scans-testing.ipynb` and
`optimisation-scans-testing-2.ipynb`, which already import a class that no
longer exists. Lab-book notebooks that use `ExpParameterManager` break; before
the PR merges, the lab book is searched and the affected notebooks are listed
in the PR description.

## 6. Testing

Following the `write-tests` skill.

A `saved_parameters` fixture in `tests/conftest.py` replaces
`get_saved_parameter`, `get_saved_parameters` and `set_saved_parameters` with
functions over a dict the test supplies, and records every write.

`tests/test_saved_parameters.py` (new):

1. A key keeps its type: an int stays an int, a float accepts an int, a wrong
   type is refused (parametrised).
2. Setting an unknown key is refused unless `create=True`.
3. Every change is logged with who, when, old, new and why; `accept` records
   the run ID.
4. Values and the log read back identical after the store is reopened.
5. An unreadable store file is moved aside, never silently overwritten.
6. The import copies all values unchanged, turns the categories into groups
   and fills the obvious units.
7. The import refuses a store that already has values, unless `--replace`.

`tests/test_experiment_manager.py`:

8. A value written with `set_saved_parameters` is returned by the next read,
   and the snapshot's `saved_parameters_version` goes up.
9. `accept_saved_parameters` refuses an unknown run ID.

`tests/test_experiment_e2e.py`:

10. A task queued before a value changes reads the new value when it starts.

`tests/test_experiments.py` (new), parametrised over every module in
`pytweezer/experiments/`, with the `saved_parameters` fixture applied:

- it imports cleanly the way the manager imports it
  (`introspect.describe`);
- every declared device exists in CONFIG (`reserved_devices()`).

These replace per-experiment tests: `tests/test_tweezer_baseline.py` is
deleted. `tests/test_optimisation.py` uses the fixture instead of patching
`ExpParams`.

`tests/test_experiment_gui.py`: the tab builds offscreen; Apply sends the value
with its reason; a live update keeps the selected key.

Not tested, ranked: importing `analysis.py` does not contact the manager; the
table filter; the units the import fills in; the bold styling for today's
changes (checked by eye).
