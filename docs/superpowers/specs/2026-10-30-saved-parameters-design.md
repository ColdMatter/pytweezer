# Saved parameters — design

Date: 2026-10-30, revised 2026-10-31. Status: design, awaiting review.
This is the first of two specs; the second, Optimisers, depends on it. The
decisions behind both are in `2026-10-30-optimisers-brainstorm.md`.

## Intent

Saved parameters are the lab's current calibrated values. Today they live in
`pytweezer/experiment/parameters.json`, a file tracked in git, read and
written through `ExpParameterManager` by whichever checkout the code runs in.
That has three problems:

- Writing a value leaves the server PC's checkout with local changes, so
  `git pull` there fails whenever someone commits the file.
- Every other PC reads its own, stale copy.
- There is no record of who changed a value, when, or why.

This spec moves saved parameters into the Experiment Manager, which runs every
task and which every PC already talks to, keeps a log of every change, and
gives the lab a tab to view and edit them. The store knows nothing about what
the values mean: it is a general facility of the task-running system.

## Goals

1. One set of saved values for the whole lab, owned by the Experiment Manager
   and the same from every PC, notebook and task.
2. Every change is logged with who, when, the old and new value, and why.
3. Code that reads a saved value gets the value saved at that moment; nothing
   is cached.
4. A GUI tab to view saved values, edit one, and see its history.
5. The current values and every current reader move over without losing a
   value.

## Non-goals

- **No inherent link between saved parameters and experiment arguments or
  MOTMaster parameters.** Code reads a value and does with it what it likes,
  including passing it to an argument as its default (section 3).
- **No per-system namespacing.** Keys are flat and unique across the lab.
- **No list or array values** until something needs them.
- **No creating or deleting keys in the GUI.** Keys are created in code.
- **No Accept.** Accepting optimiser results is the Optimisers spec; it writes
  through `set_saved_parameters(..., rid=...)`, or adds a command of its own
  when it needs one.

## 1. Data

In a new `pytweezer/experiment/saved_parameters.py`, following the pattern of
`recipes.py`.

```python
Value = float | int | bool | str


class SavedParameter(BaseModel):
    key: str
    value: Value
    group: str = ""
    unit: str = ""
    updated_at: datetime
    updated_by: str          # user@host, as submitter_name() gives it


class SavedParameterChange(BaseModel):
    time: datetime
    who: str
    key: str
    old: Value | None        # None when the key was created
    new: Value
    reason: str
    rid: int | None = None   # the run that set it, if any
```

- **Keys** are flat strings, stripped of surrounding whitespace and non-empty.
  Today's keys are kept as they are.
- **Groups** are for display only. Lookups ignore them.
- **Values** are JSON scalars. A key keeps its type: an int stays an int, a
  float accepts an int (stored as float), a bool is not an int, a str is only
  a str. A value of the wrong type, or a non-finite float, is refused.

`SavedParameterBook` holds the rules as plain data, like `RecipeBook`:

- `get(key)` raises `SavedParameterError` naming the key, with close matches
  (`difflib`) if any.
- `all()`, sorted by group then key.
- `set(values, reason, who, *, create=False, group="", unit="", rid=None)`
  applies every value or none. Without `create` an unknown key is refused; with
  it, new keys get `group` and `unit`. Returns the changes made. Setting a key
  to the value it already has is not a change and is not logged.

`SavedParameterError` is a `ValueError`, so the manager returns it as a
refusal.

`SavedParameterState` (`schema_version`, `parameters: list[SavedParameter]`)
is persisted by `SavedParameterStore` (a `ModelStore`) to
`saved_parameters.json` in the manager's data directory, next to
`recipes.json`. The change log is appended to `saved_parameters_log.jsonl` in
the same directory, one `SavedParameterChange` per line, never rewritten.

## 2. Manager commands and client functions

`ExperimentManager` gains a `SavedParameterBook`, saves it after every change,
appends to the log, and increments a `saved_parameters_version` that is
included in the published snapshot (as `recipes_version` is).

| command | fields | reply |
| --- | --- | --- |
| `saved_parameters` | | `parameters`, `version` |
| `set_saved_parameters` | `values`, `reason`, `who`, optional `create`, `group`, `unit`, `rid` | `changes` |
| `saved_parameter_history` | optional `key`, `limit` | `entries`, newest first |
| `describe_again` | `module` | |

`describe_again` makes the catalogue describe `module` afresh even though its
file has not changed; the result reaches GUIs through `catalogue_version` as
any other catalogue change does (section 3).

In `pytweezer/experiment/client.py`, next to the recipe functions, each taking
`client=` as the existing functions do:

```python
get_saved_parameter(key) -> Value              # KeyError naming the key if missing
get_saved_parameters() -> dict[str, Value]
set_saved_parameters(values, reason, *, create=False, group="", unit="", rid=None)
saved_parameter_history(key=None, *, limit=None) -> list[SavedParameterChange]
```

`who` is filled in with `submitter_name()`. Every call goes to the manager at
the address CONFIG gives; nothing is cached. With the manager unreachable they
raise `ManagerUnavailable`, as the other client functions do.

## 3. Using saved parameters in code

There is no special behaviour. Code reads values where it needs them:

```python
class Example(Experiment):
    threshold = Number(get_saved_parameter("example_threshold"))   # a default

    def prepare(self):
        self.offset = get_saved_parameter("example_offset")       # read at run time
```

- **Read in `prepare()` or `run_point()`:** the value saved when the task runs.
- **Passed to an argument as its default:** read whenever the module is
  imported. That happens:
  - when the catalogue describes the experiment, which sets the default the
    Experiments form shows;
  - when the worker imports the experiment for a task, which fills in any
    argument the request leaves out (a notebook `submit()` that does not pass
    it);
  - in a notebook or test that imports the module.
- **The Experiments form submits exactly what it shows.** Its defaults are as of
  the catalogue's last description, which may be older than the latest saved
  value. **Defaults** sends `describe_again` for the selected experiment and,
  when the new description arrives, fills the form with its defaults, so a
  changed saved value is picked up there.
- A recipe stores argument values like any other.
- Importing a module that reads a saved value contacts the manager. If the
  manager is unreachable or the key is missing, the import fails and the
  catalogue shows that error for the module, as for any other import error.
- What a run records in its measurement file is up to the experiment.

## 4. The Saved parameters tab

A new tab, `Saved parameters`, after `Results` in `bin/gui.py`, in
`pytweezer/GUI/saved_parameters.py`. Mockup: the "Saved parameters tab"
artboard on https://claude.ai/artifact/2E9MRytEvLUcr7AVahEcPJ.

- **Well, left: "Saved parameters"**, hint "N values; pick one to edit", with a
  filter box. A table grouped by group: key, value, unit, last changed (when,
  who), why. Values changed today are in bold.
- **Sheet, top right: the selected key.** Value field with its unit, a "Why"
  field (saved as the reason), Apply as the primary button, and Revert. The
  note under Apply reads "Read from now on; press Defaults in Experiments to
  use it in an open form". Apply calls `set_saved_parameters` with the reason
  prefixed `set by hand: `. A value that does not parse as the key's type is
  refused in the sheet and nothing is sent.
- **Well, bottom right: "History"** for the selected key, newest first: when,
  old, new, who and why.

The tab follows the manager's published snapshot through `ExperimentFeed`
and refetches when `saved_parameters_version` changes, updating in place:
selection, scroll position and an unsent edit survive an update. It polls
nothing while hidden. If the manager is unreachable the tab says so
("Saved parameters: the Experiment Manager is not responding") rather than
showing stale values as current.

## 5. Migration

**The values** are moved once, on the server PC, by
`import_parameters_json(path)` in `saved_parameters.py`, marked `FIXME` for
removal once the move is done. It reads a `parameters.json` and calls
`set_saved_parameters(..., create=True)` once per category, with the category
name as the group and reason `imported from parameters.json`. Running it twice
changes nothing.

On the server PC, the steps (in the PR description):

1. Copy `pytweezer/experiment/parameters.json` somewhere outside the checkout.
2. Discard its local changes and pull.
3. Start the Experiment Manager and run the import on the copy.
4. Check the values in the Saved parameters tab, filling in units by hand.

**The code**, in the same PR, reads and writes through the client functions:

- `TweezerBaseline` reads its `SAVED_MOTMASTER_PARAMETERS` in `prepare()`; its
  `parameters_file` argument is removed.
- `pytweezer/analysis/analysis.py`: the module-level read moves into the
  function that uses it; the `x_centre`/`y_centre` read and write use the
  client functions.
- `pytweezer/helper_scripts/optimisation.py` keeps writing results directly,
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

Following the `write-tests` skill. Existing tests that assume
`parameters.json` (`tests/test_tweezer_baseline.py`,
`tests/test_optimisation.py`) are changed to fit. Tests that need saved values
run a manager on a scratch data directory and point the client's CONFIG at
it; there is no fake store.

`tests/test_saved_parameters.py` (new):

1. A key keeps its type: an int stays an int, a float accepts an int; a wrong
   type or a non-finite float is refused and nothing changes (parametrised).
2. Setting an unknown key is refused unless `create=True`, all or nothing.
3. Every change is logged with who, when, old, new and why; setting a key to
   its current value is not logged.
4. The import copies every value, with its category as the group.

`tests/test_experiment_manager.py`:

5. A value written with `set_saved_parameters` is read back, survives a
   manager restart, and the snapshot's `saved_parameters_version` goes up.
6. A misspelt key in `get_saved_parameter` raises `KeyError` naming the
   closest key.
7. `describe_again` describes an unchanged module afresh.

`tests/test_experiment_gui.py`:

8. The tab builds offscreen; Apply sends the value with its reason; a value of
   the wrong type is refused before sending; a live update keeps the selected
   key and an unsent edit; an unreachable manager is said, not shown as
   current.
9. Defaults in the Experiments form asks for the experiment to be described
   again and applies the new defaults when they arrive.

Not tested, ranked: the table filter; the bold styling for today's changes
(checked by eye).
