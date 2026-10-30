# Saved Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the lab's saved parameters out of the git-tracked `parameters.json` into a store owned by the Experiment Manager, with a change log, client functions, an import tool and a GUI tab.

**Architecture:** A plain-data `SavedParameterBook` (rules) persisted by a `ModelStore` subclass plus an append-only JSONL log, both in the manager's data directory, following `recipes.py`. The manager exposes four commands; `client.py` wraps them in functions every PC, notebook and worker uses. A Qt tab follows the manager's published snapshot.

**Tech Stack:** Python 3, pydantic, ZMQ REQ/REP (existing `ExperimentManagerClient`), sipyco sync (existing `ExperimentFeed`), PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-saved-parameters-design.md` (read it first; this plan argues from it).

## Global Constraints

- UK English in code, docstrings and GUI text; Google-style docstrings describing current behaviour only.
- Logging through `get_logger(name)` from `pytweezer/logging_utils.py`, never `logging.getLogger`.
- Comment only what is surprising (CLAUDE.md "Comments and self-documenting code").
- Run `uv run ruff check` and `uv run ruff format` on touched files; `uv run pytest tests/ -q` must pass after every task.
- Store file: `saved_parameters.json`; log file: `saved_parameters_log.jsonl`; both in the manager's data directory (`ExperimentManager.root`), next to `recipes.json`.
- Keys are flat strings, stripped, non-empty. Values are JSON scalars: `float | int | bool | str`. A key keeps its type: int stays int, float accepts int (stored as float), bool is not an int, str only str.
- Exact reason strings: `accepted from run <rid>`, `set by hand: <why>`, `imported from parameters.json`, `optimisation.py`.
- Snapshot key: `saved_parameters_version`.
- Tab title `Saved parameters`, placed directly after `Results` in `bin/gui.py`. Unreachable-manager text: `Saved parameters: the Experiment Manager is not responding`.
- Import groups: `motmaster_parameters` → `MOTMaster settings`, `misc_parameters` → `Resonances`, `emccd_parameters` → `EMCCD`, `image_analysis` → `Image analysis`. Units: `V` for keys containing `VCO` or `VVA` and for `cool_vco_resonance`/`rep_vco_resonance`; `px` for `x_centre`, `y_centre`; `mm/px` for keys ending `_mm_per_px`; otherwise empty.
- Commit as you go, one commit per task at least, messages in the repo's plain-sentence style, ending with the attribution lines the session gives.

## Review Focus

1. A non-finite float (`nan`, `inf`), as a failed fit hands `optimisation.py`, is refused naming the key, not stored (JSON cannot hold it). Test in Task 1.
2. A value typed in the tab that does not parse as the key's type (`"4,05"` for a float, `"4.5"` for an int) is refused with a message in the sheet and nothing is sent. Test in Task 6.
3. A live update arriving while the user has typed but not applied a value keeps the typed value. Test in Task 6.
4. A misspelt key in `get_saved_parameter` raises `KeyError` naming the key and the closest existing key. Test in Task 3.
5. An old `parameters.json` holding a list value (notebooks used `add_parameter("site_positions", [...])`) makes the import refuse, naming the key, with nothing imported. Test in Task 4.

---

### Task 1: The store, its rules and the change log

**Files:**
- Create: `pytweezer/experiment/saved_parameters.py`
- Test: `tests/test_saved_parameters.py`

**Interfaces:**
- Produces:
  - `Value = float | int | bool | str`
  - `class SavedParameterError(ValueError)`
  - `class SavedParameter(BaseModel)`: `key: str`, `value: Value`, `group: str = ""`, `unit: str = ""`, `updated_at: datetime`, `updated_by: str`
  - `class SavedParameterChange(BaseModel)`: `time: datetime`, `who: str`, `key: str`, `old: Value | None`, `new: Value`, `reason: str`, `rid: int | None = None`
  - `class SavedParameterState(BaseModel)`: `schema_version: int = 1`, `parameters: list[SavedParameter]`
  - `class SavedParameterBook`: `__init__(state: SavedParameterState | None = None)`; `get(key: str) -> SavedParameter`; `find(group: str | None = None) -> list[SavedParameter]` (sorted by group then key); `set(values: dict[str, Value], *, reason: str, who: str, create: bool = False, group: str = "", unit: str = "", rid: int | None = None) -> list[SavedParameterChange]`; `accept(rid: int, values: dict[str, Value], who: str) -> list[SavedParameterChange]`
  - `class SavedParameterStore(ModelStore)` with `model = SavedParameterState`
  - `class SavedParameterLog`: `__init__(path: Path | str)`; `append(changes: list[SavedParameterChange]) -> None`; `read(key: str | None = None, *, rid: int | None = None, limit: int | None = None) -> list[SavedParameterChange]` (newest first)

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    "existing, new, stored",
    [(600, 650, 650), (4.1, 4, 4.0), (True, False, False), ("a", "b", "b")],
)
def test_a_key_keeps_its_type(existing, new, stored):
    book = book_with(key=existing)
    book.set({"key": new}, reason="r", who="me")
    value = book.get("key").value
    assert value == stored and type(value) is type(stored)


@pytest.mark.parametrize(
    "existing, new",
    [(600, 4.5), (600, True), (4.1, "4.2"), (4.1, float("nan")), (4.1, float("inf"))],
)
def test_a_value_of_the_wrong_type_or_not_finite_is_refused(existing, new):
    book = book_with(key=existing)
    with pytest.raises(SavedParameterError, match="key"):
        book.set({"key": new}, reason="r", who="me")
    assert book.get("key").value == existing


def test_an_unknown_key_is_refused_unless_created():
    book = book_with(key=1.0)
    with pytest.raises(SavedParameterError, match="kye"):
        book.set({"key": 2.0, "kye": 3.0}, reason="r", who="me")
    assert book.get("key").value == 1.0          # all or nothing
    book.set({"kye": 3.0}, reason="r", who="me", create=True, group="G", unit="V")
    assert (book.get("kye").group, book.get("kye").unit) == ("G", "V")


def test_every_change_is_logged_and_accept_records_the_run(tmp_path):
    book, log = book_with(key=1.0), SavedParameterLog(tmp_path / "log.jsonl")
    log.append(book.set({"key": 1.0}, reason="same", who="me"))   # not a change
    log.append(book.set({"key": 2.0}, reason="by hand", who="me"))
    log.append(book.accept(1482, {"key": 3.0}, who="you"))
    newest, older = log.read("key")
    assert (older.old, older.new, older.reason, older.who) == (1.0, 2.0, "by hand", "me")
    assert (newest.reason, newest.rid) == ("accepted from run 1482", 1482)
    assert log.read(rid=1482) == [newest]


def test_values_and_log_read_back_identical_after_reopening(tmp_path): ...
    # save via SavedParameterStore, load a new store and a new SavedParameterLog
    # at the same paths; assert state and log entries compare equal.


def test_an_unreadable_store_is_moved_aside(tmp_path): ...
    # write "not json" to saved_parameters.json; load() returns an empty state
    # and a saved_parameters.unreadable-*.json file now holds the original text.
```

`book_with(**values)` is a test helper building a book whose keys were created with `create=True`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_saved_parameters.py -q`
Expected: FAIL with `ModuleNotFoundError: pytweezer.experiment.saved_parameters`

- [ ] **Step 3: Implement `pytweezer/experiment/saved_parameters.py`**

Follow `recipes.py` (module docstring, `now()` from `pytweezer.experiment.queue` for timestamps). `get` raises `SavedParameterError` naming the key and up to three `difflib.get_close_matches`. `set` validates every value before changing any; skips values equal to the current one; `create=True` only applies `group`/`unit` to keys it creates. `accept` is `set` with `reason=f"accepted from run {rid}"` and `rid=rid`. `SavedParameterLog.append` writes one `model_dump_json()` line per change in append mode; `read` parses every line, filters, reverses, truncates.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_saved_parameters.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/saved_parameters.py tests/test_saved_parameters.py
git commit -m "Add the saved parameter store, its rules and change log"
```

---

### Task 2: Manager commands

**Files:**
- Modify: `pytweezer/servers/experiment_manager.py` (constructor near the recipe store, `_snapshot`, the `#:` layout comment, new `_cmd_*` methods next to the recipe ones)
- Test: `tests/test_experiment_manager.py`

**Interfaces:**
- Consumes: Task 1's `SavedParameterBook`, `SavedParameterStore`, `SavedParameterLog`, `SavedParameterChange`.
- Produces (commands; replies are JSON as `model_dump(mode="json")`):
  - `saved_parameters` (`group` optional) → `{"parameters": [...], "version": int}`
  - `set_saved_parameters` (`values`, `reason`, `who`, optional `create`, `group`, `unit`, `rid`) → `{"changes": [...]}`
  - `accept_saved_parameters` (`rid`, `values`, `who`) → `{"changes": [...]}`; refuses a rid `self.queue.get` does not know
  - `saved_parameter_history` (optional `key`, `rid`, `limit`) → `{"entries": [...]}`, newest first
  - snapshot gains `"saved_parameters_version"`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_written_value_is_read_back_and_bumps_the_version(make_manager):
    manager = make_manager()
    version = manager._snapshot()["saved_parameters_version"]
    reply = manager.handle({"command": "set_saved_parameters", "values": {"x_centre": 131.5},
                            "reason": "r", "who": "me", "create": True})
    assert reply["ok"], reply
    [parameter] = manager.handle({"command": "saved_parameters"})["parameters"]
    assert (parameter["key"], parameter["value"]) == ("x_centre", 131.5)
    assert manager._snapshot()["saved_parameters_version"] == version + 1
    assert make_manager().handle({"command": "saved_parameters"})["parameters"] == [parameter]


def test_accept_refuses_an_unknown_run(make_manager):
    manager = make_manager()
    reply = manager.handle({"command": "accept_saved_parameters", "rid": 999,
                            "values": {"x_centre": 1.0}, "who": "me"})
    assert not reply["ok"] and "999" in reply["error"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_manager.py -q -k "saved or accept"`
Expected: FAIL (unknown command / missing snapshot key).

- [ ] **Step 3: Implement**

In `__init__`: `self.saved_parameter_store = SavedParameterStore(self.root / "saved_parameters.json")`, `self.saved_parameters = SavedParameterBook(self.saved_parameter_store.load())`, `self.saved_parameter_log = SavedParameterLog(self.root / "saved_parameters_log.jsonl")`, `self._saved_parameters_version = 0`. A `_saved_parameters_changed(changes)` mirrors `_recipes_changed`: save the store, append the log, bump the version, mark dirty; skip all three when `changes` is empty. Add `saved_parameters_version` to `_snapshot()` and to the `#:` layout comment. `SavedParameterError` already reaches the caller as a refusal (it is a `ValueError`); check how `handle` turns `RecipeError` into `{"ok": False, "error": ...}` and rely on the same path.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_experiment_manager.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/servers/experiment_manager.py tests/test_experiment_manager.py
git commit -m "Experiment Manager: serve saved parameters and their history"
```

---

### Task 3: Client functions, the worker's endpoint and the test fixture

**Files:**
- Modify: `pytweezer/experiment/client.py`, `pytweezer/experiment/worker.py` (`main`)
- Modify: `tests/conftest.py`, `tests/helpers.py`
- Test: `tests/test_experiment_manager.py`, `tests/test_experiment_e2e.py`

**Interfaces:**
- Consumes: Task 2's commands; Task 1's models.
- Produces:
  - `ExperimentManagerClient` methods: `saved_parameters(group: str | None = None) -> list[SavedParameter]`, `set_saved_parameters(values, reason, who, *, create=False, group="", unit="", rid=None) -> list[SavedParameterChange]`, `accept_saved_parameters(rid, values, who) -> list[SavedParameterChange]`, `saved_parameter_history(key=None, *, rid=None, limit=None) -> list[SavedParameterChange]`. Add `"saved_parameters"` and `"saved_parameter_history"` to `_READ_ONLY_COMMANDS`.
  - Module functions, each with `client: ExperimentManagerClient | None = None` and filling `who` with `submitter_name()`:
    `get_saved_parameter(key: str, /, *, client=None) -> Value`;
    `get_saved_parameters(group: str | None = None, /, *, client=None) -> dict[str, Value]`;
    `set_saved_parameters(values: dict[str, Value], reason: str, /, *, create=False, group="", unit="", rid=None, client=None) -> list[SavedParameterChange]`;
    `accept_saved_parameters(rid: int, values: dict[str, Value], /, *, client=None) -> list[SavedParameterChange]`;
    `saved_parameter_history(key: str | None = None, /, *, rid=None, limit=None, client=None) -> list[SavedParameterChange]`
  - `client.default_endpoint: str | None = None`; `ExperimentManagerClient` uses `endpoint or default_endpoint or rep_endpoint()`. `worker.main` sets it to `args.endpoint`, so code inside a task reaches the manager that started it.
  - Fixture `saved_parameters` (in `tests/conftest.py`) returning a `FakeSavedParameters` (in `tests/helpers.py`) with `values: dict[str, Value]` and `writes: list[dict]`; it monkeypatches `pytweezer.experiment.client.ExperimentManagerClient` with a stand-in whose four saved-parameter methods work on `values` and append each write to `writes` as `{"values", "reason", "rid", "group", "unit", "create"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiment_manager.py
def test_a_misspelt_key_names_the_closest_one(make_manager):
    client = in_process_client(make_manager())   # ExperimentManagerClient with _req = InProcess(manager)
    set_saved_parameters({"cool_vco_resonance": 4.05}, "r", create=True, client=client)
    assert get_saved_parameter("cool_vco_resonance", client=client) == 4.05
    with pytest.raises(KeyError, match="cool_vco_resonance"):
        get_saved_parameter("cool_vco_resonanse", client=client)
```

```python
# tests/test_experiment_e2e.py — add to EXPERIMENT:
#   class ReadsSaved(Experiment):
#       def run_point(self):
#           from pytweezer.experiment.client import get_saved_parameter
#           self.record("x_centre", get_saved_parameter("x_centre"))
def test_a_task_queued_before_a_change_reads_the_new_value(harness):
    set_saved_parameters({"x_centre": 1.0}, "r", create=True, client=harness.client)
    blocker = submit("labexp.steps:Steps", Scan(repetitions=3), delay=0.5, client=harness.client)
    reader = submit("labexp.steps:ReadsSaved", client=harness.client)
    set_saved_parameters({"x_centre": 2.0}, "r", client=harness.client)
    wait(reader, client=harness.client)
    assert measurement_of(harness, reader).results["x_centre"][0] == 2.0
```

Match `submit`'s string form and the existing harness helpers for reading a finished measurement (see `test_tasks_run_in_order_and_write_measurements`).

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_manager.py tests/test_experiment_e2e.py -q -k "misspelt or queued_before"`
Expected: FAIL with `ImportError` on the new client functions.

- [ ] **Step 3: Implement the client functions, `default_endpoint`, the worker line, the fixture and `FakeSavedParameters`**

`get_saved_parameter` fetches all values and raises `KeyError` naming the key and `difflib.get_close_matches` suggestions when missing.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/client.py pytweezer/experiment/worker.py tests/conftest.py tests/helpers.py tests/test_experiment_manager.py tests/test_experiment_e2e.py
git commit -m "Client functions for saved parameters, reachable from inside a task"
```

---

### Task 4: The import tool

**Files:**
- Create: `pytweezer/experiment/saved_parameters_import.py`
- Modify: `pyproject.toml` (`[project.scripts]`: `pytweezer-saved-parameters = "pytweezer.experiment.saved_parameters_import:main"`)
- Test: `tests/test_saved_parameters.py`

**Interfaces:**
- Consumes: Task 3's `get_saved_parameters`, `set_saved_parameters`.
- Produces: `IMPORT_GROUPS: dict[str, str]` (the Global Constraints mapping); `default_unit(key: str) -> str`; `import_parameters_file(path: Path | str, *, replace: bool = False, client=None) -> int` (number of values imported); `main(argv: list[str] | None = None) -> int` with subcommand `import <file>` and flag `--replace`. A category not in `IMPORT_GROUPS` keeps its own name as the group.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_import_copies_every_value_with_groups_and_units(saved_parameters, tmp_path):
    path = write_parameters(tmp_path, {
        "motmaster_parameters": {"tImgCoolVCO": 3.0, "tMOTShimXccValue": 0.017},
        "misc_parameters": {"cool_vco_resonance": 4.79},
        "emccd_parameters": {"em_gain": 600},
        "image_analysis": {"x_centre": 131.49, "orcacam_mm_per_px": 0.0079},
    })
    assert import_parameters_file(path) == 6
    assert saved_parameters.values == {"tImgCoolVCO": 3.0, "tMOTShimXccValue": 0.017,
        "cool_vco_resonance": 4.79, "em_gain": 600, "x_centre": 131.49, "orcacam_mm_per_px": 0.0079}
    assert {w["group"] for w in saved_parameters.writes} == {
        "MOTMaster settings", "Resonances", "EMCCD", "Image analysis"}
    assert all(w["reason"] == "imported from parameters.json" for w in saved_parameters.writes)


@pytest.mark.parametrize("key, unit", [("tImgCoolVCO", "V"), ("tMOTLoadCoolVVA", "V"),
    ("cool_vco_resonance", "V"), ("x_centre", "px"), ("thorcam_mm_per_px", "mm/px"),
    ("tMOTShimXccValue", ""), ("em_gain", "")])
def test_obvious_units_are_filled_in(key, unit):
    assert default_unit(key) == unit


def test_the_import_refuses_a_store_that_has_values_unless_replacing(saved_parameters, tmp_path): ...
    # values pre-populated; import raises SavedParameterError mentioning --replace and
    # writes nothing; with replace=True the file's values win.


def test_a_list_value_stops_the_import_before_anything_is_written(saved_parameters, tmp_path): ...
    # {"misc_parameters": {"a": 1.0, "site_positions": [[1, 2]]}} → SavedParameterError
    # naming site_positions; saved_parameters.writes == [].
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_saved_parameters.py -q -k "import or unit"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — validate the whole file (every value a JSON scalar) before the first write; one `set_saved_parameters(..., create=True, group=..., unit=...)` call per (group, unit) pair.

- [ ] **Step 4: Run the tests and `uv sync`** (to install the console script), then `uv run pytweezer-saved-parameters --help`
Expected: tests pass; help lists `import`.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/saved_parameters_import.py pyproject.toml tests/test_saved_parameters.py
git commit -m "Add pytweezer-saved-parameters import, to move parameters.json into the store"
```

---

### Task 5: Move every reader over, generic experiment tests

**Files:**
- Modify: `pytweezer/experiments/tweezer_baseline.py`, `pytweezer/analysis/analysis.py:12,46-47,90-105,117`, `pytweezer/helper_scripts/optimisation.py`, `notebooks/tweezer_baseline.ipynb`, `tests/test_optimisation.py`, `.claude/skills/add-experiment/SKILL.md`
- Create: `tests/test_experiments.py`
- Delete: `pytweezer/experiment/parameters.json`, `pytweezer/experiment/experiment_parameter_manager.py`, `tests/test_tweezer_baseline.py`

**Interfaces:**
- Consumes: Task 3's client functions and `saved_parameters` fixture.

- [ ] **Step 1: Write the generic tests in `tests/test_experiments.py`**

```python
MODULES = sorted(f"pytweezer.experiments.{p.stem}" for p in EXPERIMENTS_DIR.glob("*.py")
                 if p.stem != "__init__")

@pytest.fixture(autouse=True)
def _no_manager(saved_parameters):
    """A module that reads a saved value at import must not need a running manager."""

@pytest.mark.parametrize("module", MODULES)
def test_imports_the_way_the_manager_imports_it(module):
    description = describe(module)
    assert description["error"] is None, description["error"]
    assert description["classes"]

@pytest.mark.parametrize("module", MODULES)
def test_every_declared_device_is_in_config(module):
    for experiment_cls in runnable_experiments(importlib.import_module(module)):
        experiment_cls.reserved_devices()          # raises ReservationError otherwise
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_experiments.py -q`
Expected: pass against today's code (they guard the move, not new behaviour). Investigate any failure before continuing.

- [ ] **Step 3: Move the readers**
  - `TweezerBaseline.prepare`: `self._saved_parameters = {parameter: get_saved_parameter(key) for parameter, key in SAVED_MOTMASTER_PARAMETERS.items()}`; remove `parameters_file` and its docstring sentence about `parameters.json` (say "saved parameters").
  - `analysis.py`: drop the module-level `exp_params`/`cool_vco_resonance`; the function at line 117 calls `get_saved_parameter("cool_vco_resonance")`; `set_array_centre` calls `set_saved_parameters({"x_centre": x0, "y_centre": y0}, "set_array_centre")`; `index_grid_positions` reads both with `get_saved_parameter`.
  - `optimisation.py`: remove `ExpParams`; `ExpParams.get_parameter(k)` → `get_saved_parameter(k)`; each routine collects its results and ends with one `set_saved_parameters({...}, "optimisation.py")` in place of `set_parameter` + `save_parameters`.
  - `notebooks/tweezer_baseline.ipynb`: the `conversion_factor` cell uses `set_saved_parameters`; remove the `ExpParameterManager` import.
  - `tests/test_optimisation.py`: replace the temporary-file setup with `saved_parameters.values.update(...)` and read results from `saved_parameters.values`; assertions unchanged.
  - Delete the three files listed above.
  - `.claude/skills/add-experiment/SKILL.md`: a short "Saved parameters" section: read with `get_saved_parameter`, as an argument default (`Number(get_saved_parameter("x_centre"))`) or in `prepare()`; experiments are expected only to read; tests use the `saved_parameters` fixture.

- [ ] **Step 4: Run the full suite and confirm nothing still names the old manager**

Run: `uv run pytest tests/ -q` and `git grep -n "ExpParameterManager\|parameters_file" -- ':!docs' ':!notebooks/optimisation-scans-testing*'`
Expected: tests pass; grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add -A pytweezer tests notebooks/tweezer_baseline.ipynb .claude/skills/add-experiment/SKILL.md
git commit -m "Read and write saved parameters through the Experiment Manager; drop parameters.json"
```

---

### Task 6: The Saved parameters tab

**Files:**
- Create: `pytweezer/GUI/saved_parameters.py`
- Modify: `bin/gui.py` (build the panel with the shared `feed`; tab after `Results`), `pytweezer/GUI/theme.py` only if a new `objectName` rule is needed
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: `ExperimentManagerClient.saved_parameters`, `.set_saved_parameters`, `.saved_parameter_history` (Task 3); `ExperimentFeed.queue_changed(dict)`.
- Produces: `class SavedParametersPanel(QWidget)`: `__init__(self, client=None, feed=None, parent=None)` (client defaults to `ExperimentManagerClient(timeout_ms=1000, retries=0)`, as `ExperimentsPanel` does); `select(key: str) -> None`; `apply() -> None` (the Apply button's slot).

Layout and wording are fixed by the spec's section 4 and the mockup; follow the `pytweezer-gui-design` skill (`Region("well"/"sheet")`, `PrimaryButton`, `CompactDoubleSpinBox` is not suitable here because the type varies: use a `QLineEdit` parsed by the key's type). Refetch only when `saved_parameters_version` changes and the panel is visible, and in `showEvent`.

- [ ] **Step 1: Write the failing tests** (fake client recording calls; fake feed with a `queue_changed` signal, as the existing tests in this file do)

```python
def test_saved_parameters_tab_builds_offscreen(qapp): ...
    # SavedParametersPanel(client=FakeSavedClient(VALUES), feed=Feed()) constructs and
    # lists every key.

def test_apply_sends_the_value_with_its_reason(qapp):
    panel = make_saved_panel()
    panel.select("tImgCoolVCO")
    panel.value_edit.setText("3.1"); panel.why_edit.setText("lifetime scan")
    panel.apply()
    assert panel.client.calls[-1] == ("set_saved_parameters", {"tImgCoolVCO": 3.1},
                                      "set by hand: lifetime scan")

def test_a_value_of_the_wrong_type_is_refused_before_sending(qapp):
    # "4,05" for a float key → nothing sent; the sheet's status label has state "error".

def test_a_live_update_keeps_the_selection_and_an_unsent_edit(qapp):
    # select, type "3.1", emit queue_changed with a bumped saved_parameters_version →
    # same key selected, value_edit still "3.1".

def test_an_unreachable_manager_is_said_not_shown_as_current(qapp):
    # client.saved_parameters raises ManagerUnavailable → the well's hint reads
    # "Saved parameters: the Experiment Manager is not responding".
```

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q -k saved`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the panel and register the tab**

- [ ] **Step 4: Run the tests, then screenshot and compare with the mockup**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q`, then
`uv run python .claude/skills/pytweezer-gui-design/scripts/shoot.py <fakes>:panel /tmp/saved.png --size 1440x900 --path <scratch>` with a fake client holding realistic values (several groups, a long reason, one changed today), and Read the PNG. Fix what the design skill's checklist finds.
Expected: tests pass; the screenshot matches the "Saved parameters tab" artboard's structure.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/GUI/saved_parameters.py bin/gui.py tests/test_experiment_gui.py
git commit -m "Add the Saved parameters tab"
```

---

### Task 7: Lab-book check and migration notes

**Files:** none in this repo.

- [ ] **Step 1:** Ask the user to attach `ColdMatter/CaFMOT-LabBook` (the session's `add_repo`), read its `CLAUDE.md`, and `git grep -n "ExpParameterManager\|parameters.json"` in it. List every hit by notebook.
- [ ] **Step 2:** Draft the PR description: the change, the server-PC migration steps from spec section 5 verbatim, and the lab-book list. Show it to the user; open the PR only when they ask.
