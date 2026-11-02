# Saved Parameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A store of saved parameters owned by the Experiment Manager, with a change log, client functions, an import from `parameters.json`, a Saved parameters tab, and a Defaults button in the Experiments form that picks up changed saved values.

**Architecture:** A plain-data `SavedParameterBook` (rules) persisted by a `ModelStore` subclass plus an append-only JSONL log, both in the manager's data directory, following `recipes.py`. The manager exposes three saved-parameter commands and `describe_again`; `client.py` wraps them. GUIs follow the manager's published snapshot.

**Tech Stack:** Python 3, pydantic, ZMQ REQ/REP (`ExperimentManagerClient`), sipyco sync (`ExperimentFeed`), PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-saved-parameters-design.md` (read it first).

## Global Constraints

- The Experiment Manager on the server PC runs every task. Client code, including code inside a task, reaches it at the address CONFIG gives; add no endpoint plumbing.
- Tests follow the design (CLAUDE.md): change existing tests and test helpers to fit; add nothing to production code just so a test can reach it. Tests reach a manager through the existing `make_manager` fixture and `InProcess` helper (`tests/test_experiment_manager.py`), or the e2e `Harness`.
- Out of scope: `TweezerBaseline`, `analysis.py`, `optimisation.py`, notebooks, `parameters.json` and `ExpParameterManager` stay as they are.
- UK English; Google-style docstrings describing current behaviour only; comment only what is surprising; `get_logger(name)` for logging.
- `uv run ruff check` and `uv run ruff format` on touched files; `uv run pytest tests/ -q` passes after every task.
- Files: `saved_parameters.json` and `saved_parameters_log.jsonl` in `ExperimentManager.root`, next to `recipes.json`.
- `Value = float | int | bool | str`. A key keeps its type: int stays int, float accepts int (stored as float), bool is not an int, str only str. Non-finite floats are refused.
- Exact strings: reason prefix `set by hand: `; import reason `imported from parameters.json`; snapshot key `saved_parameters_version`; tab title `Saved parameters`, directly after `Results` in `bin/gui.py`; unreachable text `Saved parameters: the Experiment Manager is not responding`; Apply note `Read from now on; press Defaults in Experiments to use it in an open form`.
- Commit after each task, plain-sentence messages, ending with the session's attribution lines.

## Review Focus

1. A non-finite float (`nan`, `inf`) from a failed fit is refused naming the key, not stored (JSON cannot hold it). Test in Task 1.
2. A misspelt key in `get_saved_parameter` raises `KeyError` naming the key and the closest existing key. Test in Task 3.
3. Pressing Defaults while the manager is unreachable leaves the form as it was and says so, rather than silently applying stale defaults. Test in Task 4.
4. A value typed in the tab that does not parse as the key's type (`"4,05"` for a float, `"4.5"` for an int, `"yes"` for a bool) is refused in the sheet and nothing is sent. Test in Task 5.
5. A live update arriving while the user has typed but not applied a value keeps the typed value and the selection. Test in Task 5.

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
  - `class SavedParameterState(BaseModel)`: `schema_version: int = 1`, `parameters: list[SavedParameter] = []`
  - `class SavedParameterBook`: `__init__(state: SavedParameterState | None = None)`; `state` attribute; `get(key: str) -> SavedParameter`; `all() -> list[SavedParameter]` (sorted by group then key); `set(values: dict[str, Value], *, reason: str, who: str, create: bool = False, group: str = "", unit: str = "", rid: int | None = None) -> list[SavedParameterChange]`
  - `class SavedParameterStore(ModelStore)` with `model = SavedParameterState`
  - `class SavedParameterLog`: `__init__(path: Path | str)`; `append(changes: list[SavedParameterChange]) -> None`; `read(key: str | None = None, *, limit: int | None = None) -> list[SavedParameterChange]` (newest first)

- [ ] **Step 1: Write the failing tests**

```python
def book_with(**values):
    book = SavedParameterBook()
    book.set(values, reason="r", who="me", create=True)
    return book


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
def test_a_wrong_type_or_non_finite_value_is_refused(existing, new):
    book = book_with(key=existing)
    with pytest.raises(SavedParameterError, match="key"):
        book.set({"key": new}, reason="r", who="me")
    assert book.get("key").value == existing


def test_an_unknown_key_is_refused_unless_created_all_or_nothing():
    book = book_with(key=1.0)
    with pytest.raises(SavedParameterError, match="kye"):
        book.set({"key": 2.0, "kye": 3.0}, reason="r", who="me")
    assert book.get("key").value == 1.0
    book.set({"kye": 3.0}, reason="r", who="me", create=True, group="G", unit="V")
    assert (book.get("kye").group, book.get("kye").unit) == ("G", "V")


def test_every_change_is_logged_but_an_unchanged_value_is_not(tmp_path):
    book, log = book_with(key=1.0), SavedParameterLog(tmp_path / "log.jsonl")
    log.append(book.set({"key": 1.0}, reason="same", who="me"))
    log.append(book.set({"key": 2.0}, reason="by hand", who="me", rid=7))
    [change] = log.read("key")
    assert (change.old, change.new, change.reason, change.who, change.rid) == (
        1.0, 2.0, "by hand", "me", 7)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_saved_parameters.py -q`
Expected: FAIL with `ModuleNotFoundError: pytweezer.experiment.saved_parameters`

- [ ] **Step 3: Implement `pytweezer/experiment/saved_parameters.py`**

Follow `recipes.py` (module docstring; `now()` from `pytweezer.experiment.queue`). `get` raises `SavedParameterError` naming the key and up to three `difflib.get_close_matches`. `set` strips keys and validates every value before changing any; skips values equal to the current one; `group`/`unit` apply only to keys it creates. `SavedParameterLog.append` writes one `model_dump_json()` line per change; `read` filters, reverses, truncates.

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
- Modify: `pytweezer/servers/experiment_manager.py` (constructor next to the recipe store; `_snapshot`; the `#:` layout comment; `_cmd_*` methods after the recipe ones; a `_saved_parameters_changed` next to `_recipes_changed`)
- Test: `tests/test_experiment_manager.py`

**Interfaces:**
- Consumes: Task 1's `SavedParameterBook`, `SavedParameterStore`, `SavedParameterLog`.
- Produces (replies are `model_dump(mode="json")`):
  - `saved_parameters` → `{"parameters": [...], "version": int}`
  - `set_saved_parameters` (`values`, `reason`, `who`, optional `create`, `group`, `unit`, `rid`) → `{"changes": [...]}`
  - `saved_parameter_history` (optional `key`, `limit`) → `{"entries": [...]}`, newest first
  - snapshot key `saved_parameters_version`

- [ ] **Step 1: Write the failing test**

```python
def test_a_written_value_is_read_back_after_a_restart_and_bumps_the_version(make_manager):
    manager = make_manager()
    version = manager._snapshot()["saved_parameters_version"]
    reply = manager.handle({"command": "set_saved_parameters", "values": {"offset": 131.5},
                            "reason": "r", "who": "me", "create": True})
    assert reply["ok"], reply
    [parameter] = manager.handle({"command": "saved_parameters"})["parameters"]
    assert (parameter["key"], parameter["value"]) == ("offset", 131.5)
    assert manager._snapshot()["saved_parameters_version"] == version + 1
    restarted = make_manager()
    assert restarted.handle({"command": "saved_parameters"})["parameters"] == [parameter]
    [entry] = restarted.handle({"command": "saved_parameter_history", "key": "offset"})["entries"]
    assert entry["new"] == 131.5
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_experiment_manager.py -q -k written_value`
Expected: FAIL with `KeyError: 'saved_parameters_version'`.

- [ ] **Step 3: Implement**

`_saved_parameters_changed(changes)` saves the store, appends the log, bumps the version and sets `_dirty`; it does nothing for an empty list. `SavedParameterError` reaches the caller as a refusal through `handle`'s existing `ValueError` branch.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_experiment_manager.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/servers/experiment_manager.py tests/test_experiment_manager.py
git commit -m "Experiment Manager: serve saved parameters and their history"
```

---

### Task 3: Client functions and the import

**Files:**
- Modify: `pytweezer/experiment/client.py`, `pytweezer/experiment/saved_parameters.py`
- Test: `tests/test_experiment_manager.py`, `tests/test_saved_parameters.py`

**Interfaces:**
- Consumes: Task 2's commands; Task 1's models.
- Produces:
  - `ExperimentManagerClient` methods: `saved_parameters() -> list[SavedParameter]`; `set_saved_parameters(values, reason, who, *, create=False, group="", unit="", rid=None) -> list[SavedParameterChange]`; `saved_parameter_history(key=None, *, limit=None) -> list[SavedParameterChange]`. Add `"saved_parameters"` and `"saved_parameter_history"` to `_READ_ONLY_COMMANDS`.
  - Module functions in `client.py`, each ending `*, client: ExperimentManagerClient | None = None` as `recipes()` does, filling `who` with `submitter_name()`:
    `get_saved_parameter(key: str, /) -> Value` (raises `KeyError` naming the key and up to three close matches);
    `get_saved_parameters() -> dict[str, Value]`;
    `set_saved_parameters(values: dict[str, Value], reason: str, /, *, create=False, group="", unit="", rid=None) -> list[SavedParameterChange]`;
    `saved_parameter_history(key: str | None = None, /, *, limit=None) -> list[SavedParameterChange]`
  - `import_parameters_json(path: Path | str, *, client=None) -> int` in `saved_parameters.py`, marked `# FIXME: remove once parameters.json has been moved into the store`; one `set_saved_parameters(values, "imported from parameters.json", create=True, group=<category>)` per top-level category; returns the number of values in the file. Import `client` functions inside it, so `client.py` may import `saved_parameters.py` at module level.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiment_manager.py
def in_process_client(manager):
    client = ExperimentManagerClient(endpoint="tcp://127.0.0.1:1")
    client._req = InProcess(manager)
    return client


def test_a_misspelt_key_names_the_closest_one(make_manager):
    client = in_process_client(make_manager())
    set_saved_parameters({"cool_resonance": 4.05}, "r", create=True, client=client)
    assert get_saved_parameter("cool_resonance", client=client) == 4.05
    with pytest.raises(KeyError, match="cool_resonance"):
        get_saved_parameter("cool_resonanse", client=client)
```

```python
# tests/test_saved_parameters.py
def test_the_import_copies_every_value_with_its_category_as_group(make_manager, tmp_path):
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"first": {"a": 3.0, "b": 600}, "second": {"c": "x"}}))
    client = in_process_client(make_manager())
    assert import_parameters_json(path, client=client) == 3
    found = {p.key: (p.value, p.group) for p in client.saved_parameters()}
    assert found == {"a": (3.0, "first"), "b": (600, "first"), "c": ("x", "second")}
    assert import_parameters_json(path, client=client) == 3
    assert len(saved_parameter_history(client=client)) == 3
```

Move `make_manager`, `StaticCatalogue`, `InProcess` and `in_process_client` from `tests/test_experiment_manager.py` into `tests/conftest.py` / `tests/helpers.py` so both test files share them.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_experiment_manager.py tests/test_saved_parameters.py -q -k "misspelt or import"`
Expected: FAIL with `ImportError` on the new functions.

- [ ] **Step 3: Implement the client methods, the module functions and `import_parameters_json`**

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/experiment/client.py pytweezer/experiment/saved_parameters.py tests/
git commit -m "Client functions for saved parameters, and an import from parameters.json"
```

---

### Task 4: Defaults picks up changed saved values

**Files:**
- Modify: `pytweezer/experiment/catalogue.py`, `pytweezer/servers/experiment_manager.py`, `pytweezer/experiment/client.py`, `pytweezer/GUI/experiments/arg_editor.py`, `pytweezer/GUI/experiments/panel.py`, `.claude/skills/add-experiment/SKILL.md`
- Modify (test helpers): `StaticCatalogue` gains `describe_again`; `FakeClient` in `tests/test_experiment_gui.py` answers `describe_again`
- Test: `tests/test_experiment_e2e.py`, `tests/test_experiment_gui.py`

**Interfaces:**
- Produces:
  - `Catalogue.describe_again(module: str) -> None`: forgets the module's stamp so the next `refresh()` describes it again; raises `KeyError` for a module `modules()` does not list.
  - Manager command `describe_again` (`module`): calls it, then `self._catalogue_changed(self.catalogue.refresh())`.
  - `ExperimentManagerClient.describe_again(module: str) -> None`.
  - `ArgumentEditor.defaults_requested = pyqtSignal(str, str)` (module, class name), emitted by the Defaults button in place of calling `reset_to_defaults` directly.
  - `ExperimentsPanel` handles it: sends `describe_again`, remembers the key, and when the next catalogue refresh for a newer version arrives, calls `editor.set_experiment(schema)` then `editor.reset_to_defaults()` for that key. If the call fails, the form is left as it is and the status line says so (the existing `_call` error path).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiment_e2e.py — a module whose default comes from outside its file
# labexp/fromfile.py:
#   from pathlib import Path
#   from pytweezer.experiment import Experiment, Number
#   class FromFile(Experiment):
#       level = Number(float((Path(__file__).parent / "level.txt").read_text()))
#       def run_point(self): pass
def test_describe_again_picks_up_a_changed_default(harness):
    harness.start()
    write_level(harness, "1.5")
    assert until(lambda: default_of(harness, "FromFile", "level") == 1.5)
    write_level(harness, "2.5")
    harness.client.describe_again("labexp.fromfile")
    assert until(lambda: default_of(harness, "FromFile", "level") == 2.5)
```

`write_level` and `default_of` are small helpers in the test file (`default_of` reads the argument's default from `harness.client.catalogue()`).

```python
# tests/test_experiment_gui.py
def test_defaults_asks_for_a_fresh_description_and_applies_it(qapp):
    # select RabiDemo; click editor.defaults_button → commands(client, "describe_again")
    # == [{"module": DEMO}]; the fake client's catalogue now returns a schema whose
    # "atoms" default is 999; emit queue_changed with a higher catalogue_version →
    # the atoms field shows 999.

def test_defaults_with_the_manager_unreachable_leaves_the_form_alone(qapp):
    # client.errors["describe_again"] = "not responding"; type 77 into atoms; click
    # Defaults → atoms still 77 and the status line shows the error.
```

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_e2e.py tests/test_experiment_gui.py -q -k "describe_again or defaults"`
Expected: FAIL (`AttributeError: describe_again`; no `describe_again` call recorded).

- [ ] **Step 3: Implement the catalogue method, command, client method, editor signal and panel handling**

- [ ] **Step 4: Add a short "Saved parameters" section to `.claude/skills/add-experiment/SKILL.md`**

Read with `get_saved_parameter`: in `prepare()`/`run_point()` for the value at run time, or passed to an argument as its default (read when the module is imported; the form's Defaults re-reads it). Importing such a module contacts the manager.

- [ ] **Step 5: Run the full suite**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pytweezer/ tests/ .claude/skills/add-experiment/SKILL.md
git commit -m "Experiments form: Defaults has the manager describe the experiment again"
```

---

### Task 5: The Saved parameters tab

**Files:**
- Create: `pytweezer/GUI/saved_parameters.py`
- Modify: `bin/gui.py` (build with the shared `feed`; tab after `Results`)
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: Task 3's `ExperimentManagerClient` methods; `ExperimentFeed.queue_changed(dict)`.
- Produces: `class SavedParametersPanel(QWidget)`: `__init__(self, client=None, feed=None, parent=None)` (client defaults to `ExperimentManagerClient(timeout_ms=1000, retries=0)`, as `ExperimentsPanel` does); attributes `value_edit`, `why_edit`; `select(key: str) -> None`; `apply() -> None`.

Layout and wording: spec section 4 and the mockup; follow the `pytweezer-gui-design` skill (`Region("well"/"sheet")`, `PrimaryButton`). The value field is a `QLineEdit` parsed by the key's current type. Refetch when `saved_parameters_version` changes while visible, and in `showEvent`.

- [ ] **Step 1: Write the failing tests** (a fake client holding a few `SavedParameter`s and recording calls; a fake feed with `queue_changed`, as the existing tests in this file do)

```python
def test_apply_sends_the_value_with_its_reason(qapp):
    panel = make_saved_panel()
    panel.select("level")
    panel.value_edit.setText("3.1"); panel.why_edit.setText("lifetime scan")
    panel.apply()
    assert panel.client.calls[-1] == ("set_saved_parameters", {"level": 3.1},
                                      "set by hand: lifetime scan")

@pytest.mark.parametrize("key, text", [("level", "4,05"), ("count", "4.5"), ("enabled", "yes")])
def test_a_value_of_the_wrong_type_is_refused_before_sending(qapp, key, text):
    # select key, type text, apply → no set_saved_parameters call; status label state "error"

def test_a_live_update_keeps_the_selection_and_an_unsent_edit(qapp):
    # select "level", type "3.1", emit queue_changed with a bumped saved_parameters_version
    # → same key selected, value_edit still "3.1"

def test_an_unreachable_manager_is_said_not_shown_as_current(qapp):
    # client.saved_parameters raises ManagerUnavailable → the well's hint reads
    # "Saved parameters: the Experiment Manager is not responding"
```

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_experiment_gui.py -q -k "apply_sends or wrong_type or live_update_keeps or unreachable_manager"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the panel and register the tab**

- [ ] **Step 4: Run the tests, then screenshot**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -q`, then screenshot the panel with a fake client holding realistic values (several groups, a long reason, one changed today) using the `pytweezer-gui-design` skill's `scripts/shoot.py`, read the PNG and fix what the skill's checklist finds.
Expected: tests pass; the screenshot matches the mockup's structure.

- [ ] **Step 5: Commit**

```bash
git add pytweezer/GUI/saved_parameters.py bin/gui.py tests/test_experiment_gui.py
git commit -m "Add the Saved parameters tab"
```
