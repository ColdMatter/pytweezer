# Experiment Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Named, shared, replayable run settings ("recipes") for experiments, kept by the Experiment Manager and usable from the Experiments tab and from notebooks.

**Architecture:** A plain-data `RecipeBook` (rules) and `RecipeStore` (JSON on disk) live in a new `pytweezer/experiment/recipes.py`. The Experiment Manager owns one, serves four new REQ commands, and publishes a `recipes_version` in its snapshot so GUIs refetch. The notebook client gets matching functions; the Experiments tab shows recipes under their experiment in the catalogue tree, loads them into the form, saves the form as a recipe, and submits or deletes them from a context menu.

**Tech Stack:** Python ≥3.11, pydantic v2, pyzmq REQ/REP, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-10-09-experiment-recipes-design.md`

## Global Constraints

- Python `^3.11.4`: no PEP 695 generics (`class Foo[T]`), no 3.12-only syntax.
- UK English in code, docstrings, UI text and docs.
- Google-style docstrings for new code; docstrings describe current behaviour only.
- Comment only what is surprising; prefer names over comments.
- Logging via `get_logger(name)` from `pytweezer/logging_utils.py`, never `logging.getLogger`.
- Lint/format with ruff (`poetry run ruff check --fix <files>` and `poetry run ruff format <files>`); the pre-commit hook also runs it.
- Run tests with `poetry run pytest tests/<file> -q`; the whole suite with `poetry run pytest tests/ -q` must stay green.
- Commit and push after each task (`git push`; the branch `experiment-recipes` already tracks `origin`). Stage only the files the task names — never `.vscode/settings.json`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- GUI wording/styling follows the `pytweezer-gui-design` skill; load it before Tasks 5–7.

## Review Focus

1. **Manager just restarted, catalogue still being read** — `submit_recipe` must say to try again shortly, not that the experiment is missing. Test in Task 3.
2. **Recipe saved from a form with "Start no earlier than" ticked** — the start time must not be stored, so a replay runs now. Test in Task 1.
3. **Notebook override with a typo** (`submit_recipe(Demo, "x", atom=3)`) — refused naming `atom` as unknown, not described as a stale recipe and never silently added. Tests in Tasks 3 and 4.
4. **Names differing only by surrounding whitespace** (`"MOT check "` vs `"MOT check"`) — the same recipe; an all-space name is refused. Test in Task 1.
5. **Recipe list refreshed (another PC saved one) while the user is editing a loaded recipe** — selection kept and the form not reloaded over their edits. Test in Task 5.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `pytweezer/experiment/queue.py` | modify | `QueueStore` becomes a subclass of a new generic `ModelStore` |
| `pytweezer/experiment/recipes.py` | create | `Recipe`, `RecipeState`, `RecipeBook`, `RecipeStore`, `RecipeError`, `unknown_arguments` |
| `pytweezer/servers/experiment_manager.py` | modify | recipe commands, persistence, `recipes_version` |
| `pytweezer/experiment/client.py` | modify | client methods and notebook functions |
| `pytweezer/GUI/experiments/catalogue_view.py` | modify | recipes in the tree, filter, context menu |
| `pytweezer/GUI/experiments/arg_editor.py` | modify | "Save as recipe…" button, recipe name in the header |
| `pytweezer/GUI/experiments/panel.py` | modify | wiring: fetch, select, save, submit now, delete |
| `.claude/skills/add-experiment/SKILL.md` | modify | "Recipes" section |
| `tests/test_experiment_recipes.py` | create | Tasks 1–2 |
| `tests/test_experiment_manager.py` | modify | Tasks 3–4 |
| `tests/test_experiment_gui.py` | modify | Tasks 5–7 |

---

### Task 1: Recipe model, book and store

**Files:**
- Modify: `pytweezer/experiment/queue.py` (the `QueueStore` class at the end of the file)
- Create: `pytweezer/experiment/recipes.py`
- Test: `tests/test_experiment_recipes.py` (create)

**Interfaces:**
- Consumes: `TaskRequest` from `pytweezer.experiment.task`; `now()` from `pytweezer.experiment.queue`.
- Produces:
  - `queue.ModelStore` — `__init__(path: Path | str)`, `load() -> BaseModel`, `save(state: BaseModel) -> None`; subclasses set class attribute `model`.
  - `recipes.RecipeError(ValueError)`
  - `recipes.Recipe(TaskRequest)` — extra fields `name: str` (stripped, non-empty), `saved_at: datetime`; property `recipe_key -> tuple[str, str, str]` = `(experiment, class_name, name)`.
  - `recipes.RecipeState(BaseModel)` — `schema_version: int = 1`, `recipes: list[Recipe]`.
  - `recipes.RecipeBook(state: RecipeState | None = None)` — `.state`, `get(experiment, class_name, name) -> Recipe`, `save(recipe, overwrite=False) -> Recipe`, `delete(experiment, class_name, name) -> None`, `find(experiment=None, class_name=None) -> list[Recipe]`.
  - `recipes.RecipeStore(ModelStore)` with `model = RecipeState`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_experiment_recipes.py`:

```python
"""Recipes as plain data: the book's rules, the store, and replaying one."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pytweezer.experiment.queue import now
from pytweezer.experiment.recipes import (
    Recipe,
    RecipeBook,
    RecipeError,
    RecipeState,
    RecipeStore,
)
from pytweezer.experiment.scan import LinearAxis, Scan

DEMO = "pytweezer.experiments.demo"


def recipe(name="check", **kwargs):
    return Recipe(experiment=DEMO, class_name="RabiDemo", name=name, **kwargs)


def test_names_are_stripped_and_required():
    assert recipe(name="  MOT check ").name == "MOT check"
    with pytest.raises(ValidationError, match="needs a name"):
        recipe(name="   ")


def test_saving_clears_the_start_time_and_stamps_the_save():
    book = RecipeBook()
    saved = book.save(
        recipe(
            due_time=datetime(2030, 1, 1).astimezone(),
            saved_at=datetime(2020, 1, 1).astimezone(),
        )
    )
    assert saved.due_time is None
    assert (now() - saved.saved_at).total_seconds() < 60
    assert book.get(DEMO, "RabiDemo", "check") == saved


def test_a_name_is_taken_until_overwritten():
    book = RecipeBook()
    book.save(recipe(label="old"))
    with pytest.raises(RecipeError, match="already exists"):
        book.save(recipe(name=" check", label="new"))
    book.save(recipe(label="new"), overwrite=True)
    [only] = book.find()
    assert only.label == "new"


def test_the_same_name_may_be_used_for_another_experiment():
    book = RecipeBook()
    book.save(recipe())
    book.save(Recipe(experiment="other.module", class_name="Other", name="check"))
    assert len(book.find()) == 2


def test_find_filters_and_sorts():
    book = RecipeBook()
    book.save(recipe(name="b"))
    book.save(recipe(name="a"))
    book.save(Recipe(experiment="a.module", class_name="Z", name="c"))
    assert [r.name for r in book.find()] == ["c", "a", "b"]
    assert [r.name for r in book.find(DEMO, "RabiDemo")] == ["a", "b"]
    assert [r.name for r in book.find(DEMO)] == ["a", "b"]


def test_missing_recipes_are_reported_by_name():
    book = RecipeBook()
    with pytest.raises(RecipeError, match="no recipe 'nope' for .*RabiDemo"):
        book.get(DEMO, "RabiDemo", "nope")
    with pytest.raises(RecipeError, match="no recipe"):
        book.delete(DEMO, "RabiDemo", "nope")
    book.save(recipe())
    book.delete(DEMO, "RabiDemo", " check ")
    assert book.find() == []


def test_store_round_trip_and_unreadable_file(tmp_path):
    path = tmp_path / "recipes.json"
    store = RecipeStore(path)
    assert store.load() == RecipeState()
    book = RecipeBook(store.load())
    book.save(
        recipe(scan=Scan(axes=[LinearAxis(argument="atoms", start=1, stop=3, n=3)]))
    )
    store.save(book.state)
    assert RecipeStore(path).load() == book.state

    path.write_text("{not json")
    assert store.load() == RecipeState()
    assert not path.exists()
    assert len(list(tmp_path.glob("recipes.unreadable-*.json"))) == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_recipes.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'pytweezer.experiment.recipes'`.

- [ ] **Step 3: Generalise `QueueStore` into `ModelStore`**

In `pytweezer/experiment/queue.py`, replace the whole `QueueStore` class with:

```python
class ModelStore:
    """Persists one pydantic model as JSON, replacing the file atomically.

    Subclasses set :attr:`model`. An unreadable file is moved aside and an
    empty model returned, so a corrupt file never stops the manager starting.
    """

    model: type[BaseModel]

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> BaseModel:
        try:
            return self.model.model_validate_json(self.path.read_text())
        except FileNotFoundError:
            return self.model()
        except (ValidationError, json.JSONDecodeError, OSError):
            aside = self.path.with_suffix(f".unreadable-{int(time.time())}.json")
            logger.exception("%s is unreadable; moved to %s", self.path, aside)
            self.path.replace(aside)
            return self.model()

    def save(self, state: BaseModel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=1))
        for attempt in range(20):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                # Windows refuses to replace a file another process has open
                # (an editor, a backup tool); it is usually released quickly.
                if attempt == 19:
                    raise
                time.sleep(0.05)


class QueueStore(ModelStore):
    """Persists a :class:`QueueState`."""

    model = QueueState
```

(The bodies are the existing `QueueStore` code with `QueueState` replaced by `self.model` and the log message made generic.)

- [ ] **Step 4: Create `pytweezer/experiment/recipes.py`**

```python
"""Recipes: named run settings for one experiment, shared through the Experiment Manager.

A recipe is a :class:`~pytweezer.experiment.task.TaskRequest` with a name. It
is replayed by loading it into the Experiments form, or queued directly from
:meth:`Recipe.replay`.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from pytweezer.experiment.queue import ModelStore, now
from pytweezer.experiment.task import TaskRequest


class RecipeError(ValueError):
    """A recipe request that can't be carried out."""


class Recipe(TaskRequest):
    """Saved run settings; ``submitter`` is whoever saved it."""

    name: str
    saved_at: datetime = Field(default_factory=now)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("a recipe needs a name")
        return name

    @property
    def recipe_key(self) -> tuple[str, str, str]:
        return (self.experiment, self.class_name, self.name)


class RecipeState(BaseModel):
    schema_version: int = 1
    recipes: list[Recipe] = Field(default_factory=list)


class RecipeBook:
    """Every saved recipe, and the rules for changing them."""

    def __init__(self, state: RecipeState | None = None) -> None:
        self.state = state or RecipeState()

    def get(self, experiment: str, class_name: str, name: str) -> Recipe:
        key = (experiment, class_name, name.strip())
        for recipe in self.state.recipes:
            if recipe.recipe_key == key:
                return recipe
        raise RecipeError(f"no recipe {key[2]!r} for {experiment}.{class_name}")

    def save(self, recipe: Recipe, overwrite: bool = False) -> Recipe:
        """Store ``recipe``, replacing one of the same name only with ``overwrite``."""
        recipe = recipe.model_copy(update={"due_time": None, "saved_at": now()})
        for index, existing in enumerate(self.state.recipes):
            if existing.recipe_key == recipe.recipe_key:
                if not overwrite:
                    raise RecipeError(
                        f"a recipe {recipe.name!r} for {recipe.experiment}."
                        f"{recipe.class_name} already exists"
                    )
                self.state.recipes[index] = recipe
                return recipe
        self.state.recipes.append(recipe)
        return recipe

    def delete(self, experiment: str, class_name: str, name: str) -> None:
        self.state.recipes.remove(self.get(experiment, class_name, name))

    def find(
        self, experiment: str | None = None, class_name: str | None = None
    ) -> list[Recipe]:
        return sorted(
            (
                recipe
                for recipe in self.state.recipes
                if experiment in (None, recipe.experiment)
                and class_name in (None, recipe.class_name)
            ),
            key=lambda recipe: recipe.recipe_key,
        )


class RecipeStore(ModelStore):
    """Persists a :class:`RecipeState`, apart from the queue state."""

    model = RecipeState
```

- [ ] **Step 5: Run the tests**

Run: `poetry run pytest tests/test_experiment_recipes.py tests/test_experiment_queue.py tests/test_experiment_manager.py -q`
Expected: all pass (the queue and manager suites confirm the `QueueStore` refactor changed nothing).

- [ ] **Step 6: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/experiment/queue.py pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
poetry run ruff format pytweezer/experiment/queue.py pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
git add pytweezer/experiment/queue.py pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
git commit -m "Add recipes: a book of named run settings and its store

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 2: Replaying a recipe and finding stale arguments

**Files:**
- Modify: `pytweezer/experiment/recipes.py`
- Test: `tests/test_experiment_recipes.py`

**Interfaces:**
- Consumes: `Recipe`, `RecipeError` (Task 1); experiment schema dicts as produced by `Experiment.schema()` — `{"arguments": {name: {...}}, "devices": {attribute: {"device", "timeout", "motmaster"?}}}`; a device is a MOTMaster when its schema has a non-`None` `"motmaster"` key.
- Produces:
  - `Recipe.replay(args: dict[str, Any] | None = None, *, priority: int | None = None, label: str | None = None, submitter: str = "") -> TaskRequest` — returns a plain `TaskRequest` (not a `Recipe`), `due_time` unset; raises `RecipeError` if `args` names a scanned argument.
  - `unknown_arguments(request: TaskRequest, schema: dict[str, Any]) -> list[str]` — sorted names (fixed and scanned) not declared by the schema, ignoring `attribute.parameter` names whose attribute is a MOTMaster.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_recipes.py` (and extend its imports: `ListAxis` from `pytweezer.experiment.scan`, `TaskRequest` from `pytweezer.experiment.task`, `unknown_arguments` from `pytweezer.experiment.recipes`, `RabiDemo` from `pytweezer.experiments.demo`):

```python
def test_replay_applies_overrides_and_keeps_the_rest():
    saved = recipe(
        args={"atoms": 5, "rabi_frequency": 1e3},
        scan=Scan(axes=[ListAxis(argument="pulse_time", values=[1e-6, 2e-6])]),
        priority=3,
        label="nightly",
        submitter="saver@pc",
    )
    request = saved.replay({"atoms": 9}, submitter="me@pc")
    assert type(request) is TaskRequest
    assert request.args == {"atoms": 9, "rabi_frequency": 1e3}
    assert request.scan == saved.scan
    assert (request.priority, request.label, request.submitter) == (
        3,
        "nightly",
        "me@pc",
    )
    assert saved.args["atoms"] == 5
    plain = saved.replay(priority=0, label="")
    assert (plain.priority, plain.label) == (0, "")


def test_replay_refuses_to_override_a_scanned_argument():
    saved = recipe(scan=Scan(axes=[ListAxis(argument="pulse_time", values=[1e-6])]))
    with pytest.raises(RecipeError, match=r"scans \['pulse_time'\]"):
        saved.replay({"pulse_time": 2e-6})


def test_unknown_arguments_lists_fixed_and_scanned_names_the_experiment_lacks():
    request = TaskRequest(
        experiment=DEMO,
        class_name="RabiDemo",
        args={"atoms": 1, "old": 2},
        scan=Scan(axes=[ListAxis(argument="gone", values=[1])]),
    )
    assert unknown_arguments(request, RabiDemo.schema()) == ["gone", "old"]


def test_unknown_arguments_leaves_motmaster_script_parameters_to_the_run():
    schema = {
        "arguments": {},
        "devices": {
            "rb": {"device": "Rb MotMaster", "timeout": None, "motmaster": {}},
            "cam": {"device": "Cam", "timeout": None},
        },
    }
    request = TaskRequest(
        experiment="m",
        class_name="C",
        args={"rb.tPulse": 1, "cam.exposure": 2, "rb": 3, "rb.": 4},
    )
    assert unknown_arguments(request, schema) == ["cam.exposure", "rb", "rb."]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_recipes.py -q`
Expected: ImportError for `unknown_arguments` (whole module fails to collect).

- [ ] **Step 3: Implement**

Add this method to `Recipe` (after `recipe_key`):

```python
    def replay(
        self,
        args: dict[str, Any] | None = None,
        *,
        priority: int | None = None,
        label: str | None = None,
        submitter: str = "",
    ) -> TaskRequest:
        """A request to run this recipe, ``args`` replacing its fixed arguments."""
        args = args or {}
        scanned = sorted(set(args) & {axis.argument for axis in self.scan.axes})
        if scanned:
            raise RecipeError(
                f"recipe {self.name!r} scans {scanned}; an override can't fix "
                "a scanned argument"
            )
        return TaskRequest(
            experiment=self.experiment,
            class_name=self.class_name,
            args={**self.args, **args},
            scan=self.scan,
            priority=self.priority if priority is None else priority,
            label=self.label if label is None else label,
            submitter=submitter,
        )
```

Add this module-level function after `RecipeStore`:

```python
def unknown_arguments(request: TaskRequest, schema: dict[str, Any]) -> list[str]:
    """Argument names ``request`` uses that the experiment ``schema`` describes doesn't declare.

    ``attribute.parameter`` names on a declared MOTMaster are left out: they
    name script parameters, which only the device can check.
    """
    motmasters = {
        attribute
        for attribute, device in schema.get("devices", {}).items()
        if device.get("motmaster") is not None
    }

    def known(name: str) -> bool:
        attribute, dot, parameter = name.partition(".")
        if dot and parameter and attribute in motmasters:
            return True
        return name in schema["arguments"]

    names = {*request.args, *(axis.argument for axis in request.scan.axes)}
    return sorted(name for name in names if not known(name))
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/test_experiment_recipes.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
poetry run ruff format pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
git add pytweezer/experiment/recipes.py tests/test_experiment_recipes.py
git commit -m "Replay recipes with overrides and find the arguments an experiment lost

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 3: Manager commands

**Files:**
- Modify: `pytweezer/servers/experiment_manager.py`
- Test: `tests/test_experiment_manager.py`

**Interfaces:**
- Consumes: `Recipe`, `RecipeBook`, `RecipeError`, `RecipeStore`, `unknown_arguments` (Tasks 1–2); `self.catalogue.entries()` (list of `{"module", "classes": [schema], "warnings", "error"}`) and `self.catalogue.busy`.
- Produces (REQ commands; replies always also carry `"ok"`):
  - `{"command": "recipes", "experiment"?: str, "class_name"?: str}` → `{"recipes": [Recipe JSON], "version": int}`
  - `{"command": "save_recipe", "recipe": Recipe JSON, "overwrite"?: bool}` → `{}`
  - `{"command": "delete_recipe", "experiment", "class_name", "name"}` → `{}`
  - `{"command": "submit_recipe", "experiment", "class_name", "name", "args"?: dict, "priority"?: int | None, "label"?: str | None, "submitter"?: str}` → `{"rid": int}`
  - Published snapshot gains `"recipes_version": int`.
  - Recipes persist in `<data root>/recipes.json`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiment_manager.py`, add imports:

```python
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.experiment.recipes import Recipe
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiments.demo import RabiDemo
```

Then append:

```python
class StaticCatalogue:
    """A catalogue whose entries are given, never read from files."""

    def __init__(self, entries):
        self._entries = entries
        self.busy = False

    def refresh(self):
        return False

    def poll(self):
        return False

    def entries(self):
        return self._entries

    def close(self):
        pass


class Sequenced(MotMasterExperiment):
    rb = MotMaster("Rb MotMaster", script="RbTweezerBasic")

    def run_point(self):
        pass


def entry(cls):
    return {
        "module": cls.__module__,
        "classes": [cls.schema()],
        "warnings": [],
        "error": None,
    }


BROKEN = {
    "module": "pytweezer.experiments.broken",
    "classes": [],
    "warnings": [],
    "error": "Traceback (most recent call last):\n  ...\nSyntaxError: invalid syntax\n",
}
DEMO = "pytweezer.experiments.demo"


@pytest.fixture
def recipe_manager(tmp_path, recording_db):
    def make():
        return em.ExperimentManager(
            root=tmp_path,
            catalogue=StaticCatalogue([entry(RabiDemo), entry(Sequenced), BROKEN]),
            bind=False,
            db=recording_db,
        )

    return make


def recipe(**kwargs):
    fields = {"experiment": DEMO, "class_name": "RabiDemo", "name": "check", **kwargs}
    return Recipe(**fields).model_dump(mode="json")


def save(manager, **kwargs):
    reply = manager.handle({"command": "save_recipe", "recipe": recipe(**kwargs)})
    assert reply == {"ok": True}


def submit_recipe(manager, **fields):
    return manager.handle(
        {
            "command": "submit_recipe",
            "experiment": DEMO,
            "class_name": "RabiDemo",
            "name": "check",
            **fields,
        }
    )


def test_recipes_are_saved_published_and_survive_a_restart(recipe_manager, tmp_path):
    manager = recipe_manager()
    version = manager._snapshot()["recipes_version"]
    save(manager, args={"atoms": 50})
    assert manager._snapshot()["recipes_version"] == version + 1
    assert (tmp_path / "recipes.json").exists()
    [saved] = recipe_manager().handle({"command": "recipes"})["recipes"]
    assert saved["name"] == "check" and saved["args"] == {"atoms": 50}


def test_saving_over_a_recipe_needs_overwrite(recipe_manager):
    manager = recipe_manager()
    save(manager)
    reply = manager.handle({"command": "save_recipe", "recipe": recipe(label="new")})
    assert not reply["ok"] and "already exists" in reply["error"]
    reply = manager.handle(
        {"command": "save_recipe", "recipe": recipe(label="new"), "overwrite": True}
    )
    assert reply["ok"]
    [saved] = manager.handle({"command": "recipes"})["recipes"]
    assert saved["label"] == "new"


def test_recipes_are_listed_per_experiment_and_deleted(recipe_manager):
    manager = recipe_manager()
    save(manager)
    save(manager, experiment=Sequenced.__module__, class_name="Sequenced", name="seq")
    listed = manager.handle(
        {"command": "recipes", "experiment": DEMO, "class_name": "RabiDemo"}
    )["recipes"]
    assert [r["name"] for r in listed] == ["check"]
    version = manager._snapshot()["recipes_version"]
    delete = {
        "command": "delete_recipe",
        "experiment": DEMO,
        "class_name": "RabiDemo",
        "name": "check",
    }
    assert manager.handle(delete) == {"ok": True}
    assert manager._snapshot()["recipes_version"] == version + 1
    assert [r["name"] for r in manager.handle({"command": "recipes"})["recipes"]] == [
        "seq"
    ]
    reply = manager.handle(delete)
    assert not reply["ok"] and "no recipe 'check'" in reply["error"]


def test_a_recipe_is_queued_with_its_settings_and_overrides(recipe_manager):
    manager = recipe_manager()
    scan = Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=1e-5, n=3)])
    save(manager, args={"atoms": 50, "rabi_frequency": 1e3}, scan=scan, priority=2)
    reply = submit_recipe(
        manager, args={"atoms": 7}, label="override", submitter="me@pc"
    )
    assert reply == {"ok": True, "rid": 1}
    task = manager.queue.get(1)
    assert task.args == {"atoms": 7, "rabi_frequency": 1e3}
    assert [axis.argument for axis in task.scan.axes] == ["pulse_time"]
    assert (task.priority, task.label, task.submitter) == (2, "override", "me@pc")
    assert task.due_time is None


def test_a_recipe_using_a_removed_argument_is_refused(recipe_manager):
    manager = recipe_manager()
    save(manager, args={"atoms": 5, "old_knob": 1})
    reply = submit_recipe(manager)
    assert not reply["ok"]
    assert "no longer has" in reply["error"] and "old_knob" in reply["error"]
    save(manager, name="scanned", scan=Scan(axes=[ListAxis(argument="old", values=[1])]))
    reply = submit_recipe(manager, name="scanned")
    assert not reply["ok"] and "'old'" in reply["error"]
    assert manager.queue.ordered() == []


def test_an_override_must_name_an_argument_and_not_a_scanned_one(recipe_manager):
    manager = recipe_manager()
    save(manager, scan=Scan(axes=[ListAxis(argument="atoms", values=[1, 2])]))
    reply = submit_recipe(manager, args={"atom": 3})
    assert not reply["ok"]
    assert "has no argument" in reply["error"] and "'atom'" in reply["error"]
    assert "no longer" not in reply["error"]
    reply = submit_recipe(manager, args={"atoms": 3})
    assert not reply["ok"] and "scans" in reply["error"]
    assert manager.queue.ordered() == []


def test_motmaster_script_parameters_are_left_to_the_run(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": Sequenced.__module__, "class_name": "Sequenced"}
    save(manager, name="seq", args={"rb.tPulse": 2e-6}, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "seq", **fields})
    assert reply["ok"]
    save(manager, name="bad", args={"cs.tPulse": 1}, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "bad", **fields})
    assert not reply["ok"] and "cs.tPulse" in reply["error"]


def test_a_recipe_for_a_missing_experiment_is_refused(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": "pytweezer.experiments.gone", "class_name": "Gone"}
    save(manager, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"] and "not in the catalogue" in reply["error"]
    manager.catalogue.busy = True
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"] and "try again" in reply["error"]


def test_a_recipe_for_a_module_that_fails_to_import_says_why(recipe_manager):
    manager = recipe_manager()
    fields = {"experiment": BROKEN["module"], "class_name": "Anything"}
    save(manager, **fields)
    reply = manager.handle({"command": "submit_recipe", "name": "check", **fields})
    assert not reply["ok"]
    assert "fails to import" in reply["error"]
    assert "SyntaxError: invalid syntax" in reply["error"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_manager.py -q`
Expected: the new tests fail with `KeyError: 'recipes_version'` or `unknown command 'save_recipe'`; existing tests still pass.

- [ ] **Step 3: Implement**

In `pytweezer/servers/experiment_manager.py`:

1. Import:

```python
from pytweezer.experiment.recipes import (
    Recipe,
    RecipeBook,
    RecipeError,
    RecipeStore,
    unknown_arguments,
)
```

2. In `__init__`, directly after `self.queue = ExperimentQueue(self.store.load())`:

```python
        self.recipe_store = RecipeStore(self.root / "recipes.json")
        self.recipes = RecipeBook(self.recipe_store.load())
```

and directly after `self._catalogue_version = 0` (it must exist before `self._snapshot()` is first called):

```python
        self._recipes_version = 0
```

3. Add the commands after `_cmd_submit`:

```python
    def _cmd_recipes(self, request):
        recipes = self.recipes.find(request.get("experiment"), request.get("class_name"))
        return {
            "recipes": [recipe.model_dump(mode="json") for recipe in recipes],
            "version": self._recipes_version,
        }

    def _cmd_save_recipe(self, request):
        recipe = self.recipes.save(
            Recipe.model_validate(request["recipe"]),
            overwrite=bool(request.get("overwrite")),
        )
        logger.info(
            "Saved recipe %r for %s.%s", recipe.name, recipe.experiment, recipe.class_name
        )
        self._recipes_changed()

    def _cmd_delete_recipe(self, request):
        self.recipes.delete(request["experiment"], request["class_name"], request["name"])
        self._recipes_changed()

    def _cmd_submit_recipe(self, request):
        recipe = self.recipes.get(
            request["experiment"], request["class_name"], request["name"]
        )
        overrides = request.get("args") or {}
        task_request = recipe.replay(
            overrides,
            priority=request.get("priority"),
            label=request.get("label"),
            submitter=request.get("submitter", ""),
        )
        unknown = unknown_arguments(
            task_request, self._schema(recipe.experiment, recipe.class_name)
        )
        if bad_overrides := [name for name in unknown if name in overrides]:
            raise RecipeError(f"{recipe.class_name} has no argument(s) {bad_overrides}")
        if unknown:
            raise RecipeError(
                f"{recipe.class_name} no longer has argument(s) {unknown}; load "
                f"recipe {recipe.name!r} into the form, check it and save it again"
            )
        task = self.queue.submit(task_request)
        logger.info("Queued task %s from recipe %r", task.rid, recipe.name)
        self._changed()
        return {"rid": task.rid}

    def _schema(self, module: str, class_name: str) -> dict[str, Any]:
        for entry in self.catalogue.entries():
            if entry["module"] != module:
                continue
            if entry.get("error"):
                reason = entry["error"].strip().splitlines()[-1]
                raise RecipeError(f"{module} fails to import: {reason}")
            for schema in entry["classes"]:
                if schema["class_name"] == class_name:
                    return schema
        if self.catalogue.busy:
            raise RecipeError(
                f"{module}.{class_name} is not in the catalogue yet; the manager "
                "is still reading the experiments, so try again shortly"
            )
        raise RecipeError(f"{module}.{class_name} is not in the catalogue")
```

4. Next to `_changed`:

```python
    def _recipes_changed(self) -> None:
        self.recipe_store.save(self.recipes.state)
        self._recipes_version += 1
        self._dirty = True
```

5. In `_snapshot`, add `"recipes_version": self._recipes_version,` after `"catalogue_version"`, and in the module-level layout comment change the first line to:

```python
#:     {"started", "catalogue_version", "recipes_version", "simulated",
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/test_experiment_manager.py tests/test_experiment_simulation.py tests/test_experiment_e2e.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/servers/experiment_manager.py tests/test_experiment_manager.py
poetry run ruff format pytweezer/servers/experiment_manager.py tests/test_experiment_manager.py
git add pytweezer/servers/experiment_manager.py tests/test_experiment_manager.py
git commit -m "Keep recipes in the Experiment Manager and queue them on request

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 4: Client methods and notebook functions

**Files:**
- Modify: `pytweezer/experiment/client.py`
- Test: `tests/test_experiment_manager.py`

**Interfaces:**
- Consumes: the four manager commands (Task 3); `Recipe` (Task 1); `StaticCatalogue`, `recipe_manager`, `DEMO` from `tests/test_experiment_manager.py` (Task 3).
- Produces:
  - `ExperimentManagerClient.recipes(experiment: str | None = None, class_name: str | None = None) -> list[Recipe]`
  - `ExperimentManagerClient.save_recipe(recipe: Recipe, overwrite: bool = False) -> None`
  - `ExperimentManagerClient.delete_recipe(experiment: str, class_name: str, name: str) -> None`
  - `ExperimentManagerClient.submit_recipe(experiment: str, class_name: str, name: str, *, args: dict | None = None, priority: int | None = None, label: str | None = None, submitter: str = "") -> int`
  - Module functions (experiment and name positional-only, so an experiment argument may itself be called `name`):
    - `save_recipe(experiment, name, scan=None, /, *, priority=0, label="", overwrite=False, client=None, **args) -> None`
    - `submit_recipe(experiment, name, /, *, priority=None, label=None, client=None, **args) -> int`
    - `recipes(experiment=None, /, *, client=None) -> list[Recipe]`
    - `delete_recipe(experiment, name, /, *, client=None) -> None`
  - `experiment` is an Experiment class or a `"module:ClassName"` string, as for `submit`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_manager.py` (add `import json` at the top and `from pytweezer.experiment.client import ExperimentManagerClient, ManagerError, delete_recipe, recipes, save_recipe, submit_recipe`):

```python
class InProcess:
    """Routes client requests straight into a manager, through JSON as on the wire."""

    def __init__(self, manager):
        self.manager = manager

    def request(self, payload):
        return self.manager.handle(json.loads(json.dumps(payload)))

    def close(self):
        pass


def client_for(manager):
    client = ExperimentManagerClient(endpoint="tcp://127.0.0.1:1")
    client._req = InProcess(manager)
    return client


def test_notebook_recipes_round_trip(recipe_manager):
    manager = recipe_manager()
    client = client_for(manager)
    scan = Scan(axes=[LinearAxis(argument="pulse_time", start=0, stop=1e-5, n=3)])
    save_recipe(RabiDemo, "check", scan, label="nightly", client=client, atoms=50)
    [saved] = recipes(RabiDemo, client=client)
    # Every argument is stored, so a later change of default can't alter the recipe.
    assert saved.args == {"rabi_frequency": 50e3, "atoms": 50, "point_delay": 0.2}
    assert saved.label == "nightly" and "@" in saved.submitter

    rid = submit_recipe(RabiDemo, "check", client=client, atoms=60)
    task = manager.queue.get(rid)
    assert task.args["atoms"] == 60 and task.label == "nightly"
    assert "@" in task.submitter

    with pytest.raises(ManagerError, match="already exists"):
        save_recipe(RabiDemo, "check", client=client)
    save_recipe(RabiDemo, "check", client=client, overwrite=True, atoms=1)
    assert recipes(client=client)[0].args["atoms"] == 1

    delete_recipe(f"{DEMO}:RabiDemo", "check", client=client)
    assert recipes(client=client) == []


def test_notebook_recipe_arguments_are_checked_before_sending(recipe_manager):
    manager = recipe_manager()
    client = client_for(manager)
    with pytest.raises(ValueError, match="no argument"):
        save_recipe(RabiDemo, "x", client=client, atom=1)
    assert recipes(client=client) == []
    save_recipe(RabiDemo, "x", client=client)
    with pytest.raises(ValueError, match="no argument"):
        submit_recipe(RabiDemo, "x", client=client, atom=1)
    assert manager.queue.ordered() == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_manager.py -q`
Expected: ImportError for `save_recipe` from `pytweezer.experiment.client`.

- [ ] **Step 3: Implement**

In `pytweezer/experiment/client.py`:

1. Import `from pytweezer.experiment.recipes import Recipe`.

2. Add to `ExperimentManagerClient` after `last_request`:

```python
    def recipes(
        self, experiment: str | None = None, class_name: str | None = None
    ) -> list[Recipe]:
        reply = self.call("recipes", experiment=experiment, class_name=class_name)
        return [Recipe.model_validate(recipe) for recipe in reply["recipes"]]

    def save_recipe(self, recipe: Recipe, overwrite: bool = False) -> None:
        self.call(
            "save_recipe", recipe=recipe.model_dump(mode="json"), overwrite=overwrite
        )

    def delete_recipe(self, experiment: str, class_name: str, name: str) -> None:
        self.call(
            "delete_recipe", experiment=experiment, class_name=class_name, name=name
        )

    def submit_recipe(
        self,
        experiment: str,
        class_name: str,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        priority: int | None = None,
        label: str | None = None,
        submitter: str = "",
    ) -> int:
        return self.call(
            "submit_recipe",
            experiment=experiment,
            class_name=class_name,
            name=name,
            args=args or {},
            priority=priority,
            label=label,
            submitter=submitter,
        )["rid"]
```

3. Replace the module/class resolution inside `submit` with a shared helper. Add above `submit`:

```python
def _resolve(
    experiment: type | str,
    scan: Scan | None = None,
    args: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """``(module, class_name)`` for ``experiment``, checking ``scan`` and ``args`` against a class."""
    if isinstance(experiment, str):
        module, _, class_name = experiment.partition(":")
    else:
        module, class_name = experiment.__module__, experiment.__qualname__
        # Fail here rather than when the task reaches the front of the queue.
        coerce_arguments(experiment, args or {})
        if scan is not None:
            scan.axis_values(experiment)
    if module == "__main__" or "<locals>" in class_name or not class_name:
        raise ValueError(
            f"{experiment!r} can't be imported by the manager; define it in a module "
            "under pytweezer/experiments/, or run it here with run_local()"
        )
    return module, class_name
```

and make the body of `submit` start:

```python
    scan = scan or Scan()
    module, class_name = _resolve(experiment, scan, args)
    request = TaskRequest(
        ...  # unchanged from here on
```

4. Add after `submit`:

```python
def save_recipe(
    experiment: type | str,
    name: str,
    scan: Scan | None = None,
    /,
    *,
    priority: int = 0,
    label: str = "",
    overwrite: bool = False,
    client: ExperimentManagerClient | None = None,
    **args: Any,
) -> None:
    """Save these settings as recipe ``name``, shared with every PC.

    With a class, every argument is stored (defaults included), so a later
    change of default does not change what the recipe runs.
    """
    scan = scan or Scan()
    module, class_name = _resolve(experiment, scan, args)
    if not isinstance(experiment, str):
        scanned = {axis.argument for axis in scan.axes}
        args = {
            key: value
            for key, value in coerce_arguments(experiment, args).items()
            if key not in scanned
        }
    recipe = Recipe(
        experiment=module,
        class_name=class_name,
        name=name,
        args=args,
        scan=scan,
        priority=priority,
        label=label,
        submitter=submitter_name(),
    )
    (client or ExperimentManagerClient()).save_recipe(recipe, overwrite)


def submit_recipe(
    experiment: type | str,
    name: str,
    /,
    *,
    priority: int | None = None,
    label: str | None = None,
    client: ExperimentManagerClient | None = None,
    **args: Any,
) -> int:
    """Queue recipe ``name``, with ``args`` replacing its fixed arguments; returns the rid."""
    module, class_name = _resolve(experiment, args=args)
    return (client or ExperimentManagerClient()).submit_recipe(
        module,
        class_name,
        name,
        args=args,
        priority=priority,
        label=label,
        submitter=submitter_name(),
    )


def recipes(
    experiment: type | str | None = None,
    /,
    *,
    client: ExperimentManagerClient | None = None,
) -> list[Recipe]:
    """Saved recipes, for one experiment or for all."""
    module = class_name = None
    if experiment is not None:
        module, class_name = _resolve(experiment)
    return (client or ExperimentManagerClient()).recipes(module, class_name)


def delete_recipe(
    experiment: type | str,
    name: str,
    /,
    *,
    client: ExperimentManagerClient | None = None,
) -> None:
    module, class_name = _resolve(experiment)
    (client or ExperimentManagerClient()).delete_recipe(module, class_name, name)
```

5. Extend the module docstring's notebook example with:

```python
    rid = submit_recipe(RabiDemo, "nightly check", atoms=300)
```

and the import line above it to `from pytweezer.experiment.client import submit, submit_recipe, wait`.

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/test_experiment_manager.py tests/test_experiment_e2e.py tests/test_experiment_runner.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/experiment/client.py tests/test_experiment_manager.py
poetry run ruff format pytweezer/experiment/client.py tests/test_experiment_manager.py
git add pytweezer/experiment/client.py tests/test_experiment_manager.py
git commit -m "Save, list, delete and submit recipes from notebooks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 5: Recipes in the catalogue tree

**Files:**
- Modify: `pytweezer/GUI/experiments/catalogue_view.py`
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: recipe dicts as returned by the `recipes` command (`Recipe.model_dump(mode="json")`: keys `experiment`, `class_name`, `name`, `args`, `scan`, `priority`, `label`, `submitter`, `due_time`, `saved_at`).
- Produces (on `CatalogueView`):
  - `recipe_selected = pyqtSignal(dict)` — a recipe item became current.
  - `recipe_action_requested = pyqtSignal(str, dict)` — `("submit" | "delete", recipe)`.
  - `recipes: list[dict]` attribute; `set_recipes(recipes: list[dict]) -> None`.
  - `select_recipe(module: str, class_name: str, name: str) -> bool`.
  - `recipe_menu(recipe: dict) -> QMenu` (actions "Submit now", separator, "Delete…").
  - `selected_key()` returns the recipe's experiment key when a recipe is current.
  - Rebuilding (from `set_modules` or `set_recipes`) keeps the current item and emits nothing.

Load the `pytweezer-gui-design` skill before this task.

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiment_gui.py` add imports `from pytweezer.experiment.recipes import Recipe` and `from pytweezer.GUI.experiments.catalogue_view import CatalogueView`, then append:

```python
def recipe_dict(name="check", **kwargs):
    return Recipe(
        experiment=SCHEMA["module"],
        class_name="Demo",
        name=name,
        submitter="me@pc",
        **kwargs,
    ).model_dump(mode="json")


def demo_item(view):
    for i in range(view.tree.topLevelItemCount()):
        module_item = view.tree.topLevelItem(i)
        for j in range(module_item.childCount()):
            if module_item.child(j).text(0) == "Demo":
                return module_item.child(j)
    raise AssertionError("Demo not in the tree")


def test_recipes_appear_under_their_experiment_and_filter(qapp):
    view = CatalogueView()
    view.set_modules(MODULES)
    view.set_recipes([recipe_dict("MOT check", args={"shots": 5})])
    demo = demo_item(view)
    assert demo.childCount() == 1
    recipe_item = demo.child(0)
    assert recipe_item.text(0) == "MOT check"
    assert recipe_item.font(0).italic()

    view.filter.setText("mot ch")
    assert not recipe_item.isHidden() and not demo.isHidden()
    view.filter.setText("Demo")
    assert not recipe_item.isHidden()
    view.filter.setText("nothing like it")
    assert recipe_item.isHidden() and demo.isHidden()


def test_selecting_a_recipe_emits_it_and_a_refresh_keeps_it_quietly(qapp):
    view = CatalogueView()
    view.set_modules(MODULES)
    view.set_recipes([recipe_dict("a"), recipe_dict("b")])
    selected, experiments = [], []
    view.recipe_selected.connect(selected.append)
    view.experiment_selected.connect(experiments.append)
    assert view.select_recipe(SCHEMA["module"], "Demo", "b")
    assert [r["name"] for r in selected] == ["b"] and experiments == []
    assert view.selected_key() == (SCHEMA["module"], "Demo")

    view.set_recipes([recipe_dict("a"), recipe_dict("b"), recipe_dict("c")])
    view.set_modules(MODULES)
    assert len(selected) == 1 and experiments == []
    assert view.tree.currentItem().text(0) == "b"


def test_the_recipe_menu_submits_or_deletes(qapp):
    view = CatalogueView()
    view.set_modules(MODULES)
    recipe = recipe_dict()
    view.set_recipes([recipe])
    actions = []
    view.recipe_action_requested.connect(lambda *args: actions.append(args))
    menu = view.recipe_menu(recipe)
    labels = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert labels == ["Submit now", "Delete…"]
    for action in menu.actions():
        if not action.isSeparator():
            action.trigger()
    assert [a[0] for a in actions] == ["submit", "delete"]
    assert actions[0][1]["name"] == "check"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_gui.py -q -k "recipe"`
Expected: `AttributeError: 'CatalogueView' object has no attribute 'set_recipes'`.

- [ ] **Step 3: Implement**

Rewrite `pytweezer/GUI/experiments/catalogue_view.py` as follows (the existing `set_modules` body moves into `_rebuild`; `schema_for` and `select` are unchanged):

```python
"""Tree of the experiments the manager can run, grouped by module, with their recipes."""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pytweezer.GUI.components import status_icon

_SCHEMA = QtCore.Qt.ItemDataRole.UserRole
_RECIPE = QtCore.Qt.ItemDataRole.UserRole + 1


class CatalogueView(QWidget):
    experiment_selected = QtCore.pyqtSignal(dict)
    recipe_selected = QtCore.pyqtSignal(dict)
    recipe_action_requested = QtCore.pyqtSignal(str, dict)
    refresh_requested = QtCore.pyqtSignal()

    def __init__(self, package="pytweezer.experiments", parent=None):
        super().__init__(parent)
        self.package = package
        self.modules = []
        self.recipes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter experiments and recipes")
        self.filter.textChanged.connect(self._apply_filter)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_requested)
        top.addWidget(self.filter, 1)
        top.addWidget(refresh)
        layout.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.tree, 1)

    def set_modules(self, modules):
        self.modules = modules
        self._rebuild()

    def set_recipes(self, recipes):
        self.recipes = recipes
        self._rebuild()

    def _rebuild(self):
        selected = self._selection()
        by_class = {}
        for recipe in self.recipes:
            by_class.setdefault(
                (recipe["experiment"], recipe["class_name"]), []
            ).append(recipe)
        self.tree.blockSignals(True)
        self.tree.clear()
        reselect = None
        for module in self.modules:
            name = module["module"].removeprefix(self.package + ".")
            module_item = QTreeWidgetItem([name])
            module_item.setToolTip(0, module["module"])
            if module.get("error"):
                module_item.setIcon(0, status_icon("crashed"))
                module_item.setToolTip(0, module["error"])
            elif module.get("warnings"):
                module_item.setIcon(0, status_icon("starting"))
                module_item.setToolTip(0, "\n".join(module["warnings"]))
            for schema in module.get("classes", []):
                item = QTreeWidgetItem([schema["class_name"]])
                item.setData(0, _SCHEMA, schema)
                item.setToolTip(0, schema.get("doc") or schema["class_name"])
                module_item.addChild(item)
                if _key(schema) == selected:
                    reselect = item
                for recipe in by_class.get(_key(schema), []):
                    recipe_item = _recipe_item(recipe)
                    item.addChild(recipe_item)
                    if _recipe_key(recipe) == selected:
                        reselect = recipe_item
            self.tree.addTopLevelItem(module_item)
        self.tree.expandAll()
        self._apply_filter(self.filter.text())
        if reselect is not None:
            self.tree.setCurrentItem(reselect)
        self.tree.blockSignals(False)

    def schema_for(self, module, class_name):
        ...  # unchanged

    def select(self, module, class_name):
        ...  # unchanged

    def select_recipe(self, module, class_name, name):
        wanted = (module, class_name, name)
        iterator = QTreeWidgetItemIterator(self.tree)
        while item := iterator.value():
            recipe = item.data(0, _RECIPE)
            if recipe and _recipe_key(recipe) == wanted:
                self.tree.setCurrentItem(item)
                return True
            iterator += 1
        return False

    def selected_key(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        if recipe := item.data(0, _RECIPE):
            return (recipe["experiment"], recipe["class_name"])
        schema = item.data(0, _SCHEMA)
        return _key(schema) if schema else None

    def _selection(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        if recipe := item.data(0, _RECIPE):
            return _recipe_key(recipe)
        schema = item.data(0, _SCHEMA)
        return _key(schema) if schema else None

    def _current_changed(self, item, _previous):
        if item is None:
            return
        if recipe := item.data(0, _RECIPE):
            self.recipe_selected.emit(recipe)
        elif schema := item.data(0, _SCHEMA):
            self.experiment_selected.emit(schema)

    def recipe_menu(self, recipe):
        menu = QMenu(self)
        submit = menu.addAction("Submit now")
        submit.triggered.connect(
            lambda: self.recipe_action_requested.emit("submit", recipe)
        )
        menu.addSeparator()
        delete = menu.addAction("Delete…")
        delete.triggered.connect(
            lambda: self.recipe_action_requested.emit("delete", recipe)
        )
        return menu

    def _context_menu(self, position):
        item = self.tree.itemAt(position)
        recipe = item.data(0, _RECIPE) if item else None
        if recipe:
            self.recipe_menu(recipe).exec(self.tree.viewport().mapToGlobal(position))

    def _apply_filter(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            module_item = self.tree.topLevelItem(i)
            module_match = text in module_item.text(0).lower()
            any_class = False
            for j in range(module_item.childCount()):
                class_item = module_item.child(j)
                class_match = module_match or text in class_item.text(0).lower()
                any_recipe = False
                for k in range(class_item.childCount()):
                    recipe_item = class_item.child(k)
                    visible = class_match or text in recipe_item.text(0).lower()
                    recipe_item.setHidden(not visible)
                    any_recipe |= visible
                class_item.setHidden(not (class_match or any_recipe))
                any_class |= class_match or any_recipe
            module_item.setHidden(not (module_match or any_class))


def _key(schema):
    return (schema["module"], schema["class_name"])


def _recipe_key(recipe):
    return (recipe["experiment"], recipe["class_name"], recipe["name"])


def _recipe_item(recipe):
    item = QTreeWidgetItem([recipe["name"]])
    item.setData(0, _RECIPE, recipe)
    font = item.font(0)
    font.setItalic(True)
    item.setFont(0, font)
    saved = recipe.get("saved_at", "")[:16].replace("T", " ")
    tooltip = f"Recipe saved by {recipe.get('submitter') or 'unknown'} on {saved}"
    if recipe.get("label"):
        tooltip += f"\nLabel: {recipe['label']}"
    item.setToolTip(0, tooltip + "\nRight-click to submit it as it is")
    return item
```

Add `QTreeWidgetItemIterator` to the `PyQt6.QtWidgets` import. Replace the two `...  # unchanged` bodies with the existing code of `schema_for` and `select` exactly as they are now.

Recipes are marked by an italic font, not a colour: the theme's `QTreeView::item { color }` rule overrides `setForeground()` (see "Qt and QSS traps" in the `pytweezer-gui-design` skill).

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/test_experiment_gui.py -q`
Expected: all pass, including the existing catalogue/panel tests.

- [ ] **Step 5: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/GUI/experiments/catalogue_view.py tests/test_experiment_gui.py
poetry run ruff format pytweezer/GUI/experiments/catalogue_view.py tests/test_experiment_gui.py
git add pytweezer/GUI/experiments/catalogue_view.py tests/test_experiment_gui.py
git commit -m "Show recipes under their experiment in the catalogue tree

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 6: "Save as recipe…" and the recipe name in the editor

**Files:**
- Modify: `pytweezer/GUI/experiments/arg_editor.py` (`ArgumentEditor` only)
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (on `ArgumentEditor`):
  - `save_recipe_requested = pyqtSignal(object)` — emits the validated `TaskRequest` of the form.
  - `save_recipe_button: QPushButton` labelled "Save as recipe…".
  - `recipe_name: str` (empty when the form is not from a recipe) and `set_recipe_name(name: str) -> None`, shown in a header label `self.recipe_label`.
  - `set_experiment` and `reset_to_defaults` clear the recipe name (so "Defaults", "Last submitted" and `load_request` all clear it; callers set it after `load_request`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_gui.py`:

```python
def test_save_as_recipe_emits_the_validated_form(editor):
    saved = []
    editor.save_recipe_requested.connect(saved.append)
    editor.rows["shots"].value.widget.setValue(6)
    editor.save_recipe_button.click()
    [request] = saved
    assert request.args["shots"] == 6 and request.class_name == "Demo"


def test_an_invalid_form_cannot_be_saved_as_a_recipe(editor):
    saved = []
    editor.save_recipe_requested.connect(saved.append)
    editor.rows["shots"].scan_button.setChecked(True)
    editor.rows["shots"].scan.mode.setCurrentText("list")
    editor.rows["shots"].scan.values.setText("1, 0")
    editor.save_recipe_button.click()
    assert saved == []
    assert "outside the allowed range" in editor.error.text()


def test_the_recipe_name_shows_until_the_form_is_reset(editor):
    editor.set_recipe_name("MOT check")
    assert editor.recipe_name == "MOT check"
    assert "MOT check" in editor.recipe_label.text()
    assert not editor.recipe_label.isHidden()
    editor.defaults_button.click()
    assert editor.recipe_name == "" and editor.recipe_label.isHidden()
    editor.set_recipe_name("again")
    editor.set_experiment(SCHEMA)
    assert editor.recipe_name == ""
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_gui.py -q -k "recipe"`
Expected: `AttributeError: ... 'save_recipe_requested'`.

- [ ] **Step 3: Implement**

In `ArgumentEditor`:

1. Signal, next to `last_requested`:

```python
    save_recipe_requested = QtCore.pyqtSignal(object)
```

2. In `__init__`, set `self.recipe_name = ""` near `self.schema = None`. In the title row, after `title_row.addWidget(self.title)`:

```python
        self.recipe_label = QLabel()
        self.recipe_label.setProperty("role", "regionHint")
        self.recipe_label.setVisible(False)
        title_row.addWidget(self.recipe_label)
```

3. Button, after `self.last_button` is created:

```python
        self.save_recipe_button = QPushButton("Save as recipe…")
        self.save_recipe_button.setToolTip(
            "Keep these settings under a name, shared with every PC"
        )
        self.save_recipe_button.clicked.connect(
            lambda: self._emit_request(self.save_recipe_requested)
        )
```

and add it to the button row right after `buttons.addWidget(self.last_button)`:

```python
        buttons.addWidget(self.save_recipe_button)
```

4. `_set_enabled` iterates `(self.defaults_button, self.last_button, self.save_recipe_button, self.submit_button)`.

5. Replace `_submit` with a shared emitter and point the Submit button at it:

```python
    def _emit_request(self, signal):
        try:
            request = self.request()
        except ValueError as error:
            self.show_error(str(error))
            return
        self.error.clear()
        signal.emit(request)
```

and change `self.submit_button.clicked.connect(self._submit)` to `self.submit_button.clicked.connect(lambda: self._emit_request(self.submit_requested))`. That is the only use of `ArgumentEditor._submit` (the panel's `_submit` is a different method and stays).

6. Add:

```python
    def set_recipe_name(self, name):
        """Name the recipe the form was loaded from; empty when it wasn't."""
        self.recipe_name = name
        self.recipe_label.setText(f"from recipe “{name}”" if name else "")
        self.recipe_label.setVisible(bool(name))
```

7. Call `self.set_recipe_name("")` as the first line of `reset_to_defaults` and in `set_experiment` right after `self.schema = schema`.

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/test_experiment_gui.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/GUI/experiments/arg_editor.py tests/test_experiment_gui.py
poetry run ruff format pytweezer/GUI/experiments/arg_editor.py tests/test_experiment_gui.py
git add pytweezer/GUI/experiments/arg_editor.py tests/test_experiment_gui.py
git commit -m "Add Save as recipe to the experiment form and name a loaded recipe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 7: Panel wiring and documentation

**Files:**
- Modify: `pytweezer/GUI/experiments/panel.py`
- Modify: `.claude/skills/add-experiment/SKILL.md`
- Test: `tests/test_experiment_gui.py`

**Interfaces:**
- Consumes: `CatalogueView.recipe_selected`, `.recipe_action_requested`, `.set_recipes`, `.recipes`, `.select_recipe` (Task 5); `ArgumentEditor.save_recipe_requested`, `.set_recipe_name`, `.recipe_name` (Task 6); manager commands `recipes`, `save_recipe`, `delete_recipe`, `submit_recipe` and snapshot key `recipes_version` (Task 3); `Recipe` (Task 1); `submitter_name()` from `pytweezer.experiment.client`.
- Produces (on `ExperimentsPanel`):
  - `refresh_recipes() -> None`
  - `ask_recipe_name(default: str) -> str` and `confirm(title: str, question: str) -> bool` — dialog hooks that tests replace.
  - Status texts: `"Saved recipe 'NAME'"`, `"Queued task RID from recipe 'NAME'"`, `"Deleted recipe 'NAME'"`; failures as `"<what>: <manager error>"` in the crashed state (existing `_call`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiment_gui.py`, replace `FakeClient` with this version (existing tests keep working: unknown commands still answer `{"ok": True}`), adding `from pytweezer.experiment.client import ManagerError` to the imports:

```python
class FakeClient:
    def __init__(self):
        self.calls = []
        self.last = None
        self.recipes = []
        self.recipes_version = 0
        self.errors = {}

    def call(self, command, **fields):
        self.calls.append((command, fields))
        if command in self.errors:
            raise ManagerError(self.errors[command])
        if command == "catalogue":
            return {"ok": True, "modules": MODULES, "version": 1}
        if command == "recipes":
            return {"ok": True, "recipes": self.recipes, "version": self.recipes_version}
        if command == "submit_recipe":
            return {"ok": True, "rid": 13}
        return {"ok": True}

    def submit(self, request):
        self.calls.append(("submit", request))
        return 12

    def last_request(self, module, class_name):
        return self.last

    def close(self):
        pass
```

Then append:

```python
def recipe_panel(qapp, *recipes):
    client, feed = FakeClient(), FakeFeed()
    client.recipes = list(recipes)
    panel = ExperimentsPanel(client=client, feed=feed)
    panel.refresh_catalogue()
    return panel, client, feed


def commands(client, name):
    return [fields for command, fields in client.calls if command == name]


def test_panel_lists_recipes_and_refetches_when_they_change(qapp):
    panel, client, feed = recipe_panel(qapp, recipe_dict("a"))
    assert [r["name"] for r in panel.catalogue.recipes] == ["a"]
    fetches = len(commands(client, "recipes"))
    snapshot = {"running": None, "queue": [], "history": [], "recipes_version": 0}
    feed.queue_changed.emit(snapshot)
    assert len(commands(client, "recipes")) == fetches
    client.recipes = [recipe_dict("a"), recipe_dict("b")]
    feed.queue_changed.emit({**snapshot, "recipes_version": 1})
    assert [r["name"] for r in panel.catalogue.recipes] == ["a", "b"]


def test_selecting_a_recipe_fills_the_form_and_names_it(qapp):
    panel, client, _feed = recipe_panel(
        qapp, recipe_dict("check", args={"shots": 5, "mode": "b"}, label="nightly")
    )
    assert panel.catalogue.select_recipe(SCHEMA["module"], "Demo", "check")
    assert panel.editor.schema["class_name"] == "Demo"
    assert panel.editor.rows["shots"].value.value() == 5
    assert panel.editor.rows["mode"].value.value() == "b"
    assert panel.editor.label.text() == "nightly"
    assert panel.editor.recipe_name == "check"


def test_a_recipe_list_refresh_does_not_reload_the_form(qapp):
    panel, client, feed = recipe_panel(qapp, recipe_dict("check", args={"shots": 5}))
    panel.catalogue.select_recipe(SCHEMA["module"], "Demo", "check")
    panel.editor.rows["shots"].value.widget.setValue(8)
    client.recipes = [recipe_dict("check", args={"shots": 5}), recipe_dict("new")]
    feed.queue_changed.emit(
        {"running": None, "queue": [], "history": [], "recipes_version": 1}
    )
    assert panel.editor.rows["shots"].value.value() == 8
    assert panel.catalogue.tree.currentItem().text(0) == "check"


def test_save_as_recipe_asks_for_a_name_and_confirms_a_replacement(qapp):
    panel, client, _feed = recipe_panel(qapp, recipe_dict("check"))
    panel.catalogue.select(SCHEMA["module"], "Demo")
    asked, confirmed = [], []
    panel.ask_recipe_name = lambda default: asked.append(default) or " new one "
    panel.confirm = lambda title, question: confirmed.append(question) or True
    panel.editor.rows["shots"].value.widget.setValue(4)
    panel.editor.save_recipe_button.click()
    [fields] = commands(client, "save_recipe")
    assert fields["recipe"]["name"] == "new one" and not fields["overwrite"]
    assert fields["recipe"]["args"]["shots"] == 4
    assert "@" in fields["recipe"]["submitter"]
    assert confirmed == [] and asked == [""]
    assert panel.editor.recipe_name == "new one"
    assert "Saved recipe 'new one'" in panel.status.text()

    panel.ask_recipe_name = lambda default: "check"
    panel.confirm = lambda title, question: False
    panel.editor.save_recipe_button.click()
    assert len(commands(client, "save_recipe")) == 1
    panel.confirm = lambda title, question: True
    panel.editor.save_recipe_button.click()
    assert commands(client, "save_recipe")[-1]["overwrite"] is True

    panel.ask_recipe_name = lambda default: "  "
    panel.editor.save_recipe_button.click()
    assert len(commands(client, "save_recipe")) == 2


def test_submit_now_queues_the_recipe_or_shows_why_not(qapp):
    recipe = recipe_dict("check")
    panel, client, _feed = recipe_panel(qapp, recipe)
    panel.catalogue.recipe_action_requested.emit("submit", recipe)
    [fields] = commands(client, "submit_recipe")
    assert (fields["experiment"], fields["class_name"], fields["name"]) == (
        SCHEMA["module"],
        "Demo",
        "check",
    )
    assert "@" in fields["submitter"]
    assert "Queued task 13 from recipe 'check'" in panel.status.text()

    client.errors["submit_recipe"] = "Demo no longer has argument(s) ['old']"
    panel.catalogue.recipe_action_requested.emit("submit", recipe)
    assert "no longer has argument(s) ['old']" in panel.status.text()
    assert panel.status.property("state") == "crashed"


def test_deleting_a_recipe_needs_confirmation(qapp):
    recipe = recipe_dict("check")
    panel, client, _feed = recipe_panel(qapp, recipe)
    panel.confirm = lambda title, question: False
    panel.catalogue.recipe_action_requested.emit("delete", recipe)
    assert commands(client, "delete_recipe") == []
    panel.confirm = lambda title, question: True
    panel.catalogue.recipe_action_requested.emit("delete", recipe)
    assert commands(client, "delete_recipe") == [
        {"experiment": SCHEMA["module"], "class_name": "Demo", "name": "check"}
    ]
    assert "Deleted recipe 'check'" in panel.status.text()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/test_experiment_gui.py -q -k "recipe"`
Expected: the panel tests fail (`catalogue.recipes` stays empty; no `ask_recipe_name`).

- [ ] **Step 3: Implement**

In `pytweezer/GUI/experiments/panel.py`:

1. Imports: add `QInputDialog` and `QMessageBox` to the `PyQt6.QtWidgets` import and `from pytweezer.experiment.recipes import Recipe`.

2. In `__init__`, after `self._catalogue_version = None`:

```python
        self._recipes_version = None
```

and with the other signal connections:

```python
        self.catalogue.recipe_selected.connect(self._recipe_selected)
        self.catalogue.recipe_action_requested.connect(self._recipe_action)
        self.editor.save_recipe_requested.connect(self._save_recipe)
```

3. At the end of `refresh_catalogue` (after `self.catalogue.set_modules(...)`), add `self.refresh_recipes()`, and add:

```python
    def refresh_recipes(self):
        reply = self._call("Listing recipes", self.client.call, "recipes")
        if reply is None:
            return
        self._recipes_version = reply.get("version")
        self.catalogue.set_recipes(reply["recipes"])
```

4. In `_queue_changed`, at the end:

```python
        recipes_version = snapshot.get("recipes_version")
        if recipes_version is not None and recipes_version != self._recipes_version:
            self.refresh_recipes()
```

5. Add a `# -- recipes --` section after `_submit`:

```python
    # -- recipes -------------------------------------------------------------

    def _recipe_selected(self, recipe):
        key = (recipe["experiment"], recipe["class_name"])
        schema = self.catalogue.schema_for(*key)
        if schema is None:
            return
        if key != self._current_key:
            self._save_draft()
            self._current_key = key
            self.editor.set_experiment(schema)
        self.editor.load_request(Recipe.model_validate(recipe))
        self.editor.set_recipe_name(recipe["name"])

    def _save_recipe(self, request):
        name = self.ask_recipe_name(self.editor.recipe_name).strip()
        if not name:
            return
        exists = any(
            (r["experiment"], r["class_name"], r["name"])
            == (request.experiment, request.class_name, name)
            for r in self.catalogue.recipes
        )
        if exists and not self.confirm(
            "Replace recipe", f"Replace the saved recipe '{name}' for every PC?"
        ):
            return
        recipe = Recipe(
            **request.model_dump(exclude={"submitter"}),
            name=name,
            submitter=submitter_name(),
        )
        reply = self._call(
            f"Saving recipe '{name}'",
            self.client.call,
            "save_recipe",
            recipe=recipe.model_dump(mode="json"),
            overwrite=exists,
        )
        if reply is None:
            return
        self.editor.set_recipe_name(name)
        self._show_status(f"Saved recipe '{name}'")
        self.refresh_recipes()

    def _recipe_action(self, action, recipe):
        name = recipe["name"]
        fields = {key: recipe[key] for key in ("experiment", "class_name", "name")}
        if action == "submit":
            reply = self._call(
                f"Submitting recipe '{name}'",
                self.client.call,
                "submit_recipe",
                submitter=submitter_name(),
                **fields,
            )
            if reply is not None:
                self._show_status(f"Queued task {reply['rid']} from recipe '{name}'")
        elif action == "delete":
            if not self.confirm(
                "Delete recipe",
                f"Delete the saved recipe '{name}' for {recipe['class_name']}? "
                "It is removed for every PC.",
            ):
                return
            reply = self._call(
                f"Deleting recipe '{name}'", self.client.call, "delete_recipe", **fields
            )
            if reply is not None:
                self._show_status(f"Deleted recipe '{name}'")
                self.refresh_recipes()

    def ask_recipe_name(self, default):
        name, ok = QInputDialog.getText(
            self, "Save as recipe", "Recipe name (shared with every PC):", text=default
        )
        return name if ok else ""

    def confirm(self, title, question):
        answer = QMessageBox.question(self, title, question)
        return answer == QMessageBox.StandardButton.Yes
```

Note: `_show_status(text)` with no state clears a previous "crashed" state, so a success after a failure reads normally.

6. In `.claude/skills/add-experiment/SKILL.md`, insert this section before `## Files`:

````markdown
## Recipes

A recipe is a named, saved form — arguments, scan, priority and label — kept by
the Experiment Manager in `{data_root}/recipes.json`, so every PC sees the same
ones. In the **Experiments** tab recipes sit under their experiment in the
tree: click one to load it into the form (the header names it), **Save as
recipe…** stores the form (asks before replacing a name), right-click → *Submit
now* queues it unchanged, *Delete…* removes it for everyone.

```python
from pytweezer.experiment.client import delete_recipe, recipes, save_recipe, submit_recipe

save_recipe(LoadingCurve, "nightly", scan, shots=2, label="nightly check")
rid = submit_recipe(LoadingCurve, "nightly", shots=3)  # overrides fixed arguments
recipes(LoadingCurve); delete_recipe(LoadingCurve, "nightly")
```

A recipe stores every argument value, not just changed ones. Submitting a
recipe that uses an argument the experiment no longer declares is refused with
the names; load it into the form, check it and save it again. An override may
not fix an argument the recipe scans. MOTMaster `attribute.parameter` names are
checked when the task runs, as for any submission.
````

- [ ] **Step 4: Run the full suite**

Run: `poetry run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: See it in the real GUI**

Load the `run-pytweezer` skill and follow it to launch `pytweezer-server` in simulation and take a screenshot of the Experiments tab with a saved recipe selected (save one through the GUI first). Check against the `pytweezer-gui-design` skill that the recipe item, header text and new button read clearly. Fix anything that doesn't, re-running `poetry run pytest tests/test_experiment_gui.py -q`.

- [ ] **Step 6: Lint, commit, push**

```bash
poetry run ruff check --fix pytweezer/GUI/experiments/panel.py tests/test_experiment_gui.py
poetry run ruff format pytweezer/GUI/experiments/panel.py tests/test_experiment_gui.py
git add pytweezer/GUI/experiments/panel.py tests/test_experiment_gui.py .claude/skills/add-experiment/SKILL.md
git commit -m "Load, save, submit and delete recipes from the Experiments tab

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 7: Open the pull request**

Every branch needs a merge request. The branch is based on `exp-handler`, which is not yet merged, so target `exp-handler`:

```bash
gh pr create --base exp-handler --title "Experiment recipes" --body "$(cat <<'EOF'
Named, shared, replayable run settings for experiments, replacing the idea of balic's Prep Station.

- Recipes kept by the Experiment Manager (`recipes.json`), visible on every PC
- Experiments tab: recipes under their experiment, load into the form, Save as recipe…, right-click Submit now / Delete
- Notebooks: `save_recipe`, `submit_recipe`, `recipes`, `delete_recipe`
- Submitting a recipe whose experiment has lost an argument is refused with the names

Spec: docs/superpowers/specs/2026-10-09-experiment-recipes-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
