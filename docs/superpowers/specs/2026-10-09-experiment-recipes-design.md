# Experiment recipes

## Goal

A recipe is a named, saved set of run settings for one experiment: argument
values, scan and queue settings. Recipes are shared by every PC, kept for weeks,
and replayed either by loading them into the Experiments form or by submitting
them directly, from the GUI or from a notebook.

This takes the useful idea from balic's Prep Station (store run settings, replay
them later) without its code or its staging-list workflow.

## Decisions

- Recipes are a library of known-good configurations, not a session staging
  list: no ordering, sleeping, bulk edit or Looper.
- The Experiment Manager owns them, so every client PC and notebook sees the
  same set.
- Replay is both "load into the form, then Submit" and "Submit now".
- Notebooks can save, list, delete and submit recipes by name.
- A recipe whose experiment has lost an argument it uses is refused on direct
  submission, with the offending names; it is never silently trimmed.
- Recipes appear under their experiment in the catalogue tree; there is no
  separate region.
- Out of scope: renaming (load, save under the new name, delete the old one),
  recording on a task which recipe it came from.

## 1. Data model

In `pytweezer/experiment/recipes.py`:

```python
class Recipe(TaskRequest):
    name: str
    saved_at: datetime
```

- A recipe is identified by `(experiment, class_name, name)`. Names are
  stripped of surrounding whitespace and must be non-empty.
- `due_time` is always cleared on save: a start time means nothing for a
  configuration replayed weeks later.
- `submitter` records who saved it (`user@host`); it is replaced by the actual
  submitter when the recipe is queued.
- A recipe stores exactly what the form (or `save_recipe`) produced: every
  argument value, not only those that differ from the defaults, so a later
  change of default does not change what a recipe runs.

`RecipeBook` holds the rules as plain data, like `ExperimentQueue`:

- `save(recipe, overwrite=False)` raises `RecipeError` if the key exists and
  `overwrite` is false.
- `get(experiment, class_name, name)` raises `KeyError` naming the recipe if
  missing.
- `delete(experiment, class_name, name)`.
- `list(experiment=None, class_name=None)`, sorted by experiment, class, name.

`RecipeState` (`schema_version`, `recipes: list[Recipe]`) is persisted by
`RecipeStore` to `recipes.json` in the manager's data directory, next to
`queue_state.json`, with the same atomic-replace and unreadable-file handling as
`QueueStore`. Recipes are kept out of `QueueState` so an unreadable queue file,
which is moved aside, never takes the recipes with it.

## 2. Manager commands

`ExperimentManager` gains a `RecipeBook`, saves it after every change, and
increments a `recipes_version` that is included in the published snapshot (as
`catalogue_version` is).

| command | fields | reply |
| --- | --- | --- |
| `recipes` | optional `experiment`, `class_name` | `recipes`, `version` |
| `save_recipe` | `recipe`, `overwrite` | — |
| `delete_recipe` | `experiment`, `class_name`, `name` | — |
| `submit_recipe` | `experiment`, `class_name`, `name`, optional `args`, `priority`, `label`, `submitter` | `rid` |

`submit_recipe` builds a `TaskRequest` from the recipe, applies the overrides,
checks it, and queues it. It refuses, naming the cause, when:

- the experiment class is not in the catalogue (or its module fails to import);
- the recipe's fixed or scanned arguments include names the experiment no
  longer declares. Arguments the experiment has gained take their defaults.
  Dotted names whose attribute is a declared MOTMaster (e.g.
  `rb.TweezerDepth`) are not checked here, since they depend on the device's
  script; they are checked when the task runs, as for any submission;
- an override names an argument the recipe scans.

## 3. Client API

In `pytweezer/experiment/client.py`, alongside `submit` and `wait`:

```python
rid = submit_recipe(Demo, "MOT check", detuning=-3.0, label="after realignment")
save_recipe(Demo, "MOT check", scan, priority=0, label="", **args)
recipes(Demo)            # -> list[Recipe]; recipes() lists all
delete_recipe(Demo, "MOT check")
```

- `experiment` is a class or a `"module:ClassName"` string, as for `submit`.
- `submit_recipe` keyword arguments are argument overrides; `priority` and
  `label` are keyword-only and override the recipe's only when given.
- `save_recipe` validates arguments and scan against the class the same way
  `submit` does, and takes `overwrite=False`.
- `ExperimentManagerClient` gets matching `recipes`, `save_recipe`,
  `delete_recipe` and `submit_recipe` methods.

## 4. GUI

All in the Experiments tab.

**Catalogue tree.** Each experiment class item gets its recipes as children,
shown with a distinct role so they read as saved settings rather than
experiments. The filter matches recipe names. The panel fetches recipes when
the catalogue loads and again whenever the snapshot's `recipes_version`
changes; selection is preserved across refreshes.

**Selecting a recipe** selects its experiment (saving the current draft as
now) and loads the recipe into the form with `load_request`. The form's header
names the loaded recipe until the experiment changes or the form is reset to
defaults or last submitted.

**Save as recipe…** is a new editor button beside "Last submitted". It asks for
a name, offering the loaded recipe's name if there is one. If that name exists
it asks before overwriting. The form is validated as for Submit; a form that
would not submit cannot be saved.

**Context menu on a recipe:** *Submit now* (calls `submit_recipe`; the status
bar shows "Queued task 42 from recipe 'MOT check'" or the refusal) and
*Delete…* (confirms first).

Wording and styling follow the `pytweezer-gui-design` skill.

## 5. Testing

- `tests/test_experiment_recipes.py`: `RecipeBook` rules (overwrite refusal,
  missing key, listing order, `due_time` cleared) and `RecipeStore`
  round-trip and unreadable-file handling.
- `tests/test_experiment_manager.py`: the four commands, `recipes_version` in
  the snapshot, and each `submit_recipe` refusal (unknown experiment, removed
  argument, removed scanned argument, override of a scanned argument), plus
  dotted MOTMaster names passing through.
- Client tests for `submit_recipe`/`save_recipe` argument handling.
- `tests/test_experiment_gui.py` (offscreen): recipes appear under their
  experiment and survive a refresh, selecting one fills the form, Save as
  recipe round-trips, Submit now reports the rid or the refusal.

## 6. Documentation

The `add-experiment` skill gains a short section on recipes: what they are,
the GUI actions and the notebook functions.
