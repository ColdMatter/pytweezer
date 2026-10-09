"""Recipes as plain data: the book's rules and the store."""

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
