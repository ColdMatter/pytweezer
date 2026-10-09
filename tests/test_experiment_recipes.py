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
    unknown_arguments,
)
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest
from pytweezer.experiments.demo import RabiDemo

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
