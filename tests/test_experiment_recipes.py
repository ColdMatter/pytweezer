from datetime import datetime

import pytest

from pytweezer.experiment.queue import now
from pytweezer.experiment.recipes import (
    Recipe,
    RecipeBook,
    RecipeError,
    RecipeStore,
    unknown_arguments,
)
from pytweezer.experiment.scan import LinearAxis, ListAxis, Scan
from pytweezer.experiment.task import TaskRequest

DEMO = "pytweezer.experiments.demo"


def recipe(name="check", **kwargs):
    return Recipe(experiment=DEMO, class_name="RabiDemo", name=name, **kwargs)


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


def test_find_filters_and_sorts():
    book = RecipeBook()
    book.save(recipe(name="b"))
    book.save(recipe(name="a"))
    # The same name may be used for another experiment.
    book.save(Recipe(experiment="a.module", class_name="Z", name="a"))
    found = [(r.experiment, r.name) for r in book.find()]
    assert found == [("a.module", "a"), (DEMO, "a"), (DEMO, "b")]
    assert [r.name for r in book.find(DEMO, "RabiDemo")] == ["a", "b"]
    assert [r.name for r in book.find(DEMO)] == ["a", "b"]


def test_store_round_trip(tmp_path):
    path = tmp_path / "recipes.json"
    book = RecipeBook()
    book.save(
        recipe(scan=Scan(axes=[LinearAxis(argument="atoms", start=1, stop=3, n=3)]))
    )
    RecipeStore(path).save(book.state)
    assert RecipeStore(path).load() == book.state


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
    with pytest.raises(RecipeError, match=r"scans \['pulse_time'\]"):
        saved.replay({"pulse_time": 2e-6})


def test_replay_keeps_the_recipes_pipeline_unless_overridden():
    saved = recipe(pipeline="CaF")
    assert saved.replay().pipeline == "CaF"
    assert saved.replay(pipeline="Rb").pipeline == "Rb"


def test_unknown_arguments_lists_names_the_experiment_lacks():
    schema = {
        "arguments": {"atoms": {}},
        "devices": {
            "rb": {"device": "Rb MotMaster", "timeout": None, "motmaster": {}},
            "cam": {"device": "Cam", "timeout": None},
        },
    }
    request = TaskRequest(
        experiment="m",
        class_name="C",
        args=dict.fromkeys(
            ["atoms", "old", "rb.tPulse", "cam.exposure", "rb", "rb."], 1
        ),
        scan=Scan(axes=[ListAxis(argument="gone", values=[1])]),
    )
    expected = ["cam.exposure", "gone", "old", "rb", "rb."]
    assert unknown_arguments(request, schema) == expected
