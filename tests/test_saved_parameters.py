import json

import numpy as np
import pytest

from pytweezer.experiment.client import saved_parameter_history
from pytweezer.experiment.saved_parameters import (
    SavedParameterBook,
    SavedParameterError,
    SavedParameterLog,
    SavedParameterStore,
    import_parameters_json,
)
from tests.helpers import in_process_client


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
        1.0,
        2.0,
        "by hand",
        "me",
        7,
    )
    assert book.get("key").reason == "by hand"


def test_the_log_reads_newest_first_up_to_the_limit(tmp_path):
    book, log = book_with(a=0, b=0), SavedParameterLog(tmp_path / "log.jsonl")
    for step in (1, 2, 3):
        log.append(book.set({"a": step, "b": step}, reason="r", who="me"))
    assert [c.new for c in log.read("a")] == [3, 2, 1]
    assert [(c.key, c.new) for c in log.read(limit=3)] == [("b", 3), ("a", 3), ("b", 2)]


def test_a_saved_book_reloads_with_every_value_in_its_type(tmp_path):
    values = {"f": 4.0, "i": 4, "flag": True, "s": "4"}
    store = SavedParameterStore(tmp_path / "parameters.json")
    store.save(book_with(**values).state)
    loaded = {p.key: p.value for p in store.load().parameters}
    assert loaded == values
    assert {k: type(v) for k, v in loaded.items()} == {
        k: type(v) for k, v in values.items()
    }
    assert {p.reason for p in store.load().parameters} == {"r"}


def test_non_ascii_text_is_stored_and_logged_as_ascii_and_reads_back(tmp_path):
    book = SavedParameterBook()
    store = SavedParameterStore(tmp_path / "parameters.json")
    log = SavedParameterLog(tmp_path / "log.jsonl")
    reason = "detuning \u0394 and Rabi \u03a9"
    changes = book.set(
        {"f": 1.0}, reason=reason, who="me", create=True, group="\u03a9", unit="\u00b5s"
    )
    store.save(book.state)
    log.append(changes)
    assert store.path.read_bytes().isascii() and log.path.read_bytes().isascii()
    [parameter] = store.load().parameters
    assert (parameter.reason, parameter.group, parameter.unit) == (
        reason,
        "\u03a9",
        "\u00b5s",
    )
    assert [c.reason for c in log.read()] == [reason]


def test_a_log_line_that_cannot_be_read_is_skipped(tmp_path):
    book, log = book_with(a=0), SavedParameterLog(tmp_path / "log.jsonl")
    log.append(book.set({"a": 1}, reason="r", who="me"))
    with log.path.open("a") as file:
        file.write('{"time": "2026-10-30T10:00:00", "who": "m\n')
    log.append(book.set({"a": 2}, reason="r", who="me"))
    assert [c.new for c in log.read("a")] == [2, 1]


def test_a_creation_with_an_unstorable_value_creates_nothing():
    book = SavedParameterBook()
    with pytest.raises(SavedParameterError, match="b"):
        book.set({"a": 1, "b": [1]}, reason="r", who="me", create=True)
    assert book.all() == []


def test_two_keys_that_strip_to_the_same_name_are_refused():
    book = SavedParameterBook()
    with pytest.raises(SavedParameterError, match="'a'"):
        book.set({"a ": 1, "a": 2}, reason="r", who="me", create=True)
    assert book.all() == []


def test_the_import_copies_every_value_with_its_category_as_group(
    make_manager, tmp_path
):
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"first": {"a": 3.0, "b": 600}, "second": {"c": "x"}}))
    client = in_process_client(make_manager())
    assert import_parameters_json(path, client=client) == 3
    found = {p.key: (p.value, p.group) for p in client.saved_parameters()}
    assert found == {"a": (3.0, "first"), "b": (600, "first"), "c": ("x", "second")}
    assert import_parameters_json(path, client=client) == 3
    assert len(saved_parameter_history(client=client)) == 3


def test_numpy_scalars_are_stored_as_python_numbers(make_manager):
    client = in_process_client(make_manager())
    client.set_saved_parameters(
        {"i": np.int64(3), "f": np.float32(0.5), "flag": np.bool_(True)},
        "r",
        "me",
        create=True,
    )
    found = {p.key: p.value for p in client.saved_parameters()}
    assert found == {"i": 3, "f": 0.5, "flag": True}
    assert {k: type(v) for k, v in found.items()} == {
        "i": int,
        "f": float,
        "flag": bool,
    }
