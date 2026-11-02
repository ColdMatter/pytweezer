import json

import numpy as np
import pytest

from pytweezer.experiment.client import (
    get_saved_parameter,
    saved_parameter_history,
    set_saved_parameters,
)
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


def test_an_acceptance_from_a_run_logs_an_unchanged_value(tmp_path):
    book, log = book_with(key=1.0), SavedParameterLog(tmp_path / "log.jsonl")
    log.append(book.set({"key": 1.0}, reason="accepted from run 7", who="me", rid=7))
    [change] = log.read(rid=7)
    assert (change.old, change.new) == (1.0, 1.0)
    assert book.get("key").reason == "r"


def test_the_log_reads_newest_first_up_to_the_limit(tmp_path):
    book, log = book_with(a=0, b=0), SavedParameterLog(tmp_path / "log.jsonl")
    for step in (1, 2, 3):
        log.append(book.set({"a": step, "b": step}, reason="r", who="me"))
    assert [c.new for c in log.read("a")] == [3, 2, 1]
    assert [(c.key, c.new) for c in log.read(limit=3)] == [("b", 3), ("a", 3), ("b", 2)]


def test_history_filters_by_run(tmp_path):
    book, log = book_with(a=0, b=0), SavedParameterLog(tmp_path / "log.jsonl")
    log.append(book.set({"a": 1}, reason="r", who="me", rid=7))
    log.append(book.set({"a": 2}, reason="r", who="me", rid=8))
    log.append(book.set({"b": 1}, reason="r", who="me", rid=7))
    assert [(c.key, c.new) for c in log.read(rid=7)] == [("b", 1), ("a", 1)]
    assert [c.new for c in log.read("a", rid=7)] == [1]


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
        book.set({"a": 1, "b": {"x": 1}}, reason="r", who="me", create=True)
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


def test_an_array_is_stored_as_nested_floats_and_reloads(tmp_path):
    book = book_with(sites=np.array([[1, 2], [3, 4]]))
    store = SavedParameterStore(tmp_path / "saved.json")
    store.save(book.state)

    value = SavedParameterBook(store.load()).get("sites").value

    assert value == [[1.0, 2.0], [3.0, 4.0]]
    assert all(type(x) is float for row in value for x in row)


@pytest.mark.parametrize(
    "new",
    [[[1.0, 2.0, 3.0]], [1.0, 2.0], 1.0],
    ids=["other columns", "other dimensions", "scalar"],
)
def test_an_array_key_keeps_its_columns_and_dimensions(new):
    book = book_with(sites=[[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(SavedParameterError):
        book.set({"sites": new}, reason="r", who="me")
    assert book.get("sites").value == [[1.0, 2.0], [3.0, 4.0]]


def test_an_array_key_takes_any_number_of_rows():
    book = book_with(sites=[[1.0, 2.0], [3.0, 4.0]])
    book.set({"sites": [[5.0, 6.0]] * 3}, reason="r", who="me")
    assert book.get("sites").value == [[5.0, 6.0]] * 3


def test_a_scalar_key_refuses_an_array():
    book = book_with(gain=1.0)
    with pytest.raises(SavedParameterError):
        book.set({"gain": [1.0]}, reason="r", who="me")


@pytest.mark.parametrize("create", [True, False])
def test_an_array_with_a_non_finite_element_is_refused(create):
    book = book_with(sites=[[1.0, 2.0]])
    with pytest.raises(SavedParameterError):
        book.set(
            {"new" if create else "sites": [[1.0, float("nan")]]},
            reason="r",
            who="me",
            create=create,
        )
    assert [p.key for p in book.all()] == ["sites"]
    assert book.get("sites").value == [[1.0, 2.0]]


def test_units_name_each_created_keys_unit():
    book = SavedParameterBook()
    book.set(
        {"a": 1.0, "b": [[1.0]]},
        reason="r",
        who="me",
        create=True,
        unit="x",
        units={"b": "px"},
    )
    assert (book.get("a").unit, book.get("b").unit) == ("x", "px")


def test_numpy_arrays_round_trip_through_the_manager(make_manager):
    client = in_process_client(make_manager())
    sites = np.array([[1.5, 2.0], [3.0, 4.25]])
    set_saved_parameters({"sites": sites}, "r", create=True, client=client)

    found = get_saved_parameter("sites", client=client)

    assert isinstance(found, np.ndarray)
    np.testing.assert_array_equal(found, sites)
