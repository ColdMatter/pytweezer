import pytest

from pytweezer.experiment.saved_parameters import (
    SavedParameterBook,
    SavedParameterError,
    SavedParameterLog,
    SavedParameterStore,
)


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
