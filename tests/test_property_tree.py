"""Addressing and editing rules of the properties tree."""

import pytest
from sipyco.sync_struct import Notifier

from pytweezer.servers.property_tree import (
    apply_ops,
    lookup,
    parse_key,
    plan_delete,
    plan_set,
)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("/A/b/c", ["A", "b", "c"]),
        ("roi/pos", ["Viewer", "Mon", "roi", "pos"]),
        ("roi/", ["Viewer", "Mon", "roi"]),
        (["x", "y"], ["x", "y"]),
        (["", "x"], ["x"]),
    ],
)
def test_parse_key(key, expected):
    assert parse_key("Viewer/Mon", key) == expected


def set_(tree, keys, value):
    apply_ops(tree, plan_set(tree, keys, value))


def test_set_creates_missing_levels_and_copies_the_value():
    tree = {}
    value = [1, 2]
    set_(tree, ["a", "b", "c"], value)
    value.append(3)
    assert tree == {"a": {"b": {"c": [1, 2]}}}


def test_set_overwrites_a_plain_leaf():
    tree = {"a": {"x": 1}}
    set_(tree, ["a", "x"], {"replaced": True})
    assert tree == {"a": {"x": {"replaced": True}}}


def test_option_property_only_accepts_its_options():
    option = {"options": ["red", "blue"], "value": "red"}
    tree = {}
    set_(tree, ["colour"], option)
    assert lookup(tree, ["colour"]) == "red"

    set_(tree, ["colour"], "blue")
    assert tree["colour"] == {"options": ["red", "blue"], "value": "blue"}
    assert plan_set(tree, ["colour"], "green") == []

    set_(tree, ["colour"], {"options": ["green"], "value": "green"})
    assert lookup(tree, ["colour"]) == "green"


def test_lookup_returns_a_copy_and_raises_for_missing():
    tree = {"a": {"b": [1]}}
    lookup(tree, ["a", "b"]).append(2)
    assert tree["a"]["b"] == [1]
    with pytest.raises(KeyError):
        lookup(tree, ["a", "missing"])


def test_delete_only_existing_entries():
    tree = {"a": {"b": 1, "c": 2}}
    apply_ops(tree, plan_delete(tree, ["a", "b"]))
    assert tree == {"a": {"c": 2}}
    assert plan_delete(tree, ["a", "b"]) == []
    assert plan_delete(tree, ["nope", "b"]) == []


def test_ops_drive_a_notifier_and_publish_each_change():
    notifier = Notifier({"a": {}})
    mods = []
    notifier.publish = mods.append
    apply_ops(notifier, plan_set(notifier.raw_view, ["a", "b", "c"], 5))
    apply_ops(notifier, plan_delete(notifier.raw_view, ["a", "b", "c"]))
    assert notifier.raw_view == {"a": {"b": {}}}
    assert [(m["action"], m["path"], m["key"]) for m in mods] == [
        ("setitem", ["a"], "b"),
        ("setitem", ["a", "b"], "c"),
        ("delitem", ["a", "b"], "c"),
    ]
