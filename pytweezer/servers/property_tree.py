"""Addressing and editing rules of the shared properties tree.

The tree is nested dicts addressed like a filesystem: ``"/Viewer/ROI/pos"`` is
absolute, while ``"ROI/pos"`` is relative to the owning process's name. A leaf
that is a dict with an ``"options"`` key is an *option property*: reads unwrap
it to its ``"value"``, and writes of a plain value only take effect if the value
is one of the options.

Edits are planned against a plain-dict view of the tree and returned as
operations, so the same rules drive both a client's local copy (a dict) and the
server's :class:`~sipyco.sync_struct.Notifier`, which broadcasts each operation.
"""

import copy
from typing import Any

from pytweezer.logging_utils import get_logger

logger = get_logger("pytweezer.servers.property_tree")

#: ``(path, key, value)``: set ``tree[path...][key] = value``, or delete it if
#: ``value is DELETE``.
Op = tuple[list[str], str, Any]
DELETE = object()


def parse_key(name: str, key: str | list[str]) -> list[str]:
    """The path of ``key``, relative to ``name`` unless it starts with ``/``."""
    if isinstance(key, list):
        keys = key
    else:
        if key[0] != "/":
            if key[-1] == "/":
                key = key[:-1]
            key = "/" + name + "/" + key
        keys = key.split("/")
    if keys[0] == "":
        keys = keys[1:]
    return keys


def is_option(node: Any) -> bool:
    return isinstance(node, dict) and "options" in node


def lookup(tree: dict, keys: list[str]) -> Any:
    """A deep copy of the entry at ``keys``, option properties unwrapped.

    Raises:
        KeyError: The entry does not exist.
    """
    node = tree
    for key in keys:
        node = node[key]
    value = copy.deepcopy(node)
    return value["value"] if is_option(value) else value


def plan_set(tree: dict, keys: list[str], value: Any) -> list[Op]:
    """Operations that set ``keys`` to ``value``, creating missing levels.

    Setting an option property to a value outside its options plans nothing.
    """
    ops: list[Op] = []
    node: Any = tree
    path: list[str] = []
    for key in keys[:-1]:
        if key in node:
            node = node[key]
        else:
            ops.append((path, key, {}))
            node = {}
        path = [*path, key]
    last = keys[-1]
    if not is_option(value) and is_option(node.get(last)):
        options = node[last]["options"]
        if value not in options:
            logger.warning(
                "%r is not one of %s's options %r; unchanged",
                value,
                "/" + "/".join(keys),
                options,
            )
            return []
        ops.append(([*path, last], "value", copy.deepcopy(value)))
    else:
        ops.append((path, last, copy.deepcopy(value)))
    return ops


def plan_delete(tree: dict, keys: list[str]) -> list[Op]:
    """Operations that delete the entry at ``keys``, if it exists."""
    node: Any = tree
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            return []
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        return []
    return [(keys[:-1], keys[-1], DELETE)]


def apply_ops(target: Any, ops: list[Op]) -> None:
    """Carry out ``ops`` on a dict or a :class:`~sipyco.sync_struct.Notifier`."""
    for path, key, value in ops:
        node = target
        for step in path:
            node = node[step]
        if value is DELETE:
            del node[key]
        else:
            node[key] = value
