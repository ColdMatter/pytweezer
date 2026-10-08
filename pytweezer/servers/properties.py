"""Client side of the shared properties tree.

Every :class:`Properties` in a process shares one live mirror of the tree held
by the Properties server (:mod:`pytweezer.servers.property_server`), so reads
are local and never block on the network. Writes apply to the local mirror at
once and are then sent to the server, which broadcasts them to every mirror.

While the server is unreachable, reads see the last state the mirror held (or
the saved file, if it never connected) and writes are applied locally only and
dropped with a warning: when the mirror reconnects, the server's tree replaces
the local one, so a stale edit never overwrites newer shared state.
"""

import copy
import queue
import threading
import time
import weakref

from sipyco.pc_rpc import Client as RPCClient

from pytweezer.configuration.config import get_config
from pytweezer.configuration.paths import load_properties
from pytweezer.logging_utils import get_logger
from pytweezer.servers.property_tree import (
    apply_ops,
    is_option,
    lookup,
    parse_key,
    plan_delete,
    plan_set,
)
from pytweezer.servers.sync import SyncMirror

_logger = get_logger("Properties")

SERVER_NAME = "Properties"
#: How long a new process waits for the server's tree before using the file.
INIT_TIMEOUT_S = 3.0
RPC_TIMEOUT_S = 2.0
DROP_WARNING_INTERVAL_S = 30.0


class PropertyAttribute:
    """
    access properties as Attributes.
    The class object using this attribute must have an attribute called _props of type Properties.
    The value property can be used to do basic manipulation of the property e.g. get, set, addition
    att = PropertyAttribute(propname, defaultval, parent=self)
    exmaples:
    getting: x = att.value. gets the property value and sets x with it
    addition: att.value += 1. this changes both the property and self.att
    indexing: element_2 = att.value[2]. this gets the second element of the property
    att.value.append() doesn't work! do:
    lst = att.value
    lst.append()
    att.value = lst

    x = att gets the property value and sets x with it
    att = x sets the property value BUT OVERWRITES att (i.e. if x is an int, att will also become an int),
    so it only works once. Therefore, using att.value is preferred.
    """

    def __init__(self, propname, defaultval, parent=None):
        """init

        Args:
            propname (str): name of the property the attribut should access
            defaultval : default value of the property in case it doesn't exist yet
                        can be any type a dictionary entry can be (hashable)
                        (needs to be compatible with json)
            parent: usually self. the object which has _props. required for the value property to function
        # TODO: pass this a props object instead. remove __get__ and __set__, only allow changing via .value for code readability. This change would mean every usage of a propertyattribute would need to change, so it's a big job
        """
        self._default = defaultval
        self._propname = propname
        self.parent = parent
        self._value = defaultval

    @property
    def value(self):
        return self._value

    @value.getter
    def value(self):
        self._value = self.parent._props.get(self._propname, self._default)
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.parent._props.set(self._propname, value)

    # these work but I've disabled them because they could be confusing
    # def append(self, value):
    #     self._value.append(value)
    #     self.parent._props.set(self._propname, self._value)
    #
    # def extend(self, values):
    #     self._value.extend(values)
    #     self.parent._props.set(self._propname, self._value)

    def __get__(self, obj, t):
        return obj._props.get(self._propname, self._default)

    def __set__(self, obj, v):
        obj._props.set(self._propname, v)


def server_address(server_name: str = SERVER_NAME) -> tuple[str, int, int]:
    """``(host, port, rpc_port)`` of the Properties server, from CONFIG."""
    conf = get_config()["Servers"][server_name]
    return conf["host"], conf["port"], conf["rpc_port"]


class _Connection:
    """This process's link to one Properties server, shared by its Properties."""

    def __init__(self, host: str, port: int, rpc_port: int):
        self.host = host
        self.rpc_port = rpc_port
        self._listeners = weakref.WeakSet()
        self._listeners_lock = threading.Lock()
        self._fallback = None
        #: Bumped on every (re)sync; an RPC connection from an older one may be
        #: to a server that has since restarted.
        self._generation = 0
        self._writes = queue.SimpleQueue()
        self._dropped = 0
        self._last_drop_warning = 0.0
        self.mirror = SyncMirror(host, port, "properties", on_mod=self._on_mod)
        if not self.mirror.wait_initialised(INIT_TIMEOUT_S):
            _logger.warning(
                "Properties server %s:%s unreachable; starting from the saved file",
                host,
                port,
            )
            self.mirror.read(self._use_file_if_unsynced)
        threading.Thread(
            target=self._write_loop, name="properties-writer", daemon=True
        ).start()

    def _use_file_if_unsynced(self, data) -> None:
        if data is None:
            self._fallback = load_properties()

    def add_listener(self, properties: "Properties") -> None:
        with self._listeners_lock:
            self._listeners.add(properties)

    def read(self, fn):
        """``fn(tree)`` under the mirror's lock; ``fn`` may edit the tree locally."""
        return self.mirror.read(
            lambda data: fn(self._fallback if data is None else data)
        )

    def send(self, method: str, *args) -> None:
        self._writes.put((method, copy.deepcopy(args)))

    def _on_mod(self, mod: dict, tree: dict) -> None:
        if mod["action"] == "init":
            self._fallback = None
            self._generation += 1
            changed = {"/", *("/" + key for key in tree)}
        else:
            changed = {_changed_key(tree, mod)}
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener._note_changes(changed)

    def _write_loop(self) -> None:
        client = None
        client_generation = None
        while True:
            method, args = self._writes.get()
            if not self.mirror.connected:
                self._drop(method, args)
                continue
            if client is not None and client_generation != self._generation:
                client.close_rpc()
                client = None
            try:
                if client is None:
                    client_generation = self._generation
                    client = RPCClient(
                        self.host, self.rpc_port, "properties", timeout=RPC_TIMEOUT_S
                    )
                getattr(client, method)(*args)
            except (OSError, EOFError):
                client = _closed(client)
                self._drop(method, args)
            except Exception:
                client = _closed(client)
                _logger.exception("Properties server refused %s%r", method, args)

    def _drop(self, method: str, args: tuple) -> None:
        self._dropped += 1
        now = time.monotonic()
        if now - self._last_drop_warning > DROP_WARNING_INTERVAL_S:
            self._last_drop_warning = now
            _logger.warning(
                "Properties server unreachable: %d edit(s) kept locally only "
                "(latest: %s %s)",
                self._dropped,
                method,
                "/" + "/".join(args[0]),
            )
            self._dropped = 0


def _closed(client) -> None:
    if client is not None:
        try:
            client.close_rpc()
        except OSError:
            pass


def _changed_key(tree, mod: dict) -> str:
    path = list(mod["path"])
    if mod["action"] == "setitem":
        node = tree
        for step in path:
            node = node[step]
        # An option property's value changing is a change to the property.
        if not (mod["key"] == "value" and is_option(node)):
            path.append(mod["key"])
    return "/" + "/".join(path)


_connections: dict[tuple[str, int, int], _Connection] = {}
_connections_lock = threading.Lock()


def _connection(address: tuple[str, int, int]) -> _Connection:
    with _connections_lock:
        if address not in _connections:
            _connections[address] = _Connection(*address)
        return _connections[address]


class Properties:
    """A process's handle on the shared properties tree.

    Args:
        name: Name of the owning process; keys not starting with ``/`` are
            relative to it, and its own entry is created if missing.
        initfromfile: Ignored; kept so existing callers still work.
    """

    def __init__(self, name, initfromfile=False):
        self.name = name
        self.recent_changes = set()
        self._changes_lock = threading.Lock()
        self._connection = _connection(server_address())
        self._connection.add_listener(self)
        if not self._connection.read(lambda tree: name in tree):
            self.get("/" + name, {})

    def _note_changes(self, keys) -> None:
        with self._changes_lock:
            self.recent_changes.update(keys)

    def set(self, key, value):
        """Set a property.

        Args:
            key (str or list of str): Path of the property, filesystem-style:
                from the root if it starts with ``/``, else from this
                process's own entry. Missing levels are created.
            value: Any JSON-serialisable value.
        """
        keys = parse_key(self.name, key)

        def edit(tree):
            try:
                if lookup(tree, keys) == value:
                    return False
            except KeyError:
                pass
            ops = plan_set(tree, keys, value)
            apply_ops(tree, ops)
            return bool(ops)

        if self._connection.read(edit):
            self._note_changes({"/" + "/".join(keys)})
            self._connection.send("set", keys, value)

    def delete(self, key):
        """Delete an entry and everything below it."""
        keys = parse_key(self.name, key)
        self._connection.read(lambda tree: apply_ops(tree, plan_delete(tree, keys)))
        self._note_changes({"/" + "/".join(keys[:-1])})
        self._connection.send("delete", keys)

    def get(self, key, defaultvalue=None):
        """A deep copy of a property's value.

        Addressing is as for :meth:`set`. If the property does not exist yet it
        is created with ``defaultvalue``, so always give a sensible default:
        that is how the tree builds and maintains itself.
        """
        if key == "/":
            return self._connection.read(copy.deepcopy)
        if key[-1] == "/":
            key = key[:-1]
        keys = parse_key(self.name, key)
        try:
            return self._connection.read(lambda tree: lookup(tree, keys))
        except KeyError:
            _logger.debug("Creating %s with its default", "/" + "/".join(keys))
            self.set(keys, defaultvalue)
            if is_option(defaultvalue):
                return copy.deepcopy(defaultvalue["value"])
            return copy.deepcopy(defaultvalue)

    def changes(self, includeparent=True):
        """Keys of all entries changed since the last call.

        Args:
            includeparent (bool): Also include every parent of a changed key.

        Returns:
            set of str
        """
        with self._changes_lock:
            changes, self.recent_changes = self.recent_changes, set()
        if not includeparent:
            return changes
        with_parents = set()
        for key in changes:
            keys = key[1:].split("/")
            for i in range(len(keys)):
                with_parents.add("/" + "/".join(keys[: i + 1]))
        return with_parents
