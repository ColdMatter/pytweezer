"""SyncServer/SyncMirror over real loopback sockets."""

import threading
import time

import pytest
from sipyco.pc_rpc import Client as RPCClient
from sipyco.sync_struct import Notifier

from pytweezer.servers.sync import SyncMirror, SyncServer

HOST = "127.0.0.1"


def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def served():
    """A server publishing ``{"items": [], "values": {}}`` as ``"state"``."""
    notifier = Notifier({"items": [], "values": {}})
    server = SyncServer({"state": notifier}, HOST, 0)
    mirrors = []

    def mirror(**kwargs):
        m = SyncMirror(HOST, server.port, "state", **kwargs)
        mirrors.append(m)
        assert m.wait_initialised(5)
        return m

    yield server, notifier, mirror
    for m in mirrors:
        m.close()
    server.close()


def test_mirror_gets_snapshot_then_changes(served):
    server, notifier, mirror = served
    server.call(notifier["items"].append, 1).result()
    m = mirror()
    assert m.snapshot() == {"items": [1], "values": {}}

    def change():
        notifier["items"].append(2)
        notifier["values"]["a"] = {"nested": [1, 2]}
        notifier["values"]["a"]["nested"].append(3)

    server.call(change).result()
    assert until(lambda: m.snapshot()["values"].get("a") == {"nested": [1, 2, 3]})
    assert m.snapshot()["items"] == [1, 2]


def test_late_joiner_sees_current_state(served):
    server, notifier, mirror = served
    early = mirror()
    server.call(notifier["values"].__setitem__, "x", 5).result()
    assert until(lambda: early.snapshot()["values"] == {"x": 5})
    late = mirror()
    assert late.snapshot()["values"] == {"x": 5}


def test_mirror_never_shares_objects_with_later_mods(served):
    server, notifier, mirror = served
    m = mirror()
    server.call(notifier["values"].__setitem__, "list", []).result()
    for i in range(3):
        server.call(notifier["values"]["list"].append, i).result()
    assert until(lambda: m.snapshot()["values"]["list"] == [0, 1, 2])
    time.sleep(0.1)
    assert m.snapshot()["values"]["list"] == [0, 1, 2]


def test_mutations_from_many_threads_stay_consistent(served):
    server, notifier, mirror = served
    m = mirror()

    def worker(offset):
        for i in range(50):
            server.call(notifier["items"].append, offset + i).result()

    threads = [threading.Thread(target=worker, args=(1000 * t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert until(lambda: len(m.snapshot()["items"]) == 200)
    assert m.snapshot()["items"] == notifier.raw_view["items"]


def test_mirror_reconnects_after_server_restart(served):
    server, notifier, mirror = served
    m = mirror(keep_mods=True)
    port = server.port
    server.close()
    assert until(lambda: not m.connected)
    m.drain()

    restarted = SyncServer(
        {"state": Notifier({"items": ["new"], "values": {}})}, HOST, port
    )
    try:
        assert until(lambda: m.connected)
        assert m.snapshot()["items"] == ["new"]
        assert [mod["action"] for mod in m.drain()] == ["init"]
    finally:
        restarted.close()


def test_mirror_waits_for_a_server_that_starts_later():
    notifier = Notifier({"n": 0})
    probe = SyncServer({"state": notifier}, HOST, 0)
    port = probe.port
    probe.close()

    m = SyncMirror(HOST, port, "state")
    try:
        assert not m.wait_initialised(0.3)
        assert m.snapshot() is None
        server = SyncServer({"state": notifier}, HOST, port)
        try:
            assert m.wait_initialised(5)
            assert m.snapshot() == {"n": 0}
        finally:
            server.close()
    finally:
        m.close()


def test_on_mod_sees_every_mod(served):
    server, notifier, mirror = served
    seen = []
    m = mirror(on_mod=lambda mod, data: seen.append(mod))
    server.call(notifier["values"].__setitem__, "k", 1).result()
    assert until(lambda: len(seen) == 2)
    assert seen[0]["action"] == "init"
    assert seen[1] == {"action": "setitem", "path": ["values"], "key": "k", "value": 1}
    assert m.connected


def test_rpc_targets_run_on_the_loop_and_publish():
    notifier = Notifier({})

    class Target:
        def put(self, key, value):
            notifier[key] = value

    server = SyncServer({"state": notifier}, HOST, 0, {"target": Target()}, 0)
    m = SyncMirror(HOST, server.port, "state")
    client = RPCClient(HOST, server.rpc_port, "target")
    try:
        assert m.wait_initialised(5)
        client.put("a", [1, 2])
        assert until(lambda: m.snapshot() == {"a": [1, 2]})
    finally:
        client.close_rpc()
        m.close()
        server.close()
