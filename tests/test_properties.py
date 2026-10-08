"""Properties clients against a real Properties server on loopback."""

import json
import time

import pytest

from pytweezer.servers import properties, property_server
from pytweezer.servers.property_server import PropertyServer

HOST = "127.0.0.1"


def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def peek(props, *keys):
    """Read a mirror without get()'s create-if-missing."""

    def walk(tree):
        for key in keys:
            if not isinstance(tree, dict) or key not in tree:
                return None
            tree = tree[key]
        return tree

    return props._connection.read(walk)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """Start/stop a server on one address; ``process()`` makes a fresh client process."""
    monkeypatch.setattr(property_server, "SAVE_QUIET_S", 0.05)
    path = str(tmp_path / "properties.json")
    with open(path, "w") as f:
        json.dump({"Saved": {"x": 1}}, f)
    state = {"server": None, "address": None, "connections": []}

    def start():
        port, rpc_port = state["address"] or (0, 0)
        server = PropertyServer(HOST, port, rpc_port, path)
        state["server"] = server
        state["address"] = (server.sync.port, server.sync.rpc_port)
        return server

    def stop():
        state["server"].close()
        state["server"] = None

    def process(name="Test"):
        # A new _connections dict is what a separate process would have.
        monkeypatch.setattr(properties, "_connections", {})
        port, rpc_port = state["address"]
        monkeypatch.setattr(
            properties, "server_address", lambda: (HOST, port, rpc_port)
        )
        props = properties.Properties(name)
        state["connections"].append(props._connection)
        return props

    start()
    yield type(
        "Lab",
        (),
        {
            "start": start,
            "stop": stop,
            "process": process,
            "path": path,
            "state": state,
        },
    )
    for connection in state["connections"]:
        connection.mirror.close()
    if state["server"] is not None:
        state["server"].close()


def test_client_starts_from_the_servers_tree_and_creates_its_entry(lab):
    p = lab.process("Viewer/Mon")
    assert p.get("/Saved/x") == 1
    assert until(
        lambda: lab.state["server"].notifier.raw_view.get("Viewer") == {"Mon": {}}
    )


def test_set_reads_back_at_once_and_reaches_other_processes(lab):
    a = lab.process("A")
    b = lab.process("B")
    a.set("roi/pos", [1, 2])
    assert a.get("roi/pos") == [1, 2]
    assert until(lambda: peek(b, "A", "roi", "pos") == [1, 2])

    b.delete("/A/roi")
    assert until(lambda: peek(a, "A", "roi") is None)


def test_late_process_gets_earlier_changes(lab):
    a = lab.process("A")
    a.set("/Shared/n", 5)
    assert until(
        lambda: lab.state["server"].notifier.raw_view.get("Shared") == {"n": 5}
    )
    late = lab.process("Late")
    assert late.get("/Shared/n") == 5


def test_changes_report_edits_from_other_processes(lab):
    a = lab.process("A")
    b = lab.process("B")
    b.changes()
    a.set("/Shared/deep/value", 3)
    assert until(lambda: peek(b, "Shared", "deep", "value") == 3)
    assert "/Shared/deep/value" in b.changes(includeparent=False)


def test_option_properties_are_enforced_by_the_server(lab):
    a = lab.process("A")
    b = lab.process("B")
    a.set("/mode", {"options": ["slow", "fast"], "value": "slow"})
    assert until(lambda: peek(b, "mode") is not None)
    b.set("/mode", "fast")
    assert until(lambda: a.get("/mode") == "fast")
    b.changes()
    b.set("/mode", "warp")
    assert b.get("/mode") == "fast"
    assert "/mode" not in b.changes()


def test_non_json_values_are_refused(lab):
    a = lab.process("A")
    a.set("/bad", {1, 2})
    a.set("/good", 1)
    assert until(lambda: lab.state["server"].notifier.raw_view.get("good") == 1)
    assert "bad" not in lab.state["server"].notifier.raw_view


def test_edits_are_saved_to_the_file(lab):
    a = lab.process("A")
    a.set("/Saved/y", [3])

    def saved():
        with open(lab.path) as f:
            return json.load(f).get("Saved", {}).get("y") == [3]

    assert until(saved)


def test_clients_resync_after_a_server_restart(lab):
    a = lab.process("A")
    a.set("/kept", 1)
    assert until(lambda: lab.state["server"].notifier.raw_view.get("kept") == 1)
    lab.stop()
    assert until(lambda: not a._connection.mirror.connected)

    a.set("/lost", 1)
    assert a.get("/lost") == 1

    lab.start()
    assert until(lambda: a._connection.mirror.connected)
    assert peek(a, "kept") == 1
    assert peek(a, "lost") is None
    assert "/" in a.changes()

    a.set("/after", 2)
    assert until(lambda: lab.state["server"].notifier.raw_view.get("after") == 2)


def test_unreachable_server_falls_back_to_the_file(lab, monkeypatch):
    lab.stop()
    monkeypatch.setattr(properties, "INIT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(properties, "load_properties", lambda: {"From": {"file": 1}})
    p = lab.process("A")
    assert p.get("/From/file") == 1
    p.set("/local", 2)
    assert p.get("/local") == 2

    lab.start()
    assert until(lambda: p._connection.mirror.connected)
    assert peek(p, "From") is None
    assert peek(p, "Saved", "x") == 1
