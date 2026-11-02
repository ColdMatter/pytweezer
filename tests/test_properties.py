"""Properties clients against a real Properties server on loopback."""

import json
from types import SimpleNamespace

import pytest

from pytweezer.servers import properties, property_server
from pytweezer.servers.property_server import PropertyServer
from tests.helpers import until

HOST = "127.0.0.1"


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
    """A server that can be stopped and restarted on the same address;
    ``process()`` makes a client as a fresh process would."""
    monkeypatch.setattr(property_server, "SAVE_QUIET_S", 0.05)
    lab = SimpleNamespace(
        path=str(tmp_path / "properties.json"), server=None, address=(0, 0)
    )
    with open(lab.path, "w") as f:
        json.dump({"Saved": {"x": 1}}, f)
    mirrors = []

    def start():
        lab.server = PropertyServer(HOST, *lab.address, lab.path)
        lab.address = (lab.server.sync.port, lab.server.sync.rpc_port)

    def stop():
        lab.server.close()
        lab.server = None

    def process(name="Test"):
        monkeypatch.setattr(properties, "_connections", {})
        monkeypatch.setattr(properties, "server_address", lambda: (HOST, *lab.address))
        props = properties.Properties(name)
        mirrors.append(props._connection.mirror)
        return props

    def served(key):
        return lab.server.notifier.raw_view.get(key)

    lab.start, lab.stop, lab.process, lab.served = start, stop, process, served
    start()
    yield lab
    # Server first: a server closing after its mirrors hung up can raise
    # ConnectionResetError from SyncServer.close().
    if lab.server is not None:
        lab.server.close()
    for mirror in mirrors:
        mirror.close()


def test_client_starts_from_the_servers_tree_and_creates_its_entry(lab):
    p = lab.process("Viewer/Mon")
    assert p.get("/Saved/x") == 1
    assert until(lambda: lab.served("Viewer") == {"Mon": {}})


def test_edits_read_back_at_once_and_reach_other_processes(lab):
    a = lab.process("A")
    b = lab.process("B")
    b.changes()
    a.set("roi/pos", [1, 2])
    assert a.get("roi/pos") == [1, 2]
    assert until(lambda: peek(b, "A", "roi", "pos") == [1, 2])
    assert "/A/roi/pos" in b.changes(includeparent=False)

    b.delete("/A/roi")
    assert until(lambda: peek(a, "A", "roi") is None)


def test_edits_are_saved_to_the_file_and_non_json_refused(lab):
    a = lab.process("A")
    a.set("/bad", {1, 2})
    a.set("/Saved/y", [3])

    def saved():
        with open(lab.path) as f:
            return json.load(f)

    assert until(lambda: saved().get("Saved", {}).get("y") == [3])
    assert "bad" not in saved()
    assert lab.served("bad") is None


def test_clients_resync_after_a_server_restart(lab):
    a = lab.process("A")
    a.set("/kept", 1)
    assert until(lambda: lab.served("kept") == 1)
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
    assert until(lambda: lab.served("after") == 2)


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
