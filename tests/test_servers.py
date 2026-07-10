"""Unit tests for the device server/client framework and the device-status
service. Everything here runs without real hardware, RPC, or ZMQ: config
lookups are stubbed, the RPC client is monkeypatched, and reachability probes
either hit a local throwaway socket or are stubbed out.
"""

import socket

import pytest

from pytweezer.servers import device_client, device_server
from pytweezer.servers.device_server import DeviceServerSpec
from pytweezer.servers.reachability import is_reachable


# --------------------------------------------------------------------------- #
# device_server: name resolution
# --------------------------------------------------------------------------- #

_FAKE_DEVICES = {
    "Devices": {
        "Rb HamCam": {"driver": "imagemx2", "host": "1.2.3.4", "port": 5000},
        "CaF MotMaster Server": {"driver": "motmaster", "host": "1.2.3.5", "port": 6000},
    }
}


@pytest.fixture
def fake_config(monkeypatch):
    monkeypatch.setattr(
        device_server.ConfigReader,
        "getConfiguration",
        staticmethod(lambda: _FAKE_DEVICES),
    )
    return _FAKE_DEVICES


def test_normalize_collapses_whitespace_and_case():
    assert device_server._normalize("Rb HamCam") == "rbhamcam"
    assert device_server._normalize("  RB   HAMCAM ") == "rbhamcam"


def test_resolve_device_exact_match(fake_config):
    name, conf = device_server.resolve_device("Rb HamCam")
    assert name == "Rb HamCam"
    assert conf["driver"] == "imagemx2"


@pytest.mark.parametrize("query", ["rbhamcam", "RB HAMCAM", "  rb   hamcam  "])
def test_resolve_device_lenient_match(fake_config, query):
    name, _ = device_server.resolve_device(query)
    assert name == "Rb HamCam"


def test_resolve_device_unknown_lists_available(fake_config):
    with pytest.raises(KeyError) as exc:
        device_server.resolve_device("nope")
    # The error message names the valid devices so a typo fails actionably.
    assert "Rb HamCam" in str(exc.value)
    assert "CaF MotMaster Server" in str(exc.value)


# --------------------------------------------------------------------------- #
# device_server: build_spec error handling
# --------------------------------------------------------------------------- #

def test_build_spec_missing_driver_key():
    with pytest.raises(KeyError, match="no 'driver' key"):
        device_server.build_spec("thing", conf={"host": "x", "port": 1})


def test_build_spec_unknown_driver():
    with pytest.raises(KeyError, match="Unknown driver"):
        device_server.build_spec("thing", conf={"driver": "does-not-exist"})


def test_build_spec_dispatches_to_registered_factory(monkeypatch):
    calls = {}

    def fake_factory(name, conf):
        calls["args"] = (name, conf)
        return DeviceServerSpec(target_name="t", target=object(), description="d")

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake", fake_factory)
    spec = device_server.build_spec("thing", conf={"driver": "fake", "x": 1})
    assert calls["args"] == ("thing", {"driver": "fake", "x": 1})
    assert spec.target_name == "t"


# --------------------------------------------------------------------------- #
# device_client
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_client_config(monkeypatch):
    monkeypatch.setattr(
        device_client.ConfigReader,
        "getConfiguration",
        staticmethod(lambda: _FAKE_DEVICES),
    )


def test_get_device_config_found(fake_client_config):
    assert device_client.get_device_config("Rb HamCam")["port"] == 5000


def test_get_device_config_unknown_lists_available(fake_client_config):
    with pytest.raises(KeyError) as exc:
        device_client.get_device_config("missing")
    assert "Rb HamCam" in str(exc.value)


@pytest.mark.parametrize("query", ["rbhamcam", "RB HAMCAM", "  rb   hamcam  "])
def test_get_device_config_lenient_match(fake_client_config, query):
    assert device_client.get_device_config(query)["port"] == 5000


def test_get_device_uses_config_host_and_port(fake_client_config, monkeypatch):
    captured = {}

    def fake_rpc_client(host, port, target_name=None, timeout=None):
        captured.update(host=host, port=port, target_name=target_name, timeout=timeout)
        return "CLIENT"

    monkeypatch.setattr(device_client, "RPCClient", fake_rpc_client)
    client = device_client.get_device("Rb HamCam")
    assert client == "CLIENT"
    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 5000


def test_get_device_lenient_match(fake_client_config, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        device_client,
        "RPCClient",
        lambda host, port, target_name=None, timeout=None: captured.update(
            host=host, port=port
        ),
    )
    device_client.get_device("rb hamcam")
    assert captured == {"host": "1.2.3.4", "port": 5000}


def test_get_device_overrides_win_over_config(fake_client_config, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        device_client,
        "RPCClient",
        lambda host, port, target_name=None, timeout=None: captured.update(
            host=host, port=port
        ),
    )
    device_client.get_device("Rb HamCam", host="9.9.9.9", port=1234)
    assert captured == {"host": "9.9.9.9", "port": 1234}


def test_get_device_no_port_raises(monkeypatch):
    monkeypatch.setattr(
        device_client.ConfigReader,
        "getConfiguration",
        staticmethod(lambda: {"Devices": {"NoPort": {"host": "1.2.3.4"}}}),
    )
    with pytest.raises(ValueError, match="no 'port' configured"):
        device_client.get_device("NoPort")


def test_get_device_async_uses_config_host_and_port(fake_client_config, monkeypatch):
    import asyncio

    captured = {}

    class FakeAsyncioClient:
        async def connect_rpc(self, host, port, target_name=None):
            captured.update(host=host, port=port, target_name=target_name)

    monkeypatch.setattr(device_client, "AsyncioClient", FakeAsyncioClient)
    client = asyncio.run(device_client.get_device_async("Rb HamCam"))
    assert isinstance(client, FakeAsyncioClient)
    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 5000


def test_get_device_async_no_port_raises(monkeypatch):
    monkeypatch.setattr(
        device_client.ConfigReader,
        "getConfiguration",
        staticmethod(lambda: {"Devices": {"NoPort": {"host": "1.2.3.4"}}}),
    )
    import asyncio

    with pytest.raises(ValueError, match="no 'port' configured"):
        asyncio.run(device_client.get_device_async("NoPort"))


# --------------------------------------------------------------------------- #
# device_client: composite sub-devices are addressed by their own names
# --------------------------------------------------------------------------- #

_COMPOSITE_DEVICES = {
    "Devices": {
        "Rb HamCam": {"driver": "imagemx2", "host": "1.2.3.4", "port": 5000},
        "Rb Feedback Rig": {
            "driver": "composite",
            "host": "1.2.3.9",
            "port": 6000,
            "devices": {
                "Rb Feedback Cam": {"driver": "imagemx2", "role": "camera"},
                "Rb Feedback DAC": {"driver": "nidac", "role": "dac"},
            },
            "coordinator": "camera_dac_feedback",
        },
    }
}


@pytest.fixture
def composite_client_config(monkeypatch):
    monkeypatch.setattr(
        device_client.ConfigReader,
        "getConfiguration",
        staticmethod(lambda: _COMPOSITE_DEVICES),
    )


@pytest.fixture
def captured_rpc(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        device_client,
        "RPCClient",
        lambda host, port, target_name=None, timeout=None: captured.update(
            host=host, port=port, target_name=target_name
        ),
    )
    return captured


@pytest.mark.parametrize(
    "name, target",
    [("Rb Feedback Cam", "rbfeedbackcam"), ("Rb Feedback DAC", "rbfeedbackdac")],
)
def test_get_device_sub_device_uses_composite_endpoint(
    composite_client_config, captured_rpc, name, target
):
    device_client.get_device(name)
    # host/port come from the composite that serves it; the target selects the device.
    assert captured_rpc == {"host": "1.2.3.9", "port": 6000, "target_name": target}


def test_get_device_composite_resolves_to_coordinator(composite_client_config, captured_rpc):
    device_client.get_device("Rb Feedback Rig")
    assert captured_rpc["target_name"] == "coordinator"


def test_get_device_plain_device_still_uses_autotarget(composite_client_config, captured_rpc):
    device_client.get_device("Rb HamCam")
    assert captured_rpc["target_name"] is device_client.AutoTarget
    assert captured_rpc["port"] == 5000


def test_get_device_explicit_target_name_wins(composite_client_config, captured_rpc):
    device_client.get_device("Rb Feedback Rig", target_name="rbfeedbackcam")
    assert captured_rpc["target_name"] == "rbfeedbackcam"


def test_get_device_config_finds_sub_device(composite_client_config):
    assert device_client.get_device_config("Rb Feedback DAC")["role"] == "dac"


def test_get_device_coordinatorless_composite_names_its_sub_devices(monkeypatch, captured_rpc):
    conf = {
        "Devices": {
            "Rig": {
                "driver": "composite",
                "host": "h",
                "port": 1,
                "devices": {"Cam": {"driver": "imagemx2"}},
            }
        }
    }
    monkeypatch.setattr(
        device_client.ConfigReader, "getConfiguration", staticmethod(lambda: conf)
    )
    with pytest.raises(KeyError) as exc:
        device_client.get_device("Rig")
    assert "has no coordinator" in str(exc.value)
    assert "Cam" in str(exc.value)


# --------------------------------------------------------------------------- #
# device_status.build_snapshot (state machine, is_reachable stubbed)
# --------------------------------------------------------------------------- #

def _status_server(devices, reachable, monkeypatch):
    from pytweezer.servers import device_status

    monkeypatch.setattr(device_status, "is_reachable", lambda h, p, t: reachable)
    # Build via __new__ to skip config reads and ZMQ socket creation.
    server = device_status.DeviceStatusServer.__new__(device_status.DeviceStatusServer)
    server.devices = devices
    server.probe_timeout = 0.1
    return server


def test_build_snapshot_disabled_device(monkeypatch):
    server = _status_server(
        {"Cam": {"active": False, "host": "h", "port": 1}}, reachable=True, monkeypatch=monkeypatch
    )
    snap = server.build_snapshot()
    assert snap["type"] == "device_status"
    assert snap["devices"]["Cam"]["state"] == "disabled"
    assert snap["devices"]["Cam"]["last_seen"] is None


def test_build_snapshot_missing_host_or_port_is_down(monkeypatch):
    server = _status_server(
        {"Cam": {"active": True, "host": None, "port": 1}}, reachable=True, monkeypatch=monkeypatch
    )
    assert server.build_snapshot()["devices"]["Cam"]["state"] == "down"


def test_build_snapshot_up_when_reachable(monkeypatch):
    server = _status_server(
        {"Cam": {"active": True, "host": "h", "port": 1}}, reachable=True, monkeypatch=monkeypatch
    )
    entry = server.build_snapshot()["devices"]["Cam"]
    assert entry["state"] == "up"
    assert entry["last_seen"] is not None


def test_build_snapshot_down_when_unreachable(monkeypatch):
    server = _status_server(
        {"Cam": {"active": True, "host": "h", "port": 1}}, reachable=False, monkeypatch=monkeypatch
    )
    assert server.build_snapshot()["devices"]["Cam"]["state"] == "down"


# --------------------------------------------------------------------------- #
# reachability (real local socket, no mocking)
# --------------------------------------------------------------------------- #

def test_is_reachable_true_against_listening_socket():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert is_reachable(host, port, timeout=1.0) is True
    finally:
        srv.close()


def test_is_reachable_false_against_closed_port():
    # Reserve then immediately release a port so nothing is listening on it.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    host, port = s.getsockname()
    s.close()
    assert is_reachable(host, port, timeout=0.3) is False
