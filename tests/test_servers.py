"""Unit tests for the device server/client framework and the device-status
service. Everything here runs without real hardware, RPC, or ZMQ: config
lookups are stubbed, the RPC client is monkeypatched, and reachability probes
either hit a local throwaway socket or are stubbed out.
"""

import socket

import pytest

from pytweezer.servers import device_client, device_server
from pytweezer.servers.reachability import is_reachable


# --------------------------------------------------------------------------- #
# device_server: name resolution
# --------------------------------------------------------------------------- #

_FAKE_DEVICES = {
    "Devices": {
        "Rb HamCam": {
            "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
            "host": "1.2.3.4",
            "port": 5000,
        },
        "CaF MotMaster Server": {
            "class": "pytweezer.drivers.motmaster:MotMasterInterface",
            "host": "1.2.3.5",
            "port": 6000,
        },
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
    assert conf["class"] == "pytweezer.drivers.imagemX2:ImagEMX2Camera"


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
# device_server: build_spec class loading and error handling
# --------------------------------------------------------------------------- #

def test_build_spec_missing_class_key():
    with pytest.raises(KeyError, match="no 'class'"):
        device_server.build_spec("thing", conf={"host": "x", "port": 1})


def test_build_spec_simulate_without_sim_class_generates_stand_in():
    # No "sim_class": build_spec generates a hardware-free stand-in from "class"
    # rather than erroring. The real SLM is never constructed (its __init__ would
    # try to load the Blink DLL); the stand-in's public methods are no-ops.
    from pytweezer.drivers.slm import SLM

    spec = device_server.build_spec(
        "SLM", conf={"class": "pytweezer.drivers.slm:SLM", "simulate": True}
    )
    backend = spec.target
    assert type(backend)._simulates is SLM
    assert not isinstance(backend, SLM)
    assert backend.update_mask(object()) is None  # stubbed no-op, no hardware


def test_build_spec_loads_and_constructs_class_from_config():
    # A backend named "module:Class" is imported and constructed from the config
    # keys that match its __init__ parameters; a "teardown" method is wired up.
    spec = device_server.build_spec(
        "SLM",
        conf={
            "class": "pytweezer.drivers.slm:SimulatedSLM",
            "teardown": "close",
            "width": 64,
            "height": 32,
        },
    )
    slm = spec.target
    assert slm.get_dimensions() == {"width": 64, "height": 32, "depth": 8}
    assert slm.is_connected()
    spec.teardown()
    assert not slm.is_connected()


def test_build_spec_sim_class_used_when_simulating():
    spec = device_server.build_spec(
        "SLM",
        conf={
            "class": "pytweezer.drivers.slm:SLM",
            "sim_class": "pytweezer.drivers.slm:SimulatedSLM",
            "simulate": True,
        },
    )
    from pytweezer.drivers.slm import SimulatedSLM

    assert isinstance(spec.target, SimulatedSLM)


# --------------------------------------------------------------------------- #
# device_server: a composite degrades around sub-devices that won't start
# --------------------------------------------------------------------------- #

#: A sub-device whose backend class cannot even be imported, standing in for any
#: device that is absent at build time (no hardware, missing driver library, ...).
_UNAVAILABLE = {"class": "no.such.module:Nope"}
_WORKING_SLM = {"class": "pytweezer.drivers.slm:SimulatedSLM", "teardown": "close",
                "role": "slm", "width": 8, "height": 8}


def _rig(sub_confs, coordinator=None):
    conf = {"host": "h", "port": 1, "devices": sub_confs}
    if coordinator is not None:
        conf["coordinator"] = coordinator
    return conf


def test_composite_serves_the_sub_devices_that_did_start():
    spec = device_server.build_spec(
        "Rig", conf=_rig({"Good SLM": _WORKING_SLM, "Dead Cam": _UNAVAILABLE})
    )
    assert set(spec.targets) == {"goodslm"}
    assert spec.failed == ("Dead Cam",)


def test_composite_skips_coordinator_when_a_sub_device_failed():
    # The coordinator drives every backend, so a partial rig gets none: the
    # composite's own name is unserved while its survivors stay addressable.
    spec = device_server.build_spec(
        "Rig",
        conf=_rig(
            {"Good SLM": _WORKING_SLM, "Dead Cam": _UNAVAILABLE},
            coordinator="pytweezer.coordinators.base:Coordinator",
        ),
    )
    assert set(spec.targets) == {"goodslm"}


def test_composite_builds_coordinator_when_every_sub_device_started():
    spec = device_server.build_spec(
        "Rig",
        conf=_rig(
            {"Good SLM": _WORKING_SLM},
            coordinator="pytweezer.coordinators.base:Coordinator",
        ),
    )
    assert set(spec.targets) == {"goodslm", "coordinator"}


def test_composite_coordinator_failure_leaves_sub_devices_served():
    spec = device_server.build_spec(
        "Rig", conf=_rig({"Good SLM": _WORKING_SLM}, coordinator="no.such.module:Nope")
    )
    assert set(spec.targets) == {"goodslm"}
    assert spec.failed == ()  # the sub-device is fine; only the coordinator isn't


def test_composite_with_nothing_available_still_builds_a_spec():
    # The server binds anyway so the rig reads as "running, all parts failed"
    # rather than silently vanishing from the status feed.
    spec = device_server.build_spec("Rig", conf=_rig({"Dead Cam": _UNAVAILABLE}))
    assert spec.targets == {}
    assert spec.failed == ("Dead Cam",)


def test_composite_tears_down_survivors_of_a_partial_start():
    spec = device_server.build_spec(
        "Rig", conf=_rig({"Good SLM": _WORKING_SLM, "Dead Cam": _UNAVAILABLE})
    )
    slm = spec.targets["goodslm"]
    assert slm.is_connected()
    spec.teardown()
    assert not slm.is_connected()


def test_composite_still_rejects_a_nested_composite():
    # A structural config error is a typo, not a missing device: it fails loudly.
    with pytest.raises(ValueError, match="may not"):
        device_server.build_spec(
            "Rig", conf=_rig({"Inner": {"devices": {"X": _WORKING_SLM}}})
        )


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
        "Rb HamCam": {
            "class": "pytweezer.drivers.imagemX2:ImagEMX2Camera",
            "host": "1.2.3.4",
            "port": 5000,
        },
        "Rb Feedback Rig": {
            "host": "1.2.3.9",
            "port": 6000,
            "devices": {
                "Rb Feedback Cam": {"class": "pkg:Cam", "role": "camera"},
                "Rb Feedback DAC": {"class": "pkg:Dac", "role": "dac"},
            },
            "coordinator": "pkg:CameraDacFeedback",
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
                "host": "h",
                "port": 1,
                "devices": {"Cam": {"class": "pkg:Cam"}},
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

def _status_server(devices, reachable, monkeypatch, targets=None):
    from pytweezer.servers import device_status

    monkeypatch.setattr(device_status, "is_reachable", lambda h, p, t: reachable)
    # ``targets`` stands in for the sipyco handshake that lists what a composite
    # serves; None means the handshake did not complete.
    monkeypatch.setattr(device_status, "server_targets", lambda h, p, t: targets)
    # Build via __new__ to skip config reads and ZMQ socket creation.
    server = device_status.DeviceStatusServer.__new__(device_status.DeviceStatusServer)
    server.devices = devices
    server.probe_timeout = 0.1
    server.target_timeout = 0.1
    server._last_targets = {}
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
# device_status: composites report each sub-device separately
# --------------------------------------------------------------------------- #

_STATUS_RIG = {
    "Rig": {
        "active": True,
        "host": "h",
        "port": 1,
        "devices": {"Rig Cam": {"class": "pkg:Cam"}, "Rig SLM": {"class": "pkg:Slm"}},
        "coordinator": "pkg:Coord",
    }
}


def test_build_snapshot_composite_reports_each_sub_device(monkeypatch):
    server = _status_server(
        _STATUS_RIG, reachable=True, monkeypatch=monkeypatch,
        targets={"rigcam", "rigslm", "coordinator"},
    )
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == "up"
    assert devices["Rig"]["children"] == ["Rig Cam", "Rig SLM"]
    assert devices["Rig Cam"]["state"] == "up"
    assert devices["Rig SLM"] == {
        "state": "up", "host": "h", "port": 1,
        "last_seen": devices["Rig SLM"]["last_seen"], "parent": "Rig",
    }


def test_build_snapshot_composite_marks_the_missing_sub_device_failed(monkeypatch):
    # The rig is serving, but the SLM never opened: it is failed, not stopped, and
    # the coordinator stood down so the rig itself is only degraded.
    server = _status_server(
        _STATUS_RIG, reachable=True, monkeypatch=monkeypatch, targets={"rigcam"}
    )
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == "degraded"
    assert devices["Rig Cam"]["state"] == "up"
    assert devices["Rig SLM"]["state"] == "failed"
    assert devices["Rig SLM"]["last_seen"] is None


def test_build_snapshot_composite_unknown_when_handshake_never_answered(monkeypatch):
    server = _status_server(
        _STATUS_RIG, reachable=True, monkeypatch=monkeypatch, targets=None
    )
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == "unknown"
    assert devices["Rig Cam"]["state"] == "unknown"


def test_build_snapshot_composite_reuses_targets_while_the_server_is_busy(monkeypatch):
    # A long synchronous RPC leaves the handshake unanswered though the port still
    # accepts connections; the last known answer stands in rather than flapping
    # every sub-device to "failed".
    from pytweezer.servers import device_status

    server = _status_server(
        _STATUS_RIG, reachable=True, monkeypatch=monkeypatch,
        targets={"rigcam", "rigslm", "coordinator"},
    )
    server.build_snapshot()
    monkeypatch.setattr(device_status, "server_targets", lambda h, p, t: None)
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == "up"
    assert devices["Rig Cam"]["state"] == "up"


def test_build_snapshot_composite_sub_devices_follow_a_down_rig(monkeypatch):
    server = _status_server(
        _STATUS_RIG, reachable=False, monkeypatch=monkeypatch, targets={"rigcam"}
    )
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == "down"
    assert devices["Rig Cam"]["state"] == "down"
    assert devices["Rig SLM"]["state"] == "down"


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
