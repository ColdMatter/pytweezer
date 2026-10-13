"""Device server/client framework and the device-status service, with config
lookups stubbed and no real RPC."""

import asyncio
import socket

import pytest

from pytweezer.drivers.slm import SLM, SimulatedSLM
from pytweezer.servers import device_client, device_server, device_status
from pytweezer.servers.reachability import is_reachable

_DEVICES = {
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
        "Rb Feedback Rig": {
            "host": "1.2.3.9",
            "port": 6000,
            "devices": {
                "Rb Feedback Cam": {"class": "pkg:Cam", "role": "camera"},
                "Rb Feedback DAC": {"class": "pkg:Dac", "role": "dac"},
            },
            "coordinator": "pkg:CameraDacFeedback",
        },
        "Bare Rig": {"host": "h", "port": 1, "devices": {"Cam": {"class": "pkg:Cam"}}},
        "NoPort": {"host": "1.2.3.4"},
    }
}


@pytest.fixture
def fake_config(monkeypatch):
    monkeypatch.setattr(device_server, "get_config", lambda: _DEVICES)


@pytest.fixture
def captured_rpc(fake_config, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        device_client,
        "RPCClient",
        lambda host, port, target_name=None, timeout=None: captured.update(
            host=host, port=port, target_name=target_name
        ),
    )
    return captured


# --------------------------------------------------------------------------- #
# Name resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["Rb HamCam", "rbhamcam", "  RB   HAMCAM "])
def test_resolve_device_matches_leniently(fake_config, query):
    name, conf = device_server.resolve_device(query)
    assert name == "Rb HamCam"
    assert conf["port"] == 5000


@pytest.mark.parametrize(
    "lookup", [device_server.resolve_device, device_client.get_device_config]
)
def test_unknown_device_error_lists_available(fake_config, lookup):
    with pytest.raises(KeyError, match="Rb HamCam"):
        lookup("nope")


def test_get_device_config_finds_sub_device(fake_config):
    assert device_client.get_device_config("Rb Feedback DAC")["role"] == "dac"


# --------------------------------------------------------------------------- #
# build_spec
# --------------------------------------------------------------------------- #


def test_build_spec_missing_class_key():
    with pytest.raises(KeyError):
        device_server.build_spec("thing", conf={"host": "x", "port": 1})


def test_build_spec_constructs_class_from_config_and_wires_teardown():
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
    spec.teardown()
    assert not slm.is_connected()


def test_build_spec_simulation_uses_sim_class_else_a_generated_stand_in():
    conf = {"class": "pytweezer.drivers.slm:SLM", "simulate": True}
    spec = device_server.build_spec(
        "SLM", conf={**conf, "sim_class": "pytweezer.drivers.slm:SimulatedSLM"}
    )
    assert isinstance(spec.target, SimulatedSLM)

    # Without "sim_class" the real SLM is never constructed (its __init__ would
    # load the Blink DLL); the stand-in's public methods are no-ops.
    stand_in = device_server.build_spec("SLM", conf=conf).target
    assert type(stand_in)._simulates is SLM
    assert not isinstance(stand_in, SLM)
    assert stand_in.update_mask(object()) is None


#: A sub-device whose backend cannot even be imported, standing in for any device
#: absent at build time (no hardware, missing driver library, ...).
_UNAVAILABLE = {"class": "no.such.module:Nope"}
_WORKING_SLM = {
    "class": "pytweezer.drivers.slm:SimulatedSLM",
    "teardown": "close",
    "role": "slm",
    "width": 8,
    "height": 8,
}
_COORDINATOR = "pytweezer.coordinators.base:Coordinator"


@pytest.mark.parametrize(
    "sub_confs, coordinator, targets, failed",
    [
        # A coordinator drives every backend, so a partial rig gets none.
        (
            {"Good SLM": _WORKING_SLM, "Dead Cam": _UNAVAILABLE},
            _COORDINATOR,
            {"goodslm"},
            ("Dead Cam",),
        ),
        ({"Good SLM": _WORKING_SLM}, _COORDINATOR, {"goodslm", "coordinator"}, ()),
        ({"Good SLM": _WORKING_SLM}, "no.such.module:Nope", {"goodslm"}, ()),
        # Still builds so the rig reads as "running, all parts failed" rather
        # than vanishing from the status feed.
        ({"Dead Cam": _UNAVAILABLE}, None, set(), ("Dead Cam",)),
    ],
)
def test_composite_serves_what_started(sub_confs, coordinator, targets, failed):
    conf = {"host": "h", "port": 1, "devices": sub_confs}
    if coordinator is not None:
        conf["coordinator"] = coordinator
    spec = device_server.build_spec("Rig", conf=conf)
    assert set(spec.targets) == targets
    assert spec.failed == failed

    survivor = spec.targets.get("goodslm")
    spec.teardown()
    assert survivor is None or not survivor.is_connected()


def test_composite_rejects_a_nested_composite():
    with pytest.raises(ValueError):
        device_server.build_spec(
            "Rig",
            conf={"host": "h", "port": 1, "devices": {"Inner": {"devices": {}}}},
        )


# --------------------------------------------------------------------------- #
# device_client endpoints
# --------------------------------------------------------------------------- #

_AUTO = device_client.AutoTarget


@pytest.mark.parametrize(
    "name, kwargs, expected",
    [
        ("Rb HamCam", {}, ("1.2.3.4", 5000, _AUTO)),
        ("rb hamcam", {}, ("1.2.3.4", 5000, _AUTO)),
        ("Rb HamCam", {"host": "9.9.9.9", "port": 1234}, ("9.9.9.9", 1234, _AUTO)),
        # A sub-device is reached at its composite's endpoint, selected by target.
        ("Rb Feedback Cam", {}, ("1.2.3.9", 6000, "rbfeedbackcam")),
        ("Rb Feedback DAC", {}, ("1.2.3.9", 6000, "rbfeedbackdac")),
        ("Rb Feedback Rig", {}, ("1.2.3.9", 6000, "coordinator")),
        (
            "Rb Feedback Rig",
            {"target_name": "rbfeedbackcam"},
            ("1.2.3.9", 6000, "rbfeedbackcam"),
        ),
    ],
)
def test_get_device_endpoint(captured_rpc, name, kwargs, expected):
    device_client.get_device(name, **kwargs)
    assert (
        captured_rpc["host"],
        captured_rpc["port"],
        captured_rpc["target_name"],
    ) == expected


@pytest.mark.parametrize(
    "name, error",
    [
        ("NoPort", ValueError),
        # A coordinatorless composite's own name addresses nothing.
        ("Bare Rig", KeyError),
    ],
)
def test_get_device_unaddressable(captured_rpc, name, error):
    with pytest.raises(error):
        device_client.get_device(name)


def test_get_device_async_uses_config_host_and_port(fake_config, monkeypatch):
    captured = {}

    class FakeAsyncioClient:
        async def connect_rpc(self, host, port, target_name=None):
            captured.update(host=host, port=port)

    monkeypatch.setattr(device_client, "AsyncioClient", FakeAsyncioClient)
    client = asyncio.run(device_client.get_device_async("Rb HamCam"))
    assert isinstance(client, FakeAsyncioClient)
    assert captured == {"host": "1.2.3.4", "port": 5000}


# --------------------------------------------------------------------------- #
# device_status.build_snapshot (probes stubbed)
# --------------------------------------------------------------------------- #


@pytest.fixture
def status_server(monkeypatch):
    def make(devices, reachable, targets=None):
        monkeypatch.setattr(device_status, "is_reachable", lambda h, p, t: reachable)
        # ``targets`` stands in for the sipyco handshake listing what a composite
        # serves; None means the handshake did not complete.
        monkeypatch.setattr(device_status, "server_targets", lambda h, p, t: targets)
        server = device_status.DeviceStatusServer.__new__(
            device_status.DeviceStatusServer
        )
        server.devices = devices
        server.probe_timeout = server.target_timeout = 0.1
        server._last_targets = {}
        return server

    return make


@pytest.mark.parametrize(
    "params, reachable, state",
    [
        ({"active": False, "host": "h", "port": 1}, True, "disabled"),
        ({"active": True, "host": None, "port": 1}, True, "down"),
        ({"active": True, "host": "h", "port": 1}, True, "up"),
        ({"active": True, "host": "h", "port": 1}, False, "down"),
    ],
)
def test_build_snapshot_plain_device(status_server, params, reachable, state):
    entry = status_server({"Cam": params}, reachable).build_snapshot()["devices"]["Cam"]
    assert entry["state"] == state
    assert (entry["last_seen"] is not None) == (state == "up")


_STATUS_RIG = {
    "Rig": {
        "active": True,
        "host": "h",
        "port": 1,
        "devices": {"Rig Cam": {"class": "pkg:Cam"}, "Rig SLM": {"class": "pkg:Slm"}},
        "coordinator": "pkg:Coord",
    }
}
_ALL_TARGETS = {"rigcam", "rigslm", "coordinator"}


@pytest.mark.parametrize(
    "reachable, targets, rig, cam, slm",
    [
        (True, _ALL_TARGETS, "up", "up", "up"),
        # The rig serves, but the SLM never opened: failed, not stopped, and the
        # coordinator stood down so the rig itself is only degraded.
        (True, {"rigcam"}, "degraded", "up", "failed"),
        (True, None, "unknown", "unknown", "unknown"),
        (False, {"rigcam"}, "down", "down", "down"),
    ],
)
def test_build_snapshot_composite(status_server, reachable, targets, rig, cam, slm):
    devices = status_server(_STATUS_RIG, reachable, targets).build_snapshot()["devices"]
    assert devices["Rig"]["children"] == ["Rig Cam", "Rig SLM"]
    assert [devices[n]["state"] for n in ("Rig", "Rig Cam", "Rig SLM")] == [
        rig,
        cam,
        slm,
    ]
    assert devices["Rig SLM"]["parent"] == "Rig"
    assert (devices["Rig SLM"]["last_seen"] is not None) == (slm == "up")


def test_build_snapshot_composite_reuses_targets_while_the_server_is_busy(
    status_server, monkeypatch
):
    # A long synchronous RPC leaves the handshake unanswered though the port still
    # accepts connections; the last known answer stands in rather than flapping
    # every sub-device to "failed".
    server = status_server(_STATUS_RIG, True, _ALL_TARGETS)
    server.build_snapshot()
    monkeypatch.setattr(device_status, "server_targets", lambda h, p, t: None)
    devices = server.build_snapshot()["devices"]
    assert devices["Rig"]["state"] == devices["Rig Cam"]["state"] == "up"


def test_is_reachable():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert is_reachable(host, port, timeout=1.0) is True
    finally:
        srv.close()
    assert is_reachable(host, port, timeout=0.3) is False
