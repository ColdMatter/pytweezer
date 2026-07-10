"""Unit tests for composite device servers: the multi-target ``DeviceServerSpec``
and the ``_make_composite`` factory. No RPC, no ZMQ, no hardware — sub-drivers are
fake factories registered into ``DRIVER_REGISTRY``, mirroring ``test_servers.py``.
"""

import pytest

from pytweezer.servers import device_server
from pytweezer.servers.device_server import DeviceServerSpec


# --------------------------------------------------------------------------- #
# DeviceServerSpec: single-target back-compat + multi-target form
# --------------------------------------------------------------------------- #

def test_spec_single_target_normalizes_to_targets():
    target = object()
    spec = DeviceServerSpec(target_name="camera", target=target, description="d")
    assert spec.targets == {"camera": target}


def test_spec_multi_target_form():
    targets = {"camera": object(), "dac": object()}
    spec = DeviceServerSpec(targets=targets, description="d")
    assert spec.targets == targets


def test_spec_rejects_both_forms():
    with pytest.raises(ValueError, match="not both"):
        DeviceServerSpec(target_name="camera", target=object(), targets={"dac": object()})


def test_spec_rejects_neither_form():
    with pytest.raises(ValueError, match="target_name/target or targets"):
        DeviceServerSpec(description="d")


# --------------------------------------------------------------------------- #
# Fake sub-drivers
# --------------------------------------------------------------------------- #

class FakeBackend:
    def __init__(self, conf=None):
        self.conf = conf or {}


@pytest.fixture
def fake_drivers(monkeypatch):
    """Register two fake drivers; record the confs they were built with."""
    seen = {}

    def make_a(name, conf):
        seen[name] = conf
        backend = FakeBackend(conf)
        return DeviceServerSpec(target_name="its_own_name", target=backend, description="a")

    def make_b(name, conf):
        seen[name] = conf
        return DeviceServerSpec(target_name="also_ignored", target=FakeBackend(conf), description="b")

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_a", make_a)
    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_b", make_b)
    return seen


def _conf(**overrides):
    conf = {
        "driver": "composite",
        "devices": {"camera": {"driver": "fake_a"}, "dac": {"driver": "fake_b"}},
    }
    conf.update(overrides)
    return conf


# --------------------------------------------------------------------------- #
# _make_composite: target assembly
# --------------------------------------------------------------------------- #

def test_composite_target_names_come_from_sub_keys(fake_drivers):
    spec = device_server.build_spec("Rig", conf=_conf())
    # The sub-device name wins over whatever target_name the sub-factory chose,
    # which is what lets two of the same driver coexist under different names.
    assert set(spec.targets) == {"camera", "dac"}


def test_composite_target_name_is_the_normalized_device_name(fake_drivers):
    conf = {"driver": "composite", "devices": {"Rb Feedback Cam": {"driver": "fake_a"}}}
    spec = device_server.build_spec("Rig", conf=conf)
    # sipyco target names cannot contain whitespace, so the display name is folded.
    assert list(spec.targets) == ["rbfeedbackcam"]


def test_composite_dispatches_each_sub_driver(fake_drivers):
    device_server.build_spec("Rig", conf=_conf())
    assert sorted(fake_drivers) == ["camera", "dac"]


def test_composite_sub_devices_inherit_simulate(fake_drivers):
    device_server.build_spec("Rig", conf=_conf(simulate=True))
    assert fake_drivers["camera"]["simulate"] is True
    assert fake_drivers["dac"]["simulate"] is True


def test_composite_sub_device_simulate_overrides_parent(fake_drivers):
    conf = _conf(simulate=True)
    conf["devices"]["dac"]["simulate"] = False
    device_server.build_spec("Rig", conf=conf)
    assert fake_drivers["camera"]["simulate"] is True
    assert fake_drivers["dac"]["simulate"] is False


def test_composite_passes_backends_to_coordinator_by_role(monkeypatch):
    captured = {}

    class FakeCoordinator:
        def shutdown(self):
            pass

    def make_a(name, conf):
        return DeviceServerSpec(target_name="x", target=FakeBackend(conf), description="a")

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_a", make_a)
    monkeypatch.setitem(
        device_server.COORDINATOR_REGISTRY, "fake_coord",
        lambda targets, conf: captured.update(targets=targets) or FakeCoordinator(),
    )

    conf = {
        "driver": "composite",
        "devices": {"Rb Feedback Cam": {"driver": "fake_a", "role": "camera"}},
        "coordinator": "fake_coord",
    }
    spec = device_server.build_spec("Rig", conf=conf)
    # The coordinator looks its backend up by role; RPC clients address it by name.
    assert list(captured["targets"]) == ["camera"]
    assert set(spec.targets) == {"coordinator", "rbfeedbackcam"}


def test_composite_role_defaults_to_device_name(monkeypatch):
    captured = {}

    class FakeCoordinator:
        def shutdown(self):
            pass

    monkeypatch.setitem(
        device_server.DRIVER_REGISTRY, "fake_a",
        lambda name, conf: DeviceServerSpec(target_name="x", target=object(), description="a"),
    )
    monkeypatch.setitem(
        device_server.COORDINATOR_REGISTRY, "fake_coord",
        lambda targets, conf: captured.update(targets=targets) or FakeCoordinator(),
    )

    conf = {
        "driver": "composite",
        "devices": {"camera": {"driver": "fake_a"}},
        "coordinator": "fake_coord",
    }
    device_server.build_spec("Rig", conf=conf)
    assert list(captured["targets"]) == ["camera"]


def test_composite_does_not_mutate_config(fake_drivers):
    conf = _conf(simulate=True)
    device_server.build_spec("Rig", conf=conf)
    assert "simulate" not in conf["devices"]["camera"]


# --------------------------------------------------------------------------- #
# _make_composite: teardown chaining
# --------------------------------------------------------------------------- #

def test_composite_teardown_runs_all_even_when_one_raises(monkeypatch):
    calls = []

    def boom():
        calls.append("boom")
        raise RuntimeError("teardown failed")

    def make_a(name, conf):
        return DeviceServerSpec(target_name="a", target=object(), description="a", teardown=boom)

    def make_b(name, conf):
        return DeviceServerSpec(
            target_name="b", target=object(), description="b",
            teardown=lambda: calls.append("b"),
        )

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_a", make_a)
    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_b", make_b)

    spec = device_server.build_spec("Rig", conf=_conf())
    spec.teardown()
    # Reversed order, and the raising teardown does not abort the rest (_safe).
    assert calls == ["b", "boom"]


def test_composite_coordinator_teardown_runs_before_backends(monkeypatch):
    calls = []

    class FakeCoordinator:
        def shutdown(self):
            calls.append("coordinator")

    def make_a(name, conf):
        return DeviceServerSpec(
            target_name="a", target=object(), description="a",
            teardown=lambda: calls.append("camera"),
        )

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_a", make_a)
    monkeypatch.setitem(
        device_server.COORDINATOR_REGISTRY, "fake_coord",
        lambda targets, conf: FakeCoordinator(),
    )

    conf = {
        "driver": "composite",
        "devices": {"camera": {"driver": "fake_a"}},
        "coordinator": "fake_coord",
    }
    spec = device_server.build_spec("Rig", conf=conf)
    assert set(spec.targets) == {"camera", "coordinator"}
    spec.teardown()
    assert calls == ["coordinator", "camera"]


def test_composite_partial_build_failure_tears_down_what_opened(monkeypatch):
    calls = []

    def make_a(name, conf):
        return DeviceServerSpec(
            target_name="a", target=object(), description="a",
            teardown=lambda: calls.append("camera closed"),
        )

    def make_b(name, conf):
        raise RuntimeError("dac failed to open")

    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_a", make_a)
    monkeypatch.setitem(device_server.DRIVER_REGISTRY, "fake_b", make_b)

    with pytest.raises(RuntimeError, match="dac failed to open"):
        device_server.build_spec("Rig", conf=_conf())
    assert calls == ["camera closed"]


# --------------------------------------------------------------------------- #
# _make_composite: rejections
# --------------------------------------------------------------------------- #

def test_composite_requires_devices():
    with pytest.raises(KeyError, match="non-empty 'devices' dict"):
        device_server.build_spec("Rig", conf={"driver": "composite"})


def test_composite_rejects_empty_devices():
    with pytest.raises(KeyError, match="non-empty 'devices' dict"):
        device_server.build_spec("Rig", conf={"driver": "composite", "devices": {}})


def test_composite_rejects_nested_composite(fake_drivers):
    conf = {
        "driver": "composite",
        "devices": {"inner": {"driver": "composite", "devices": {}}},
    }
    with pytest.raises(ValueError, match="itself be a composite"):
        device_server.build_spec("Rig", conf=conf)


@pytest.mark.parametrize("bad_name", ["", "   ", "\t"])
def test_composite_rejects_nameless_sub_device(fake_drivers, bad_name):
    conf = {"driver": "composite", "devices": {bad_name: {"driver": "fake_a"}}}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        device_server.build_spec("Rig", conf=conf)
    # Rejected before any sub-driver was built, so no hardware was opened.
    assert fake_drivers == {}


def test_composite_rejects_sub_devices_sharing_a_target(fake_drivers):
    # Both fold to the same wire-safe target name.
    conf = {
        "driver": "composite",
        "devices": {"Rb Cam": {"driver": "fake_a"}, "rbcam": {"driver": "fake_b"}},
    }
    with pytest.raises(ValueError, match="both map to RPC target"):
        device_server.build_spec("Rig", conf=conf)
    assert fake_drivers == {}


def test_composite_rejects_callable_target(monkeypatch):
    class CallableBackend:
        def __call__(self):
            pass

    monkeypatch.setitem(
        device_server.DRIVER_REGISTRY, "fake_callable",
        lambda name, conf: DeviceServerSpec(
            target_name="x", target=CallableBackend(), description="d"
        ),
    )
    conf = {"driver": "composite", "devices": {"camera": {"driver": "fake_callable"}}}
    with pytest.raises(TypeError, match="is callable"):
        device_server.build_spec("Rig", conf=conf)


def test_composite_rejects_coordinator_name_collision(fake_drivers):
    conf = {
        "driver": "composite",
        "devices": {"coordinator": {"driver": "fake_a"}},
        "coordinator": "camera_dac_feedback",
    }
    with pytest.raises(ValueError, match="collides with sub-device"):
        device_server.build_spec("Rig", conf=conf)


def test_composite_unknown_coordinator_lists_registered(fake_drivers):
    conf = _conf(coordinator="does-not-exist")
    with pytest.raises(KeyError) as exc:
        device_server.build_spec("Rig", conf=conf)
    assert "camera_dac_feedback" in str(exc.value)


# --------------------------------------------------------------------------- #
# nidac driver factory
# --------------------------------------------------------------------------- #

def test_make_nidac_simulated():
    spec = device_server.build_spec(
        "DAC", conf={"driver": "nidac", "simulate": True, "channels": ["Dev1/ao0"]}
    )
    assert list(spec.targets) == ["dac"]
    assert spec.targets["dac"].get_last_values() == {"Dev1/ao0": 0.0}


def test_make_nidac_real_hardware_not_implemented():
    with pytest.raises(NotImplementedError, match="ni_dac.py"):
        device_server.build_spec("DAC", conf={"driver": "nidac", "channels": ["Dev1/ao0"]})


# --------------------------------------------------------------------------- #
# Flat device addressing: sub-devices share the top-level device namespace
# --------------------------------------------------------------------------- #

_FLAT_CONFIG = {
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
def flat_config(monkeypatch):
    monkeypatch.setattr(
        device_server.ConfigReader, "getConfiguration", staticmethod(lambda: _FLAT_CONFIG)
    )


def test_device_index_flattens_sub_devices(flat_config):
    names = sorted(a.name for a in device_server.device_index().values())
    assert names == ["Rb Feedback Cam", "Rb Feedback DAC", "Rb Feedback Rig", "Rb HamCam"]


def test_resolve_address_sub_device_points_at_its_composite(flat_config):
    address = device_server.resolve_address("Rb Feedback Cam")
    assert address.is_sub_device
    assert address.owner_name == "Rb Feedback Rig"
    assert address.owner_conf["port"] == 6000       # the serving process's port
    assert address.target_name == "rbfeedbackcam"
    assert address.conf["role"] == "camera"         # its own driver config


def test_resolve_address_composite_resolves_to_its_coordinator(flat_config):
    address = device_server.resolve_address("Rb Feedback Rig")
    assert not address.is_sub_device
    assert address.target_name == "coordinator"


def test_resolve_address_plain_device_has_no_target(flat_config):
    address = device_server.resolve_address("Rb HamCam")
    assert address.target_name is None              # AutoTarget picks the sole target


@pytest.mark.parametrize("query", ["rbfeedbackcam", "RB FEEDBACK CAM", " rb  feedback cam "])
def test_resolve_address_matches_leniently(flat_config, query):
    assert device_server.resolve_address(query).name == "Rb Feedback Cam"


def test_resolve_address_unknown_lists_sub_devices_too(flat_config):
    with pytest.raises(KeyError) as exc:
        device_server.resolve_address("nope")
    assert "Rb Feedback Cam" in str(exc.value)


def test_duplicate_device_name_across_composite_is_rejected(monkeypatch):
    conf = {
        "Devices": {
            "Rb HamCam": {"driver": "imagemx2", "host": "h", "port": 1},
            "Rig": {
                "driver": "composite",
                "host": "h",
                "port": 2,
                "devices": {"rb hamcam": {"driver": "imagemx2"}},
            },
        }
    }
    monkeypatch.setattr(
        device_server.ConfigReader, "getConfiguration", staticmethod(lambda: conf)
    )
    with pytest.raises(KeyError, match="Duplicate device name"):
        device_server.device_index()


def test_resolve_device_rejects_sub_device_with_a_hint(flat_config):
    """Sub-devices are addressable by clients but have no server to launch."""
    with pytest.raises(KeyError) as exc:
        device_server.resolve_device("Rb Feedback Cam")
    assert "sub-device of composite 'Rb Feedback Rig'" in str(exc.value)
    assert "launch 'Rb Feedback Rig' instead" in str(exc.value)


def test_resolve_device_still_finds_the_composite(flat_config):
    name, conf = device_server.resolve_device("rb feedback rig")
    assert name == "Rb Feedback Rig"
    assert conf["driver"] == "composite"
