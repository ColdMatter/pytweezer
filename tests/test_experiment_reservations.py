"""Which CONFIG devices an experiment class reserves, with CONFIG faked."""

import sys
import types

import pytest

from pytweezer.experiment.arguments import Device
from pytweezer.experiment.experiment import Experiment
from pytweezer.experiment.introspect import describe
from pytweezer.experiment.motmaster import MotMaster, MotMasterExperiment
from pytweezer.experiment.reservations import ReservationError, device_reservation
from pytweezer.experiment.runner import run_local
from pytweezer.servers import device_client, device_server
from pytweezer.servers.device_client import (
    AutoTarget,
    UndeclaredDeviceError,
    restrict_devices,
)

_DEVICES = {
    "Devices": {
        "Cam": {"class": "pkg:Cam", "host": "1.2.3.4", "port": 5000},
        "Old Cam": {
            "active": False,
            "class": "pkg:Cam",
            "host": "1.2.3.4",
            "port": 5001,
        },
        "Rig": {
            "host": "1.2.3.5",
            "port": 5002,
            "devices": {
                "Rig Cam": {"class": "pkg:Cam", "role": "camera"},
                "Rig Dac": {"class": "pkg:Dac", "role": "dac"},
            },
        },
        "CaF MotMaster": {
            "class": "pkg:MM",
            "host": "1.2.3.6",
            "port": 5003,
            "linked_partner": "Rb MotMaster",
        },
        "Rb MotMaster": {"class": "pkg:MM", "host": "1.2.3.7", "port": 5004},
        "K MotMaster": {"class": "pkg:MM", "host": "1.2.3.8", "port": 5005},
    }
}


@pytest.fixture
def fake_config(monkeypatch):
    monkeypatch.setattr(device_server, "get_config", lambda: _DEVICES)


def test_plain_device_reserves_itself(fake_config):
    assert device_reservation("Cam") == {"Cam"}


def test_names_are_canonicalised(fake_config):
    assert device_reservation(" cam ") == {"Cam"}


def test_composite_reserves_its_members_but_a_member_only_itself(fake_config):
    assert device_reservation("Rig") == {"Rig", "Rig Cam", "Rig Dac"}
    assert device_reservation("Rig Cam") == {"Rig Cam"}


def test_unknown_and_inactive_devices_are_refused(fake_config):
    with pytest.raises(ReservationError, match="not found"):
        device_reservation("Nope")
    with pytest.raises(ReservationError, match="inactive"):
        device_reservation("Old Cam")


def test_experiment_reserves_the_union_of_its_devices(fake_config):
    class Both(Experiment):
        rig = Device("Rig")
        cam = Device("Cam")

        def run_point(self):
            pass

    assert Both.reserved_devices() == ["Cam", "Rig", "Rig Cam", "Rig Dac"]


def test_linked_motmaster_reserves_its_partner(fake_config):
    class Linked(MotMasterExperiment):
        mm = MotMaster("CaF MotMaster", script="S", linked=True)

        def run_point(self):
            pass

    assert Linked.reserved_devices() == ["CaF MotMaster", "Rb MotMaster"]


def test_unlinked_motmaster_reserves_only_itself(fake_config):
    class Unlinked(MotMasterExperiment):
        mm = MotMaster("CaF MotMaster", script="S")

        def run_point(self):
            pass

    assert Unlinked.reserved_devices() == ["CaF MotMaster"]


def test_linked_motmaster_without_partner_is_refused(fake_config):
    class NoPartner(MotMasterExperiment):
        mm = MotMaster("K MotMaster", script="S", linked=True)

        def run_point(self):
            pass

    with pytest.raises(ReservationError, match="linked_partner"):
        NoPartner.reserved_devices()


def test_describe_reports_reservations_and_errors(fake_config, monkeypatch):
    class Good(Experiment):
        cam = Device("Cam")

        def run_point(self):
            pass

    class Bad(Experiment):
        cam = Device("Old Cam")

        def run_point(self):
            pass

    module = types.ModuleType("fake_reservation_experiments")
    for cls in (Good, Bad):
        cls.__module__ = module.__name__
        setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    schemas = {s["class_name"]: s for s in describe(module.__name__)["classes"]}

    assert schemas["Good"]["reserves"] == ["Cam"]
    assert schemas["Good"]["reserve_error"] is None
    assert schemas["Bad"]["reserves"] is None
    assert "inactive" in schemas["Bad"]["reserve_error"]


class Sneaky(Experiment):
    cam = Device("Cam")

    def run_point(self):
        self.device("Rig Dac")


def test_undeclared_device_is_refused_in_run_local(fake_config, fake_devices):
    measurement = run_local(Sneaky)

    assert measurement.status == "failed"
    assert "without declaring it" in measurement.attrs["error"]
    assert [client.name for client in fake_devices] == []


def test_declared_and_composite_member_devices_are_allowed(fake_config, fake_devices):
    class Declared(Experiment):
        rig = Device("Rig")
        cam = Device("Cam")

        def run_point(self):
            self.device("Rig Cam")
            self.device("cam", fresh=True)

    assert run_local(Declared).status == "completed"
    assert sorted(client.name for client in fake_devices) == ["Rig Cam", "cam"]


def test_undeclared_device_is_refused_in_simulation(fake_config):
    measurement = run_local(Sneaky, simulate=True)

    assert measurement.status == "failed"
    assert "without declaring it" in measurement.attrs["error"]


def test_restrict_devices_blocks_direct_get_device(fake_config):
    restrict_devices({"Cam"})
    try:
        with pytest.raises(UndeclaredDeviceError):
            device_client._resolve_endpoint("Rig Dac", None, None, AutoTarget)
        device_client._resolve_endpoint("cam", None, None, AutoTarget)
    finally:
        restrict_devices(None)
