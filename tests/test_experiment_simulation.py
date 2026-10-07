"""Simulation mode: in-process simulated devices and a separate data root."""

import pytest

from pytweezer.experiment import Device, Experiment, run_local
from pytweezer.experiment.catalogue import Catalogue
from pytweezer.experiment.simulation import SimulatedDevices
from pytweezer.experiment.storage import data_root
from pytweezer.servers import device_server
from pytweezer.servers import experiment_manager as em

TORN_DOWN = []


class RealCamera:
    def __init__(self, **kwargs):
        raise AssertionError("simulation must never build the real backend")


class SimCamera:
    def __init__(self, gain=1):
        self.gain = gain

    def snap(self):
        return [[self.gain]]

    def close(self):
        TORN_DOWN.append(self)


class SimDac:
    def __init__(self):
        self.volts = 0.0

    def set(self, volts):
        self.volts = volts


DEVICES = {
    "Cam": {
        "class": f"{__name__}:RealCamera",
        "sim_class": f"{__name__}:SimCamera",
        "teardown": "close",
        "simulate": False,
        "gain": 3,
    },
    "Rig": {
        "simulate": False,
        "devices": {
            "Rig Cam": {
                "class": f"{__name__}:RealCamera",
                "sim_class": f"{__name__}:SimCamera",
                "simulate": False,
            },
            "Rig Dac": {
                "class": f"{__name__}:RealCamera",
                "sim_class": f"{__name__}:SimDac",
            },
        },
    },
}


@pytest.fixture
def devices(monkeypatch):
    monkeypatch.setattr(device_server, "get_config", lambda: {"Devices": DEVICES})
    TORN_DOWN.clear()


def test_simulated_backends_are_built_even_when_config_says_real(devices):
    sim = SimulatedDevices()
    camera = sim.get("cam")  # lenient name matching, as get_device
    assert isinstance(camera, SimCamera) and camera.gain == 3
    assert sim.get("Cam") is camera
    rig_cam, rig_dac = sim.get("Rig Cam"), sim.get("Rig Dac")
    assert isinstance(rig_cam, SimCamera) and isinstance(rig_dac, SimDac)
    sim.close()
    assert TORN_DOWN == [camera]
    with pytest.raises(KeyError, match="not found"):
        sim.get("Nope")


def test_run_local_simulated_experiment(devices):
    class Snap(Experiment):
        camera = Device("Cam")
        dac = Device("Rig Dac")

        def run_point(self):
            assert self.device("Cam", fresh=True) is self.camera
            self.dac.set(1.5)
            self.record("image", self.camera.snap())
            self.record("volts", self.dac.volts)

    measurement = run_local(Snap, simulate=True)
    assert measurement.status == "completed", measurement.attrs["error"]
    assert measurement.attrs["simulated"] is True
    assert measurement.results["image"].tolist() == [[[3]]]
    assert measurement.results["volts"].tolist() == [1.5]
    assert len(TORN_DOWN) == 1
    assert run_local(Snap, simulate=False).attrs["simulated"] is False


def test_simulated_data_root_is_separate(monkeypatch, tmp_path):
    conf = {"data_root": str(tmp_path), "simulate": False}
    monkeypatch.setattr(
        "pytweezer.configuration.config.get_config",
        lambda: {"Servers": {"Experiment Manager": conf}},
    )
    monkeypatch.delenv("PYTWEEZER_DATA_DIR", raising=False)
    assert data_root() == tmp_path
    conf["simulate"] = True
    assert data_root() == tmp_path / "simulated"
    # Even when pointed at the real data share, simulated runs stay apart.
    monkeypatch.setenv("PYTWEEZER_DATA_DIR", str(tmp_path / "share"))
    assert data_root() == tmp_path / "share" / "simulated"


def test_manager_tells_workers_and_guis_it_is_simulating(monkeypatch, tmp_path):
    conf = {"host": "127.0.0.1", "port": 1, "pub_port": 2, "simulate": True}
    monkeypatch.setattr(
        em, "get_config", lambda: {"Servers": {"Experiment Manager": conf}}
    )
    manager = em.ExperimentManager(
        root=tmp_path,
        catalogue=Catalogue(package="none", directory=tmp_path / "none"),
        bind=False,
    )
    assert manager.handle({"command": "ping"})["simulated"] is True
    assert manager.handle({"command": "snapshot"})["snapshot"]["simulated"] is True
    manager.handle(
        {
            "command": "submit",
            "request": {"experiment": "m", "class_name": "C"},
        }
    )
    manager.queue.mark_started(
        1, em.WorkerRecord(rid=1, token="t", pid=0, create_time=0), "x.h5"
    )
    started = manager.handle(
        {"command": "worker", "rid": 1, "token": "t", "event": "started"}
    )
    assert started["simulate"] is True
