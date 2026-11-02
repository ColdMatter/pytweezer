"""Measurement files served by the Experiment Manager over sipyco RPC."""

import h5py
import numpy as np
import pytest
from sipyco.sync_struct import Notifier

from pytweezer.experiment import Choice, Experiment, ListAxis, Number, Scan, run_local
from pytweezer.experiment.archive import (
    RPC_TARGET,
    ArchiveTarget,
    ArchiveUnavailable,
    RemoteArchive,
    ResultsArchive,
    default_archive,
)
from pytweezer.servers.sync import SyncServer

HOST = "127.0.0.1"
DAY = "2026/10/07"


class Mixed(Experiment):
    x = Number(0.0)
    mode = Choice(["a", "b"])

    def prepare(self):
        self.record("calibration", np.arange(3.0), unit="V")

    def run_point(self):
        self.record("y", self.x * 2, unit="counts")
        self.record("image", np.full((3, 2), self.point.index, dtype=np.uint16))
        self.record("note", f"point {self.point.index}")
        self.record("fit", {"ok": True, "x0": self.x})


SCAN = Scan(
    axes=[
        ListAxis(argument="x", values=[1.0, 2.0]),
        ListAxis(argument="mode", values=["a", "b"]),
    ]
)


def make_file(root, rid=1, label=""):
    path = root / DAY / f"{rid:06d}_Mixed.h5"
    run_local(Mixed, SCAN, path=path, label=label)
    return path


@pytest.fixture
def remote(tmp_path):
    server = SyncServer(
        {"unused": Notifier({})},
        HOST,
        0,
        rpc_targets={RPC_TARGET: ArchiveTarget(ResultsArchive(tmp_path))},
        rpc_port=0,
    )
    archive = RemoteArchive((HOST, server.rpc_port), timeout=5)
    yield archive
    archive.close()
    server.close()


def test_a_fetched_measurement_reads_back_as_the_file_does(tmp_path, remote):
    path = make_file(tmp_path)
    local = ResultsArchive(tmp_path).load(f"{DAY}/{path.name}")
    fetched = remote.load(f"{DAY}/{path.name}")

    assert fetched.attrs == local.attrs
    assert fetched.arguments == local.arguments
    assert fetched.scan == local.scan
    assert fetched.units == local.units
    assert (
        fetched.result_shapes
        == local.result_shapes
        == {
            "y": (),
            "image": (3, 2),
            "note": (),
            "fit": (),
        }
    )
    np.testing.assert_array_equal(fetched.results["image"], local.results["image"])
    assert fetched.results["image"].dtype == np.uint16
    np.testing.assert_array_equal(fetched.results["y"], [2.0, 2.0, 4.0, 4.0])
    assert list(fetched.results["note"]) == list(local.results["note"])
    assert fetched.results["fit"] == local.results["fit"]
    assert list(fetched.points["mode"]) == ["a", "b", "a", "b"]
    np.testing.assert_array_equal(fetched.constants["calibration"], np.arange(3.0))
    assert fetched.source == local.source


def test_only_the_requested_results_are_fetched(tmp_path, remote):
    path = make_file(tmp_path)
    fetched = remote.load(f"{DAY}/{path.name}", results=["y"])
    assert set(fetched.results) == {"y"}
    assert set(fetched.result_shapes) == {"y", "image", "note", "fit"}


def test_planned_points_include_those_not_yet_run(tmp_path, remote):
    path = make_file(tmp_path)
    with h5py.File(path, "r+") as f:
        f.attrs.update(status="running", n_done=1)
    assert remote.load(f"{DAY}/{path.name}").n_done == 1
    planned = remote.planned_points(f"{DAY}/{path.name}")
    assert list(planned["mode"]) == ["a", "b", "a", "b"]
    np.testing.assert_array_equal(planned["x"], [1.0, 1.0, 2.0, 2.0])


def test_runs_follow_a_running_measurement_to_its_end(tmp_path, remote):
    make_file(tmp_path, 1, label="first")
    running = make_file(tmp_path, 2)
    with h5py.File(running, "r+") as f:
        f.attrs["status"] = "running"
    assert remote.days() == [DAY]
    assert remote.runs(DAY) == [
        {"path": f"{DAY}/000002_Mixed.h5", "status": "running", "label": ""},
        {"path": f"{DAY}/000001_Mixed.h5", "status": "completed", "label": "first"},
    ]
    with h5py.File(running, "r+") as f:
        f.attrs["status"] = "completed"
    assert remote.runs(DAY)[0]["status"] == "completed"


def test_a_rid_is_found_by_its_file(tmp_path, remote):
    make_file(tmp_path, 7)
    assert remote.find(7) == f"{DAY}/000007_Mixed.h5"
    with pytest.raises(FileNotFoundError):
        remote.find(8)


@pytest.mark.parametrize("path", ["../secret.h5", "/etc/passwd", f"{DAY}/notes.txt"])
def test_files_outside_the_data_root_are_refused(tmp_path, remote, path):
    (tmp_path.parent / "secret.h5").write_bytes(b"")
    with pytest.raises(ValueError):
        remote.load(path)


def test_an_unreachable_manager_is_reported_then_reconnected(tmp_path):
    make_file(tmp_path)
    server = SyncServer(
        {"unused": Notifier({})},
        HOST,
        0,
        rpc_targets={RPC_TARGET: ArchiveTarget(ResultsArchive(tmp_path))},
        rpc_port=0,
    )
    port = server.rpc_port
    archive = RemoteArchive((HOST, port), timeout=5)
    try:
        assert archive.days() == [DAY]
        server.close()
        with pytest.raises(ArchiveUnavailable):
            archive.days()
        server = SyncServer(
            {"unused": Notifier({})},
            HOST,
            0,
            rpc_targets={RPC_TARGET: ArchiveTarget(ResultsArchive(tmp_path))},
            rpc_port=port,
        )
        assert archive.days() == [DAY]
    finally:
        archive.close()
        server.close()


def test_the_manager_pc_reads_locally_and_others_fetch(monkeypatch, tmp_path):
    conf = {"host": "10.0.0.9", "results_port": 1234, "data_root": str(tmp_path)}
    for module in ("pytweezer.configuration.config", "pytweezer.experiment.client"):
        monkeypatch.setattr(
            f"{module}.get_config", lambda: {"Servers": {"Experiment Manager": conf}}
        )
    monkeypatch.delenv("PYTWEEZER_DATA_DIR", raising=False)
    remote = default_archive()
    assert isinstance(remote, RemoteArchive)
    assert (remote.host, remote.port) == ("10.0.0.9", 1234)

    conf["host"] = "127.0.0.1"
    assert default_archive().root == tmp_path
    conf["host"] = "10.0.0.9"
    monkeypatch.setenv("PYTWEEZER_DATA_DIR", str(tmp_path / "env"))
    assert default_archive().root == tmp_path / "env"
