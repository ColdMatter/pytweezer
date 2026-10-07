from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from pytweezer.experiment import Experiment, ListAxis, Number, Scan, load_measurement
from pytweezer.experiment.runner import open_writer
from pytweezer.experiment.storage import (
    RecordError,
    data_root,
    highest_rid,
    measurement_relpath,
)


class Sweep(Experiment):
    x = Number(0.0, unit="V")

    def run_point(self):
        pass


SCAN = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])])


def make_writer(path):
    return open_writer(path, Sweep, {}, SCAN, {"rid": 7, "label": "t"})


def test_file_is_never_overwritten(tmp_path):
    path = tmp_path / "m.h5"
    _, _, writer = make_writer(path)
    writer.close()
    with pytest.raises(FileExistsError):
        make_writer(path)


def test_header_arguments_and_planned_points(tmp_path):
    _, points, writer = make_writer(tmp_path / "m.h5")
    writer.close()
    with h5py.File(tmp_path / "m.h5", "r") as f:
        assert f.attrs["rid"] == 7
        assert f.attrs["class_name"] == "Sweep"
        assert f.attrs["status"] == "running"
        assert f.attrs["n_points"] == 3
        assert list(f["points/x"][:]) == [1.0, 2.0, 3.0]
        assert "test_experiment_storage" in next(iter(f["source"]))


def test_partial_file_is_readable_mid_run(tmp_path):
    path = tmp_path / "m.h5"
    _, points, writer = make_writer(path)
    writer.begin_point(points[0])
    writer.record("y", 1.5, unit="counts")
    writer.end_point()
    writer.begin_point(points[1])
    writer.record("y", 2.5)
    # point 1 not ended: a reader sees only point 0
    measurement = load_measurement(path)
    assert measurement.status == "running"
    assert measurement.n_done == 1
    assert list(measurement.results["y"]) == [1.5]
    assert measurement.units["y"] == "counts"
    assert list(measurement.points["x"]) == [1.0]
    writer.close()


def test_missing_values_are_nan_and_flagged(tmp_path):
    _, points, writer = make_writer(None)
    for point in points:
        writer.begin_point(point)
        if point.index != 1:
            writer.record("y", point.index)
            writer.record("flag", True)
        writer.end_point()
    writer.finalise("completed")
    measurement = load_measurement(writer.file)
    y = measurement.results["y"]
    assert y.dtype.kind == "i"  # first record was an int
    assert list(measurement.recorded["y"]) == [True, False, True]
    assert measurement.status == "completed"
    writer.close()


def test_float_results_fill_with_nan(tmp_path):
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    writer.end_point()
    writer.begin_point(points[1])
    writer.record("y", 2.0)
    writer.end_point()
    measurement = load_measurement(writer.file)
    assert np.isnan(measurement.results["y"][0])
    assert measurement.results["y"][1] == 2.0
    writer.close()


def test_shape_and_type_mismatch_raise(tmp_path):
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    writer.record("img", np.zeros((2, 3)))
    writer.record("count", 3)
    writer.end_point()
    writer.begin_point(points[1])
    with pytest.raises(RecordError, match=r"shape \(2, 3\).*\(3, 2\)"):
        writer.record("img", np.zeros((3, 2)))
    with pytest.raises(RecordError, match="dtype int64"):
        writer.record("count", 2.5)
    writer.record("count", np.uint8(4))  # widening is fine
    with pytest.raises(RecordError, match="twice"):
        writer.record("count", 5)
    writer.close()


def test_strings_dicts_and_constants(tmp_path):
    _, points, writer = make_writer(None)
    writer.record("reference", np.arange(4))
    writer.record("settings", {"gain": 2})
    writer.begin_point(points[0])
    writer.record("note", "first")
    writer.record("meta", {"a": 1})
    writer.end_point()
    measurement = load_measurement(writer.file)
    assert list(measurement.constants["reference"]) == [0, 1, 2, 3]
    assert measurement.constants["settings"] == {"gain": 2}
    assert list(measurement.results["note"]) == ["first"]
    assert measurement.results["meta"] == [{"a": 1}]
    with pytest.raises(RecordError, match="already recorded"):
        writer.abandon_point()
        writer.record("reference", 1)
    writer.close()


@pytest.mark.parametrize("name", ["", "a/b", "__schema__"])
def test_bad_record_names(name):
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    with pytest.raises(RecordError, match="invalid record name"):
        writer.record(name, 1)
    writer.close()


def test_unsupported_type():
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    with pytest.raises(RecordError, match="unsupported type"):
        writer.record("obj", object())
    writer.close()


def test_paths_and_rid_scan(tmp_path, monkeypatch):
    rel = measurement_relpath(42, "Sweep", datetime(2026, 3, 4, 5, 6, tzinfo=UTC))
    assert rel == Path("2026", "03", "04", "000042_Sweep.h5")
    assert highest_rid(tmp_path) == 0
    for rid in (3, 41, 12):
        path = tmp_path / measurement_relpath(
            rid, "Sweep", datetime(2026, 1, rid % 28 + 1, tzinfo=UTC)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "2026" / "01" / "04" / "notes.h5").touch()
    assert highest_rid(tmp_path) == 41

    monkeypatch.setenv("PYTWEEZER_DATA_DIR", str(tmp_path))
    assert data_root() == tmp_path


def test_images_keep_their_native_dtype():
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    writer.record("image", np.ones((4, 4), dtype=np.uint16))
    writer.end_point()
    writer.begin_point(points[1])
    with pytest.raises(RecordError, match="uint16"):
        writer.record("image", np.ones((4, 4), dtype=np.int32))
    assert writer.file["results/image"].dtype == np.uint16
    writer.close()


def test_load_only_selected_results_but_know_all_shapes():
    _, points, writer = make_writer(None)
    writer.begin_point(points[0])
    writer.record("image", np.zeros((8, 8)))
    writer.record("y", 1.0)
    writer.end_point()
    measurement = load_measurement(writer.file, results=["y"])
    assert list(measurement.results) == ["y"]
    assert measurement.result_shapes == {"image": (8, 8), "y": ()}
    assert measurement.result_kinds == {"image": "f", "y": "f"}
    writer.close()
