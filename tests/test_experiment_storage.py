from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from pytweezer.experiment import Experiment, ListAxis, Number, Scan, load_measurement
from pytweezer.experiment.runner import open_writer
from pytweezer.experiment.storage import (
    RecordError,
    highest_rid,
    measurement_relpath,
    read_planned_points,
)


class Sweep(Experiment):
    x = Number(0.0, unit="V")

    def run_point(self):
        pass


SCAN = Scan(axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])])


def make_writer(path=None):
    """Returns ``(points, writer)`` for a three-point scan of ``Sweep``."""
    _, points, writer = open_writer(path, Sweep, {}, SCAN, {"rid": 7, "label": "t"})
    return points, writer


def test_header_and_planned_points_and_no_overwrite(tmp_path):
    path = tmp_path / "m.h5"
    make_writer(path)[1].close()
    with h5py.File(path, "r") as f:
        assert f.attrs["rid"] == 7
        assert f.attrs["class_name"] == "Sweep"
        assert f.attrs["status"] == "running"
        assert f.attrs["n_points"] == 3
        assert list(f["points/x"][:]) == [1.0, 2.0, 3.0]
        assert "test_experiment_storage" in next(iter(f["source"]))
    with pytest.raises(FileExistsError):
        make_writer(path)


def test_partial_file_is_readable_mid_run(tmp_path):
    path = tmp_path / "m.h5"
    points, writer = make_writer(path)
    writer.begin_point(points[0])
    writer.record("y", 1.5, unit="counts")
    writer.end_point()
    writer.begin_point(points[1])
    writer.record("y", 2.5)
    # point 1 not ended: a reader sees only point 0, but every planned point
    measurement = load_measurement(path)
    assert measurement.status == "running"
    assert measurement.n_done == 1
    assert list(measurement.results["y"]) == [1.5]
    assert measurement.units["y"] == "counts"
    assert list(measurement.points["x"]) == [1.0]
    planned = read_planned_points(path)
    assert list(planned["x"]) == [1.0, 2.0, 3.0]
    assert list(planned["index"]) == [0, 1, 2]
    writer.close()


def test_missing_values_are_flagged_and_float_gaps_are_nan():
    points, writer = make_writer()
    for point in points:
        writer.begin_point(point)
        if point.index != 1:
            writer.record("count", point.index)
            writer.record("y", float(point.index))
        writer.end_point()
    measurement = load_measurement(writer.file)
    assert measurement.results["count"].dtype.kind == "i"
    assert list(measurement.recorded["count"]) == [True, False, True]
    assert np.isnan(measurement.results["y"][1])
    assert measurement.results["y"][2] == 2.0
    writer.close()


def test_complex_results_read_back_with_nan_gaps():
    points, writer = make_writer()
    for point in points:
        writer.begin_point(point)
        if point.index != 1:
            writer.record("amplitude", complex(point.index, -0.5))
        writer.end_point()
    amplitude = load_measurement(writer.file).results["amplitude"]
    assert amplitude[0] == complex(0, -0.5)
    assert np.isnan(amplitude[1].real) and np.isnan(amplitude[1].imag)
    assert amplitude[2] == complex(2, -0.5)
    writer.close()


def test_a_result_keeps_its_first_shape_and_dtype():
    points, writer = make_writer()
    writer.begin_point(points[0])
    writer.record("image", np.ones((2, 3), dtype=np.uint16))
    writer.record("count", 3)
    writer.end_point()
    writer.begin_point(points[1])
    with pytest.raises(RecordError, match="shape"):
        writer.record("image", np.ones((3, 2), dtype=np.uint16))
    with pytest.raises(RecordError, match="uint16"):
        writer.record("image", np.ones((2, 3), dtype=np.int32))
    with pytest.raises(RecordError, match="dtype int64"):
        writer.record("count", 2.5)
    writer.record("count", np.uint8(4))  # widening is fine
    with pytest.raises(RecordError, match="twice"):
        writer.record("count", 5)
    assert writer.file["results/image"].dtype == np.uint16
    writer.close()


def test_strings_dicts_and_constants():
    points, writer = make_writer()
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
        writer.record("reference", 1)
    writer.close()


@pytest.mark.parametrize(
    "first, second",
    [({"a": 1}, "plain"), ("plain", {"a": 1})],
    ids=["dict-str", "str-dict"],
)
def test_a_text_result_cannot_switch_between_string_and_dict(first, second):
    points, writer = make_writer()
    writer.begin_point(points[0])
    writer.record("note", first)
    writer.end_point()
    writer.begin_point(points[1])
    with pytest.raises(RecordError, match="note"):
        writer.record("note", second)
    writer.end_point()
    assert load_measurement(writer.file).results["note"] == [first, None]
    writer.close()


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("", 1, "invalid record name"),
        ("a/b", 1, "invalid record name"),
        ("__schema__", 1, "invalid record name"),
        ("obj", object(), "unsupported type"),
    ],
)
def test_bad_records_are_refused(name, value, message):
    points, writer = make_writer()
    writer.begin_point(points[0])
    with pytest.raises(RecordError, match=message):
        writer.record(name, value)
    writer.close()


def test_paths_and_rid_scan(tmp_path):
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


def test_load_only_selected_results_but_know_all_shapes():
    points, writer = make_writer()
    writer.begin_point(points[0])
    writer.record("image", np.zeros((8, 8)))
    writer.record("y", 1.0)
    writer.end_point()
    measurement = load_measurement(writer.file, results=["y"])
    assert list(measurement.results) == ["y"]
    assert measurement.result_shapes == {"image": (8, 8), "y": ()}
    assert measurement.result_kinds == {"image": "f", "y": "f"}
    writer.close()
