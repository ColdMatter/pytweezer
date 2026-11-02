"""Dataset: getting a measurement's points and results out for analysis."""

import numpy as np
import pytest

from pytweezer.experiment import Dataset, Experiment, ListAxis, Number, Scan
from pytweezer.experiment.runner import open_writer, run_points
from pytweezer.experiment.task import Action


class Seen(Experiment):
    x = Number(0.0)

    def run_point(self):
        self.record("seen_x", self.x)
        self.record("count", 10 * self.point.index)
        self.record("image", np.full((2, 2), self.point.index, dtype=np.uint16))
        self.record("note", f"point {self.point.index}")


POINTS_RUN = 4
POINT_COLUMNS = ["index", "repetition", "x", "t_start", "t_end"]


@pytest.fixture
def dataset(tmp_path):
    """A shuffled 3-value scan with 2 repetitions, terminated after 4 of its 6 points."""
    scan = Scan(
        axes=[ListAxis(argument="x", values=[1.0, 2.0, 3.0])],
        repetitions=2,
        order="shuffle",
    )
    path = tmp_path / "m.h5"
    experiment, points, writer = open_writer(path, Seen, {}, scan, {"rid": 3})

    def control(progress):
        return Action.TERMINATE if progress.done == POINTS_RUN else Action.CONTINUE

    run_points(experiment, points, writer, control)
    writer.close()
    return Dataset(path)


def test_dataframe_has_a_row_per_point_run_in_run_order(dataset):
    frame = dataset.get_dataframe()
    assert list(frame.columns[:5]) == POINT_COLUMNS
    assert set(frame.columns[5:]) == {"seen_x", "count", "note"}
    assert list(frame["index"]) == list(range(POINTS_RUN))
    assert list(frame["x"]) == list(frame["seen_x"])
    assert list(frame["count"]) == [0, 10, 20, 30]
    # >= because Windows' clock can stamp consecutive points identically.
    assert (frame["t_start"].diff().dropna() >= 0).all()


def test_dict_stacks_each_points_array_result_in_run_order(dataset):
    images = dataset.get_dict()["image"]
    assert images.shape == (POINTS_RUN, 2, 2)
    for index, image in enumerate(images):
        np.testing.assert_array_equal(image, np.full((2, 2), index))


def test_array_holds_only_numeric_per_point_values(dataset):
    array = dataset.get_array()
    assert list(array.dtype.names[:5]) == POINT_COLUMNS
    assert set(array.dtype.names[5:]) == {"seen_x", "count"}
    assert list(array["count"]) == [0, 10, 20, 30]
