import numpy as np
import pytest

from pytweezer.GUI.viewers.live_plot import ResultHistory

LIMIT = 100


def point(rid, index, scalars, **values):
    return {"rid": rid, "index": index, "values": values, "scalars": scalars}


def history(*points, limit=LIMIT):
    results = ResultHistory()
    for p in points:
        results.add(p, limit)
    return results


def test_results_accumulate_across_tasks_against_the_newest_scanned_argument():
    results = history(
        point(1, 0, {"counts": 5.0}, detuning=2e6),
        point(2, 0, {"counts": 7.0}, detuning=-1e6),
    )
    argument = results.x_argument({"counts"})
    x, y, rids = results.curve("counts", argument)
    assert argument == "detuning"
    np.testing.assert_array_equal(x, [2e6, -1e6])
    np.testing.assert_array_equal(y, [5.0, 7.0])
    assert rids == {1, 2}


def test_points_that_scanned_another_argument_are_left_out():
    results = history(
        point(1, 0, {"counts": 5.0}, power=1.0),
        point(2, 0, {"counts": 7.0}, detuning=-1e6),
    )
    x, y, rids = results.curve("counts", results.x_argument({"counts"}))
    np.testing.assert_array_equal(y, [7.0])
    assert rids == {2}


def test_the_x_axis_follows_the_newest_point_carrying_a_chosen_result():
    results = history(
        point(1, 0, {"counts": 5.0}, detuning=2e6),
        point(2, 0, {"temperature": 1e-5}, power=1.0),
    )
    assert results.x_argument({"counts"}) == "detuning"


@pytest.mark.parametrize(
    "values", [{}, {"mode": "fast"}], ids=["nothing scanned", "non-numeric scan"]
)
def test_without_a_numeric_scan_results_are_drawn_in_arrival_order(values):
    results = history(
        point(1, 3, {"counts": 1.0}, detuning=2e6),
        point(2, 0, {"counts": 2.0}, **values),
    )
    argument = results.x_argument({"counts"})
    x, y, _ = results.curve("counts", argument)
    assert argument is None
    np.testing.assert_array_equal(x, [0, 1])
    np.testing.assert_array_equal(y, [1.0, 2.0])


def test_points_without_the_result_are_left_out():
    results = history(
        point(1, 0, {"counts": 5.0}, t=1.0),
        point(1, 1, {"other": 9.0}, t=2.0),
        point(1, 2, {"counts": 6.0}, t=3.0),
    )
    x, y, _ = results.curve("counts", "t")
    np.testing.assert_array_equal(x, [1.0, 3.0])
    np.testing.assert_array_equal(y, [5.0, 6.0])


def test_a_point_delivered_twice_is_kept_once():
    results = history(point(1, 0, {"counts": 5.0}), point(1, 0, {"counts": 5.0}))
    _, y, _ = results.curve("counts", None)
    np.testing.assert_array_equal(y, [5.0])


def test_only_the_newest_points_are_kept_beyond_the_history_limit():
    results = history(*(point(1, i, {"counts": float(i)}) for i in range(5)), limit=3)
    _, y, _ = results.curve("counts", None)
    np.testing.assert_array_equal(y, [2.0, 3.0, 4.0])
