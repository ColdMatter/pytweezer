from itertools import pairwise

import pytest
from pydantic import ValidationError

from pytweezer.experiment import (
    Choice,
    Experiment,
    Integer,
    LinearAxis,
    ListAxis,
    Number,
    Scan,
)


class Grid(Experiment):
    a = Number(0.0)
    b = Number(0.0)
    n = Integer(1)
    mode = Choice(["x", "y"])


def _two_by_three(**kwargs):
    return Scan(
        axes=[
            ListAxis(argument="a", values=[0, 1]),
            ListAxis(argument="b", values=[0, 1, 2]),
        ],
        **kwargs,
    )


def indices(points):
    return [p.axis_indices for p in points]


def test_no_axes_is_a_single_point_per_repetition():
    points = Scan(repetitions=3).points(Grid)
    assert [(p.index, p.repetition, p.values) for p in points] == [
        (0, 0, {}),
        (1, 1, {}),
        (2, 2, {}),
    ]


def test_nested_first_axis_outermost():
    assert indices(_two_by_three().points(Grid)) == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
    ]  # fmt: skip


def test_snake_changes_one_axis_by_one_step():
    points = Scan(
        axes=[
            ListAxis(argument="a", values=[0, 1]),
            ListAxis(argument="b", values=[0, 1]),
            ListAxis(argument="n", values=[1, 2, 3]),
        ],
        order="snake",
    ).points(Grid)
    seq = indices(points)
    assert len(set(seq)) == 12
    for before, after in pairwise(seq):
        assert sum(abs(x - y) for x, y in zip(before, after)) == 1


def test_shuffle_gets_a_seed_and_is_reproducible():
    scan = _two_by_three(order="shuffle")
    assert scan.seed is not None
    again = Scan.model_validate_json(scan.model_dump_json())
    assert indices(scan.points(Grid)) == indices(again.points(Grid))
    assert sorted(indices(scan.points(Grid))) == indices(_two_by_three().points(Grid))


def test_repeat_point_vs_scan():
    by_point = _two_by_three(repetitions=2, repeat="point").points(Grid)
    assert [(p.repetition, p.axis_indices) for p in by_point[:3]] == [
        (0, (0, 0)), (1, (0, 0)), (0, (0, 1)),
    ]  # fmt: skip
    by_scan = _two_by_three(repetitions=2, repeat="scan").points(Grid)
    assert [p.repetition for p in by_scan] == [0] * 6 + [1] * 6
    assert [p.index for p in by_scan] == list(range(12))


def test_values_are_coerced_against_the_declared_argument():
    points = Scan(axes=[ListAxis(argument="mode", values=["y", "x"])]).points(Grid)
    assert [p.values for p in points] == [{"mode": "y"}, {"mode": "x"}]
    with pytest.raises(ValueError, match="not one of"):
        Scan(axes=[ListAxis(argument="mode", values=["z"])]).points(Grid)


def test_linear_integer_axis_rounds_and_rejects_duplicates():
    points = Scan(axes=[LinearAxis(argument="n", start=1, stop=5, n=3)]).points(Grid)
    assert [p.values["n"] for p in points] == [1, 3, 5]
    with pytest.raises(ValueError, match="repeats values"):
        Scan(axes=[LinearAxis(argument="n", start=1, stop=2, n=5)]).points(Grid)


def test_unknown_or_duplicate_axes_are_rejected():
    with pytest.raises(ValueError, match="no argument 'c'"):
        Scan(axes=[ListAxis(argument="c", values=[1])]).points(Grid)
    with pytest.raises(ValidationError, match="scanned twice"):
        Scan(
            axes=[
                ListAxis(argument="a", values=[1]),
                ListAxis(argument="a", values=[2]),
            ]
        )


def test_json_round_trip_keeps_axis_kinds():
    scan = Scan(
        axes=[
            LinearAxis(argument="a", start=0, stop=1, n=5),
            ListAxis(argument="b", values=[3, 4]),
        ],
        repetitions=2,
    )
    again = Scan.model_validate_json(scan.model_dump_json())
    assert again == scan
    assert isinstance(again.axes[0], LinearAxis)
