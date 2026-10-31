import pytest

from pytweezer.experiment import Experiment, Integer, ListAxis, Scan, run_local
from pytweezer.experiment.client import _resolve
from pytweezer.experiment.scan import Point
from pytweezer.experiments.demo import RabiDemo


class Counts(Experiment):
    level = Integer(3, min=1)

    def planned_points(self, scan):
        if scan.axes or scan.repetitions != 1:
            raise ValueError("Counts sets its own points")
        return [Point(i, 0, (), {}) for i in range(self.level)]

    def run_point(self):
        self.record("i", self.point.index)


def test_an_ordinary_experiment_runs_its_scan():
    scan = Scan(axes=[ListAxis(argument="pulse_time", values=[1e-6, 2e-6, 3e-6])])
    assert run_local(RabiDemo, scan, point_delay=0).n_done == 3


def test_an_experiment_can_set_its_own_number_of_points():
    measurement = run_local(Counts, level=5)
    assert measurement.n_done == 5
    assert list(measurement.results["i"]) == [0, 1, 2, 3, 4]


def test_a_scan_an_experiment_refuses_is_refused_at_submission():
    with pytest.raises(ValueError, match="sets its own points"):
        _resolve(Counts, Scan(repetitions=2))
