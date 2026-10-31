import math
from typing import ClassVar

import pytest

from pytweezer.experiment import (
    Experiment,
    Integer,
    ListAxis,
    Scan,
    TaskStatus,
    load_measurement,
    run_local,
)
from pytweezer.experiment.client import _resolve, recipes, save_recipe
from pytweezer.experiment.runner import open_writer, run_points
from pytweezer.experiment.scan import Point
from pytweezer.experiment.task import Action
from pytweezer.experiments.demo import DemoStagedScan, RabiDemo
from tests.helpers import in_process_client


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


class SpyStagedScan(DemoStagedScan):
    calls: ClassVar[list[dict]] = []

    def measure(self, **settings):
        SpyStagedScan.calls.append(settings)
        return super().measure(**settings)


class LowestSetting(DemoStagedScan):
    def best(self, settings, objectives):
        return min(settings)


def test_stage_two_runs_at_stage_ones_best_value():
    SpyStagedScan.calls.clear()
    m = run_local(SpyStagedScan, detuning_points=11, power_points=4)
    best_detuning = m.constants["proposed.demo_resonance"]
    assert best_detuning == pytest.approx(1.0, abs=1.0)
    assert [c["detuning"] for c in SpyStagedScan.calls[11:]] == [best_detuning] * 4


def test_the_default_best_averages_repeats_and_ignores_nan():
    optimiser = DemoStagedScan()
    assert optimiser.best([1, 1, 2, 2], [10, 0, 8, 8]) == 2
    assert optimiser.best([1, 1, 2, 2], [10, math.nan, 8, 8]) == 1


def test_an_overridden_best_replaces_the_default():
    m = run_local(LowestSetting, detuning_points=5, power_points=3)
    assert m.constants["proposed.demo_resonance"] == -5.0
    assert m.constants["proposed.demo_power"] == pytest.approx(0.1)


def test_a_single_point_stage_proposes_its_value():
    m = run_local(
        DemoStagedScan,
        detuning_start=0.5,
        detuning_stop=0.5,
        detuning_points=1,
        power_points=2,
    )
    assert m.constants["proposed.demo_resonance"] == 0.5


def test_the_points_argument_sets_the_number_of_points():
    assert (
        run_local(DemoStagedScan, detuning_points=3, power_points=2, repeats=2).n_done
        == 10
    )


def _run_until(done_limit, **args):
    experiment, points, writer = open_writer(
        None, DemoStagedScan, args, Scan(), {"rid": -1, "label": ""}
    )

    def control(progress):
        return Action.TERMINATE if progress.done >= done_limit else Action.CONTINUE

    try:
        status, error = run_points(experiment, points, writer, control)
        assert (status, error) == (TaskStatus.TERMINATED, None)
        return load_measurement(writer.file)
    finally:
        writer.close()


def test_a_run_terminated_in_stage_two_proposes_only_stage_one():
    m = _run_until(7, detuning_points=5, power_points=4)
    assert "proposed.demo_resonance" in m.constants
    assert "proposed.demo_power" not in m.constants
    assert (m.constants["stages_done"], m.constants["stages_total"]) == (1, 2)


def test_a_run_terminated_in_stage_one_proposes_nothing():
    m = _run_until(2, detuning_points=5, power_points=4)
    assert not [name for name in m.constants if name.startswith("proposed.")]
    assert m.constants["stages_done"] == 0


def test_a_recipe_stores_the_generated_arguments(make_manager):
    client = in_process_client(make_manager())
    save_recipe(DemoStagedScan, "quick", client=client, detuning_points=5)
    [recipe] = recipes(DemoStagedScan, client=client)
    assert recipe.args["detuning_points"] == 5


def test_the_schema_lists_generated_arguments_and_proposals():
    schema = DemoStagedScan.schema()
    assert schema["arguments"]["detuning_points"]["default"] == 21
    assert schema["optimiser"] == {
        "kind": "staged",
        "proposes": ["demo_resonance", "demo_power"],
    }
