import math
from typing import ClassVar

import numpy as np
import pytest
import skopt

from pytweezer.experiment import (
    Experiment,
    Integer,
    ListAxis,
    Scan,
    TaskStatus,
    load_measurement,
    optimisers,
    run_local,
)
from pytweezer.experiment.client import _resolve, recipes, save_recipe
from pytweezer.experiment.optimisers import (
    BayesianOptimisation,
    Dimension,
    Fit,
    Stage,
    StagedScan,
)
from pytweezer.experiment.runner import open_writer, run_points
from pytweezer.experiment.scan import Point
from pytweezer.experiment.task import Action
from pytweezer.experiments.demo_resonance import DemoResonance
from pytweezer.experiments.rabi_demo import RabiDemo
from pytweezer.optimisers.demo_bayesian import DemoBayesian
from pytweezer.optimisers.demo_staged_scan import DemoStagedScan
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


class LowestDetuningHighestPower(DemoStagedScan):
    def best(self, settings, objectives):
        return min(settings) if self.stage.setting == "detuning" else max(settings)


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


def test_an_overridden_best_replaces_the_default_and_knows_its_stage():
    m = run_local(LowestDetuningHighestPower, detuning_points=5, power_points=3)
    assert m.constants["proposed.demo_resonance"] == -5.0
    assert m.constants["proposed.demo_power"] == pytest.approx(1.0)


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


def _run_until(done_limit, experiment_cls=DemoStagedScan, **args):
    experiment, points, writer = open_writer(
        None, experiment_cls, args, Scan(), {"rid": -1, "label": ""}
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


class SilentStage(StagedScan, DemoResonance):
    objective = "counts"
    stages: ClassVar[list[Stage]] = [
        Stage(
            "detuning",
            start=-5.0,
            stop=5.0,
            points=3,
            save_as="demo_resonance",
            run=False,
        ),
    ]


def test_a_stage_switched_off_is_not_scanned():
    m = run_local(DemoStagedScan, power_run=False)
    assert set(m.results["stage"]) == {0}
    assert "proposed.demo_resonance" in m.constants
    assert "proposed.demo_power" not in m.constants
    assert m.constants["stages_total"] == 1


def test_a_later_stage_runs_at_the_best_of_the_enabled_earlier_ones():
    SpyStagedScan.calls.clear()
    run_local(SpyStagedScan, detuning_run=False, power_points=4)
    assert len(SpyStagedScan.calls) == 4
    assert all(set(call) == {"power"} for call in SpyStagedScan.calls)


def test_a_run_with_no_stage_enabled_is_refused():
    m = run_local(DemoStagedScan, detuning_run=False, power_run=False)
    assert m.status == "failed"
    assert "detuning_run" in m.attrs["error"]
    assert "AttributeError" not in m.attrs["error"]


def test_stage_run_defaults_follow_the_declaration():
    assert SilentStage.schema()["arguments"]["detuning_run"]["default"] is False
    assert DemoStagedScan.schema()["arguments"]["detuning_run"]["default"] is True


class AroundSavedDetuning(StagedScan, DemoResonance):
    objective = "counts"
    stages: ClassVar[list[Stage]] = [
        Stage("detuning", span=2.0, points=5, save_as="demo_resonance"),
    ]


@pytest.fixture
def saved_resonance(monkeypatch):
    monkeypatch.setattr(
        "pytweezer.experiment.client.get_saved_parameters",
        lambda: {"demo_resonance": 1.0},
    )


def test_a_span_stage_scans_around_its_saved_value(saved_resonance):
    m = run_local(AroundSavedDetuning)
    assert m.status == "completed", m.attrs["error"]
    assert list(m.results["setting"]) == [-1, 0, 1, 2, 3]


def test_a_span_stage_scans_around_a_chosen_centre(monkeypatch):
    def unread():
        raise AssertionError("the saved parameter was read")

    monkeypatch.setattr("pytweezer.experiment.client.get_saved_parameters", unread)
    m = run_local(AroundSavedDetuning, detuning_centre=10.0)
    assert m.status == "completed", m.attrs["error"]
    assert list(m.results["setting"]) == [8, 9, 10, 11, 12]


def test_the_centre_used_is_in_the_file(saved_resonance):
    m = run_local(AroundSavedDetuning)
    assert m.arguments["detuning_centre"] == 1.0


def test_a_span_stage_without_its_saved_parameter_names_the_key(monkeypatch):
    monkeypatch.setattr("pytweezer.experiment.client.get_saved_parameters", dict)
    with pytest.raises(KeyError, match="demo_resonance"):
        open_writer(None, AroundSavedDetuning, {}, Scan(), {"rid": -1, "label": ""})


def test_a_span_stage_has_a_span_and_a_centre_but_no_start_or_stop():
    arguments = AroundSavedDetuning.schema()["arguments"]
    assert arguments["detuning_span"]["default"] == 2.0
    assert arguments["detuning_centre"]["saved"] == "demo_resonance"
    assert "detuning_start" not in arguments
    assert "detuning_stop" not in arguments


@pytest.mark.parametrize(
    "stage",
    [
        {"span": 1.0, "start": 0.0, "stop": 1.0},
        {},
        {"span": 0.0},
        {"span": -1.0},
        {"start": 0.0},
        {"stop": 1.0},
    ],
    ids=[
        "span and range",
        "neither",
        "zero span",
        "negative span",
        "only start",
        "only stop",
    ],
)
def test_bad_span_declarations_are_refused(stage):
    with pytest.raises(TypeError, match="detuning"):
        _staged([Stage("detuning", points=3, save_as="a", **stage)])


class NanOnThirdCall(DemoBayesian):
    def prepare(self):
        self.calls = 0
        super().prepare()

    def measure(self, **settings):
        self.calls += 1
        if self.calls == 3:
            return {"counts": math.nan}
        return super().measure(**settings)


def test_bayesian_finds_the_simulated_resonance():
    m = run_local(DemoBayesian, n_calls=25, seed=1)
    assert m.constants["proposed.demo_resonance"] == pytest.approx(1.3, abs=0.5)
    assert m.constants["evaluations_done"] == 25


def test_bayesian_proposes_the_best_measured_point():
    m = run_local(DemoBayesian, n_calls=6, n_initial=3)
    best = int(m.results["objective"].argmax())
    assert m.constants["proposed.demo_resonance"] == m.results["detuning"][best]
    assert m.constants["proposed.demo_power"] == m.results["power"][best]


class AroundSavedResonance(DemoBayesian):
    dimensions: ClassVar[list[Dimension]] = [
        Dimension("detuning", span=0.5, save_as="demo_resonance"),
        Dimension("power", low=0.1, high=1.0, save_as="demo_power"),
    ]


def test_a_spanned_dimension_searches_only_around_its_saved_value(monkeypatch):
    monkeypatch.setattr(
        optimisers, "get_saved_parameters", lambda: {"demo_resonance": 2.0}
    )
    m = run_local(AroundSavedResonance, n_calls=8, n_initial=4, detuning_span=0.25)
    assert m.status == "completed", m.attrs["error"]
    assert np.all(np.abs(m.results["detuning"] - 2.0) <= 0.25)
    assert list(m.constants["search_bounds"]["detuning"]) == [1.75, 2.25]


class LowestCounts(DemoBayesian):
    goal = "min"


def test_a_bayesian_minimiser_proposes_the_lowest_measured_point():
    m = run_local(LowestCounts, n_calls=6, n_initial=3)
    best = int(m.results["objective"].argmin())
    assert m.constants["proposed.demo_resonance"] == m.results["detuning"][best]
    assert m.constants["proposed.demo_power"] == m.results["power"][best]


def test_xi_is_annealed_from_start_to_end():
    m = run_local(DemoBayesian, n_calls=5, n_initial=2)
    assert list(m.results["xi"]) == pytest.approx(
        [0.1, 0.1 / 10**0.5, 0.01, 0.01 / 10**0.5, 0.001]
    )


def test_each_points_xi_is_the_one_that_chose_it(monkeypatch):
    tell = skopt.Optimizer._tell
    chosen = []

    def spy(optimizer, *args, **kwargs):
        result = tell(optimizer, *args, **kwargs)
        if hasattr(optimizer, "_next_x"):
            xi = optimizer.acq_func_kwargs["xi"]
            chosen.append((xi, [float(v) for v in optimizer._next_x]))
        return result

    monkeypatch.setattr(skopt.Optimizer, "_tell", spy)
    m = run_local(DemoBayesian, n_calls=4, n_initial=2)
    recorded = [
        (m.results["xi"][i], [m.results["detuning"][i], m.results["power"][i]])
        for i in (2, 3)
    ]
    assert recorded == chosen[:2]
    assert recorded[-1][0] == pytest.approx(0.001)


def test_a_nan_objective_does_not_stop_the_run():
    m = run_local(NanOnThirdCall, n_calls=6, n_initial=3)
    assert m.status == TaskStatus.COMPLETED
    assert math.isnan(m.results["objective"][2])
    assert math.isfinite(m.constants["proposed.demo_resonance"])
    assert math.isfinite(m.constants["proposed.demo_power"])


def test_a_bayesian_run_terminated_before_any_evaluation_proposes_nothing():
    m = _run_until(0, DemoBayesian, n_calls=5)
    assert not [name for name in m.constants if name.startswith("proposed.")]
    assert (m.constants["evaluations_done"], m.constants["evaluations_total"]) == (0, 5)


def test_a_recipe_stores_the_generated_arguments(make_manager):
    client = in_process_client(make_manager())
    save_recipe(DemoStagedScan, "quick", client=client, detuning_points=5)
    [recipe] = recipes(DemoStagedScan, client=client)
    assert recipe.args["detuning_points"] == 5


@pytest.mark.parametrize(
    ("optimiser", "argument", "default", "kind"),
    [
        (DemoStagedScan, "detuning_points", 21, "staged"),
        (DemoBayesian, "detuning_low", -5.0, "bayesian"),
    ],
)
def test_the_schema_lists_generated_arguments_and_proposals(
    optimiser, argument, default, kind
):
    schema = optimiser.schema()
    assert schema["arguments"][argument]["default"] == default
    assert schema["optimiser"] == {
        "kind": kind,
        "goal": "max",
        "objective": "counts",
        "proposes": ["demo_resonance", "demo_power"],
        "settings": ["detuning", "power"],
        "units": {"demo_resonance": "", "demo_power": ""},
        "group": "",
    }


class SharedBase(StagedScan, DemoResonance):
    objective = "counts"


def test_a_base_without_stages_is_not_listed_as_an_optimiser():
    assert "optimiser" not in SharedBase.schema()


def _staged(stages, goal="max"):
    return type("Bad", (StagedScan, DemoResonance), {"stages": stages, "goal": goal})


def _bayesian(dimensions):
    return type(
        "Bad", (BayesianOptimisation, DemoResonance), {"dimensions": dimensions}
    )


@pytest.mark.parametrize(
    ("define", "message"),
    [
        (
            lambda: _staged(
                [
                    Stage("detuning", start=0, stop=1, points=3, save_as="a"),
                    Stage("detuning", start=0, stop=1, points=3, save_as="b"),
                ]
            ),
            "detuning",
        ),
        (
            lambda: _staged(
                [
                    Stage("detuning", start=0, stop=1, points=3, save_as="a"),
                    Stage("power", start=0, stop=1, points=3, save_as="a"),
                ]
            ),
            "'a'",
        ),
        (
            lambda: _staged(
                [Stage("detuning", start=0, stop=1, points=3, save_as="a")],
                goal="maximum",
            ),
            "maximum",
        ),
        (
            lambda: _bayesian([Dimension("detuning", low=1, high=1, save_as="a")]),
            "detuning",
        ),
        (
            lambda: _bayesian(
                [Dimension("detuning", low=0, high=1, span=0.5, save_as="a")]
            ),
            "both a span and low/high",
        ),
    ],
    ids=[
        "duplicate setting",
        "duplicate save_as",
        "unknown goal",
        "empty range",
        "span and range",
    ],
)
def test_a_bad_declaration_is_refused_when_the_class_is_defined(define, message):
    with pytest.raises(TypeError, match=message):
        define()


class DemoFit(Fit, Experiment):
    proposal_units: ClassVar[dict[str, str]] = {"demo_offset": "V", "demo_grid": "px"}
    proposal_group = "Demo"
    fails = Integer(0)

    def fit(self):
        if self.fails:
            raise RuntimeError("no fit")
        self.record("residual", 0.25)
        return {"demo_offset": 1.5, "demo_grid": np.ones((3, 2))}


def test_a_fit_runs_one_point_and_proposes_what_it_returns():
    measurement = run_local(DemoFit)
    assert measurement.status == "completed", measurement.attrs.get("error")
    assert len(measurement.points["index"]) == 1
    assert measurement.results["residual"][0] == 0.25
    assert measurement.constants["proposed.demo_offset"] == 1.5
    np.testing.assert_array_equal(
        measurement.constants["proposed.demo_grid"], np.ones((3, 2))
    )


def test_a_fit_that_raises_proposes_nothing():
    measurement = run_local(DemoFit, fails=1)
    assert measurement.status == "failed"
    assert not any(key.startswith("proposed.") for key in measurement.constants)


def test_the_schema_gives_a_fits_proposals_with_their_units_and_group():
    assert DemoFit.schema()["optimiser"] == {
        "kind": "fit",
        "goal": "max",
        "objective": None,
        "proposes": ["demo_offset", "demo_grid"],
        "settings": [],
        "units": {"demo_offset": "V", "demo_grid": "px"},
        "group": "Demo",
    }
