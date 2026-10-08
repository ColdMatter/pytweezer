import json

import pytest

from pytweezer.experiment import (
    Bool,
    Choice,
    Device,
    Experiment,
    Integer,
    ListAxis,
    Number,
    Scan,
    Text,
    run_local,
)
from pytweezer.experiment.arguments import coerce_arguments


class Base(Experiment):
    detuning = Number(-12e6, unit="MHz", scale=1e6, min=-50e6, max=0)
    shots = Integer(10, min=1)
    camera = Device("Rb ThorCam")


class Child(Base):
    shots = Integer(5, min=1)
    imaging = Bool(True)
    mode = Choice(["absorption", "fluorescence"])
    note = Text()

    def __init__(self, args=None):
        raise AssertionError("introspection must not instantiate")


def test_number_coerces_and_enforces_limits():
    assert Base.detuning.coerce(-1) == -1.0
    with pytest.raises(ValueError, match="above the maximum"):
        Base.detuning.coerce(1e6)
    with pytest.raises(ValueError, match="not a number"):
        Base.detuning.coerce("1")
    with pytest.raises(ValueError, match="not a number"):
        Base.detuning.coerce(True)
    with pytest.raises(ValueError, match="not finite"):
        Base.detuning.coerce(float("nan"))


def test_integer_rejects_fractions_but_accepts_whole_floats():
    assert Base.shots.coerce(3.0) == 3
    assert isinstance(Base.shots.coerce(3.0), int)
    with pytest.raises(ValueError, match="whole number"):
        Base.shots.coerce(2.5)
    with pytest.raises(ValueError, match="minimum"):
        Base.shots.coerce(0)


def test_bool_choice_text():
    assert Child.imaging.coerce(1) is True
    with pytest.raises(ValueError):
        Child.imaging.coerce("yes")
    assert Child.mode.default == "absorption"
    with pytest.raises(ValueError, match="not one of"):
        Child.mode.coerce("other")
    with pytest.raises(ValueError):
        Child.note.coerce(3)


def test_bad_default_is_rejected_at_declaration():
    with pytest.raises(ValueError):
        Number(5, max=1)


def test_schema_is_read_from_the_class_without_instantiating():
    schema = Child.schema()
    json.dumps(schema)
    assert list(schema["arguments"]) == ["detuning", "shots", "imaging", "mode", "note"]
    assert schema["arguments"]["shots"]["default"] == 5
    assert schema["arguments"]["detuning"] | {} == {
        "kind": "number",
        "default": -12e6,
        "tooltip": "",
        "group": "",
        "unit": "MHz",
        "scale": 1e6,
        "min": -50e6,
        "max": 0,
        "step": None,
        "ndecimals": None,
    }
    assert schema["devices"] == {"camera": {"device": "Rb ThorCam", "timeout": None}}


def test_coerce_arguments_fills_defaults_and_rejects_unknown_names():
    assert coerce_arguments(Base, {"shots": 2.0}) == {"detuning": -12e6, "shots": 2}
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Base, {"shot": 2})


def test_instance_attributes_hold_effective_values():
    class Plain(Base):
        def run_point(self):
            pass

    experiment = Plain({"shots": 4})
    assert experiment.shots == 4
    assert Plain.shots.default == 10
    assert experiment.argument_values() == {"detuning": -12e6, "shots": 4}


def test_device_is_resolved_lazily_once_and_closed(monkeypatch):
    opened = []

    class FakeClient:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close_rpc(self):
            self.closed = True

    def fake_get_device(name, timeout):
        opened.append(FakeClient(name))
        return opened[-1]

    monkeypatch.setattr("pytweezer.experiment.experiment._get_device", fake_get_device)

    class Plain(Base):
        def run_point(self):
            pass

    experiment = Plain()
    assert opened == []
    assert experiment.camera is experiment.camera
    assert experiment.device("Rb ThorCam") is experiment.camera
    extra = experiment.device("Rb ThorCam", fresh=True)
    assert extra is not experiment.camera
    assert len(opened) == 2

    experiment.close_devices()
    assert all(client.closed for client in opened)
    assert "camera" not in experiment.__dict__


class Dotted(Experiment):
    base = Number(1.0)

    @classmethod
    def extra_argument(cls, name):
        if name.startswith("x."):
            argument = Number(0.0)
            argument.name = name
            return argument
        return None

    def run_point(self):
        self.record("seen", self.__dict__["x.a"])


def test_hook_names_coerce_and_unknown_names_still_fail():
    assert coerce_arguments(Dotted, {"x.a": 3})["x.a"] == 3.0
    with pytest.raises(ValueError, match="no argument"):
        coerce_arguments(Dotted, {"y.a": 3})


def test_hook_names_can_be_scanned_and_are_stored():
    scan = Scan(axes=[ListAxis(argument="x.a", values=[1, 2])])
    measurement = run_local(Dotted, scan, **{"x.b": 5})
    assert measurement.status == "completed", measurement.attrs["error"]
    assert list(measurement.points["x.a"]) == [1.0, 2.0]
    assert list(measurement.results["seen"]) == [1.0, 2.0]
    assert measurement.arguments["x.b"] == 5.0


def test_scanning_an_unknown_name_still_fails():
    scan = Scan(axes=[ListAxis(argument="y.a", values=[1])])
    with pytest.raises(ValueError, match="no argument 'y.a'"):
        scan.points(Dotted)
